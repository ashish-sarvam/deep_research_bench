import json
import os
import argparse
import logging
import time

# Set a dummy JINA_API_KEY to avoid import errors (we don't need it for RACE evaluation)
if "JINA_API_KEY" not in os.environ:
    os.environ["JINA_API_KEY"] = "dummy_key_for_import"

from config.llm import LLMClient
from utils.io_utils import load_jsonl
from utils.score_calculator import calculate_weighted_scores
from utils.json_extractor import extract_json_from_markdown

# Import scoring prompts
from prompt.score_prompt_zh import (
    generate_merged_score_prompt as zh_merged_score_prompt,
)
from prompt.score_prompt_en import (
    generate_merged_score_prompt as en_merged_score_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fixed configuration parameters
CRITERIA_FILE = "data/criteria_data/criteria.jsonl"
REFERENCE_FILE = "data/test_data/cleaned_data/reference.jsonl"
QUERY_FILE = "data/prompt_data/query.jsonl"
MAX_RETRIES = 10


def format_criteria_list(criteria_data):
    """Format evaluation criteria list as JSON string, without weight information"""
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})

    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            logger.warning(
                f"Value for dimension '{dim}' is not a list. Skipping."
            )
            continue

        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if (
                isinstance(crit_item, dict)
                and "criterion" in crit_item
                and "explanation" in crit_item
            ):
                criteria_for_prompt[dim].append(
                    {
                        "criterion": crit_item["criterion"],
                        "explanation": crit_item["explanation"],
                    }
                )
            else:
                logger.warning(
                    f"Invalid criteria format in dimension '{dim}'. Skipping item."
                )

    try:
        return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)
    except TypeError as e:
        raise ValueError(f"Failed to serialize criteria to JSON: {e}")


def load_query_data(query_id):
    """Load query data for a specific query ID"""
    logger.info(f"Loading query data for ID {query_id}...")

    # Load query information
    all_queries = load_jsonl(QUERY_FILE)
    query_data = None
    for query in all_queries:
        if query.get("id") == query_id:
            query_data = query
            break

    if not query_data:
        raise ValueError(f"Query ID {query_id} not found in {QUERY_FILE}")

    logger.info(f"Found query {query_id}: {query_data['prompt'][:100]}...")
    return query_data


def load_report_data(report_dir, query_id, query_prompt):
    """Load the generated report for the specific query"""
    logger.info(f"Loading report data from {report_dir}...")

    # Try different possible file patterns
    possible_files = [
        os.path.join(report_dir, "reports.jsonl"),
        os.path.join(report_dir, "output.jsonl"),
        report_dir,  # If report_dir is the file itself
    ]

    # Also check if report_dir itself is a file
    if os.path.isfile(report_dir):
        possible_files.insert(0, report_dir)

    report_data = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            logger.info(f"Checking file: {file_path}")
            try:
                reports = load_jsonl(file_path)
                for report in reports:
                    if report.get("id") == query_id:
                        report_data = report
                        logger.info(
                            f"Found report for query {query_id} in {file_path}"
                        )
                        break
                if report_data:
                    break
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue

    if not report_data:
        # If not found by ID, try to match by prompt
        logger.info("Report not found by ID, trying to match by prompt...")
        for file_path in possible_files:
            if os.path.exists(file_path):
                try:
                    reports = load_jsonl(file_path)
                    for report in reports:
                        if report.get("prompt") == query_prompt:
                            report_data = report
                            logger.info(
                                f"Found report matching prompt in {file_path}"
                            )
                            break
                    if report_data:
                        break
                except Exception as e:
                    continue

    if not report_data:
        raise ValueError(
            f"Report for query ID {query_id} not found in {report_dir}. "
            f"Checked files: {possible_files}"
        )

    return report_data


def load_evaluation_data(query_id, query_prompt):
    """Load criteria and reference data for evaluation"""
    logger.info("Loading evaluation criteria and reference data...")

    # Load criteria
    all_criteria = load_jsonl(CRITERIA_FILE)
    criteria_data = None
    for criteria in all_criteria:
        if (
            criteria.get("id") == query_id
            or criteria.get("prompt") == query_prompt
        ):
            criteria_data = criteria
            break

    if not criteria_data:
        raise ValueError(
            f"Evaluation criteria for query ID {query_id} not found"
        )

    # Load reference data
    all_references = load_jsonl(REFERENCE_FILE)
    reference_data = None
    for reference in all_references:
        if (
            reference.get("id") == query_id
            or reference.get("prompt") == query_prompt
        ):
            reference_data = reference
            break

    if not reference_data:
        raise ValueError(f"Reference data for query ID {query_id} not found")

    return criteria_data, reference_data


def evaluate_single_query(
    query_data, report_data, criteria_data, reference_data, llm_client
):
    """Evaluate a single query using the RACE methodology"""
    query_id = query_data["id"]
    prompt = query_data["prompt"]
    language = query_data.get("language", "en")

    logger.info(
        f"Starting evaluation for query {query_id} (language: {language})"
    )

    target_article = report_data.get("article", "")
    reference_article = reference_data.get("article", "")
    print("target_article == ", target_article[:200])
    print("reference_article == ", reference_article[:200])

    if not target_article:
        raise ValueError("Target article is empty")
    if not reference_article:
        raise ValueError("Reference article is empty")

    # Format evaluation criteria
    try:
        criteria_list_str = format_criteria_list(criteria_data)
    except ValueError as e:
        raise ValueError(f"Failed to format criteria: {str(e)}")

    # Choose scoring prompt based on language
    merged_score_prompt = (
        zh_merged_score_prompt if language == "zh" else en_merged_score_prompt
    )

    # Prepare LLM prompt
    user_prompt = merged_score_prompt.format(
        task_prompt=prompt,
        article_1=target_article,
        article_2=reference_article,
        criteria_list=criteria_list_str,
    )

    # Call LLM for evaluation
    logger.info("Calling LLM for evaluation...")
    success = False
    retry_count = 0

    while retry_count < MAX_RETRIES and not success:
        try:
            llm_response_str = llm_client.generate(
                user_prompt=user_prompt, system_prompt=""
            )

            # Extract JSON from response
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            if not json_str_extracted:
                raise ValueError("Failed to extract JSON from LLM response")

            llm_output_json = json.loads(json_str_extracted)

            # Check if all required dimensions exist
            expected_dims = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            if not all(dim in llm_output_json for dim in expected_dims):
                missing_dims = [
                    dim for dim in expected_dims if dim not in llm_output_json
                ]
                raise ValueError(
                    f"Missing expected dimensions: {missing_dims}"
                )

            success = True

        except Exception as e:
            retry_count += 1
            if retry_count < MAX_RETRIES:
                logger.warning(f"Retry {retry_count}/{MAX_RETRIES} - {str(e)}")
                time.sleep(1.5**retry_count)
            else:
                raise Exception(
                    f"Failed after {MAX_RETRIES} retries - {str(e)}"
                )

    # Calculate weighted scores
    try:
        scores = calculate_weighted_scores(
            llm_output_json, criteria_data, language
        )

        # Calculate overall score = target / (target + reference)
        target_total = scores["target"]["total"]
        reference_total = scores["reference"]["total"]
        overall_score = 0
        if target_total + reference_total > 0:
            overall_score = target_total / (target_total + reference_total)

        # Calculate normalized dimension scores
        normalized_dims = {}
        for dim in [
            "comprehensiveness",
            "insight",
            "instruction_following",
            "readability",
        ]:
            dim_key = f"{dim}_weighted_avg"
            if dim_key in scores["target"]["dims"]:
                target_score = scores["target"]["dims"][dim_key]
                reference_score = scores["reference"]["dims"][dim_key]
                if target_score + reference_score > 0:
                    normalized_dims[dim] = target_score / (
                        target_score + reference_score
                    )
                else:
                    normalized_dims[dim] = 0
            else:
                logger.warning(f"Missing dimension {dim_key} in scores")
                normalized_dims[dim] = 0

    except Exception as e:
        raise Exception(f"Error calculating scores: {str(e)}")

    # Prepare final result
    final_result = {
        "id": query_id,
        "prompt": prompt,
        "comprehensiveness": normalized_dims.get("comprehensiveness", 0),
        "insight": normalized_dims.get("insight", 0),
        "instruction_following": normalized_dims.get(
            "instruction_following", 0
        ),
        "readability": normalized_dims.get("readability", 0),
        "overall_score": overall_score,
        "raw_llm_response": llm_response_str,
        "detailed_scores": scores,
    }

    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single query from DeepResearch Bench"
    )
    parser.add_argument(
        "--query_id",
        type=int,
        required=True,
        help="Query ID to evaluate (e.g., 55)",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help="Directory containing your generated report file, or path to the report file itself",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/single_query",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check API key
    # if not os.getenv("GEMINI_API_KEY"):
    #     logger.error("GEMINI_API_KEY environment variable is required")
    #     return 1

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = LLMClient()

        # Load query data
        query_data = load_query_data(args.query_id)

        # Load report data
        report_data = load_report_data(
            args.report_dir,
            args.query_id,
            query_data["prompt"],
        )

        # Load evaluation data
        criteria_data, reference_data = load_evaluation_data(
            args.query_id, query_data["prompt"]
        )

        # Run evaluation
        result = evaluate_single_query(
            query_data, report_data, criteria_data, reference_data, llm_client
        )

        # Save detailed results
        output_file = os.path.join(
            args.output_dir, f"query_{args.query_id}_detailed_results.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Save summary results
        summary_file = os.path.join(
            args.output_dir, f"query_{args.query_id}_summary.txt"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results for Query {args.query_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Query: {query_data['prompt'][:100]}...\n\n")
            f.write(f"Comprehensiveness: {result['comprehensiveness']:.4f}\n")
            f.write(f"Insight: {result['insight']:.4f}\n")
            f.write(
                f"Instruction Following: {result['instruction_following']:.4f}\n"
            )
            f.write(f"Readability: {result['readability']:.4f}\n")
            f.write(f"Overall Score: {result['overall_score']:.4f}\n")

        # Print results
        logger.info("\n" + "=" * 50)
        logger.info(f"EVALUATION RESULTS FOR QUERY {args.query_id}")
        logger.info("=" * 50)
        logger.info(
            f"Comprehensiveness:      {result['comprehensiveness']:.4f}"
        )
        logger.info(f"Insight:                {result['insight']:.4f}")
        logger.info(
            f"Instruction Following:  {result['instruction_following']:.4f}"
        )
        logger.info(f"Readability:            {result['readability']:.4f}")
        logger.info(f"Overall Score:          {result['overall_score']:.4f}")
        logger.info("=" * 50)
        logger.info(f"Detailed results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
