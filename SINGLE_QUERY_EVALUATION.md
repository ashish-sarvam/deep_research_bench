# Single Query Evaluation Guide

This guide explains how to use the custom `evaluate_single_query.py` script to evaluate a specific query from the DeepResearch Bench.

## Overview

The `evaluate_single_query.py` script allows you to:
- Evaluate a single query by ID (e.g., query 55) instead of running the full benchmark
- Use your own generated reports in any directory structure
- Get detailed RACE evaluation scores for your specific query

## Prerequisites

1. **API Key**: Set your Gemini API key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

2. **Report Format**: Your report file should be in JSONL format with entries like:
   ```json
   {"id": 55, "prompt": "your query text", "article": "your generated report"}
   ```

## Usage

### Basic Usage

```bash
python evaluate_single_query.py --query_id 55 --report_dir /path/to/your/reports --model_name your-model-name
```

### Parameters

- `--query_id`: The ID of the query you want to evaluate (required)
- `--report_dir`: Directory containing your report file, or path to the report file itself (required)
- `--model_name`: Name of your model for output file naming (optional, default: "custom-model")
- `--output_dir`: Directory to save results (optional, default: "results/single_query")
- `--verbose`: Enable detailed logging (optional)

### Examples

#### Example 1: Evaluate query 55 with existing data
```bash
python evaluate_single_query.py \
  --query_id 55 \
  --report_dir data/test_data/raw_data \
  --model_name claude-3-7-sonnet-latest
```

#### Example 2: Evaluate with your custom report file
```bash
python evaluate_single_query.py \
  --query_id 55 \
  --report_dir /path/to/your/custom_reports.jsonl \
  --model_name my-custom-model \
  --output_dir my_evaluation_results
```

#### Example 3: Verbose evaluation
```bash
python evaluate_single_query.py \
  --query_id 55 \
  --report_dir data/test_data/raw_data \
  --model_name claude-3-7-sonnet-latest \
  --verbose
```

## Report File Formats

The script can automatically find your report in several ways:

### Option 1: Directory with model-named file
```
your_reports/
├── your-model-name.jsonl
```

### Option 2: Directory with standard names
```
your_reports/
├── reports.jsonl
├── output.jsonl
```

### Option 3: Direct file path
```bash
--report_dir /path/to/your/specific/report.jsonl
```

## Output

The script generates two output files:

### 1. Detailed Results (`query_55_detailed_results.json`)
Contains complete evaluation data including:
- Individual dimension scores
- Raw LLM response
- Detailed scoring breakdown

### 2. Summary Results (`query_55_summary.txt`)
Contains a human-readable summary:
```
Evaluation Results for Query 55
==================================================

Query: While the market features diverse quantitative strategies...

Comprehensiveness: 0.7234
Insight: 0.6891
Instruction Following: 0.8123
Readability: 0.7456
Overall Score: 0.7426
```

## Console Output

The script also prints results directly to the console:
```
==================================================
EVALUATION RESULTS FOR QUERY 55
==================================================
Comprehensiveness:      0.7234
Insight:                0.6891
Instruction Following:  0.8123
Readability:            0.7456
Overall Score:          0.7426
==================================================
```

## Troubleshooting

### Common Issues

1. **"Query ID not found"**
   - Make sure the query ID exists in `data/prompt_data/query.jsonl`
   - Valid query IDs are 1-100

2. **"Report not found"**
   - Check that your report file contains an entry with the correct ID
   - The script will also try to match by prompt text if ID matching fails

3. **"GEMINI_API_KEY not set"**
   - Set your API key: `export GEMINI_API_KEY="your_key"`

4. **"Evaluation criteria not found"**
   - This means the query ID doesn't have corresponding evaluation criteria
   - Check `data/criteria_data/criteria.jsonl`

### Verbose Mode

Use `--verbose` flag to see detailed logging:
```bash
python evaluate_single_query.py --query_id 55 --report_dir your_reports --verbose
```

## Query 55 Specific Information

Query 55 is an English query about developing evaluation frameworks for quantitative trading strategies:

**Query Text**: "While the market features diverse quantitative strategies like multi-factor and high-frequency trading, it lacks a single, standardized benchmark for assessing their performance across multiple dimensions such as returns, risk, and adaptability to market conditions. Could we develop a general yet rigorous evaluation framework to enable accurate comparison and analysis of various advanced quant strategies?"

**Language**: English
**Topic**: Finance & Business

## Integration with Existing Workflow

This script is designed to work alongside the existing DeepResearch Bench infrastructure:
- Uses the same evaluation criteria and reference data
- Follows the same RACE methodology
- Produces comparable scores to the full benchmark

You can use this for rapid iteration and testing of specific queries before running the full evaluation suite.
