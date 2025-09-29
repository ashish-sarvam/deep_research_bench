import json
from pathlib import Path


def txt_to_jsonl(
    txt_path: str, jsonl_path: str, record_id: int = 1, prompt: str = ""
):
    txt_file = Path(txt_path)
    if not txt_file.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    with open(txt_file, "r", encoding="utf-8") as f:
        article_content = f.read().strip()

    record = {
        "id": record_id,
        "prompt": prompt,
        "article": article_content,
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ JSONL written to {jsonl_path}")


# Example usage:
prompt = "Summarize the global investments, key initiatives, and outputs related to Artificial Intelligence (AI) by major international consulting firms (e.g., Big Four, Accenture, MBB, IBM, Capgemini). Cover aspects such as AI-driven products/services, client case studies, application scenarios, strategic directions, and talent development programs."
txt_to_jsonl("57_v2.txt", "report.jsonl", record_id=57, prompt=prompt)
