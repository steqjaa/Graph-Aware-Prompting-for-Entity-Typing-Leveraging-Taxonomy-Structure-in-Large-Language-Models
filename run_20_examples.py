from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# Configuration
# =========================================================

MODEL = "llama-3.3-70b-versatile"

PROMPT_FOLDER = Path("generated_20_prompts")
MANIFEST_FILE = PROMPT_FOLDER / "manifest.csv"
RESULT_FOLDER = Path("generated_20_results")

TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 400

# 20 entities x 4 prompt conditions = 80 API calls.
PAUSE_SECONDS = 3
MAX_RETRIES = 3


# =========================================================
# Helpers
# =========================================================

def normalize_label(text):
    """
    Normalize a short label for an exact-label comparison.

    This does not solve synonym differences such as coast/coastline.
    Those still require manual semantic evaluation.
    """
    text = text.strip().lower()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = text.strip(" \t\n\r.,;:!?\"'")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_prediction(response_text):
    """
    Try to extract the final entity type from either a direct answer
    or an Explanation/Answer response.
    """
    answer_matches = re.findall(
        r"(?im)^\s*answer\s*:\s*(.+?)\s*$",
        response_text,
    )

    if answer_matches:
        return answer_matches[-1].strip()

    nonempty_lines = [
        line.strip()
        for line in response_text.splitlines()
        if line.strip()
    ]

    if not nonempty_lines:
        return ""

    return nonempty_lines[-1]


def request_with_retries(client, prompt):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = 10 * attempt

            print(
                f"  Request failed on attempt {attempt}. "
                f"Waiting {wait_seconds} seconds before retrying."
            )
            print(" ", error)

            time.sleep(wait_seconds)

    raise last_error


# =========================================================
# Main program
# =========================================================

def main():
    load_dotenv(override=True)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Check your .env file."
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run generate_20_examples.py first."
        )

    client = Groq(api_key=api_key)
    RESULT_FOLDER.mkdir(exist_ok=True)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        manifest_rows = list(csv.DictReader(file))

    summary_rows = []

    for call_number, row in enumerate(manifest_rows, start=1):
        example_number = int(row["example_number"])
        condition = row["condition"]
        prompt_path = Path(row["prompt_path"])

        result_subfolder = (
            RESULT_FOLDER / f"example_{example_number:02d}"
        )
        result_subfolder.mkdir(exist_ok=True)

        result_json_path = (
            result_subfolder / f"{condition}_result.json"
        )
        response_text_path = (
            result_subfolder / f"{condition}_response.txt"
        )

        print("=" * 70)
        print(
            f"Call {call_number}/{len(manifest_rows)} | "
            f"Example {example_number:02d} | {condition}"
        )
        print("Entity:", row["entity_label"])

        # Resume safely after interruption.
        if result_json_path.exists():
            print("  Existing result found; skipping this call.")

            existing = json.loads(
                result_json_path.read_text(encoding="utf-8")
            )

            summary_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "description": row["description"],
                    "reference_class_id": row["reference_class_id"],
                    "reference_type": row["reference_type"],
                    "condition": condition,
                    "prediction": existing.get("prediction", ""),
                    "exact_label_match": existing.get(
                        "exact_label_match",
                        False,
                    ),
                    "status": existing.get("status", "completed"),
                    "error": existing.get("error", ""),
                }
            )
            continue

        prompt = prompt_path.read_text(encoding="utf-8")

        try:
            response = request_with_retries(client, prompt)

            response_text = (
                response.choices[0].message.content or ""
            ).strip()

            prediction = extract_prediction(response_text)

            exact_match = (
                normalize_label(prediction)
                == normalize_label(row["reference_type"])
            )

            usage = response.usage

            result_record = {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "completed",
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "example_number": example_number,
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "description": row["description"],
                "reference_class_id": row["reference_class_id"],
                "reference_type": row["reference_type"],
                "condition": condition,
                "prompt": prompt,
                "response": response_text,
                "prediction": prediction,
                "exact_label_match": exact_match,
                "prompt_tokens": getattr(
                    usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    usage,
                    "total_tokens",
                    None,
                ),
                "error": "",
            }

            response_text_path.write_text(
                response_text,
                encoding="utf-8",
            )

            result_json_path.write_text(
                json.dumps(
                    result_record,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print("Prediction:", prediction)
            print("Reference:", row["reference_type"])
            print("Exact label match:", exact_match)

            summary_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "description": row["description"],
                    "reference_class_id": row["reference_class_id"],
                    "reference_type": row["reference_type"],
                    "condition": condition,
                    "prediction": prediction,
                    "exact_label_match": exact_match,
                    "status": "completed",
                    "error": "",
                }
            )

        except Exception as error:
            error_text = str(error)

            print("Request failed permanently:")
            print(error_text)

            result_record = {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "failed",
                "model": MODEL,
                "example_number": example_number,
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "reference_type": row["reference_type"],
                "condition": condition,
                "error": error_text,
            }

            result_json_path.write_text(
                json.dumps(
                    result_record,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "description": row["description"],
                    "reference_class_id": row["reference_class_id"],
                    "reference_type": row["reference_type"],
                    "condition": condition,
                    "prediction": "",
                    "exact_label_match": False,
                    "status": "failed",
                    "error": error_text,
                }
            )

        time.sleep(PAUSE_SECONDS)

    summary_path = RESULT_FOLDER / "summary.csv"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "example_number",
            "entity_id",
            "entity_label",
            "description",
            "reference_class_id",
            "reference_type",
            "condition",
            "prediction",
            "exact_label_match",
            "status",
            "error",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nAll available prompts have been processed.")
    print("Summary:", summary_path)
    print(
        "\nImportant: exact_label_match does not recognize synonyms "
        "such as coast and coastline. Add a separate manual semantic "
        "correctness column during analysis."
    )


if __name__ == "__main__":
    main()