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

PROMPT_FOLDER = Path("generated_ranked_prompts")
MANIFEST_FILE = PROMPT_FOLDER / "manifest.csv"

# Start with the smaller model. Later run the same prompts with:
# LLAMA_MODEL=llama-3.3-70b-versatile python3 run_ranked_prompts.py
DEFAULT_MODEL = "llama-3.1-8b-instant"

TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 500
TOP_K = 10

PAUSE_SECONDS = 3
MAX_RETRIES = 3


# =========================================================
# Helpers
# =========================================================

def safe_model_name(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def normalize_label(text):
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def deduplicate_labels(labels):
    unique = []
    seen = set()

    for label in labels:
        if not isinstance(label, str):
            continue

        cleaned = label.strip()
        normalized = normalize_label(cleaned)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique.append(cleaned)

        if len(unique) == TOP_K:
            break

    return unique


def extract_json_object(text):
    stripped = text.strip()

    # Remove a single Markdown code fence if present.
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        stripped = fenced.group(1)

    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to the first plausible JSON object.
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)

    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def extract_ranked_types(response_text):
    parsed = extract_json_object(response_text)

    if parsed is not None:
        for key in ("types", "entity_types", "predictions"):
            value = parsed.get(key)

            if isinstance(value, list):
                return deduplicate_labels(value)

    # Fallback for numbered or bulleted lists.
    labels = []

    for line in response_text.splitlines():
        match = re.match(
            r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$",
            line,
        )

        if match:
            labels.append(match.group(1))

    return deduplicate_labels(labels)


def request_with_retries(client, model, prompt):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
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
                f"Waiting {wait_seconds} seconds."
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
            "Run generate_ranked_prompts.py first."
        )

    model = os.getenv("LLAMA_MODEL", DEFAULT_MODEL)

    result_folder = (
        Path("generated_ranked_results")
        / safe_model_name(model)
    )
    result_folder.mkdir(parents=True, exist_ok=True)

    client = Groq(api_key=api_key)

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
        metadata_path = Path(row["metadata_path"])

        result_subfolder = (
            result_folder / f"example_{example_number:02d}"
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
        print("Model:", model)
        print("Entity:", row["entity_label"])

        if result_json_path.exists():
            print("  Existing result found; skipping.")
            existing = json.loads(
                result_json_path.read_text(encoding="utf-8")
            )

            summary_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "condition": condition,
                    "model": model,
                    "predictions_json": json.dumps(
                        existing.get("predictions", []),
                        ensure_ascii=False,
                    ),
                    "status": existing.get(
                        "status",
                        "completed",
                    ),
                    "error": existing.get("error", ""),
                }
            )
            continue

        prompt = prompt_path.read_text(encoding="utf-8")
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        try:
            response = request_with_retries(
                client,
                model,
                prompt,
            )

            response_text = (
                response.choices[0].message.content or ""
            ).strip()

            predictions = extract_ranked_types(response_text)
            usage = response.usage

            result_record = {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "status": "completed",
                "model": model,
                "temperature": TEMPERATURE,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "example_number": example_number,
                "entity_id": metadata["entity_id"],
                "entity_label": metadata["entity_label"],
                "description": metadata["description"],
                "condition": condition,
                "prompt": prompt,
                "raw_response": response_text,
                "predictions": predictions,
                "metadata_path": str(metadata_path),
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

            print("Predictions:")
            for rank, prediction in enumerate(
                predictions,
                start=1,
            ):
                print(f"  {rank}. {prediction}")

            if len(predictions) != TOP_K:
                print(
                    f"Warning: parsed {len(predictions)} "
                    f"predictions instead of {TOP_K}."
                )

            summary_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": metadata["entity_id"],
                    "entity_label": metadata["entity_label"],
                    "condition": condition,
                    "model": model,
                    "predictions_json": json.dumps(
                        predictions,
                        ensure_ascii=False,
                    ),
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
                "model": model,
                "example_number": example_number,
                "entity_id": metadata["entity_id"],
                "entity_label": metadata["entity_label"],
                "condition": condition,
                "metadata_path": str(metadata_path),
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
                    "entity_id": metadata["entity_id"],
                    "entity_label": metadata["entity_label"],
                    "condition": condition,
                    "model": model,
                    "predictions_json": "[]",
                    "status": "failed",
                    "error": error_text,
                }
            )

        time.sleep(PAUSE_SECONDS)

    summary_path = result_folder / "raw_summary.csv"

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
            "condition",
            "model",
            "predictions_json",
            "status",
            "error",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nAll available ranked prompts have been processed.")
    print("Model:", model)
    print("Results:", result_folder)
    print("Raw summary:", summary_path)


if __name__ == "__main__":
    main()