from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv


# =========================================================
# Configuration
# =========================================================

DEFAULT_PROMPT_FOLDER = Path("generated_ranked_prompts")

# Cloudflare-hosted Meta Llama 3.1 8B Instruct.
DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"

TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 800
PAUSE_SECONDS = 1
MAX_RETRIES = 3


# =========================================================
# Helpers
# =========================================================

def safe_model_name(model):
    """
    Convert a model name into something safe to use as a folder name.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def normalize_label(text):
    """
    Normalize a predicted label so duplicate labels can be detected.
    """
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)

    return text.strip(
        " \t\n\r.,;:!?\"'"
    )


def deduplicate_labels(labels):
    """
    Remove duplicate predicted labels while preserving their ranking.
    """
    unique = []
    seen = set()

    for label in labels:
        if not isinstance(label, str):
            continue

        cleaned = label.strip()
        normalized = normalize_label(cleaned)

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(cleaned)

    return unique


def extract_json_object(text):
   
    stripped = text.strip()

    # Remove a Markdown JSON code fence if the model used one.
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        stripped = fenced.group(1)

    # First try parsing the complete response.
    try:
        value = json.loads(stripped)

        if isinstance(value, dict):
            return value

        return None

    except json.JSONDecodeError:
        pass

    # Otherwise search for the first plausible JSON object.
    match = re.search(
        r"\{.*\}",
        stripped,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        value = json.loads(match.group(0))

        if isinstance(value, dict):
            return value

        return None

    except json.JSONDecodeError:
        return None


def extract_ranked_types(response_text):
    """
    Extract the ranked entity types from the model response.
    """
    parsed = extract_json_object(response_text)

    if parsed is not None:
        for key in (
            "types",
            "entity_types",
            "predictions",
        ):
            value = parsed.get(key)

            if isinstance(value, list):
                return deduplicate_labels(value)

    # Fallback in case the model returns a numbered or bulleted list.
    labels = []

    for line in response_text.splitlines():
        match = re.match(
            r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$",
            line,
        )

        if match:
            labels.append(
                match.group(1)
            )

    return deduplicate_labels(labels)


def extract_explanation(response_text):
    """
    Extract the taxonomy-grounded explanation from CoT responses.

    Standard prompts do not contain an explanation, so this will simply
    return an empty string for them.
    """
    parsed = extract_json_object(response_text)

    if parsed is None:
        return ""

    explanation = parsed.get(
        "explanation",
        "",
    )

    if isinstance(explanation, str):
        return explanation.strip()

    return ""


def request_with_retries(
    account_id,
    token,
    model,
    prompt,
):
    
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )

    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_COMPLETION_TOKENS,
        }
    ).encode("utf-8")

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": (
                        f"Bearer {token}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                },
                method="POST",
            )

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                payload = json.load(
                    response
                )

            if not payload.get(
                "success",
                False,
            ):
                raise RuntimeError(
                    "Cloudflare API error: "
                    f"{payload.get('errors', [])}"
                )

            result = payload.get(
                "result",
                {},
            )

            # Cloudflare currently returns an OpenAI-compatible
            # choices structure for this Llama endpoint.
            choices = result.get(
                "choices",
                [],
            )

            if choices:
                response_text = (
                    choices[0]
                    .get("message", {})
                    .get("content", "")
                )

            else:
                # Fallback for Cloudflare responses that use
                # the simpler "response" field.
                response_value = result.get(
                    "response",
                    "",
                )

                if isinstance(
                    response_value,
                    str,
                ):
                    response_text = (
                        response_value
                    )

                else:
                    response_text = json.dumps(
                        response_value,
                        ensure_ascii=False,
                    )

            return {
                "response_text": (
                    response_text.strip()
                ),
                "usage": result.get(
                    "usage",
                    {},
                ),
                "provider_model": (
                    result.get(
                        "model",
                        model,
                    )
                ),
            }

        except urllib.error.HTTPError as error:
            # Try to retrieve Cloudflare's actual error message.
            try:
                error_body = (
                    error.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                error_body = ""

            if error_body:
                last_error = RuntimeError(
                    f"HTTP {error.code}: "
                    f"{error_body}"
                )
            else:
                last_error = error

        except Exception as error:
            last_error = error

        if attempt == MAX_RETRIES:
            break

        wait_seconds = (
            10 * attempt
        )

        print(
            f"  Request failed on attempt "
            f"{attempt}. "
            f"Waiting {wait_seconds} seconds."
        )

        print(
            " ",
            last_error,
        )

        time.sleep(
            wait_seconds
        )

    raise last_error


# =========================================================
# Main program
# =========================================================

def main():
    load_dotenv(
        ".env",
        override=True,
    )

    # -----------------------------------------------------
    # Cloudflare authentication
    # -----------------------------------------------------

    account_id = os.getenv(
        "CLOUDFLARE_ACCOUNT_ID"
    )

    token = os.getenv(
        "CLOUDFLARE_AUTH_TOKEN"
    )

    if not account_id:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID was not found. "
            "Check your .env file."
        )

    if not token:
        raise RuntimeError(
            "CLOUDFLARE_AUTH_TOKEN was not found. "
            "Check your .env file."
        )

    # -----------------------------------------------------
    # Input prompt folder
    # -----------------------------------------------------

    prompt_folder = Path(
        os.getenv(
            "PROMPT_FOLDER",
            str(
                DEFAULT_PROMPT_FOLDER
            ),
        )
    )

    manifest_file = (
        prompt_folder
        / "manifest.csv"
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_file}\n"
            "Run generate_ranked_prompts.py "
            "first or set PROMPT_FOLDER."
        )

    # -----------------------------------------------------
    # Model and run configuration
    # -----------------------------------------------------

    model = os.getenv(
        "CLOUDFLARE_MODEL",
        DEFAULT_MODEL,
    )

    run_tag = os.getenv(
        "RUN_TAG",
        "",
    ).strip()

    # Optional testing limit.
    #
    # Example:
    #
    # MAX_CALLS=1 python3 run_ranked_prompts.py
    #
    # If MAX_CALLS is not supplied, all prompts are run.
    max_calls_text = os.getenv(
        "MAX_CALLS",
        "",
    ).strip()

    if max_calls_text:
        try:
            max_calls = int(
                max_calls_text
            )

            if max_calls <= 0:
                raise ValueError

        except ValueError:
            raise RuntimeError(
                "MAX_CALLS must be a "
                "positive integer."
            )

    else:
        max_calls = None

    # -----------------------------------------------------
    # Result directory
    # -----------------------------------------------------

    folder_name = safe_model_name(
        model
    )

    if run_tag:
        folder_name += (
            "__"
            + safe_model_name(
                run_tag
            )
        )

    result_folder = (
        Path(
            "generated_ranked_results"
        )
        / folder_name
    )

    result_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    with open(
        manifest_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        manifest_rows = list(
            csv.DictReader(file)
        )

    if max_calls is not None:
        rows_to_process = (
            manifest_rows[:max_calls]
        )
    else:
        rows_to_process = (
            manifest_rows
        )

    print(
        "Cloudflare endpoint:",
        model,
    )

    print(
        "Prompt folder:",
        prompt_folder,
    )

    print(
        "Prompts to process:",
        len(rows_to_process),
    )

    print()

    summary_rows = []

    # -----------------------------------------------------
    # Run prompts
    # -----------------------------------------------------

    for call_number, row in enumerate(
        rows_to_process,
        start=1,
    ):
        example_number = int(
            row["example_number"]
        )

        condition = row[
            "condition"
        ]

        prompt_path = Path(
            row["prompt_path"]
        )

        metadata_path = Path(
            row["metadata_path"]
        )

        result_subfolder = (
            result_folder
            / (
                f"example_"
                f"{example_number:02d}"
            )
        )

        result_subfolder.mkdir(
            exist_ok=True
        )

        result_json_path = (
            result_subfolder
            / (
                f"{condition}"
                "_result.json"
            )
        )

        response_text_path = (
            result_subfolder
            / (
                f"{condition}"
                "_response.txt"
            )
        )

        print(
            "=" * 70
        )

        print(
            f"Call "
            f"{call_number}/"
            f"{len(rows_to_process)} "
            f"| Example "
            f"{example_number:02d} "
            f"| {condition}"
        )

        print(
            "Endpoint:",
            model,
        )

        print(
            "Entity:",
            row["entity_label"],
        )

        # -------------------------------------------------
        # Skip already completed prompts
        # -------------------------------------------------

        if result_json_path.exists():
            print(
                "  Existing result found; "
                "skipping."
            )

            existing = json.loads(
                result_json_path.read_text(
                    encoding="utf-8"
                )
            )

            summary_rows.append(
                {
                    "example_number": (
                        example_number
                    ),
                    "entity_id": (
                        row["entity_id"]
                    ),
                    "entity_label": (
                        row["entity_label"]
                    ),
                    "condition": (
                        condition
                    ),
                    "model": model,
                    "provider_model": (
                        existing.get(
                            "provider_model",
                            "",
                        )
                    ),
                    "predictions_json": (
                        json.dumps(
                            existing.get(
                                "predictions",
                                [],
                            ),
                            ensure_ascii=False,
                        )
                    ),
                    "neurons": (
                        existing.get(
                            "neurons",
                            "",
                        )
                    ),
                    "status": (
                        existing.get(
                            "status",
                            "completed",
                        )
                    ),
                    "error": (
                        existing.get(
                            "error",
                            "",
                        )
                    ),
                }
            )

            continue

        # -------------------------------------------------
        # Read prompt + metadata
        # -------------------------------------------------

        prompt = (
            prompt_path.read_text(
                encoding="utf-8"
            )
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        # -------------------------------------------------
        # Request
        # -------------------------------------------------

        try:
            response = (
                request_with_retries(
                    account_id,
                    token,
                    model,
                    prompt,
                )
            )

            response_text = response[
                "response_text"
            ]

            predictions = (
                extract_ranked_types(
                    response_text
                )
            )

            explanation = (
                extract_explanation(
                    response_text
                )
            )

            usage = response[
                "usage"
            ]

            provider_model = response[
                "provider_model"
            ]

            # ---------------------------------------------
            # Save complete result
            # ---------------------------------------------

            result_record = {
                "timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "status": (
                    "completed"
                ),
                "model": model,
                "provider_model": (
                    provider_model
                ),
                "run_tag": (
                    run_tag
                ),
                "prompt_folder": (
                    str(prompt_folder)
                ),
                "temperature": (
                    TEMPERATURE
                ),
                "max_completion_tokens": (
                    MAX_COMPLETION_TOKENS
                ),
                "example_number": (
                    example_number
                ),
                "entity_id": (
                    metadata["entity_id"]
                ),
                "entity_label": (
                    metadata[
                        "entity_label"
                    ]
                ),
                "description": (
                    metadata[
                        "description"
                    ]
                ),
                "condition": (
                    condition
                ),
                "prompt": prompt,
                "raw_response": (
                    response_text
                ),
                "predictions": (
                    predictions
                ),
                "explanation": (
                    explanation
                ),
                "metadata_path": (
                    str(metadata_path)
                ),
                "prompt_tokens": (
                    usage.get(
                        "prompt_tokens"
                    )
                ),
                "completion_tokens": (
                    usage.get(
                        "completion_tokens"
                    )
                ),
                "total_tokens": (
                    usage.get(
                        "total_tokens"
                    )
                ),
                "neurons": (
                    usage.get(
                        "neurons"
                    )
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

            # ---------------------------------------------
            # Terminal output
            # ---------------------------------------------

            print(
                "Provider model:",
                provider_model,
            )

            if explanation:
                print(
                    "Explanation:",
                    explanation,
                )

            print(
                "Predictions:"
            )

            for rank, prediction in enumerate(
                predictions,
                start=1,
            ):
                print(
                    f"  {rank}. "
                    f"{prediction}"
                )

            if usage.get(
                "neurons"
            ) is not None:
                print(
                    "Neurons:",
                    usage.get(
                        "neurons"
                    ),
                )

            summary_rows.append(
                {
                    "example_number": (
                        example_number
                    ),
                    "entity_id": (
                        metadata[
                            "entity_id"
                        ]
                    ),
                    "entity_label": (
                        metadata[
                            "entity_label"
                        ]
                    ),
                    "condition": (
                        condition
                    ),
                    "model": model,
                    "provider_model": (
                        provider_model
                    ),
                    "predictions_json": (
                        json.dumps(
                            predictions,
                            ensure_ascii=False,
                        )
                    ),
                    "neurons": (
                        usage.get(
                            "neurons",
                            "",
                        )
                    ),
                    "status": (
                        "completed"
                    ),
                    "error": "",
                }
            )

        # -------------------------------------------------
        # Failed request
        # -------------------------------------------------

        except Exception as error:
            error_text = str(
                error
            )

            print(
                "Request failed permanently:"
            )

            print(
                error_text
            )

            result_record = {
                "timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "status": (
                    "failed"
                ),
                "model": model,
                "provider_model": "",
                "run_tag": (
                    run_tag
                ),
                "prompt_folder": (
                    str(prompt_folder)
                ),
                "example_number": (
                    example_number
                ),
                "entity_id": (
                    metadata[
                        "entity_id"
                    ]
                ),
                "entity_label": (
                    metadata[
                        "entity_label"
                    ]
                ),
                "condition": (
                    condition
                ),
                "metadata_path": (
                    str(metadata_path)
                ),
                "error": (
                    error_text
                ),
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
                    "example_number": (
                        example_number
                    ),
                    "entity_id": (
                        metadata[
                            "entity_id"
                        ]
                    ),
                    "entity_label": (
                        metadata[
                            "entity_label"
                        ]
                    ),
                    "condition": (
                        condition
                    ),
                    "model": model,
                    "provider_model": "",
                    "predictions_json": (
                        "[]"
                    ),
                    "neurons": "",
                    "status": (
                        "failed"
                    ),
                    "error": (
                        error_text
                    ),
                }
            )

        # Small delay between requests.
        time.sleep(
            PAUSE_SECONDS
        )

    # -----------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------

    summary_path = (
        result_folder
        / "raw_summary.csv"
    )

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
            "provider_model",
            "predictions_json",
            "neurons",
            "status",
            "error",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print(
        "All requested ranked prompts "
        "have been processed."
    )

    print(
        "Cloudflare endpoint:",
        model,
    )

    print(
        "Prompt folder:",
        prompt_folder,
    )

    print(
        "Run tag:",
        run_tag or "(none)",
    )

    print(
        "Results:",
        result_folder,
    )

    print(
        "Raw summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()