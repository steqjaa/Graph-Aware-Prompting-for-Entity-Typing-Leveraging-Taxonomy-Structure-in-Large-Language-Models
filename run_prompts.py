from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

from dotenv import load_dotenv
from groq import Groq


MODEL = "llama-3.3-70b-versatile"

PROMPT_FOLDER = Path("generated_prompts")
RESULT_FOLDER = Path("generated_results")


# ---------------------------------------------------------
# 1. Load API key
# ---------------------------------------------------------

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

client = Groq(api_key=api_key)


# ---------------------------------------------------------
# 2. Find the generated prompt files
# ---------------------------------------------------------

prompt_files = sorted(PROMPT_FOLDER.glob("*.txt"))

if not prompt_files:
    raise RuntimeError(
        "No prompt files were found in generated_prompts."
    )

RESULT_FOLDER.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 3. Send every prompt to the same model
# ---------------------------------------------------------

for prompt_file in prompt_files:
    prompt = prompt_file.read_text(encoding="utf-8")

    print("=" * 70)
    print("Running:", prompt_file.name)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_completion_tokens=400,
        )

        answer = response.choices[0].message.content.strip()

        print("\nResponse:")
        print(answer)

        # Save the raw text response.
        text_output = RESULT_FOLDER / prompt_file.name.replace(
            "_prompt.txt",
            "_response.txt",
        )

        text_output.write_text(
            answer,
            encoding="utf-8",
        )

        # Save useful experiment metadata separately.
        usage = response.usage

        result_record = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model": MODEL,
            "prompt_file": prompt_file.name,
            "prompt": prompt,
            "response": answer,
            "temperature": 0,
            "max_completion_tokens": 400,
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
        }

        json_output = RESULT_FOLDER / prompt_file.name.replace(
            "_prompt.txt",
            "_result.json",
        )

        json_output.write_text(
            json.dumps(
                result_record,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as error:
        print("\nRequest failed:")
        print(error)

    # A small pause helps avoid free-tier rate-limit errors.
    time.sleep(2)


print("\nAll available prompts have been processed.")
print("Results saved in:", RESULT_FOLDER)