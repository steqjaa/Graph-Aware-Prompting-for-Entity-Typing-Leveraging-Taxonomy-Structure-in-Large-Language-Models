import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": (
                "Entity: Trabocchi Coast\n"
                "Description: place in Abruzzo, Italy\n\n"
                "What is the most specific entity type?\n"
                "Answer with only the entity type."
            ),
        }
    ],
    temperature=0,
    max_completion_tokens=100,
)

answer = response.choices[0].message.content

print("Llama 3.3 70B answer:")
print(answer)