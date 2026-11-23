"""
Simple isolated test for LLM generation latency.
Benchmarks multiple OpenAI models on the same input using the same timing method as app.py.
No memory, no context, no retrieval.

Usage:
    python .testFiles/test_generation_latency.py
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=API_KEY)

USER_MESSAGE = "Quand j'étais petit, mon film préféré, c'était « Retour vers le futur »."

MODELS_TO_TEST = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano"
]

MAX_TOKENS = 150

print(f"Input: {USER_MESSAGE}")
print(f"{'─' * 80}")
print(f"{'Model':<20} {'Time (s)':>8}  {'In tok':>7}  {'Out tok':>7}  Answer")
print(f"{'─' * 80}")

for model in MODELS_TO_TEST:
    try:
        now = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": USER_MESSAGE}],
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - now

        answer = completion.choices[0].message.content.replace("\n", " ")
        in_tok = completion.usage.prompt_tokens
        out_tok = completion.usage.completion_tokens

        # Truncate answer for display
        short_answer = answer[:60] + "…" if len(answer) > 60 else answer
        print(f"{model:<20} {elapsed:>8.2f}  {in_tok:>7}  {out_tok:>7}  {short_answer}")

    except Exception as e:
        print(f"{model:<20} {'ERROR':>8}  {'─':>7}  {'─':>7}  {e}")

print(f"{'─' * 80}")

