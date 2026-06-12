from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from datasets import load_dataset
from tqdm import tqdm

import argparse
import os
import asyncio
import time
import pandas as pd

load_dotenv()

project_id = os.getenv("SLURM_JOB_ACCOUNT", "project_462000131")

DATASET_NAME="AGBonnet/augmented-clinical-notes"
DATASET_CACHE_DIR=f"/scratch/{project_id}/data/"

dataset = load_dataset(DATASET_NAME, cache_dir=DATASET_CACHE_DIR)

df = dataset["train"].to_pandas()

main_text_col = df.full_note

def messages_for_llm(text_to_include: str):
    return [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text":
            """
You are a clinical documentation assistant. Your task is to convert unstructured clinical notes into a standardized structured format.

**Instructions:**
- Extract relevant information from the input clinical note.
- Organize the information into the predefined sections below.
- Use clear, concise medical language.
- If information is missing, write: "Not specified".
- Preserve clinical meaning and terminology.
- Avoid duplication across sections.
- Infer future details in to a follow-up plan.

**Output Format**
Return the output strictly in the following structure:

REASON FOR VISIT:
<Brief summary of why the patient is seeking care>

PATIENT DETAILS AND HISTORY:
<Age, gender, relevant demographics, relevant past medical history, conditions, medications, surgeries, lifestyle factors>

CURRENT STATUS:
<Current symptoms, findings, vitals, clinical observations>

TREATMENTS/ACTIONS:
<Medications prescribed, procedures performed, advice given>

FOLLOW-UP PLAN:
<Next steps, monitoring, referrals, timelines. Follow-up plan should not include "future" details that are mentioned in the note, but rather should infer what the next steps would be based on the found future details.>

**Additional rules**
- Normalize vague expressions:
    - “a few days ago” → keep as-is (do not convert to exact dates)
- Keep clinical abbreviations if commonly used (e.g., BP, HR), but ensure clarity.
- Do not include personal opinions or interpretations beyond the note.
- Do not use bullet points but rather full sentences."""
        }]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Here is the clinical note: {text_to_include}"
            }
        ]
    }
]

df["text_for_llm"] = main_text_col.apply(messages_for_llm)

async def process_note(msgs_for_llm, model, max_retries=5):
    delay = 1

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()

            response = await openai_client.chat.completions.create(
                model=model,
                messages=msgs_for_llm,
            )

            latency = time.perf_counter() - start_time

            # --- SAFE EXTRACTION ---
            output_text = ""
            if (
                response
                and getattr(response, "choices", None)
                and len(response.choices) > 0
                and response.choices[0].message
            ):
                output_text = response.choices[0].message.content or ""

            usage = getattr(response, "usage", None)

            return {
                "input_text": msgs_for_llm,
                "output_text": output_text,
                "latency_sec": latency,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            
            # If API provides retry-after, use it
            retry_after = getattr(e, "retry_after", None)

            wait_time = retry_after if retry_after else delay
            print(f"Rate limited. Waiting {wait_time} seconds...")

            await asyncio.sleep(wait_time)

            delay *= 2  # exponential backoff

        except Exception as e:
            # Log and return safe fallback instead of crashing whole batch
            print(f"Error: {e}")

            return {
                "input_text": msgs_for_llm,
                "output_text": "",
                "latency_sec": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

async def process_batch(batch, model):
    tasks = [process_note(note, model) for note in batch]
    return await asyncio.gather(*tasks)

async def main(batch_size, model):
    all_results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]['text_for_llm'].tolist()
        results = await process_batch(batch, model)
        all_results.extend(results)
        if i == 0:
            print(results[0])

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Model name to use for LLM inference")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing notes")

    parser.add_argument("--backend", type=str, default=None, help="Whether to use LLM hosted by own vllm server")

    parser.add_argument("--json_name", type=str, help="Name of the output JSON file to save results")

    parser.add_argument("--api-url")

    args, _ = parser.parse_known_args()

    if args.backend == "vllm":
        openai_client = AsyncOpenAI(
            base_url=args.api_url,
            api_key="EMPTY",
        )
    else:
        AITTA_API_URL=os.getenv("AITTA_API_URL")
        AITTA_API_KEY=os.getenv("AITTA_API_KEY")
        openai_client = AsyncOpenAI(
            base_url=AITTA_API_URL,
            api_key=AITTA_API_KEY,
        )

    results = asyncio.run(main(batch_size=args.batch_size, model=args.model))

    df["structured_note"] = [result["output_text"] for result in results]

    df_results = pd.DataFrame(results)

    df.to_json(path_or_buf=os.path.join(DATASET_CACHE_DIR, args.json_name))

    # df_results.to_csv(os.path.join(DATASET_CACHE_DIR, "health_case_dataset.csv"))