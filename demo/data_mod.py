from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from datasets import load_dataset
from tqdm import tqdm

import argparse
import json
import os
import asyncio
import time
from datetime import datetime

import numpy as np
import pandas as pd

load_dotenv()

project_id = os.getenv("SLURM_JOB_ACCOUNT", "project_462000131")
user = os.getenv("USER")

DATASET_NAME="AGBonnet/augmented-clinical-notes"

# All data files of the pipeline (source dataset cache, generated splits,
# models, predictions, metrics) live under this per-user directory.
DATA_ROOT=f"/scratch/{project_id}/{user}/data"
DATASET_CACHE_DIR=f"{DATA_ROOT}/"

# Columns written to train/validation/test. Only "conversation" and
# "structured_note" are needed for training and evaluation; "idx" identifies
# the source row and "full_note" is kept as reference.
SPLIT_COLUMNS = ["idx", "conversation", "structured_note", "full_note"]

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

# ---------------------------------------------------------------------------
# Saving and splitting
# ---------------------------------------------------------------------------

def save_jsonl(frame: pd.DataFrame, path: str) -> None:
    """One JSON object per line (JSON Lines)."""
    frame.to_json(path, orient="records", lines=True, force_ascii=False)


def split_dataframe(frame: pd.DataFrame, val_size: float, test_size: float, seed: int) -> dict:
    """Random train/validation/test split, reproducible with the seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))

    n_test = int(round(len(frame) * test_size))
    n_val = int(round(len(frame) * val_size))

    test_rows = np.sort(perm[:n_test])
    val_rows = np.sort(perm[n_test:n_test + n_val])
    train_rows = np.sort(perm[n_test + n_val:])

    return {
        "train": frame.iloc[train_rows].reset_index(drop=True),
        "validation": frame.iloc[val_rows].reset_index(drop=True),
        "test": frame.iloc[test_rows].reset_index(drop=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Model name to use for LLM inference")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing notes")

    parser.add_argument("--backend", type=str, default=None, help="Whether to use LLM hosted by own vllm server")

    parser.add_argument("--out_name", type=str, default="structured_notes",
                        help=f"Name of the output directory under {DATA_ROOT}")

    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of rows for the validation split")

    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of rows for the test split")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")

    parser.add_argument("--max_rows", type=int, default=None,
                        help="Process only the first N rows (for testing)")

    parser.add_argument("--api-url")

    args, _ = parser.parse_known_args()

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Test run: using only the first {len(df)} rows")

    if not 0 <= args.val_size + args.test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")

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

    out_dir = os.path.join(DATA_ROOT, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1) Save all rows first, so the expensive LLM run is never lost ─────
    # text_for_llm is left out: it repeats the same system prompt on every
    # row. The prompt is stored once in dataset_info.json instead.
    full_path = os.path.join(out_dir, "structured_notes_full.jsonl")
    save_jsonl(df.drop(columns=["text_for_llm"]), full_path)
    print(f"\nSaved all {len(df)} rows -> {full_path}")

    # ── 2) Drop failed LLM calls (empty structured_note) ───────────────────
    empty = df["structured_note"].fillna("").str.strip() == ""
    n_empty = int(empty.sum())

    keep_cols = [c for c in SPLIT_COLUMNS if c in df.columns]
    for required in ("conversation", "structured_note"):
        if required not in keep_cols:
            raise KeyError(f"Required column '{required}' missing. Columns: {list(df.columns)}")

    df_clean = df.loc[~empty, keep_cols].reset_index(drop=True)

    # ── 3) Split ───────────────────────────────────────────────────────────
    splits = split_dataframe(df_clean, args.val_size, args.test_size, args.seed)

    # ── 4) Save splits ─────────────────────────────────────────────────────
    for name, split_df in splits.items():
        save_jsonl(split_df, os.path.join(out_dir, f"{name}.jsonl"))

    # ── 5) Save and print dataset information ──────────────────────────────
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": DATASET_NAME,
        "llm_model": args.model,
        "seed": args.seed,
        "columns": keep_cols,
        "rows_total": len(df),
        "rows_dropped_empty": n_empty,
        "splits": {name: len(split_df) for name, split_df in splits.items()},
    }

    info_path = os.path.join(out_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nDataset info:")
    print(json.dumps(info, indent=2))
    print(f"Saved -> {info_path}")