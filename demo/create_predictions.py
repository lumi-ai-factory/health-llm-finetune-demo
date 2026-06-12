import os
import json
import gc
import time
import argparse
from datetime import datetime

import torch
from vllm import LLM, SamplingParams


OUTPUT_DIR      = "/scratch/project_462001520/demo/inference" 
TENSOR_PARALLEL = 4


SYSTEM_PROMPT = """You are a medical clinical documentation assistant. 
You task is to convert a dialogue between a doctor and patient into a structured clinical note in the following output format:
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
"""


SAMPLING = dict(temperature=0.5, max_tokens=1024)

# Helper functions

def load_json(filepath: str) -> list:
    with open(filepath, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(data)} examples from {filepath}")
    return data


def save_json(data, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {filepath}")


def build_messages(example: dict) -> list[dict]:
    """Build chat messages from a val example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["conversation"]},
    ]


def run_inference(model_path: str, val_data: list[dict],
                  tensor_parallel: int) -> list[str]:
    """Load one model, generate predictions, free GPU memory, return texts."""
    print(f"\n  Loading: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
    )
    print(f"  Model loaded in {round(time.time() - t0, 1)}s")

    all_messages    = [build_messages(ex) for ex in val_data]
    sampling_params = SamplingParams(**SAMPLING)

    print(f"  Running inference on {len(all_messages)} samples...")
    t1      = time.time()
    outputs = llm.chat(all_messages, sampling_params, use_tqdm=True)
    elapsed = round(time.time() - t1, 1)
    print(f"  Done in {elapsed}s  ({round(elapsed / len(all_messages), 2)}s/sample)")

    predictions = [o.outputs[0].text.strip() for o in outputs]

    # ── Free GPU memory before loading the next model ──────────────────────
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


def merge_predictions(
    val_data:       list[dict],
    base_4b_preds:  list[str],
    ft_preds:       list[str],
    base_27b_preds: list[str],
) -> list[dict]:
    """
    One record per example: original fields first, then three prediction fields.
    """
    records = []
    for example, b4b, ft, b27b in zip(val_data, base_4b_preds, ft_preds, base_27b_preds):
        record = {
            "conversation":      example["conversation"],
            "reference":         example["structured_note"],
            "base_4b_response":  b4b,
            "finetuned_response": ft,
            "base_27b_response": b27b,
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference for 4B-base, 4B-finetuned, and 27B-base models."
    )
    parser.add_argument("--base-model-4b",   required=True,
                        help="HF ID or local path for the 4B base model")
    parser.add_argument("--base-model-27b",  required=True,
                        help="HF ID or local path for the 27B base model")
    parser.add_argument("--finetuned-model", required=True,
                        help="Local path to the merged finetuned model directory")
    parser.add_argument("--val-data",        required=True,
                        help="Path to val_dataset.json produced by finetune SLURM job")
    parser.add_argument("--output-dir",      default=OUTPUT_DIR)
    parser.add_argument("--tensor-parallel", type=int, default=TENSOR_PARALLEL)
    args = parser.parse_args()

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")

    print(f"\n{'=' * 60}")
    print(f"4B base model:    {args.base_model_4b}")
    print(f"27B base model:   {args.base_model_27b}")
    print(f"Finetuned model:  {args.finetuned_model}")
    print(f"Val data:         {args.val_data}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Tensor parallel:  {args.tensor_parallel}")
    print(f"{'=' * 60}\n")

    # ── Load val data (CPU, before any GPU work) ───────────────────────────
    print("Loading val data...")
    val_data = load_json(args.val_data)

    # ── MODEL 1: 4B BASE ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL 1 — 4B BASE: {args.base_model_4b}")
    print(f"{'=' * 60}")
    base_4b_preds = run_inference(args.base_model_4b, val_data, args.tensor_parallel)

    # ── MODEL 2: 27B BASE ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL 2 — 27B BASE: {args.base_model_27b}")
    print(f"{'=' * 60}")
    base_27b_preds = run_inference(args.base_model_27b, val_data, args.tensor_parallel)

    # ── MODEL 3: FINETUNED ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL 3 — FINETUNED: {args.finetuned_model}")
    print(f"{'=' * 60}")
    ft_preds = run_inference(args.finetuned_model, val_data, args.tensor_parallel)

    # ── Merge and save predictions ─────────────────────────────────────────
    print(f"\nMerging predictions...")
    records = merge_predictions(val_data, base_4b_preds, ft_preds, base_27b_preds)

    out_path = os.path.join(
        args.output_dir,
        f"predictions_{slurm_id}.json"
    )
    save_json(records, out_path)

    # Preview first example
    first = records[0]
    print(f"\n--- First example preview ---")
    print(f"  conversation:        {first['conversation'][:120]}...")
    print(f"  reference:           {first['reference'][:120]}...")
    print(f"  base_4b_response:    {first['base_4b_response'][:120]}...")
    print(f"  finetuned_response:  {first['finetuned_response'][:120]}...")
    print(f"  base_27b_response:   {first['base_27b_response'][:120]}...")

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "run_id":           run_id,
        "slurm_job_id":     slurm_id,
        "base_model_4b":    args.base_model_4b,
        "base_model_27b":   args.base_model_27b,
        "finetuned_model":  args.finetuned_model,
        "val_data":         args.val_data,
        "val_size":         len(val_data),
        "tensor_parallel":  args.tensor_parallel,
        "sampling":         SAMPLING,
        "predictions_file": out_path,
    }
    save_json(manifest, os.path.join(args.output_dir, f"manifest_{run_id}.json"))

    print(f"\n{'=' * 60}")
    print(f"All done. Results saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()