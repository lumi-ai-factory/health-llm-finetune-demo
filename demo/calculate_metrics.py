import os
import json
import argparse
import glob
from datetime import datetime

import evaluate as hf_evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
PREDICTIONS_DIR = "/scratch/project_462001520/data"
OUTPUT_DIR      = "/scratch/project_462001520/demo/metrics"

BERTSCORE_LANG  = "en"   # change to "fi" if evaluating Finnish text

# Display names for the three models
MODEL_NAMES = {
    "base_4b":   "Medgemma-4B-Base",
    "finetuned": "Medgemma-4B-Finetuned",
    "base_27b":  "Medgemma-27B-Base",
}

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "base_4b":   "#4C72B0",   # blue
    "finetuned": "#DD8452",   # orange
    "base_27b":  "#55A868",   # green
}

BAR_METRICS = ["rougeL", "bertscore_f1", "bleu"]

METRIC_LABELS = {
    "rougeL":       "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "bleu":         "BLEU",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(filepath: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Load a predictions.json produced by create_predictions.py.
    Returns (base_4b_preds, finetuned_preds, base_27b_preds, references).
    """
    print(f"  Loading predictions from: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        records = json.load(f)

    base_4b   = [r["base_4b_response"]    for r in records]
    finetuned = [r["finetuned_response"]  for r in records]
    base_27b  = [r["base_27b_response"]   for r in records]
    refs      = [r["reference"]           for r in records]

    print(f"  Loaded {len(records)} records.")
    return base_4b, finetuned, base_27b, refs


def save_json(data, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {filepath}")

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    print("    Computing ROUGE...")
    rouge  = hf_evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references,
                           use_stemmer=False)
    return {
        "rougeL": round(result["rougeL"], 4),
    }


def compute_bertscore(predictions: list[str], references: list[str],
                      lang: str = BERTSCORE_LANG) -> dict:
    print(f"    Computing BERTScore (lang={lang})...")
    bertscore = hf_evaluate.load("bertscore")
    result    = bertscore.compute(predictions=predictions, references=references,
                                  lang=lang)
    return {
        "bertscore_f1": round(sum(result["f1"]) / len(result["f1"]), 4),
    }


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    print("    Computing BLEU (sacrebleu)...")
    bleu   = hf_evaluate.load("sacrebleu")
    result = bleu.compute(predictions=predictions,
                          references=[[r] for r in references])
    return {"bleu": round(result["score"] / 100.0, 4)}


def compute_all(predictions: list[str], references: list[str],
                skip_bertscore: bool = False,
                skip_bleu: bool = False) -> dict:
    metrics = {}
    metrics.update(compute_rouge(predictions, references))
    if not skip_bertscore:
        metrics.update(compute_bertscore(predictions, references))
    if not skip_bleu:
        metrics.update(compute_bleu(predictions, references))
    return metrics

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def bar_plot(model_metrics: dict[str, dict], out_path: str) -> None:
    """Grouped bar chart — one bar per model per metric."""
    model_keys  = list(model_metrics.keys())
    metric_keys = [k for k in BAR_METRICS if k in next(iter(model_metrics.values()))]
    labels      = [METRIC_LABELS[k] for k in metric_keys]

    x       = np.arange(len(metric_keys))
    n       = len(model_keys)
    width   = 0.25
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, mk in zip(offsets, model_keys):
        vals = [model_metrics[mk][k] for k in metric_keys]
        bars = ax.bar(x + offset, vals, width,
                      label=MODEL_NAMES[mk], color=COLORS[mk], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    title_models = " vs ".join(MODEL_NAMES[mk] for mk in model_keys)
    ax.set_title(f"Evaluation metrics — Structured Clinical Notes\n{title_models}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    all_vals = [model_metrics[mk][k] for mk in model_keys for k in metric_keys]
    ax.set_ylim(0, max(all_vals) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def delta_plot(model_metrics: dict[str, dict], out_path: str) -> None:
    """
    Horizontal bar chart showing Δ vs 4B-base for finetuned and 27B-base.
    Green = improvement over base, red = regression.
    """
    base_m = model_metrics["base_4b"]
    metric_keys = [k for k in BAR_METRICS if k in base_m]
    labels      = [METRIC_LABELS[k] for k in metric_keys]

    comparisons = {
        "finetuned": ("Δ Finetuned − 4B-Base", "#DD8452"),
        "base_27b":  ("Δ 27B-Base − 4B-Base",  "#55A868"),
    }

    fig, axes = plt.subplots(1, len(comparisons), figsize=(13, 4), sharey=True)

    for ax, (mk, (title, color)) in zip(axes, comparisons.items()):
        deltas = [model_metrics[mk][k] - base_m[k] for k in metric_keys]
        bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        bars = ax.barh(labels, deltas, color=bar_colors, alpha=0.85)
        for bar, d in zip(bars, deltas):
            ax.text(d + (0.001 if d >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{d:+.4f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score difference")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Δ vs Medgemma-4B-Base  (green = improvement)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_results(model_metrics: dict[str, dict]) -> None:
    model_keys  = list(model_metrics.keys())
    metric_keys = list(next(iter(model_metrics.values())).keys())
    col    = max(len(METRIC_LABELS.get(k, k)) for k in metric_keys) + 2
    name_w = 24

    header = f"  {'Metric':<{col}}"
    for mk in model_keys:
        header += f"  {MODEL_NAMES[mk]:>{name_w}}"
    header += f"  {'Δ ft−4B':>12}  {'Δ 27B−4B':>12}"

    print(f"\n{'=' * (len(header) + 2)}")
    print("  Structured Clinical Notes")
    print(f"{'=' * (len(header) + 2)}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for key in metric_keys:
        name = METRIC_LABELS.get(key, key)
        row  = f"  {name:<{col}}"
        for mk in model_keys:
            row += f"  {model_metrics[mk][key]:>{name_w}}"
        d_ft  = model_metrics["finetuned"][key] - model_metrics["base_4b"][key]
        d_27b = model_metrics["base_27b"][key]  - model_metrics["base_4b"][key]
        row  += f"  {'+' if d_ft  >= 0 else ''}{d_ft:.4f}"
        row  += f"  {'+' if d_27b >= 0 else ''}{d_27b:.4f}"
        print(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 4B-base, 4B-finetuned, and 27B-base structured-note predictions."
    )
    parser.add_argument("--predictions-dir", default=PREDICTIONS_DIR)
    parser.add_argument("--output-dir",      default=OUTPUT_DIR)
    parser.add_argument("--skip-bertscore",  action="store_true")
    parser.add_argument("--skip-bleu",       action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find the most recent predictions file
    pred_files = glob.glob(os.path.join(args.predictions_dir, "predictions*.json"))
    if not pred_files:
        raise FileNotFoundError(
            f"No predictions*.json found in: {args.predictions_dir}"
        )
    pred_file = max(pred_files, key=os.path.getmtime)
    print(f"\nUsing predictions file: {pred_file}")

    base_4b_preds, ft_preds, base_27b_preds, references = load_predictions(pred_file)
    print(f"  Samples: {len(references)}")

    # ── Compute metrics ────────────────────────────────────────────────────
    print(f"\n  BASE 4B model:")
    base_4b_metrics = compute_all(base_4b_preds, references,
                                  args.skip_bertscore, args.skip_bleu)

    print(f"\n  FINETUNED model:")
    ft_metrics = compute_all(ft_preds, references,
                             args.skip_bertscore, args.skip_bleu)

    print(f"\n  BASE 27B model:")
    base_27b_metrics = compute_all(base_27b_preds, references,
                                   args.skip_bertscore, args.skip_bleu)

    model_metrics = {
        "base_4b":   base_4b_metrics,
        "finetuned": ft_metrics,
        "base_27b":  base_27b_metrics,
    }

    # ── Console table ──────────────────────────────────────────────────────
    print_results(model_metrics)

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics_out = {
        "timestamp":        timestamp,
        "slurm_job_id":     SLURM_JOB_ID,
        "predictions_file": pred_file,
        "n_samples":        len(references),
        "models":           MODEL_NAMES,
        "base_4b":          base_4b_metrics,
        "finetuned":        ft_metrics,
        "base_27b":         base_27b_metrics,
    }
    save_json(metrics_out,
              os.path.join(args.output_dir, f"metrics_{SLURM_JOB_ID}.json"))

    # ── Plots ──────────────────────────────────────────────────────────────
    if not args.skip_bertscore:
        bar_plot(model_metrics,
                 os.path.join(args.output_dir, f"metrics_barplot_{SLURM_JOB_ID}.png"))
        delta_plot(model_metrics,
                   os.path.join(args.output_dir, f"metrics_delta_{SLURM_JOB_ID}.png"))

    print(f"\n{'=' * 60}")
    print(f"All done. Results saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()