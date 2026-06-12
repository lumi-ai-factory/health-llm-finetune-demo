import argparse
import os
import sys
import time
import torch
import mlflow
import pandas as pd

from pathlib import Path
from datasets import Dataset #load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AutoProcessor
)
from functools import partial

def preprocess(examples, tokenizer, max_tokens=2048):
    """
    Convert chat examples into tokenized causal-LM training examples.

    Safeguards:
    - Skip samples where the assistant response is completely truncated.
    - Ensure labels and input_ids always have identical lengths.
    - Ensure at least one non-masked label token exists.
    """

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    skipped_no_response = 0
    skipped_empty_labels = 0

    for conversation, structured_note in zip(
        examples["conversation"],
        examples["structured_note"],
    ):

        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": """You are a medical clinical documentation assistant. 
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
"""},
            {"role": "user", "content": conversation},
            {"role": "assistant", "content": structured_note},
        ]

        # Apply chat template — tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_tokens,
            padding=False,
            add_special_tokens=False,
        )

        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_tokens,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Safety: prompt length cannot exceed actual sequence length
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        # Assistant response completely truncated
        if prompt_len >= len(input_ids):
            skipped_no_response += 1
            continue

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Safety check
        if len(labels) != len(input_ids):
            raise ValueError(
                f"Label length mismatch: labels={len(labels)}, "
                f"input_ids={len(input_ids)}"
            )

        valid_label_count = sum(label != -100 for label in labels)

        if valid_label_count == 0:
            skipped_empty_labels += 1
            continue

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    #print(
    #    f"Skipped {skipped_no_response} samples "
    #    f"(assistant response truncated)"
    #)
    #print(
    #    f"Skipped {skipped_empty_labels} samples "
    #    f"(no valid label tokens)"
    #)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model_output_name", type=str, required=True)
    parser.add_argument("--json-file", type=str, required=True, help="Path to train JSON file")
    parser.add_argument("--val-json-output", type=str, default=None, help="Path to save val dataset as JSON")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True)
    parser.add_argument("--mlflow_experiment", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--peft", action="store_true")
    args, _ = parser.parse_known_args()

    rank            = int(os.environ["RANK"])
    local_rank      = int(os.environ["LOCAL_RANK"])
    world_size      = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    if rank == 0:
        #mlflow_tracking_uri = os.path.join(args.output_path, "mlruns")
        mlflow_tracking_uri = args.mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        #mlflow.set_experiment(args.model_output_name)
        mlflow.set_experiment(args.mlflow_experiment)
        print(f"MLflow tracking URI: {mlflow_tracking_uri}")
        print(f"MLflow Experiment name: {args.mlflow_experiment}")

    output_model_dir = os.path.join(args.output_path, args.model_output_name)

    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Output dir: {output_model_dir}")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"Using GPU {local_rank}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if rank == 0:
            print("No GPU found, using CPU")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} must be a multiple of num GPUs={world_size}")
        sys.exit(1)

    # ── Load tokenizer and model ──────────────────────────────

    start = time.time()

    if rank == 0:
        print(f"Loading model: {args.input_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.input_model, use_fast=True)

    #download processor -> needed for finetuned model usage (medgemma is multimodal)
    processor = AutoProcessor.from_pretrained(args.input_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    if args.peft:
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if rank == 0:
            print("Using LoRA (PEFT)")
            model.print_trainable_parameters()

    stop = time.time()
    if rank == 0:
        print(f"Loading model took: {stop - start:.2f}s")

    # ── Load and tokenize datasets ────────────────────────────

    if rank == 0:
        print(f"Loading datasets...")
        print(f"  Train: {args.json_file}")

    df = pd.read_json(args.json_file)

    dataset = Dataset.from_pandas(df, preserve_index=False)

    split = dataset.train_test_split(test_size=0.1, seed=42)

    raw_train = split["train"]
    raw_val = split["test"]

    if rank == 0:
        print(f"  Train size: {len(raw_train)}")
        print(f"  Val size:   {len(raw_val)}")

    preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=args.max_tokens)

    tokenized_train = raw_train.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_train.column_names,
        num_proc=args.num_workers,
    )

    if rank == 0:
            print(f"  Train tokenized: {len(tokenized_train)} samples "
                  f"(from {len(raw_train)}, skipped {len(raw_train) - len(tokenized_train)})")

    tokenized_val = raw_val.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_val.column_names,
        num_proc=args.num_workers,
    )

    if rank == 0:
            print(f"  Val tokenized:   {len(tokenized_val)} samples "
                  f"(from {len(raw_val)}, skipped {len(raw_val) - len(tokenized_val)})")

    if rank == 0:
        tokenized_val.save_to_disk(f"{Path(args.json_file).parent / 'tokenized_val'}")
        raw_val.save_to_disk(f"{Path(args.json_file).parent / 'raw_val'}")
        print(f"Tokenized train size: {len(tokenized_train)}")
        print(f"Tokenized val size:   {len(tokenized_val)}")

        # Save raw_val as JSON to a custom path
        if args.val_json_output:
            raw_val.to_json(args.val_json_output)
            print(f"Val dataset saved to: {args.val_json_output}")

    # ── Training arguments ────────────────────────────────────

    training_args = TrainingArguments(
        disable_tqdm=True,
        output_dir=output_model_dir,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,
        load_best_model_at_end=False,
        per_device_train_batch_size=args.batch_size // world_size,
        per_device_eval_batch_size=args.batch_size // world_size,
        dataloader_num_workers=args.num_workers,
        #ddp_find_unused_parameters=False,
        ddp_find_unused_parameters=True,
        dataloader_pin_memory=True,
        metric_for_best_model="eval_loss",
        eval_strategy="steps",
        eval_steps=1000,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        report_to=["mlflow"],
        logging_steps=100,
        logging_strategy="steps",
        run_name=f"{args.model_output_name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
    )

    # Use DataCollatorForSeq2Seq to handle labels padding correctly
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────

    start_train = time.time()
    if rank == 0:
        print("Training starting...")

    trainer.train(resume_from_checkpoint=args.resume)

    stop_train = time.time()
    if rank == 0:
        elapsed = stop_train - start_train
        hours   = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"Training took: {hours}h {minutes}m {seconds}s")

    # ── Save ──────────────────────────────────────────────────

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    processor.save_pretrained(output_model_dir)

    if rank == 0:
        print(f"\nModel saved to: {output_model_dir}")
        print(f"MLflow data:    {mlflow_tracking_uri}")
    
    if args.peft and rank == 0:
        merged_output_dir = os.path.join(
            args.output_path,
            f"{args.model_output_name}_merged"
        )

        merged_model = model.merge_and_unload()

        merged_model.save_pretrained(
            merged_output_dir
        )
        tokenizer.save_pretrained(merged_output_dir)
        processor.save_pretrained(merged_output_dir)

        print("Merged model saved successfully.")