#!/bin/bash
#SBATCH --job-name=inference_job
#SBATCH --account=project_462000131
#SBATCH -p small-g
#SBATCH --time 10:00:00
#SBATCH --tasks-per-node 1
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1
#SBATCH --mem 240G
#SBATCH --output=./log/inference/%j_output.log
#SBATCH --error=./log/inference/%j_error.log


# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif


# MIOpen cache in a unique directory in node-local /tmp to avoid collisions
# with other users on the same node; removed at the end of the job
MIOPEN_DIR=$(mktemp -d -p /tmp "${USER}-miopen-XXXXXX")
export MIOPEN_USER_DB_PATH=$MIOPEN_DIR
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR
trap 'rm -rf "$MIOPEN_DIR"' EXIT


# prequisites: accept medgemma terms and create hf token
export HF_TOKEN_PATH=~/.cache/huggingface/token

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache

# All data files live under the user's data directory
DATA_ROOT=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/data

# Model paths
BASE_MODEL_4B="google/medgemma-1.5-4b-it"
BASE_MODEL_27B="google/medgemma-27b-it"
FINETUNED_MODEL="$DATA_ROOT/models/medgemma-1.5-4b-it-structured_note_merged"


# Test split created by data_mod.py
TEST_DATA="$DATA_ROOT/structured_notes/test.jsonl"
OUTPUT_DIR="$DATA_ROOT/predictions"



export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash
# Shared model download cache for the whole project; models go to $HF_HOME/hub

export TORCH_COMPILE_DISABLE=1
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

mkdir -p log

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun singularity run $SIF python create_predictions.py \
    --base-model-4b   "$BASE_MODEL_4B"   \
    --base-model-27b  "$BASE_MODEL_27B"  \
    --finetuned-model "$FINETUNED_MODEL" \
    --test-data       "$TEST_DATA"       \
    --output-dir      "$OUTPUT_DIR"

echo "Job ended at $(date)"