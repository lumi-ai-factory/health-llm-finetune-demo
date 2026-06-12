#!/bin/bash
#SBATCH --job-name=inference_job
#SBATCH --account=project_462001520
#SBATCH -p small-g
#SBATCH --time 10:00:00
#SBATCH --tasks-per-node 1
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1
#SBATCH --mem 240G
#SBATCH --output=./log/inference/%j_output.log
#SBATCH --error=./log/inference/%j_error.log


# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings


# Set MIOPEN temp folder to avoid collisions with other users on the same node
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# prequisites: accept medgemma terms and create hf token
export HF_TOKEN_PATH=~/.cache/huggingface/token

# Model paths
BASE_MODEL_4B="google/medgemma-1.5-4b-it"
BASE_MODEL_27B="google/medgemma-27b-it"
FINETUNED_MODEL="/scratch/${SLURM_JOB_ACCOUNT}/demo/ft_model/medgemma-1.5-4b-it-structured_note_merged"


# Path to the val JSON saved by the finetune job
VAL_DATA="/scratch/${SLURM_JOB_ACCOUNT}/demo/val_dataset.json"


export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub
export TORCH_COMPILE_DISABLE=1
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

mkdir -p log

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun singularity run $CONTAINER_IMAGE python create_predictions.py \
    --base-model-4b   "$BASE_MODEL_4B"   \
    --base-model-27b  "$BASE_MODEL_27B"  \
    --finetuned-model "$FINETUNED_MODEL" \
    --val-data        "$VAL_DATA"

echo "Job ended at $(date)"