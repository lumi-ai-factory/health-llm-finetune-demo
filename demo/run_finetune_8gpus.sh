#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=./log/ft_medgemma_mlflow/%j_output.log
#SBATCH --error=./log/ft_medgemma_mlflow/%j_error.log


module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif    

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache

export HF_TOKEN_PATH=~/.cache/huggingface/token

# MIOpen cache in node-local /tmp (single-node job)
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
 
cleanup() {
    echo "[$(date)] Cleaning up MIOpen cache at $MIOPEN_USER_DB_PATH"
    rm -rf "$MIOPEN_USER_DB_PATH"
}
trap cleanup EXIT


# All data files live under the user's data directory
DATA_ROOT=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/data
DATA_DIR=$DATA_ROOT/structured_notes       # created by data_mod.py
OUTPUT_DIR=$DATA_ROOT/models
MLFLOW_MLRUNS_DIR=$DATA_ROOT/mlruns
mkdir -p $OUTPUT_DIR $MLFLOW_MLRUNS_DIR
 
MODEL_NAME="google/medgemma-1.5-4b-it"


MLFLOW_EXPERIMENT="${MODEL_NAME##*/}structured-note-finetuned"

# Newer MLflow versions refuse the file-based tracking store (mlruns/) unless this is set.
export MLFLOW_ALLOW_FILE_STORE=true


export TOKENIZERS_PARALLELISM=false


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

#export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

set -xv

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


srun singularity run $SIF python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    finetune.py "$@" \
    --input-model "$MODEL_NAME" \
    --output-path $OUTPUT_DIR \
    --data-dir $DATA_DIR \
    --mlflow_tracking_uri $MLFLOW_MLRUNS_DIR \
    --mlflow_experiment $MLFLOW_EXPERIMENT \
    --model_output_name="${MODEL_NAME##*/}-structured_note" \
    --num-workers 7 \
    --batch_size=8 \
    --peft
