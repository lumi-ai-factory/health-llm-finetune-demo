#!/bin/bash
#SBATCH --account=project_462001520
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

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub
mkdir -p $HF_HOME

singularity run $SIF bash -c "python -m venv --system-site-packages ./venv && source ./venv/bin/activate && pip install -U transformers==5.5.4"

export PYTHONPATH=$PYTHONPATH:./venv/lib/python3.12/site-packages

export HF_TOKEN_PATH=~/.cache/huggingface/token

#######################################################
#set hf-cache to Temporary (SSD) storage on login node
LOCAL_HF_CACHE="${TMPDIR}/hf_cache_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_HF_CACHE}"
export TRANSFORMERS_CACHE="${LOCAL_HF_CACHE}"

# Add tmp clean up!!!
cleanup() {
    echo "[$(date)] Cleaning up local cache at $LOCAL_HF_CACHE"
    rm -rf "$LOCAL_HF_CACHE"
}

# Run cleanup on:
#  - EXIT (normal termination), ERR (command failure), SIGTERM (job cancelled), SIGINT (Ctrl+C)
trap cleanup EXIT ERR SIGTERM SIGINT
#######################################################



OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/demo/ft_model
mkdir -p $OUTPUT_DIR

MODEL_NAME="google/medgemma-1.5-4b-it"

MLFLOW_MLRUNS_DIR=$OUTPUT_DIR/mlruns
mkdir -p $MLFLOW_MLRUNS_DIR

MLFLOW_EXPERIMENT="${MODEL_NAME##*/}structured-note-finetuned"

# NOTE!!! Use the same json file name you used when submitting the data_mod_vllm job
JSON_FILE=/scratch/${SLURM_JOB_ACCOUNT}/data/structured_notes.json
VAL_JSON_PATH="/scratch/${SLURM_JOB_ACCOUNT}/data/val_dataset.json"
export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

set -xv

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

#Check which one works!
#srun singularity run $SIF python -m torch.distributed.run \
srun singularity run $SIF   ./venv/bin/python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    finetune.py $* \
    --input-model "$MODEL_NAME" \
    --output-path $OUTPUT_DIR \
    --val-json-output $VAL_JSON_PATH \
    --mlflow_tracking_uri $MLFLOW_MLRUNS_DIR \
    --mlflow_experiment $MLFLOW_EXPERIMENT \
    --json-file $JSON_FILE \
    --model_output_name="${MODEL_NAME##*/}-structured_note" \
    --num-workers $SLURM_CPUS_PER_TASK \
    --batch_size=8 \
    --peft 