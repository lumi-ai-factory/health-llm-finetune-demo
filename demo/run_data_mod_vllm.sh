#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --output=./log/data_mod/%j_output.log
#SBATCH --error=./log/data_mod/%j_error.log
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --nodes=1

# Usage: sbatch run_data_mod_vllm.sh <model> <out_name> <batch_size> [max_rows]
#   e.g. sbatch run_data_mod_vllm.sh openai/gpt-oss-120b structured_notes 256
# Test run with only the first 200 rows:
#   sbatch --time=01:00:00 run_data_mod_vllm.sh openai/gpt-oss-120b structured_notes_test 64 200
# Results go to /scratch/<project>/<user>/data/<out_name>/:
#   structured_notes_full.jsonl, train.jsonl, validation.jsonl, test.jsonl,
#   dataset_info.json

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

#export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache
mkdir -p $HF_HOME
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export TORCH_COMPILE_DISABLE=1

VLLM_LOG=$PWD/log/data_mod/${SLURM_JOB_ID}_vllm.log
mkdir -p $(dirname $VLLM_LOG)

MODEL=$1
OUT_NAME=$2
BATCH_SIZE=$3
MAX_ROWS=${4:-}   # optional: process only the first N rows (for testing)

srun singularity run $SIF vllm serve $MODEL \
--tensor-parallel-size 2 \
--chat-template-content-format openai \
--load-format runai_streamer \
--port 8000 > $VLLM_LOG &

VLLM_PID=$!

cleanup() {
    echo "Cleaning up vLLM process $VLLM_PID"
    kill $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# Wait until vLLM is running
sleep 20
while ! curl http://0.0.0.0:8000 >/dev/null 2>&1
do
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        echo "vLLM crashed"
        exit 1
    fi
    sleep 10
done

# Run the actual Python job
singularity exec $SIF bash -c "
    export CUDA_VISIBLE_DEVICES='' && \
    python data_mod.py \
        --backend vllm \
        --batch_size $BATCH_SIZE \
        --model $MODEL \
        --api-url http://0.0.0.0:8000/v1 \
        --out_name $OUT_NAME \
        ${MAX_ROWS:+--max_rows $MAX_ROWS}
"
Q_EXIT_CODE=$?

# Return the same exit code as data_mod.py (the EXIT trap stops vLLM)
exit $Q_EXIT_CODE