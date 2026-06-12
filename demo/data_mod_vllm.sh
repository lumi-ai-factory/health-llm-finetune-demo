#!/bin/bash
#SBATCH --account=project_xxx
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --output=./log/data_mod/%j/output.log
#SBATCH --error=./log/data_mod/%j/error.log
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --nodes=1

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

export HF_HUB_CACHE=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub/
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export TORCH_COMPILE_DISABLE=1

VLLM_LOG=$PWD/log/data_mod/${SLURM_JOB_ID}/vllm.log
mkdir -p $(dirname $VLLM_LOG)

MODEL=$1
JSON_NAME=$2
BATCH_SIZE=$3

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
        --json_name $JSON_NAME
"
Q_EXIT_CODE=$?

# Return the same exit code as data_mod.py
exit $Q_EXIT_CODE

kill $VLLM_PID
