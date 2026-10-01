#!/bin/bash
#SBATCH --job-name=eval_metrics_structured_notes
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=./log/metrics/%j_output.log
#SBATCH --error=./log/metrics/%j_error.log


module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif    
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache

# The container's OpenBLAS supports at most 64 threads; more threads make
# BERTScore crash with "malloc(): corrupted top size". Limit the thread
# counts to the CPUs of the job (64).
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

 
# The metric packages are not in the container: install them into a venv on
# top of the container's packages. Created only on the first run.
if [ ! -d ./venv ]; then
    singularity run $SIF bash -c "python -m venv --system-site-packages ./venv && source ./venv/bin/activate && pip install evaluate rouge-score bert-score sacrebleu"
fi

#export PYTHONPATH=$PYTHONPATH:./venv/lib/python3.12/site-packages

echo "Starting evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"

srun singularity run $SIF bash -c "source ./venv/bin/activate && python calculate_metrics.py"

echo "Evaluation finished at $(date)"