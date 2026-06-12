#!/bin/bash
#SBATCH --job-name=eval_metrics_structured_notes
#SBATCH --account=project_462001520
#SBATCH --partition=small
#SBATCH --time=3:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=./log_metrics/%j_output.log
#SBATCH --error=./log_metrics/%j_error.log


module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif    
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub

mkdir -p $HF_HOME
mkdir -p log_metrics

singularity run $SIF bash -c "python -m venv --system-site-packages ./venv && source ./venv/bin/activate && pip install evaluate rouge-score bert-score sacrebleu"

export PYTHONPATH=$PYTHONPATH:./venv/lib/python3.12/site-packages

echo "Starting evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"

srun singularity run $SIF bash -c "source ./venv/bin/activate && python calculate_metrics.py"

echo "Evaluation finished at $(date)"