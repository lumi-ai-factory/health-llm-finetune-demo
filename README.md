# Get started with LLM finetuning on LUMI

### Notebooks
- **1_finetuning_nb.ipynb** – Fine-tuning workflow using a single GPU (Jupyter)
- **2_inference_nb.ipynb** – Generating predictions (inference)
- **3_test_finetuned_medgemma.ipynb** – Testing the fine-tuned model

### Scripts
- **train.py** – Training script for multi-GPU setup  
- **create_predictions.py** – Script for generating predictions  
- **run_train_8gpus.sh** – SLURM script for training with 8 GPUs  
- **run_create_predictions.sh** – SLURM script for inference with 4 GPUs  

---

## Running the Notebooks (1 GPU Setup)

The notebooks are designed for interactive use via Jupyter on LUMI.

### 1. Launch Jupyter

Go to: https://www.lumi.csc.fi/ and select **Jupyter** from the app menu.

### 2. Configure the session

Use the following settings:

- **Project**: `project_462001520`  
- **Partition**: `dev-g`  
- **Cores**: `7`  
- **Memory**: `60 GiB`  
- **GPUs (MI250)**: `1`  
- **Time**: `02:00:00`  
- **Working directory**: `/scratch/project_462001520`  
- **Module**: `lumi-multitorch`  

This setup is suitable for:
- Running fine-tuning experiments on a small scale  
- Performing inference interactively  

---

## Multi-GPU Training (8 GPUs)

To run large-scale training, use:

```bash
sbatch run_train_8gpus.sh
```

## Inference

```bash
sbatch run_create_predictions.sh
```
