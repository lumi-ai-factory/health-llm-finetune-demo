## Running the Notebooks (1 GPU Setup)

The notebooks are designed for interactive use via Jupyter on LUMI.

### Notebooks
- **1_finetuning.ipynb** – Fine-tuning workflow using a single GPU (Jupyter)
- **2_test_ft_qwen.ipynb** – Generating predictions (inference)
- **3_test_ft_medgemma.ipynb** – Testing the fine-tuned model
4_inspect_demo_predictions.ipynb


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
