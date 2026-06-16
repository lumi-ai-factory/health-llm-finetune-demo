## Running the Notebooks (1 GPU Setup)

The notebooks are designed for interactive use via Jupyter on LUMI.

### Notebooks
- **1_finetuning.ipynb** – Fine-tuning workflow using a single GPU (Jupyter)
- **2_test_ft_qwen.ipynb** –  Testing the fine-tuned model with one-time inference
- **3_inspect_demo_prediction.ipynb** – Dispaly model responses generated in the [demo](../demo/README.md)


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
- **Working directory**: `/scratch/$PROJECT`  
- **Module**: `lumi-multitorch`  (see more info [here](https://lumi-supercomputer.github.io/LUMI-training-materials/User-Coffee-Breaks/20260326-user-coffee-break-LAIF-software-environment/) and [here](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/))

This setup is suitable for:
- Running fine-tuning experiments on a small scale  
- Performing inference interactively  


### 3. Open Jupyter App terminal 

Paste these command one by one to the terminal.

1. Create a personal folder:
````
mkdir /scratch/$SLURM_JOB_ACCOUNT/$USER
````

2. Copy the exercise notebooks into your folder:
````
cp -r /scratch/$SLURM_JOB_ACCOUNT/notebooks /scratch/$SLURM_JOB_ACCOUNT/$USER 
````

-----

**Authors:**
- Henri Meriläinen
- Emma Hintsala

**Acknowledgements**

Fine-tuning code is based on [CSCfi/llm-fine-tuning-examples](https://github.com/CSCfi/llm-fine-tuning-examples).