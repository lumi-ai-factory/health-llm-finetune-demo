## Running the Notebooks (1 GPU Setup)

The notebooks are designed for interactive use via Jupyter on LUMI.

### Notebooks
- **1_finetuning.ipynb** – Fine-tuning workflow using a single GPU (Jupyter)
- **2_test_ft_qwen.ipynb** – Testing the fine-tuned model with one-time inference
- **3_inspect_demo_predictions.ipynb** – Display model responses generated in the [demo](../demo/README.md)

Run the notebooks in this order: notebook 2 uses the model trained in notebook 1. Notebook 3 does not depend on the other two.

⚠️ **Run one notebook at a time**, and restart the kernel before moving to the next one (**Kernel → Restart Kernel**). Several notebooks running at the same time may run out of GPU memory.


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


### 3. Get the notebooks

Open a terminal in the Jupyter app (**File → New → Terminal**) and choose one of the options below.

#### Option A: Guided session

In a guided session, the notebooks are ready in the shared LUMI project of the session. Copy them into a personal folder by pasting these commands one by one to the terminal:

1. Create a personal folder:
```
mkdir -p /scratch/$SLURM_JOB_ACCOUNT/$USER
```

2. Copy the exercise notebooks into your folder:
```
cp -r /scratch/$SLURM_JOB_ACCOUNT/notebooks /scratch/$SLURM_JOB_ACCOUNT/$USER
```

Then open the notebooks from `/scratch/$SLURM_JOB_ACCOUNT/$USER/notebooks` in the Jupyter file browser.

#### Option B: On your own

If you are not in a guided session, clone this repository into your project's scratch directory:

```
mkdir -p /scratch/$SLURM_JOB_ACCOUNT/$USER
cd /scratch/$SLURM_JOB_ACCOUNT/$USER
git clone https://github.com/lumi-ai-factory/health-llm-finetune-demo.git
```

Then open the `notebooks` folder of the cloned repository in the Jupyter file browser.

In both options, run the notebooks in order, starting from `1_finetuning.ipynb`.

### Data

The notebooks read their data directly from a public [LUMI-O](https://docs.lumi-supercomputer.eu/storage/lumio/) bucket, so there is nothing to download before you start. The address of the bucket is set in the `BUCKET_URL` variable in notebooks 1 and 3.

- **1_finetuning.ipynb** uses `validation.jsonl`: the validation set of the [demo](../demo/README.md) dataset, about 3,000 doctor–patient conversations with structured clinical notes. The notebook takes 2,000 of them and splits them into 1,900 for training and 100 for validation.
- **3_inspect_demo_predictions.ipynb** uses `predictions.json`: the notes generated in the demo by three models for the test set.

The bucket also contains a `README.md` that describes the files, how they were created, and the sources and licenses of the data. The conversations are synthetic: they come from the [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) dataset (MIT License) and contain no real patient data.

If you would like a local copy of the data, run the optional download cell at the end of notebook 3. The files are saved next to the notebooks; keep the `README.md` together with the data files, as it contains the license information.


### Outputs

Everything the notebooks create is saved in the same folder as the notebooks:

- **1_finetuning.ipynb** saves the fine-tuning checkpoints and the merged fine-tuned model to the `ft_model` folder (about 8 GB).
- **2_test_ft_qwen.ipynb** loads the merged model from `ft_model`, so run notebook 1 first.
- **3_inspect_demo_predictions.ipynb** saves files only if you run the optional download cell.

The downloaded base model (Qwen3-4B-Instruct-2507) is stored in the project's shared Hugging Face cache, `/scratch/$SLURM_JOB_ACCOUNT/hf-cache`, so it is downloaded only once per project.

-----

**Authors:**
- Henri Meriläinen
- Emma Hintsala

**Acknowledgements**

Fine-tuning code is based on [CSCfi/llm-fine-tuning-examples](https://github.com/CSCfi/llm-fine-tuning-examples).