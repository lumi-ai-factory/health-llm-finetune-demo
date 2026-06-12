# Get started with LLM finetuning on LUMI

This repo includes the Juypyter notebook exercises and LLM finetuning demo scripts. 


### [notebooks](./notebooks/README.md)
- **1_finetuning_nb.ipynb** – Fine-tuning workflow using a single GPU (Jupyter)
- **2_inference_nb.ipynb** – Generating predictions (inference)
- **3_test_finetuned_medgemma.ipynb** – Testing the fine-tuned model

### [Scripts](./demo/README.md)
- **train.py** – Training script for multi-GPU setup  
- **create_predictions.py** – Script for generating predictions  
- **run_train_8gpus.sh** – SLURM script for training with 8 GPUs  
- **run_create_predictions.sh** – SLURM script for inference with 4 GPUs  


-----

**Authors of the original repository:**
- Henri Meriläinen
- Emma Hintsala

**Acknowledgements**

Fine-tuning code is based on [CSCfi/llm-fine-tuning-examples](https://github.com/CSCfi/llm-fine-tuning-examples).

