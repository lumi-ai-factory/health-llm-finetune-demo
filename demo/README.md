# Health LLM Finetuning Demo
![demo_overview](../images/demo_overview.png)

This demo fine-tunes [MedGemma-1.5-4B](google/medgemma-1.5-4b-it) on doctor–patient conversations to generate structured clinical notes.

It covers four steps: data preprocessing, fine-tuning, inference, and evaluation.

## Files

| File | Description |
|---|---|
| `create_predictions.py` | Runs inference and saves predictions to JSON |
| `calculate_metrics.py` | Computes BLEU, ROUGE-L, and BERTScore |
| `data_mod.py` | Data preprocessing script |
| `finetune.py` | Fine-tuning script (MedGemma-1.5-4B, 8 GPUs, PEFT) |
| `run_create_predictions.sh` | SLURM script for inference |
| `run_calculate_metrics.sh` | SLURM script for evaluation |
| `run_data_mod_vllm.sh` | SLURM script for data preprocessing |
| `run_finetune_8gpus.sh` | SLURM script for fine-tuning |

## Steps

### 1. Data preprocessing
Creates the training dataset of (dialogue → structured note) pairs using a large LLM to augment the data. Created structured note is created based on the `full_note` column of the [original dataset](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes).

```bash
sbatch run_data_mod_vllm.sh openai/gpt-oss-120b structured_notes.json 256
```
Script arguments explained:
* openai/gpt-oss-120b - LLM used to augment the dataset
* structured_notes.json - JSON file name (used in finetuning codes)
* 256 - batch size (how many queries sent to vLLM server at once)

### 2. Fine-tuning
Fine-tunes MedGemma-1.5-4B using PEFT on 8 GPUs.

```bash
sbatch run_finetune_8gpus.sh
```

**Monitor GPU usage during training:**

Check the jobid of the job with the `squeue --me` command.  

Open an interactive parallel session (replace XXXXXXX with the jobid of your job) with the following command:

````
srun --jobid XXXXXXX --interactive --pty /bin/bash
````

This will open a shell on the compute node where the job is running. We can now use the rocm-smi tool to monitor the GPU usage. The following command will show the GPU usage updated every second:

`watch -n1 rocm-smi`

<details>
  <summary>example_output.log</summary>

Here is (part of) the slurm output log from succeded training run. 
````
Job started at ke 10.6.2026 18.59.54 +0300
Running on node: nid007960
Job ID: 19154866
MLflow tracking URI: /scratch/project_462000131/demo/ft_model/mlruns
MLflow Experiment name: medgemma-1.5-4b-itstructured-note-finetuned
Using 8 GPUs
Output dir: /scratch/project_462000131/demo/ft_model/medgemma-1.5-4b-it-structured_note
Using GPU 0: AMD Instinct MI250X
Loading model: google/medgemma-1.5-4b-it
Using LoRA (PEFT)
trainable params: 38,497,792 || all params: 4,338,577,264 || trainable%: 0.8873
Loading model took: 32.36s
Loading datasets...
  Train: /scratch/project_462000131/data/structured_notes.json
  Train size: 27000
  Val size:   3000
  Train tokenized: 26830 samples (from 27000, skipped 170)
  Val tokenized:   2985 samples (from 3000, skipped 15)
Tokenized train size: 26830
Tokenized val size:   2985
Val dataset saved to: /scratch/project_462000131/data/val_dataset.json
Training starting...
{'loss': 1.5457, 'grad_norm': 0.3182925283908844, 'learning_rate': 1.9409660107334526e-05, 'epoch': 0.02981514609421586}
{'loss': 1.2424, 'grad_norm': 0.33986878395080566, 'learning_rate': 1.881335718545021e-05, 'epoch': 0.05963029218843172}
{'loss': 1.1935, 'grad_norm': 0.3536140024662018, 'learning_rate': 1.8217054263565892e-05, 'epoch': 0.08944543828264759}
{'loss': 1.1698, 'grad_norm': 0.5306793451309204, 'learning_rate': 1.7620751341681576e-05, 'epoch': 0.11926058437686345}
{'loss': 1.1415, 'grad_norm': 0.4296364188194275, 'learning_rate': 1.702444841979726e-05, 'epoch': 0.1490757304710793}
{'loss': 1.1258, 'grad_norm': 0.44555070996284485, 'learning_rate': 1.6428145497912942e-05, 'epoch': 0.17889087656529518}
{'loss': 1.1148, 'grad_norm': 0.43608298897743225, 'learning_rate': 1.5831842576028623e-05, 'epoch': 0.20870602265951102}
{'loss': 1.1004, 'grad_norm': 0.47608131170272827, 'learning_rate': 1.5235539654144307e-05, 'epoch': 0.2385211687537269}
{'loss': 1.1073, 'grad_norm': 0.4293484687805176, 'learning_rate': 1.4639236732259989e-05, 'epoch': 0.26833631484794274}
{'loss': 1.0961, 'grad_norm': 0.4295450747013092, 'learning_rate': 1.4042933810375671e-05, 'epoch': 0.2981514609421586}
{'eval_loss': 1.0912601947784424, 'eval_runtime': 112.9733, 'eval_samples_per_second': 26.422, 'eval_steps_per_second': 3.311, 'epoch': 0.2981514609421586}
{'loss': 1.0905, 'grad_norm': 0.5018987059593201, 'learning_rate': 1.3446630888491354e-05, 'epoch': 0.3279666070363745}
{'loss': 1.0672, 'grad_norm': 0.46132710576057434, 'learning_rate': 1.2850327966607037e-05, 'epoch': 0.35778175313059035}
{'loss': 1.0707, 'grad_norm': 0.494373083114624, 'learning_rate': 1.225402504472272e-05, 'epoch': 0.3875968992248062}
{'loss': 1.0826, 'grad_norm': 0.5018445253372192, 'learning_rate': 1.1657722122838402e-05, 'epoch': 0.41741204531902204}
{'loss': 1.059, 'grad_norm': 0.5192534327507019, 'learning_rate': 1.1061419200954087e-05, 'epoch': 0.4472271914132379}
{'loss': 1.062, 'grad_norm': 0.5206125378608704, 'learning_rate': 1.046511627906977e-05, 'epoch': 0.4770423375074538}
{'loss': 1.0607, 'grad_norm': 0.510175347328186, 'learning_rate': 9.86881335718545e-06, 'epoch': 0.5068574836016696}
{'loss': 1.0493, 'grad_norm': 0.5446339249610901, 'learning_rate': 9.272510435301133e-06, 'epoch': 0.5366726296958855}
{'loss': 1.0592, 'grad_norm': 0.5435701608657837, 'learning_rate': 8.676207513416816e-06, 'epoch': 0.5664877757901013}
{'loss': 1.059, 'grad_norm': 0.5825333595275879, 'learning_rate': 8.079904591532499e-06, 'epoch': 0.5963029218843172}
{'eval_loss': 1.058499813079834, 'eval_runtime': 113.0612, 'eval_samples_per_second': 26.402, 'eval_steps_per_second': 3.308, 'epoch': 0.5963029218843172}
{'loss': 1.0588, 'grad_norm': 0.537396252155304, 'learning_rate': 7.483601669648182e-06, 'epoch': 0.6261180679785331}
{'loss': 1.0613, 'grad_norm': 0.5493167042732239, 'learning_rate': 6.887298747763864e-06, 'epoch': 0.655933214072749}
{'loss': 1.0637, 'grad_norm': 0.5460458397865295, 'learning_rate': 6.290995825879548e-06, 'epoch': 0.6857483601669648}
{'loss': 1.0662, 'grad_norm': 0.5493756532669067, 'learning_rate': 5.694692903995231e-06, 'epoch': 0.7155635062611807}
{'loss': 1.0457, 'grad_norm': 0.5400915145874023, 'learning_rate': 5.098389982110913e-06, 'epoch': 0.7453786523553966}
{'loss': 1.0414, 'grad_norm': 0.5336266756057739, 'learning_rate': 4.502087060226595e-06, 'epoch': 0.7751937984496124}
{'loss': 1.0551, 'grad_norm': 0.5538650751113892, 'learning_rate': 3.905784138342278e-06, 'epoch': 0.8050089445438283}
{'loss': 1.052, 'grad_norm': 0.8979543447494507, 'learning_rate': 3.309481216457961e-06, 'epoch': 0.8348240906380441}
{'loss': 1.041, 'grad_norm': 0.5568894743919373, 'learning_rate': 2.7131782945736433e-06, 'epoch': 0.86463923673226}
{'loss': 1.0534, 'grad_norm': 0.5635836720466614, 'learning_rate': 2.1168753726893265e-06, 'epoch': 0.8944543828264758}
{'eval_loss': 1.047757625579834, 'eval_runtime': 115.6403, 'eval_samples_per_second': 25.813, 'eval_steps_per_second': 3.234, 'epoch': 0.8944543828264758}
{'loss': 1.0562, 'grad_norm': 0.5930586457252502, 'learning_rate': 1.520572450805009e-06, 'epoch': 0.9242695289206917}
{'loss': 1.0369, 'grad_norm': 0.5996881723403931, 'learning_rate': 9.242695289206919e-07, 'epoch': 0.9540846750149076}
{'loss': 1.048, 'grad_norm': 0.5297402143478394, 'learning_rate': 3.2796660703637447e-07, 'epoch': 0.9838998211091234}
{'train_runtime': 2904.556, 'train_samples_per_second': 9.237, 'train_steps_per_second': 1.155, 'train_loss': 1.0951672965170873, 'epoch': 1.0}
Training took: 0h 48m 46s

Model saved to: /scratch/project_462001520/demo/ft_model/medgemma-1.5-4b-it-structured_note
MLflow data:    /scratch/project_462001520/demo/ft_model/mlruns
Merged model saved successfully.
[ke 10.6.2026 19.59.57 +0300] Cleaning up local cache at /tmp/hf_cache_19154866
````

</details>


### 3. Inference
Generates predictions on the validation set (3,000 samples) for three models: MedGemma-1.5-4B original, MedGemma-1.5-4B fine-tuned, and MedGemma-27B original. Results are saved to a JSON file.

```bash
sbatch run_create_predictions.sh
```

### 4. Evaluation
Computes BLEU, ROUGE-L, and BERTScore against reference answers using the predictions JSON from the previous step.

```bash
sbatch run_calculate_metrics.sh
```
-----

**Authors:**
- Henri Meriläinen
- Emma Hintsala

**Acknowledgements**

Fine-tuning code is based on [CSCfi/llm-fine-tuning-examples](https://github.com/CSCfi/llm-fine-tuning-examples).