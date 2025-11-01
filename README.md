# LoRA Fine-Tuning for Text Classification (Transformers + PEFT)

A clean, scalable LoRA training pipeline for text classification.  
Tested CPU-only on a tiny local dataset; ready to scale to GPUs, bigger models, and large datasets.

## Features
- PEFT LoRA on any encoder model (BERT/DistilBERT/RoBERTa…).
- Trainer-based pipeline with evaluation + checkpoints.
- Deterministic seeds; tiny local dataset for quick tests.
- GPU support (Accelerate / bf16 / gradient checkpointing ready).
- Inference script for saved checkpoints.

## Quick Start (tiny local dataset, CPU)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train on the tiny CSV (15 train / 5 val) with a tiny model (fast)
python train.py \
  --model_name prajjwal1/bert-tiny \
  --train_csv data/tiny.csv \
  --eval_csv data/tiny.csv \
  --text_col text --label_col label \
  --output_dir outputs/tiny-bert \
  --epochs 1 --batch_size 8 --lr 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05

# Inference demo
python infer.py \
  --model_dir outputs/tiny-bert \
  --texts "I love this!" "This is boring and bad."
```

## Scale Up (GPU, larger model & dataset)
Use a bigger model, e.g. distilbert-base-uncased or roberta-base.

Switch to a public dataset (e.g., SST-2) by replacing --train_csv/--eval_csv with a datasets loader in train.py (see comments).

Enable GPU with Accelerate:

```bash
accelerate config
accelerate launch train.py \
  --model_name distilbert-base-uncased \
  --train_csv /path/to/sst2_train.csv \
  --eval_csv /path/to/sst2_dev.csv \
  --text_col sentence --label_col label \
  --output_dir outputs/distil-sst2 \
  --epochs 3 --batch_size 32 --lr 2e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16 True --gradient_checkpointing True
```

Use --per_device_train_batch_size, --gradient_accumulation_steps to fit memory.

For multi-GPU: accelerate launch --multi_gpu train.py ....

## Repo I/O
Input: CSV with columns text, label (int class id).

Output: --output_dir with LoRA adapters merged weights for inference.

## Notes
The tiny dataset is only for tests. For real training, use a proper dataset (SST-2, AG News, etc.).