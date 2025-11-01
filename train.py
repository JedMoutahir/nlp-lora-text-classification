import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import set_seed, ensure_dir


@dataclass
class Config:
    model_name: str
    train_csv: Optional[str]
    eval_csv: Optional[str]
    text_col: str
    label_col: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    bf16: bool
    gradient_checkpointing: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    p.add_argument("--train_csv", type=str, default="data/tiny.csv")
    p.add_argument("--eval_csv", type=str, default="data/tiny.csv")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--output_dir", type=str, default="outputs/tiny-bert")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=lambda x: str(x).lower()=="true", default=False)
    p.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower()=="true", default=False)
    args = p.parse_args()
    return Config(**vars(args))


def load_local_csv(train_csv: str, eval_csv: str, text_col: str, label_col: str):
    ds_train = load_dataset("csv", data_files=train_csv)["train"]
    ds_eval  = load_dataset("csv", data_files=eval_csv)["train"]

    # cast labels to int
    def cast_label(ex):
        ex[label_col] = int(ex[label_col])
        return ex

    ds_train = ds_train.map(cast_label)
    ds_eval = ds_eval.map(cast_label)

    # build label2id/id2label from data
    labels = sorted(set(ds_train[label_col]) | set(ds_eval[label_col]))
    id2label = {i: str(i) for i in labels}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(labels)

    return ds_train, ds_eval, num_labels, label2id, id2label


def tokenize_function(example, tokenizer: AutoTokenizer, text_col: str):
    return tokenizer(example[text_col], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    # ----- Data -----
    # For public datasets (e.g., sst2), replace load_local_csv with load_dataset("glue","sst2") etc.
    train_ds, eval_ds, num_labels, label2id, id2label = load_local_csv(
        cfg.train_csv, cfg.eval_csv, cfg.text_col, cfg.label_col
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tok(ex):
        return tokenize_function(ex, tokenizer, cfg.text_col)

    train_tokenized = train_ds.map(tok, batched=True, remove_columns=[cfg.text_col])
    eval_tokenized  = eval_ds.map(tok, batched=True, remove_columns=[cfg.text_col])

    # ----- Model + LoRA -----
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # PEFT LoRA config — target common modules (query/key/value/output) where present
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value", "output", "q_proj", "k_proj", "v_proj", "o_proj", "dense"]
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----- Training -----
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        seed=cfg.seed,
        bf16=cfg.bf16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # Save final for inference
    trainer.save_model(cfg.output_dir)  # merges adapters for inference-ready use

if __name__ == "__main__":
    main()
