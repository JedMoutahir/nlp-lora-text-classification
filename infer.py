import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--texts", nargs="+", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tok(args.texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs).logits
    probs = out.softmax(-1).tolist()
    preds = out.argmax(-1).tolist()

    for txt, pred, prob in zip(args.texts, preds, probs):
        print(f"[{pred}] p={max(prob):.3f} :: {txt}")

if __name__ == "__main__":
    main()
