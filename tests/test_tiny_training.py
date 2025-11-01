import os
import shutil
import subprocess
import sys
import time

def test_runs_tiny_training(tmp_path):
    out = tmp_path / "out"
    cmd = [
        sys.executable, "train.py",
        "--model_name", "prajjwal1/bert-tiny",
        "--train_csv", "data/tiny.csv",
        "--eval_csv", "data/tiny.csv",
        "--text_col", "text", "--label_col", "label",
        "--output_dir", str(out),
        "--epochs", "1",
        "--batch_size", "8",
        "--lr", "2e-4",
        "--lora_r", "8", "--lora_alpha", "16", "--lora_dropout", "0.05",
        "--seed", "7"
    ]
    env = os.environ.copy()
    # keep runs quick
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    took = time.time() - t0
    # sanity checks
    assert (out / "config.json").exists()
    assert took < 120  # should finish in under ~2 min CPU on tiny set
