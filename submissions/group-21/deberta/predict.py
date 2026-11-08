import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import MAX_LENGTH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    test = pd.read_csv(args.test_csv)
    if len(test) != 3000 or not test["id"].is_unique:
        raise ValueError("Expected 3,000 test rows with unique IDs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()