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

    passages = test["text"].tolist()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(passages), args.batch_size):
            batch = tokenizer(
                passages[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            logits = model(**batch).logits
            ai_scores = logits.float().softmax(dim=-1)[:, 1]
            probabilities.extend(ai_scores.cpu().tolist())

    submission = pd.DataFrame({"id": test["id"], "label": probabilities})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output_csv}")

if __name__ == "__main__":
    main()