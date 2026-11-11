# Group 21 | DistilRoBERTa and robust feature blend

**Team members:** Maanasi Ganapathy (mg2563), Ke Dong (kd563), Reuben Ronald Miranda (rrm226)

**Kaggle team name:** Group Number 21

**Model contributor:** Reuben Ronald Miranda (rrm226)

The selected model combines 75% of a standardized v2 score with 25% of a standardized fine-tuned DistilRoBERTa margin. V2 averages standardized scores from word TF-IDF logistic regression and a gradient-boosted classifier using style and language-model features. A sigmoid converts the final blended margin to an AI-authorship score.

## Setup

Run all commands from `submissions/group-21/` in the team repository. Use Python 3.11 or 3.12 and a CUDA GPU for training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place `train.csv`, `unlabeled.csv`, `test.csv`, and `sample_submission.csv` in `./data/`.

The entrypoints use the supporting Python files under `distilroberta/`. Keep that directory alongside `train.py` and `predict.py`. The root requirements file loads `distilroberta/requirements.txt`.

## Train / validate

```bash
RUN_TRAIN_DIAGNOSTIC=1 python train.py --data-dir data --out-dir outputs
```

This command evaluates the fixed 8,000/2,000 split, runs the baseline and robustness experiments, records the selected blend's training and validation AUROC, and fits the final models on all 10,000 labeled passages.

Training saves the v2 estimators and fitted vocabulary to `outputs/distilroberta_v1/v2.joblib`. The transformer checkpoint includes its tokenizer and is identified by `outputs/distilroberta_v1/final_checkpoint.json`.

## Predict (writes Kaggle submission)

```bash
python predict.py --data-dir data --out-dir outputs
```

Prediction loads the saved models and reads `test.csv` and `sample_submission.csv`. It writes `outputs/submission.csv` with exactly 3,000 rows, columns `id,label`, and one probability per test ID in sample-submission order. Class 1 means AI-generated. Prediction supports CUDA and CPU.

## Seeds and notes

- Global seed is 42.
- Validation uses a fixed stratified 2,000-row holdout, with 1,000 passages per class. Split IDs are stored in `outputs/val_ids.csv`.
- The fitting partition contains 8,000 original labeled passages. V2 also trains on edited copies of fitting passages. DistilRoBERTa trains on original passages.
- V2's word vocabulary and IDF use all labeled and unlabeled text, including validation text without labels. This is a transductive feature-fitting procedure.
- DistilRoBERTa uses `distilbert/distilroberta-base`, revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, maximum length 320 tokens, learning rate 0.00002, two epochs, training batch size 16, and mixed precision.
- Copy the historical `pretrained_revision.txt` into `outputs/distilroberta_v1/` to reuse that revision. A new output folder otherwise records the model revision resolved at run time.
- V2 uses token-predictability features from `distilgpt2`. The comparison experiments also use `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- The paired robustness study applies identical edited passages to both components before comparing them. Blend weights were selected using validation experiments.
- Final prediction models use all 10,000 labels after model selection.
- Keep raw competition files, exported passage text, prediction CSVs, checkpoints, and credentials outside Git. Retain the original split IDs and model revision when replaying the historical experiment.

| Selected blend result                          |    AUROC |
| ---------------------------------------------- | -------: |
| Training, 8,000 original fitting passages      | 1.000000 |
| Validation, 2,000 original held-out passages   | 0.996868 |
| Paired lowercase and punctuation-removal probe | 0.978895 |
| Reported Kaggle public score                   |  0.95934 |

A fresh validation-stage DistilRoBERTa fit took 130.8624 seconds on a Tesla T4 for two epochs over 8,000 rows. This measurement excludes download and dataset tokenization and covers the transformer component. Total v2 and blend runtimes were not recorded.

Verification covered the saved validation scores, ID alignment, prediction CSV format, and checkpoint-based prediction with local test models. A complete GPU replay of the packaged training pipeline remains unverified. Numerical differences can arise from hardware and dependency versions.

## AI assistance

ChatGPT and Codex assisted with experiment code, debugging, explanations, result analysis, packaging, and report drafting. The team ran the experiments and selected the final approach. We checked the selected blend's held-out AUROC, training AUROC, length-bin results, error examples, and submission IDs against saved model outputs. Passage scores were produced by the documented models. No external LLM API was used to label or predict competition test passages.
