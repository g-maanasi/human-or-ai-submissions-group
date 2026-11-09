# Group 21 | Human or AI? Kaggle Competition

**Team members**

- Maanasi Ganapathy ([mg2563]) — DeBERTa-v3-base
- Ke Dong(kd563) — Linear SVM with TF-IDF features
- [Team member 3] ([NetID]) — [Model or contribution]

**Kaggle team name:** `[Kaggle team name]`

## Setup

Run these commands from `submissions/group-21/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `train.csv`, `unlabeled.csv`, and `test.csv` in `group-21`. Keep the data and trained checkpoints out of Git. The DeBERTa run documented below uses `train.csv` for training and validation and `test.csv` for prediction.

## DeBERTa-v3-base | Maanasi Ganapathy

I fine-tuned `microsoft/deberta-v3-base` for binary text classification over three epochs. Passages were tokenized with the pretrained DeBERTa tokenizer. The model outputs a probability for class `1`, which represents AI-generated text. Class `0` represents human-written text.

### Train and validate

```bash
python deberta/train.py --help
```

**Training command:** `[Paste the exact command used for the reported run]`

- **Seed:** `42`
- **Split:** Stratified split of `train.csv` into 8,000 training rows and 2,000 validation rows
- **Epochs:** `3`
- **Saved checkpoint:** `~/runs/group21-deberta/best_model`

| Metric           | Result                                        |
| ---------------- | --------------------------------------------- |
| Training AUROC   | [Measure on the original 8,000 training rows] |
| Validation AUROC | 0.991956                                      |
| Training time    | 1 hr 1 min 36 sec                             |
| Kaggle AUROC     | 0.93123                                       |

### Generate Kaggle predictions

```bash
python deberta/predict.py --help
```

**Prediction command:** `[Paste the exact command used to generate submission.csv]`

Prediction uses the saved `best_model` checkpoint and writes one probability of AI authorship per test ID to `submission.csv`.

## Linear SVM | Ke Dong

- **Model and preprocessing:** Word TF-IDF with unigrams and bigrams, `max_features=50000`, `min_df=2`, and `sublinear_tf=True`, followed by `LinearSVC`. Compared `C=[0.1, 1, 10]` and selected `C=1` using validation AUROC. AI probabilities are obtained through sigmoid calibration with three-fold stratified cross-validation within the training partition. Each calibration fold fits its own complete TF-IDF and SVM pipeline.
- **Training and validation command:** From the repository root, run `python submissions/group-21/svm/svm_model.py`.
- **Prediction command:** The same command also refits the calibrated model on all 10,000 labeled examples and generates `submissions/group-21/svm/results/submission.csv`. Training and prediction are implemented in one script; no separate prediction command is required.
- **Seed and validation split:** Random seed `42`; fixed stratified split of 8,000 training examples and 2,000 validation examples from the original `train.csv`. TF-IDF fitting and probability calibration exclude the held-out validation data. Split IDs are saved in `results/split.csv` within the SVM directory.
- **Results:** Selected `C=1`. Training AUROC: **1.000000**; validation AUROC: **0.942530**. Calibrated validation AUROC: **0.942530**. Selected uncalibrated pipeline training time: **3.85 seconds**, excluding evaluation, probability calibration, hyperparameter search, and final refitting.
  
## [Model 3] | [Team member 3]

- **Model and preprocessing:** [Describe the approach]
- **Training and validation command:** `[Command]`
- **Prediction command:** `[Command]`
- **Seed and validation split:** [Details]
- **Results:** [Training AUROC, validation AUROC, and training time]
