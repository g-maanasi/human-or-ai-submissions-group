# Group 21 | Human or AI? Kaggle Competition

**Team members**

- Maanasi Ganapathy (mg2563) — DeBERTa-v3-base
- Ke Dong (kd563) — Linear SVM with TF-IDF features
- [Team member 3] ([NetID]) — [Model or contribution]

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

**Training command:**
Mac:
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 caffeinate -i python3 submissions/group-21/deberta/train.py \
  --train-csv submissions/group-21/train.csv \
  --output-dir ~/runs/group21-deberta \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-batch-size 2`

Windows:
`python submissions/group-21/deberta/train.py \
  --train-csv submissions/group-21/train.csv \
  --output-dir "$HOME/runs/group21-deberta" \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-batch-size 2`

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

## Linear SVM | Ke Dong (kd563)

- **Model and preprocessing:** The selected model combines word and character TF-IDF features, followed by `LinearSVC(C=1)`. Word features use unigrams and bigrams, `max_features=50000`, and `min_df=2`. Character features preserve case and use 3–6-grams, `max_features=200000`, and `min_df=3`. Both feature blocks use `sublinear_tf=True` and L2 normalization. AI probabilities are obtained through sigmoid calibration with three-fold stratified cross-validation within the training partition, using `ensemble=False`. Each calibration fold fits its own complete TF-IDF and SVM pipeline.

- **Training and validation command:** The training entrypoint is `submissions/group-21/svm/train.py`. Run it from the repository root with the data and output arguments supported by the script. 

- **Prediction command:** The prediction entrypoint is `submissions/group-21/svm/predict.py`. 

- **Seed and validation split:** Random seed `42`; a fixed stratified split of the original `train.csv` into 8,000 training examples and 2,000 validation examples. Feature fitting and probability calibration exclude the held-out validation data. The validation set is used for model selection. The selected Round 2 submission was generated after refitting the calibrated pipeline on all 10,000 labeled examples.

- **Experiments and model selection:** The initial word-only SVM compared `C=[0.1, 1, 10]`. Subsequent experiments added character features and compared alternative character representations. In Round 2, `C=1` was held fixed while testing a larger lowercased character vocabulary, word-boundary character features, and case-sensitive 3–6-gram character features. The case-sensitive 3–6-gram configuration achieved the highest validation AUROC and was selected.

- **Results:**

  | Model | Validation AUROC | Kaggle Public AUROC |
  | Original word-only TF-IDF SVM | 0.942530 | 0.84925 |
  | Word + lowercase character 3–5-gram TF-IDF SVM | 0.970462 | 0.87440 |
  | Selected: word + case-sensitive character 3–6-gram TF-IDF SVM*| 0.970807| 0.89627 |

  For the selected configuration, uncalibrated training AUROC was **1.000000**. Validation AUROC before and after sigmoid calibration was **0.970807**. Fitting the selected uncalibrated TF-IDF and SVM pipeline took **43.84 seconds**, excluding evaluation, probability calibration, other experimental configurations, and final refitting.

- **Reproducibility:** Keep raw competition data and trained model artifacts out of Git. Document the actual data location and saved model path in the commands above. The saved prediction pipeline must include the fitted TF-IDF features, SVM classifier, and probability calibration. Dependencies are listed in the team’s `requirements.txt`.

- **Interpretation and limitations:** Adding character features improved both validation and public leaderboard performance relative to the word-only baseline. However, a substantial validation-to-leaderboard gap remains. This is consistent with limited generalization to unfamiliar domains, generators, and modified passages, although the exact causes have not been established. The small validation improvement from the first combined-feature model to Round 2 was not conclusive in the paired bootstrap comparison. Public leaderboard AUROC is not the final private leaderboard score.



## [Model 3] | [Team member 3]

- **Model and preprocessing:** [Describe the approach]
- **Training and validation command:** `[Command]`
- **Prediction command:** `[Command]`
- **Seed and validation split:** [Details]
- **Results:** [Training AUROC, validation AUROC, and training time]
