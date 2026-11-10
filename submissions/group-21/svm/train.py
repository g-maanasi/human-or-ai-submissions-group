from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"

SEED = 42
C_VALUES = [0.1, 1.0, 10.0]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load training data
data = pd.read_csv(DATA_DIR / "train.csv")

assert data["id"].is_unique, "Training IDs must be unique."
assert data[["text", "label"]].notna().all().all(), (
    "Training data contains missing values."
)
assert set(data["label"].unique()) == {0, 1}, (
    "Labels must be 0 (human) or 1 (AI)."
)


# 3. Create a fixed, stratified train/validation split
train_idx, val_idx = train_test_split(
    np.arange(len(data)),
    test_size=0.2,
    random_state=SEED,
    stratify=data["label"]
)

train = data.iloc[train_idx]
val = data.iloc[val_idx]

assert len(val) == 2000, (
    "Expected 2,000 validation rows from the full training dataset."
)

# Save IDs so teammates can verify the shared split.
split = data[["id"]].copy()
split["split"] = "train"
split.loc[val_idx, "split"] = "validation"
split.to_csv(OUTPUT_DIR / "split.csv", index=False)


# Check for normalized exact duplicates across the split.
def normalize(text):
    return (
        text.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


overlap = set(normalize(train["text"])) & set(normalize(val["text"]))

if overlap:
    raise ValueError(
        "Duplicate texts found across training and validation sets. "
        "Resolve the shared split with teammates before proceeding."
    )


# 4. Define the TF-IDF + Linear SVM pipeline
def build_model(c):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "svm",
            LinearSVC(
                C=c,
                dual="auto",
                max_iter=10000,
                random_state=SEED
            )
        )
    ])


# 5. Compare regularization settings using validation AUROC
records = []

for c in C_VALUES:
    model = build_model(c)

    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    train_seconds = time.perf_counter() - start

    train_scores = model.decision_function(train["text"])
    val_scores = model.decision_function(val["text"])

    records.append({
        "C": c,
        "Train AUROC": roc_auc_score(train["label"], train_scores),
        "Validation AUROC": roc_auc_score(val["label"], val_scores),
        "Training Time (s)": train_seconds
    })

comparison = pd.DataFrame(records)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("Linear SVM: Hyperparameter Comparison")
print(
    comparison.to_string(
        index=False,
        formatters={
            "C": lambda value: f"{value:g}",
            "Train AUROC": lambda value: f"{value:.6f}",
            "Validation AUROC": lambda value: f"{value:.6f}",
            "Training Time (s)": lambda value: f"{value:.2f}"
        }
    )
)

best_c = float(
    comparison.loc[comparison["Validation AUROC"].idxmax(), "C"]
)


# 6. Calibrate probabilities using training-only cross-validation
# Each fold fits its own complete TF-IDF + SVM pipeline.
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)

calibrated_model = CalibratedClassifierCV(
    estimator=build_model(best_c),
    method="sigmoid",
    cv=cv,
    ensemble=False,
    n_jobs=1
)

start = time.perf_counter()
calibrated_model.fit(train["text"], train["label"])
calibrated_training_seconds = time.perf_counter() - start

train_prob = calibrated_model.predict_proba(train["text"])[:, 1]
val_prob = calibrated_model.predict_proba(val["text"])[:, 1]

train_auc = roc_auc_score(train["label"], train_prob)
val_auc = roc_auc_score(val["label"], val_prob)
val_pred = (val_prob >= 0.5).astype(int)

summary = pd.DataFrame([{
    "Selected C": best_c,
    "Train AUROC": train_auc,
    "Validation AUROC": val_auc,
    "Training + Calibration Time (s)": calibrated_training_seconds
}])
summary.to_csv(OUTPUT_DIR / "selected_model_metrics.csv", index=False)

print(f"\nSelected C: {best_c:g}")
print(f"Calibrated Train AUROC: {train_auc:.6f}")
print(f"Calibrated Validation AUROC: {val_auc:.6f}")

cm = pd.DataFrame(
    confusion_matrix(val["label"], val_pred, labels=[0, 1]),
    index=["Actual: Human", "Actual: AI"],
    columns=["Predicted: Human", "Predicted: AI"]
)

print("\nValidation Confusion Matrix (threshold = 0.5)")
print(cm.to_string())

cm.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

validation_results = val[["id", "label"]].copy()
validation_results["p_ai"] = val_prob
validation_results["prediction"] = val_pred
validation_results.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)


# 7. Refit on all labeled data after validation and save the final model
calibrated_model.fit(data["text"], data["label"])

model_path = OUTPUT_DIR / "svm_model.joblib"
joblib.dump(calibrated_model, model_path)

print(f"\nModel saved: {model_path.resolve()}")
