from pathlib import Path

import joblib
import pandas as pd


# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_PATH = OUTPUT_DIR / "svm_model.joblib"


# 2. Load the trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train.py first."
    )

# Load only a trusted model, such as the file produced by train.py.
calibrated_model = joblib.load(MODEL_PATH)


# 3. Load test data
test = pd.read_csv(DATA_DIR / "test.csv")

assert test["id"].is_unique, "Test IDs must be unique."
assert test["text"].notna().all(), (
    "Test text contains missing values."
)

sample_path = DATA_DIR / "sample_submission.csv"
sample = (
    pd.read_csv(sample_path)
    if sample_path.exists()
    else test[["id"]].copy()
)


# 4. Predict AI probabilities
ai_class_index = list(calibrated_model.classes_).index(1)

test_results = pd.DataFrame({
    "id": test["id"],
    "label": calibrated_model.predict_proba(
        test["text"]
    )[:, ai_class_index]
})


# 5. Match the sample submission order
assert sample["id"].is_unique, (
    "Sample submission IDs must be unique."
)
assert set(sample["id"]) == set(test["id"]), (
    "Sample submission IDs must match test IDs."
)

submission = sample[["id"]].merge(
    test_results,
    on="id",
    how="left",
    validate="one_to_one"
)

assert len(submission) == len(test)
assert submission["label"].notna().all()
assert submission["label"].between(0, 1).all()


# 6. Save submission
submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\nSubmission saved: {len(submission):,} test predictions")
print(f"Submission file: {submission_path.resolve()}")
