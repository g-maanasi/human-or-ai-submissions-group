"""Compatibility entry point for the original all-in-one train and predict run."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train import main

    main(export_test_predictions=True)
