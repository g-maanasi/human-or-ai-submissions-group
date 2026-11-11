"""Train and evaluate Group 21's selected DistilRoBERTa blend on a CUDA machine."""
import argparse
import gc
import json
import os
from pathlib import Path


NOTEBOOKS = Path(__file__).resolve().parent / "distilroberta" / "notebooks"


def run_notebooks(names, export_test_predictions=False):
    # The research notebooks share fitted estimators within a stage. Start the
    # transformer stage in a fresh namespace to release the sparse feature matrices.
    namespace = {"__name__": "__main__", "display": print}
    for name in names:
        with (NOTEBOOKS / name).open() as stream:
            notebook = json.load(stream)
        for index, cell in enumerate(notebook["cells"], start=1):
            tags = cell.get("metadata", {}).get("tags", [])
            if "export-selected-test-predictions" in tags and not export_test_predictions:
                continue
            if cell["cell_type"] == "code":
                location = f"{name}, cell {index}"
                print(f"Running {location}", flush=True)
                source = "".join(cell["source"])
                exec(compile(source, location, "exec"), namespace)


def main(export_test_predictions=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stage", choices=["all", "v2", "transformer", "blend"],
                        default="all", help="blend reuses the v2 predictions and paired validation exports")
    args = parser.parse_args()
    expected = ["train.csv", "test.csv", "sample_submission.csv"]
    if args.stage in {"all", "v2"}:
        expected.append("unlabeled.csv")
    for filename in expected:
        if not (args.data_dir / filename).is_file():
            parser.error(f"Missing {args.data_dir / filename}")
    if args.stage == "blend":
        for filename in ("submission_v2.csv", "val_ids.csv",
                         "distilroberta_v1/v2_validation_probes.csv.gz",
                         "distilroberta_v1/char_validation_scores.csv"):
            if not (args.out_dir / filename).is_file():
                parser.error(f"Run --stage v2 first to create {filename}")
    import torch
    if not torch.cuda.is_available():
        parser.error("Choose a CUDA GPU. For Colab, select a T4 runtime.")
    args.out_dir.mkdir(exist_ok=True, parents=True)
    os.environ.update(DATA_DIR=str(args.data_dir.resolve()), OUT_DIR=str(args.out_dir.resolve()),
                      MPLBACKEND="Agg", FINAL_FIT="1")
    if args.stage in {"v2", "all"}:
        run_notebooks(["01_baselines.ipynb", "02_robustness.ipynb"])
        gc.collect()
        torch.cuda.empty_cache()
    if args.stage in {"transformer", "blend", "all"}:
        notebooks = ["03_distilroberta.ipynb"]
        if args.stage in {"all", "blend"}:
            notebooks.append("04_paired_blend.ipynb")
        run_notebooks(notebooks, export_test_predictions=export_test_predictions)


if __name__ == "__main__":
    main()
