"""Train three candidate models on Spaceship Titanic, compare, and package the winner.

Saves the winning sklearn Pipeline as model_artifact/model.tar.gz, ready to
upload to S3 for SageMaker deployment.

Run from a SageMaker Notebook Instance:
    pip install xgboost scikit-learn pandas joblib
    python pipeline.py

Make sure train.csv is in the same directory as pipeline.py before running.
"""

import os
import sys
import tarfile

import joblib

sys.path.insert(0, "src")
from data import CLASS_NAMES, load_dataset, split_data
from evaluate import evaluate_pipeline, print_classification_report, print_comparison, select_best
from models import build_pipelines


ARTIFACT_DIR    = "model_artifact"
MODEL_FILENAME  = "model.joblib"
TARBALL_PATH    = os.path.join(ARTIFACT_DIR, "model.tar.gz")
DATA_PATH       = "train.csv"


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution (train): "
          f"Not Transported={int((y_train==0).sum())}, "
          f"Transported={int((y_train==1).sum())}")

    pipelines = build_pipelines()
    results   = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)

    print_comparison(results)

    best_name     = select_best(results, metric="test_accuracy")
    best_pipeline = pipelines[best_name]
    print(f"\nWinner: {best_name}")
    print(f"\nDetailed report for {best_name}:")
    print_classification_report(best_pipeline, X_test, y_test, CLASS_NAMES)

    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    joblib.dump(best_pipeline, model_path)
    print(f"Saved: {model_path}")

    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(model_path, arcname=MODEL_FILENAME)
    print(f"Packaged: {TARBALL_PATH}")

    print("\nNext steps:")
    print("  Create S3 bucket:\n    aws s3 mb s3://your-bucket-name --region us-east-1")
    print(f"\n  Upload model:\n    aws s3 cp {TARBALL_PATH} s3://your-bucket-name/spaceship/model.tar.gz\n")


if __name__ == "__main__":
    main()
