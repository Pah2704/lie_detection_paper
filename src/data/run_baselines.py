from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluate import binary_metrics
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-learning and metadata baselines.")
    parser.add_argument("--train", default="data/processed/splits/train.csv")
    parser.add_argument("--val", default="data/processed/splits/val.csv")
    parser.add_argument("--test", default="data/processed/splits/test.csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--out-dir", default="outputs/metrics/baselines")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r} in {path}")
    return df.dropna(subset=[label_col]).copy()


def eval_model(name: str, clf, train: pd.DataFrame, target: pd.DataFrame, feature_cols: list[str], label_col: str) -> dict:
    x_train = train[feature_cols].fillna(0).to_numpy() if feature_cols else np.zeros((len(train), 1))
    y_train = train[label_col].astype(int).to_numpy()
    x_target = target[feature_cols].fillna(0).to_numpy() if feature_cols else np.zeros((len(target), 1))
    y_target = target[label_col].astype(int).to_numpy()
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(x_target)[:, 1]
    else:
        score = clf.decision_function(x_target)
        score = (score - score.min()) / max(score.max() - score.min(), 1e-12)
    return {"name": name, **binary_metrics(y_target, score)}


def main() -> None:
    args = parse_args()
    train = load_split(args.train, args.label_column)
    val = load_split(args.val, args.label_column)
    test = load_split(args.test, args.label_column)
    out_dir = ensure_dir(args.out_dir)

    feature_cols = [c for c in ("duration_sec", "fps", "num_frames", "width", "height", "file_size_bytes") if c in train.columns]
    baselines = {
        "majority": (DummyClassifier(strategy="most_frequent"), []),
        "random_stratified": (DummyClassifier(strategy="stratified", random_state=args.seed), []),
    }
    if feature_cols:
        baselines["metadata_logreg"] = (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed)),
            feature_cols,
        )
        baselines["metadata_svm"] = (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=args.seed)),
            feature_cols,
        )

    results = {"feature_cols": feature_cols, "val": {}, "test": {}}
    for name, (clf, cols) in baselines.items():
        results["val"][name] = eval_model(name, clf, train, val, cols, args.label_column)
        results["test"][name] = eval_model(name, clf, train, test, cols, args.label_column)

    write_json(Path(out_dir) / "non_learning_baselines.json", results)
    print(f"Wrote baseline metrics to {out_dir}")


if __name__ == "__main__":
    main()
