from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import binary_metrics, bootstrap_ci
from src.utils import ensure_dir, set_seed, write_json


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    texts: list[str] = []
    for transcript_path in df["transcript_path"]:
        path_obj = Path(transcript_path)
        texts.append(path_obj.read_text(encoding="utf-8", errors="ignore") if path_obj.exists() else "")
    df = df.copy()
    df["text"] = texts
    return df


def build_models(seed: int) -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )
    return {
        "tfidf_logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def score_model(model: Pipeline, texts: pd.Series) -> Any:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    scores = model.decision_function(texts)
    return scores


def evaluate_model(name: str, model: Pipeline, split_df: pd.DataFrame, out_dir: Path, bootstrap: bool, seed: int) -> dict[str, Any]:
    scores = score_model(model, split_df["text"])
    pred_df = split_df[["video_id", "group_id", "label"]].copy()
    pred_df["score_lie"] = scores
    metrics = binary_metrics(pred_df["label"].to_numpy(), pred_df["score_lie"].to_numpy())
    if bootstrap:
        metrics["bootstrap_95ci"] = bootstrap_ci(
            pred_df["label"].to_numpy(),
            pred_df["score_lie"].to_numpy(),
            seed=seed,
        )
    pred_df.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    write_json(out_dir / f"{name}_metrics.json", metrics)
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    train_df = load_split(args.train)
    val_df = load_split(args.val)
    test_df = load_split(args.test)

    results: dict[str, Any] = {}
    for name, model in build_models(args.seed).items():
        model.fit(train_df["text"], train_df["label"])
        results[name] = {
            "val": evaluate_model(name + "_val", model, val_df, out_dir, False, args.seed),
            "test": evaluate_model(name + "_test", model, test_df, out_dir, True, args.seed),
        }

    write_json(out_dir / "text_baselines.json", results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transcript TF-IDF baselines.")
    parser.add_argument("--train", default="data/processed/splits/real_life/train.csv")
    parser.add_argument("--val", default="data/processed/splits/real_life/val.csv")
    parser.add_argument("--test", default="data/processed/splits/real_life/test.csv")
    parser.add_argument("--out-dir", default="outputs/metrics/real_life_text_baselines")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(args)
    print(results)


if __name__ == "__main__":
    main()
