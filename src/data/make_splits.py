from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create subject-disjoint Bag-of-Lies splits.")
    parser.add_argument("--metadata", default="outputs/metrics/data_check/metadata.csv", help="Metadata CSV from data_check.py.")
    parser.add_argument("--out-dir", default="data/processed/splits", help="Split output directory.")
    parser.add_argument("--train-subjects", type=int, default=25)
    parser.add_argument("--val-subjects", type=int, default=5)
    parser.add_argument("--test-subjects", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--stratified", action="store_true", help="Try to balance subject label ratios across splits.")
    parser.add_argument(
        "--balanced-search",
        action="store_true",
        help="Random-search group-disjoint splits that balance video counts and labels.",
    )
    parser.add_argument("--search-iterations", type=int, default=300000)
    return parser.parse_args()


def subject_stats(df: pd.DataFrame, subject_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, group in df.groupby(subject_col, dropna=False):
        labels = group[label_col].dropna().astype(int) if label_col in group else pd.Series(dtype=int)
        n_lie = int((labels == 1).sum())
        n_truth = int((labels == 0).sum())
        total = int(len(group))
        rows.append(
            {
                "subject_id": subject,
                "num_videos": total,
                "truth": n_truth,
                "lie": n_lie,
                "lie_ratio": n_lie / max(n_truth + n_lie, 1),
            }
        )
    return pd.DataFrame(rows)


def stratified_subject_split(stats: pd.DataFrame, sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    stats = stats.sample(frac=1.0, random_state=seed).sort_values(["lie_ratio", "num_videos"], ascending=[False, False])
    targets = {name: sizes[name] for name in ("train", "val", "test")}
    splits: dict[str, list[str]] = {name: [] for name in targets}
    label_totals = {name: {"lie": 0, "truth": 0, "videos": 0} for name in targets}

    for _, row in stats.iterrows():
        candidates = [name for name, target in targets.items() if len(splits[name]) < target]
        if not candidates:
            break

        def score(name: str) -> tuple[float, float]:
            after_lie = label_totals[name]["lie"] + int(row["lie"])
            after_truth = label_totals[name]["truth"] + int(row["truth"])
            after_total = max(after_lie + after_truth, 1)
            ratio = after_lie / after_total
            size_pressure = len(splits[name]) / max(targets[name], 1)
            return (abs(ratio - 0.5) + size_pressure * 0.1, rng.random() * 1e-6)

        chosen = min(candidates, key=score)
        subject = str(row["subject_id"])
        splits[chosen].append(subject)
        label_totals[chosen]["lie"] += int(row["lie"])
        label_totals[chosen]["truth"] += int(row["truth"])
        label_totals[chosen]["videos"] += int(row["num_videos"])

    return splits


def random_subject_split(subjects: list[str], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end : val_end + sizes["test"]]),
    }


def balanced_search_split(
    stats: pd.DataFrame,
    sizes: dict[str, int],
    seed: int,
    iterations: int,
    target_ratios: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    target_ratios = target_ratios or {"train": 0.6, "val": 0.2, "test": 0.2}
    rng = np.random.default_rng(seed)
    stats = stats.reset_index(drop=True)
    subjects = stats["subject_id"].astype(str).tolist()
    n = stats["num_videos"].to_numpy()
    lie = stats["lie"].to_numpy()
    total_videos = float(n.sum())
    global_lie_ratio = float(lie.sum() / max(total_videos, 1.0))

    # Keep the largest group in train; otherwise one large group can dominate val/test.
    largest_subject = subjects[int(np.argmax(n))]
    eligible = [s for s in subjects if s != largest_subject]
    index = {s: i for i, s in enumerate(subjects)}
    best_score = float("inf")
    best: dict[str, list[str]] | None = None

    for _ in range(iterations):
        val = set(rng.choice(eligible, size=sizes["val"], replace=False).tolist())
        remaining = [s for s in eligible if s not in val]
        test = set(rng.choice(remaining, size=sizes["test"], replace=False).tolist())
        train = set(subjects) - val - test
        if len(train) != sizes["train"]:
            continue

        score = 0.0
        valid = True
        for name, group in (("train", train), ("val", val), ("test", test)):
            ids = [index[s] for s in group]
            split_n = int(n[ids].sum())
            split_lie = int(lie[ids].sum())
            split_truth = split_n - split_lie
            if split_n <= 0 or split_lie <= 0 or split_truth <= 0:
                valid = False
                break
            split_ratio = split_lie / split_n
            score += ((split_n / total_videos) - target_ratios[name]) ** 2 * 12.0
            score += (split_ratio - global_lie_ratio) ** 2 * 2.0
        if not valid:
            continue
        if score < best_score:
            best_score = score
            best = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    if best is None:
        raise RuntimeError("Could not find a balanced split. Try more iterations or different split sizes.")
    return best


def assert_disjoint(splits: dict[str, list[str]]) -> None:
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert len(train & val) == 0, f"Subject leakage train/val: {sorted(train & val)}"
    assert len(train & test) == 0, f"Subject leakage train/test: {sorted(train & test)}"
    assert len(val & test) == 0, f"Subject leakage val/test: {sorted(val & test)}"


def split_summary(df: pd.DataFrame, subject_col: str, label_col: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_videos": int(len(df)),
        "num_subjects": int(df[subject_col].nunique()),
    }
    if label_col in df:
        counts = df[label_col].value_counts(dropna=False).to_dict()
        summary["label_counts"] = {str(k): int(v) for k, v in counts.items()}
    return summary


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    out_dir = ensure_dir(args.out_dir)
    df = pd.read_csv(metadata_path)

    required = {args.subject_column, args.label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {sorted(missing)}")

    df = df.dropna(subset=[args.subject_column, args.label_column]).copy()
    df[args.label_column] = df[args.label_column].astype(int)
    stats = subject_stats(df, args.subject_column, args.label_column)
    subjects = sorted(stats["subject_id"].astype(str).tolist())
    sizes = {"train": args.train_subjects, "val": args.val_subjects, "test": args.test_subjects}
    requested = sum(sizes.values())
    if requested > len(subjects):
        raise ValueError(f"Requested {requested} subjects, but metadata only has {len(subjects)}.")

    if args.balanced_search:
        splits = balanced_search_split(stats, sizes, args.seed, args.search_iterations)
    elif args.stratified:
        splits = stratified_subject_split(stats, sizes, args.seed)
        for key in splits:
            splits[key] = sorted(splits[key])
    else:
        splits = random_subject_split(subjects, sizes, args.seed)

    assert_disjoint(splits)

    summaries: dict[str, Any] = {
        "seed": args.seed,
        "stratified": args.stratified,
        "balanced_search": args.balanced_search,
        "subjects": splits,
        "splits": {},
    }
    for split_name, split_subjects in splits.items():
        split_df = df[df[args.subject_column].astype(str).isin(split_subjects)].copy()
        split_df["split"] = split_name
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
        summaries["splits"][split_name] = split_summary(split_df, args.subject_column, args.label_column)

    stats.to_csv(out_dir / "subject_stats.csv", index=False)
    write_json(out_dir / "subjects.json", splits)
    write_json(out_dir / "split_summary.json", summaries)

    print(f"Wrote splits to {out_dir}")
    print(json.dumps(summaries["splits"], indent=2))


if __name__ == "__main__":
    main()
