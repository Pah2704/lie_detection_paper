from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path("outputs/metrics/paper_ablation_summary")
MANUAL_APPENDIX_MARKER = "## Exploratory Adaptive Prior-KL Pilot"


EXPERIMENTS = {
    "static_only": {
        "label": "Static only",
        "metrics_dir": Path("outputs/metrics/paper_ablation_static"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42],
        "group": "single_stream",
    },
    "flow_only": {
        "label": "Flow only",
        "metrics_dir": Path("outputs/metrics/paper_ablation_flow"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42],
        "group": "single_stream",
    },
    "audio_only": {
        "label": "Audio only",
        "metrics_dir": Path("outputs/metrics/paper_ablation_audio"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42],
        "group": "single_stream",
    },
    "gated_no_prior": {
        "label": "Gated prior-init w/o Prior-KL",
        "metrics_dir": Path("outputs/metrics/paper_gated_no_prior"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42, 123, 2025],
        "group": "prior_kl",
    },
    "gated_neutral_no_prior": {
        "label": "Gated neutral no-prior",
        "metrics_dir": Path("outputs/metrics/paper_gated_neutral_no_prior"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42, 123, 2025],
        "group": "prior_kl",
    },
    "gated_prior_kl": {
        "label": "Gated Prior-KL",
        "metrics_dir": Path("outputs/metrics/paper_gated_prior_kl"),
        "folds": ["fold1", "fold2", "fold3"],
        "seeds": [42, 123, 2025],
        "group": "prior_kl",
    },
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(exp_key: str, exp: dict[str, Any], fold: str, seed: int) -> dict[str, Any] | None:
    summary_path = exp["metrics_dir"] / f"{fold}_seed{seed}_summary.json"
    summary = load_json(summary_path)
    if summary is None:
        return None
    test = summary["test_metrics"]
    calibrated = test.get("calibrated_threshold_metrics", {})
    return {
        "experiment": exp_key,
        "method": exp["label"],
        "group": exp["group"],
        "fold": fold,
        "seed": seed,
        "summary_path": str(summary_path),
        "accuracy_at_0p5": test.get("accuracy"),
        "balanced_accuracy_at_0p5": test.get("balanced_accuracy"),
        "f1_lie_at_0p5": test.get("f1_lie"),
        "macro_f1_at_0p5": test.get("macro_f1"),
        "auc_roc": test.get("auc_roc"),
        "auc_pr": test.get("auc_pr"),
        "calibrated_accuracy": calibrated.get("accuracy"),
        "calibrated_balanced_accuracy": calibrated.get("balanced_accuracy"),
        "calibrated_precision_lie": calibrated.get("precision_lie"),
        "calibrated_recall_lie": calibrated.get("recall_lie"),
        "calibrated_f1_lie": calibrated.get("f1_lie"),
        "calibrated_macro_f1": calibrated.get("macro_f1"),
        "calibrated_threshold": summary.get("threshold_calibration", {}).get("threshold"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_metric": summary.get("best_val_metric"),
        "loss_name": summary.get("loss_name"),
    }


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 100:.2f}"


def mean_std(group: pd.DataFrame, column: str) -> str:
    values = group[column].dropna()
    if values.empty:
        return ""
    if len(values) == 1:
        return pct(float(values.iloc[0]))
    return f"{values.mean() * 100:.2f} +- {values.std(ddof=1) * 100:.2f}"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc_roc",
        "balanced_accuracy_at_0p5",
        "calibrated_balanced_accuracy",
        "calibrated_f1_lie",
        "calibrated_macro_f1",
    ]
    summary_rows: list[dict[str, Any]] = []
    for (group_name, experiment, method), part in rows.groupby(["group", "experiment", "method"], sort=False):
        item: dict[str, Any] = {
            "group": group_name,
            "experiment": experiment,
            "method": method,
            "num_runs": len(part),
            "folds": ",".join(sorted(part["fold"].unique())),
            "seeds": ",".join(str(seed) for seed in sorted(part["seed"].unique())),
        }
        for metric in metrics:
            values = part[metric].dropna()
            item[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
            item[f"{metric}_paper"] = mean_std(part, metric)
        summary_rows.append(item)
    return pd.DataFrame(summary_rows)


def markdown_table(df: pd.DataFrame, group_name: str) -> str:
    part = df[df["group"] == group_name].copy()
    if part.empty:
        return "_No completed runs yet._"
    cols = [
        "method",
        "num_runs",
        "folds",
        "seeds",
        "auc_roc_paper",
        "balanced_accuracy_at_0p5_paper",
        "calibrated_balanced_accuracy_paper",
        "calibrated_f1_lie_paper",
        "calibrated_macro_f1_paper",
    ]
    out = part[cols].rename(
        columns={
            "method": "Method",
            "num_runs": "Runs",
            "folds": "Folds",
            "seeds": "Seeds",
            "auc_roc_paper": "AUC",
            "balanced_accuracy_at_0p5_paper": "BA@0.5",
            "calibrated_balanced_accuracy_paper": "Cal. BA",
            "calibrated_f1_lie_paper": "Cal. F1 Lie",
            "calibrated_macro_f1_paper": "Cal. Macro F1",
        }
    )
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


PAIRED_METRICS = [
    ("auc_roc", "AUC"),
    ("balanced_accuracy_at_0p5", "BA@0.5"),
    ("calibrated_balanced_accuracy", "Cal. BA"),
    ("calibrated_f1_lie", "Cal. F1 Lie"),
    ("calibrated_macro_f1", "Cal. Macro F1"),
]


def paired_prior_delta(raw: pd.DataFrame, baseline_experiment: str = "gated_no_prior") -> pd.DataFrame:
    key_cols = ["fold", "seed"]
    value_cols = [metric for metric, _ in PAIRED_METRICS]
    no_prior = raw[raw["experiment"] == baseline_experiment][key_cols + value_cols]
    prior = raw[raw["experiment"] == "gated_prior_kl"][key_cols + value_cols]
    if no_prior.empty or prior.empty:
        return pd.DataFrame()
    paired = prior.merge(no_prior, on=key_cols, suffixes=("_prior_kl", "_no_prior"))
    for metric, _ in PAIRED_METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_prior_kl"] - paired[f"{metric}_no_prior"]
    return paired


def paired_delta_summary_table(paired: pd.DataFrame) -> str:
    if paired.empty:
        return "_No paired Prior-KL/no-prior runs yet._"
    headers = ["Metric", "Mean delta pp", "Std pp", "Prior-KL wins", "No-prior wins", "Ties"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for metric, label in PAIRED_METRICS:
        deltas = paired[f"{metric}_delta"].dropna()
        if deltas.empty:
            continue
        wins = int((deltas > 1e-12).sum())
        losses = int((deltas < -1e-12).sum())
        ties = int(len(deltas) - wins - losses)
        std = deltas.std(ddof=1) if len(deltas) > 1 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{deltas.mean() * 100:+.2f}",
                    f"{std * 100:.2f}",
                    f"{wins}/{len(deltas)}",
                    f"{losses}/{len(deltas)}",
                    f"{ties}/{len(deltas)}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def paired_delta_detail_table(paired: pd.DataFrame) -> str:
    if paired.empty:
        return "_No paired Prior-KL/no-prior runs yet._"
    headers = ["Fold", "Seed"] + [label for _, label in PAIRED_METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in paired.sort_values(["fold", "seed"]).iterrows():
        values = [str(row["fold"]), str(int(row["seed"]))]
        values.extend(f"{row[f'{metric}_delta'] * 100:+.2f}" for metric, _ in PAIRED_METRICS)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paper_reading_notes(summary: pd.DataFrame, paired_prior_init: pd.DataFrame, paired_neutral: pd.DataFrame) -> list[str]:
    notes = [
        "## How to Read for Paper",
        "",
        "- `Gated neutral no-prior` uses uniform `model.gate_init_weights=[1/3, 1/3, 1/3]` and `training.gate_prior_weight=0.0`.",
        "- `Gated prior-init w/o Prior-KL` keeps `model.gate_init_weights=[0.10, 0.45, 0.45]` but sets `training.gate_prior_weight=0.0`.",
        "- `Gated Prior-KL` uses `training.gate_prior_weight=0.2` and prior `[0.10, 0.45, 0.45]`, adding a KL penalty only when the weight is positive.",
        "- Read AUC as ranking quality, Cal. BA as the validation-calibrated operating point, and Cal. F1 Lie as lie-class detection quality.",
    ]
    no_prior = summary[summary["experiment"] == "gated_no_prior"]
    prior = summary[summary["experiment"] == "gated_prior_kl"]
    if no_prior.empty or prior.empty:
        return notes
    no_row = no_prior.iloc[0]
    prior_row = prior.iloc[0]
    deltas = {
        label: (float(prior_row[f"{metric}_mean"]) - float(no_row[f"{metric}_mean"])) * 100
        for metric, label in PAIRED_METRICS
        if pd.notna(prior_row[f"{metric}_mean"]) and pd.notna(no_row[f"{metric}_mean"])
    }
    if deltas:
        notes.extend(
            [
                "",
                "Short interpretation:",
                "",
                f"- Prior-KL is essentially tied on AUC ({deltas['AUC']:+.2f} pp) and slightly lower on Cal. BA ({deltas['Cal. BA']:+.2f} pp).",
                f"- Prior-KL improves lie-class F1 by {deltas['Cal. F1 Lie']:+.2f} pp and macro F1 by {deltas['Cal. Macro F1']:+.2f} pp.",
                "- The safest claim is that Prior-KL shifts the gated model toward better lie-class behavior, not that it uniformly improves every metric.",
            ]
        )
    if not paired_prior_init.empty:
        lie_wins = int((paired_prior_init["calibrated_f1_lie_delta"] > 1e-12).sum())
        notes.append(f"- Against prior-init w/o Prior-KL, Prior-KL improves Cal. F1 Lie in {lie_wins}/{len(paired_prior_init)} paired runs.")
    if not paired_neutral.empty:
        lie_wins = int((paired_neutral["calibrated_f1_lie_delta"] > 1e-12).sum())
        notes.append(f"- Against neutral no-prior, Prior-KL improves Cal. F1 Lie in {lie_wins}/{len(paired_neutral)} paired runs.")
    return notes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_md_path = OUT_DIR / "paper_ablation_summary.md"
    manual_appendix: str | None = None
    if summary_md_path.exists():
        existing = summary_md_path.read_text(encoding="utf-8")
        marker_index = existing.find(MANUAL_APPENDIX_MARKER)
        if marker_index >= 0:
            manual_appendix = existing[marker_index:].rstrip()

    rows = [
        row
        for exp_key, exp in EXPERIMENTS.items()
        for fold in exp["folds"]
        for seed in exp["seeds"]
        if (row := metric_row(exp_key, exp, fold, seed)) is not None
    ]
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "paper_ablation_runs.csv", index=False)
    if raw.empty:
        summary_md_path.write_text("# Paper Ablation Summary\n\nNo completed runs yet.\n", encoding="utf-8")
        return

    summary = summarize(raw)
    summary.to_csv(OUT_DIR / "paper_ablation_summary.csv", index=False)
    paired_prior_init = paired_prior_delta(raw, "gated_no_prior")
    paired_neutral = paired_prior_delta(raw, "gated_neutral_no_prior")
    if not paired_prior_init.empty:
        paired_prior_init.to_csv(OUT_DIR / "paper_prior_kl_vs_prior_init_paired_deltas.csv", index=False)
    if not paired_neutral.empty:
        paired_neutral.to_csv(OUT_DIR / "paper_prior_kl_vs_neutral_paired_deltas.csv", index=False)

    lines = [
        "# Paper Ablation Summary",
        "",
        "Source configs:",
        "",
        "- Single-stream: `configs/paper_ablation_static.yaml`, `configs/paper_ablation_flow.yaml`, `configs/paper_ablation_audio.yaml`.",
        "- Prior comparison: `configs/paper_gated_neutral_no_prior.yaml`, `configs/paper_gated_no_prior.yaml`, `configs/paper_gated_prior_kl.yaml`.",
        "- Exploratory vote-score-sum: `configs/paper_gated_vote_score_sum_g05_floor005.yaml`.",
        "- Focal pilot: `configs/paper_gated_vote_score_sum_g05_floor005_focal.yaml`.",
        "",
        *paper_reading_notes(summary, paired_prior_init, paired_neutral),
        "",
        "Main-method interpretation after the vote-score-sum expansion:",
        "",
        "- `Vote-score-sum gamma=0.5 floor=0.05` is stronger than fixed Prior-KL on AUC (+1.90 pp), Cal. BA (+2.05 pp), Cal. F1 Lie (+1.78 pp), and Cal. Macro F1 (+2.43 pp).",
        "- It is also stronger than prior-init w/o Prior-KL on AUC (+1.86 pp), Cal. BA (+1.74 pp), Cal. F1 Lie (+5.77 pp), and Cal. Macro F1 (+2.70 pp).",
        "- Its weak point is raw threshold-0.5 Lie F1, mainly because some runs are poorly calibrated at 0.5 despite good AUC. Use validation-calibrated threshold metrics for the operating-point claim.",
        "",
        "## Single-Stream Fold1-Fold3",
        "",
        markdown_table(summary, "single_stream"),
        "",
        "## Prior-KL Contribution",
        "",
        markdown_table(summary, "prior_kl"),
        "",
        "## Paired Prior-KL Delta vs Prior-Init w/o Prior-KL",
        "",
        "Delta is `Gated Prior-KL - Gated prior-init w/o Prior-KL` in percentage points, paired by fold and seed.",
        "",
        paired_delta_summary_table(paired_prior_init),
        "",
        "## Paired Delta Details vs Prior-Init w/o Prior-KL",
        "",
        paired_delta_detail_table(paired_prior_init),
        "",
        "## Paired Prior-KL Delta vs Neutral No-Prior",
        "",
        "Delta is `Gated Prior-KL - Gated neutral no-prior` in percentage points, paired by fold and seed.",
        "",
        paired_delta_summary_table(paired_neutral),
        "",
        "## Paired Delta Details vs Neutral No-Prior",
        "",
        paired_delta_detail_table(paired_neutral),
    ]
    if manual_appendix:
        lines.extend(["", manual_appendix])
    else:
        lines.extend(["", "## Missing Runs", ""])
        missing: list[str] = []
        for exp_key, exp in EXPERIMENTS.items():
            for fold in exp["folds"]:
                for seed in exp["seeds"]:
                    path = exp["metrics_dir"] / f"{fold}_seed{seed}_summary.json"
                    if not path.exists():
                        missing.append(f"- `{exp_key}` {fold} seed {seed}: `{path}`")
        lines.extend(missing or ["All planned runs are complete."])
    lines.append("")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
