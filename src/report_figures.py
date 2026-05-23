from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(".")
DEFAULT_OUT_DIR = Path("docs/figures/report")
FINAL_REPORT_DIR = Path("outputs/metrics/final_report_clean_temporal_mask_soft_cross")


FIGURE_SPECS = [
    ("fig01_problem_pipeline", "Problem pipeline"),
    ("fig02_micro_expression_regions", "Micro-expression facial regions schematic"),
    ("fig03_dolos_internal_variation", "DOLOS internal variation schematic"),
    ("fig04_preprocessing_pipeline", "Preprocessing pipeline"),
    ("fig05_face_valid_timeline", "Face-valid timeline example"),
    ("fig06_optical_flow_schematic", "Optical-flow schematic"),
    ("fig07_window_aggregation", "Clip-to-window aggregation"),
    ("fig08_three_stream_architecture", "Three-stream architecture"),
    ("fig09_cross_attention_block", "Cross-attention block"),
    ("fig10_gated_logit_fusion", "Gated logit fusion"),
    ("fig11_prediction_level_ensemble", "Prediction-level ensemble"),
    ("fig12_dolos_protocol", "DOLOS-only evaluation protocol"),
    ("graph01_dolos_label_distribution", "DOLOS label distribution"),
    ("graph02_dolos_method_comparison", "DOLOS method comparison"),
    ("graph03_dolos_per_fold_auc_ba", "DOLOS per-fold AUC/BA"),
    ("graph04_ours_vs_paper_auc", "Ours vs DOLOS paper AUC"),
    ("graph05_stream_ablation", "Fold3 stream ablation"),
    ("graph06_error_by_host", "Error by host"),
    ("graph07_episode_error_heatmap", "Worst episode BA heatmap"),
    ("graph08_confusion_matrices", "Final ensemble confusion matrices"),
    ("graph11_dolos_score_distribution", "DOLOS final score distribution"),
    ("graph12_dolos_roc_curve", "DOLOS final ROC curves"),
    ("graph13_dolos_pr_curve", "DOLOS final precision-recall curve"),
    ("graph14_dolos_threshold_sweep", "DOLOS threshold sweep"),
    ("graph15_dolos_error_type_counts", "DOLOS error type counts"),
    ("graph16_contamination_metric_comparison", "Contamination metric comparison"),
    ("graph17_dolos_score_by_host", "DOLOS score by host"),
]


COLORS = {
    "blue": "#2F6B9A",
    "orange": "#D9853B",
    "green": "#4E8F5A",
    "red": "#B54A4A",
    "purple": "#7A5AA6",
    "gray": "#6C757D",
    "light": "#EEF2F5",
    "dark": "#263238",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures and graphs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def pct(series: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(series, dtype=float) * 100.0


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    fc: str = "#EEF2F5",
    ec: str = "#263238",
    fontsize: int = 10,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        wrap=True,
    )
    return box


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#263238") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    ax.add_patch(arrow)


def diagram_canvas(width: float = 12, height: float = 5) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def simple_pipeline(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(12, 3)
    boxes = [
        (0.04, "Input video"),
        (0.22, "Face crop\n+ audio\n+ optical flow"),
        (0.44, "Three-stream\nmodel"),
        (0.64, "Fusion\nattention/gate"),
        (0.82, "P(lie)\ntruth/lie"),
    ]
    for x, label in boxes:
        add_box(ax, (x, 0.34), 0.14, 0.32, label, fc=COLORS["light"])
    for i in range(len(boxes) - 1):
        add_arrow(ax, (boxes[i][0] + 0.14, 0.5), (boxes[i + 1][0], 0.5))
    ax.set_title("Micro-Expression for Lie Detection: Problem Pipeline", y=0.92)
    save(fig, out_dir, "fig01_problem_pipeline", dpi)


def micro_expression_regions(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(6, 6)
    ax.add_patch(Ellipse((0.5, 0.52), 0.52, 0.68, facecolor="#F4D6C6", edgecolor=COLORS["dark"], linewidth=1.5))
    ax.add_patch(Ellipse((0.39, 0.61), 0.11, 0.055, facecolor="white", edgecolor=COLORS["dark"], linewidth=1.1))
    ax.add_patch(Ellipse((0.61, 0.61), 0.11, 0.055, facecolor="white", edgecolor=COLORS["dark"], linewidth=1.1))
    ax.add_patch(Circle((0.39, 0.61), 0.015, color=COLORS["dark"]))
    ax.add_patch(Circle((0.61, 0.61), 0.015, color=COLORS["dark"]))
    ax.plot([0.35, 0.45], [0.70, 0.72], color=COLORS["dark"], linewidth=2)
    ax.plot([0.55, 0.65], [0.72, 0.70], color=COLORS["dark"], linewidth=2)
    ax.plot([0.5, 0.49, 0.52], [0.58, 0.48, 0.48], color=COLORS["dark"], linewidth=1.2)
    ax.plot([0.42, 0.58], [0.36, 0.35], color=COLORS["red"], linewidth=2)
    regions = [
        ((0.16, 0.76), "Brows\nAU1/AU2", (0.35, 0.71)),
        ((0.78, 0.64), "Eyes\nblink/gaze", (0.64, 0.61)),
        ((0.12, 0.42), "Mouth\nAU12/AU15", (0.43, 0.36)),
        ((0.73, 0.27), "Head/face\nmotion", (0.59, 0.43)),
    ]
    for text_xy, label, target in regions:
        add_box(ax, text_xy, 0.18, 0.10, label, fc="#F7F3E8", fontsize=9)
        add_arrow(ax, (text_xy[0] + 0.09, text_xy[1] + 0.05), target, COLORS["gray"])
    ax.set_title("Facial Regions Used by Micro-Expression Cues", y=0.96)
    save(fig, out_dir, "fig02_micro_expression_regions", dpi)


def dolos_internal_variation(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(11, 5)
    add_box(ax, (0.08, 0.62), 0.24, 0.18, "Host / identity\nappearance bias\nspeaking style", fc="#E8F2FA")
    add_box(ax, (0.38, 0.62), 0.24, 0.18, "Episode / stage\nlighting\ncamera angle", fc="#F7F3E8")
    add_box(ax, (0.68, 0.62), 0.24, 0.18, "Clip / shot\nface stability\nwindow length", fc="#EAF4EA")
    add_box(ax, (0.19, 0.22), 0.26, 0.18, "Audio variation\nprosody\nbackground mix", fc="#F2ECF8")
    add_box(ax, (0.55, 0.22), 0.26, 0.18, "Visual variation\nface crop\nmotion / flow", fc="#FFF1E6")
    center = (0.50, 0.50)
    ax.text(center[0], center[1], "DOLOS\n3-fold\nvariance", ha="center", va="center", fontsize=14, weight="bold", color=COLORS["dark"])
    for target in [(0.20, 0.62), (0.50, 0.62), (0.80, 0.62), (0.32, 0.40), (0.68, 0.40)]:
        add_arrow(ax, center, target, COLORS["gray"])
    ax.set_title("Internal Variation Within DOLOS", y=0.95)
    save(fig, out_dir, "fig03_dolos_internal_variation", dpi)


def preprocessing_pipeline(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(13, 4)
    boxes = [
        (0.03, "Video clip"),
        (0.18, "Audio\n16 kHz WAV"),
        (0.34, "MediaPipe\nface mesh"),
        (0.50, "Dominant\nface track"),
        (0.66, "face_valid\nmask"),
        (0.82, "TV-L1\noptical flow"),
    ]
    for x, label in boxes:
        add_box(ax, (x, 0.42), 0.12, 0.25, label, fc=COLORS["light"], fontsize=9)
    for i in range(len(boxes) - 1):
        add_arrow(ax, (boxes[i][0] + 0.12, 0.545), (boxes[i + 1][0], 0.545))
    add_box(ax, (0.35, 0.12), 0.30, 0.16, "Windowed dataset\nRGB + flow + audio + face_valid", fc="#EAF6EA", fontsize=10)
    add_arrow(ax, (0.5, 0.42), (0.5, 0.28), COLORS["green"])
    ax.set_title("Preprocessing Pipeline", y=0.90)
    save(fig, out_dir, "fig04_preprocessing_pipeline", dpi)


def face_valid_timeline(out_dir: Path, dpi: int) -> None:
    candidates = sorted(Path("data/processed/faces_224_clean/dolos/face_valid").glob("*.csv"))
    if candidates:
        path = candidates[0]
        df = pd.read_csv(path).head(180)
        x = df["frame_index"] if "frame_index" in df.columns else np.arange(len(df))
        y = df["face_valid"].astype(float) if "face_valid" in df.columns else np.ones(len(df))
    else:
        x = np.arange(180)
        y = np.ones(180)
        y[:20] = 0
        y[80:92] = 0
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(x, 0, y, step="mid", color=COLORS["green"], alpha=0.45)
    ax.plot(x, y, color=COLORS["green"], linewidth=1.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("face_valid")
    ax.set_title("Face Validity Timeline Example")
    ax.text(0.01, 0.90, "No raw/cropped frame is shown; only validity metadata.", transform=ax.transAxes, fontsize=9, color=COLORS["gray"])
    save(fig, out_dir, "fig05_face_valid_timeline", dpi)


def optical_flow_schematic(out_dir: Path, dpi: int) -> None:
    x, y = np.meshgrid(np.linspace(-2, 2, 18), np.linspace(-2, 2, 18))
    u = -y * np.exp(-(x**2 + y**2) / 5)
    v = x * np.exp(-(x**2 + y**2) / 5)
    fig, ax = plt.subplots(figsize=(6, 5))
    speed = np.sqrt(u**2 + v**2)
    ax.quiver(x, y, u, v, speed, cmap="viridis", angles="xy")
    ax.add_patch(Ellipse((0, 0), 2.0, 2.6, fill=False, linewidth=1.8, edgecolor=COLORS["dark"]))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optical Flow Captures Facial Motion")
    save(fig, out_dir, "fig06_optical_flow_schematic", dpi)


def window_aggregation(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(12, 4)
    add_box(ax, (0.05, 0.58), 0.88, 0.12, "Full clip", fc="#E8F2FA")
    starts = [0.08, 0.22, 0.36, 0.50, 0.64]
    scores = [0.41, 0.55, 0.62, 0.58, 0.47]
    for i, (x, s) in enumerate(zip(starts, scores, strict=True), start=1):
        add_box(ax, (x, 0.30), 0.16, 0.16, f"Window {i}\nscore={s:.2f}", fc="#FFF1E6", fontsize=9)
        add_arrow(ax, (x + 0.08, 0.58), (x + 0.08, 0.46), COLORS["gray"])
    add_box(ax, (0.79, 0.12), 0.16, 0.12, "Mean score\n0.53", fc="#EAF6EA", fontsize=10)
    for x in starts:
        add_arrow(ax, (x + 0.08, 0.30), (0.87, 0.24), COLORS["green"])
    ax.set_title("Clip-to-Window Evaluation and Score Aggregation", y=0.92)
    save(fig, out_dir, "fig07_window_aggregation", dpi)


def three_stream_architecture(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(13, 6)
    streams = [
        (0.08, 0.72, "Face RGB\nframes", "ViT\nspatial stream", "#E8F2FA"),
        (0.08, 0.45, "Optical\nflow", "Flow CNN\nmotion stream", "#EAF6EA"),
        (0.08, 0.18, "Audio\nwaveform", "Wav2Vec2\naudio stream", "#FFF1E6"),
    ]
    for x, y, inp, model, color in streams:
        add_box(ax, (x, y), 0.14, 0.13, inp, fc=color, fontsize=9)
        add_box(ax, (0.30, y), 0.16, 0.13, model, fc=color, fontsize=9)
        add_arrow(ax, (x + 0.14, y + 0.065), (0.30, y + 0.065))
        add_arrow(ax, (0.46, y + 0.065), (0.58, 0.50))
    add_box(ax, (0.58, 0.39), 0.16, 0.22, "Fusion\nattention / gate", fc="#F4EEF8", fontsize=10)
    add_box(ax, (0.82, 0.43), 0.12, 0.14, "Lie/truth\nlogits", fc=COLORS["light"], fontsize=10)
    add_arrow(ax, (0.74, 0.50), (0.82, 0.50))
    ax.set_title("Three-Stream Multimodal Architecture", y=0.95)
    save(fig, out_dir, "fig08_three_stream_architecture", dpi)


def cross_attention_block(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(12, 4.5)
    add_box(ax, (0.08, 0.64), 0.18, 0.13, "Audio tokens\nQ", fc="#FFF1E6")
    add_box(ax, (0.08, 0.28), 0.18, 0.13, "Visual tokens\nK, V", fc="#E8F2FA")
    add_box(ax, (0.39, 0.43), 0.18, 0.18, "Multi-head\nattention", fc="#F4EEF8")
    add_box(ax, (0.66, 0.43), 0.14, 0.18, "Residual\n+ LayerNorm", fc="#EAF6EA")
    add_box(ax, (0.88, 0.43), 0.10, 0.18, "Classifier", fc=COLORS["light"])
    add_arrow(ax, (0.26, 0.705), (0.39, 0.56))
    add_arrow(ax, (0.26, 0.345), (0.39, 0.48))
    add_arrow(ax, (0.57, 0.52), (0.66, 0.52))
    add_arrow(ax, (0.80, 0.52), (0.88, 0.52))
    ax.set_title("Cross-Attention Fusion Block", y=0.93)
    save(fig, out_dir, "fig09_cross_attention_block", dpi)


def gated_logit_fusion(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(13, 5)
    ys = [0.72, 0.48, 0.24]
    labels = [("Spatial features", "logits_s"), ("Flow features", "logits_f"), ("Audio features", "logits_a")]
    colors = ["#E8F2FA", "#EAF6EA", "#FFF1E6"]
    for y, (feat, logit_label), color in zip(ys, labels, colors, strict=True):
        add_box(ax, (0.06, y), 0.16, 0.12, feat, fc=color, fontsize=9)
        add_box(ax, (0.33, y), 0.13, 0.12, logit_label, fc=color, fontsize=9)
        add_arrow(ax, (0.22, y + 0.06), (0.33, y + 0.06))
        add_arrow(ax, (0.46, y + 0.06), (0.69, 0.50))
    add_box(ax, (0.33, 0.40), 0.18, 0.12, "Gate MLP\nsoftmax weights", fc="#F4EEF8", fontsize=9)
    for y in ys:
        add_arrow(ax, (0.22, y + 0.06), (0.33, 0.46), COLORS["gray"])
    add_box(ax, (0.69, 0.40), 0.17, 0.20, "Weighted sum\nof logits", fc="#EAF6EA")
    add_box(ax, (0.91, 0.44), 0.08, 0.12, "CE loss", fc=COLORS["light"], fontsize=9)
    add_arrow(ax, (0.86, 0.50), (0.91, 0.50))
    ax.set_title("Gated Prior-KL Logit Fusion", y=0.93)
    save(fig, out_dir, "fig10_gated_logit_fusion", dpi)


def prediction_level_ensemble(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(12, 4)
    add_box(ax, (0.08, 0.62), 0.22, 0.12, "Cross-attention\nscore", fc="#E8F2FA")
    add_box(ax, (0.08, 0.28), 0.22, 0.12, "Gated prior-KL\nscore", fc="#FFF1E6")
    add_box(ax, (0.43, 0.43), 0.22, 0.16, "Grid-search weights\non DOLOS val", fc="#F4EEF8")
    add_box(ax, (0.78, 0.43), 0.16, 0.16, "Final score\n+ threshold", fc="#EAF6EA")
    add_arrow(ax, (0.30, 0.68), (0.43, 0.53))
    add_arrow(ax, (0.30, 0.34), (0.43, 0.49))
    add_arrow(ax, (0.65, 0.51), (0.78, 0.51))
    ax.set_title("Prediction-Level Ensemble", y=0.92)
    save(fig, out_dir, "fig11_prediction_level_ensemble", dpi)


def dolos_protocol(out_dir: Path, dpi: int) -> None:
    fig, ax = diagram_canvas(12, 4)
    add_box(ax, (0.05, 0.55), 0.22, 0.18, "DOLOS train\nlearn parameters", fc="#E8F2FA")
    add_box(ax, (0.34, 0.55), 0.22, 0.18, "DOLOS val\nselect checkpoint\nthreshold/weights", fc="#EAF6EA", fontsize=9)
    add_box(ax, (0.63, 0.55), 0.18, 0.18, "DOLOS test\nin-domain result", fc="#FFF1E6", fontsize=9)
    add_box(ax, (0.63, 0.18), 0.25, 0.16, "Final report\n3-fold DOLOS mean", fc="#F8EAEA", fontsize=9)
    add_arrow(ax, (0.27, 0.64), (0.34, 0.64))
    add_arrow(ax, (0.56, 0.64), (0.63, 0.64))
    add_arrow(ax, (0.45, 0.55), (0.68, 0.34), COLORS["red"])
    ax.set_title("Evaluation Protocol: DOLOS 3-Fold Only", y=0.92)
    save(fig, out_dir, "fig12_dolos_protocol", dpi)


def graph_label_distribution(out_dir: Path, dpi: int) -> None:
    rows = []
    for fold in ["fold1", "fold2", "fold3"]:
        for split in ["train", "val", "test"]:
            path = Path(f"data/processed/splits/dolos/cache_filtered/{fold}_{split}.csv")
            if not path.exists():
                continue
            df = pd.read_csv(path)
            counts = df["label"].astype(int).value_counts().to_dict()
            rows.append({"fold": fold, "split": split, "truth": counts.get(0, 0), "lie": counts.get(1, 0)})
    data = pd.DataFrame(rows)
    data_long = data.melt(id_vars=["fold", "split"], value_vars=["truth", "lie"], var_name="label", value_name="clips")
    split_order = {"train": 0, "val": 1, "test": 2}
    data_long["fold_split"] = data_long["fold"] + "\n" + data_long["split"]
    ordered = (
        data.assign(split_order=data["split"].map(split_order))
        .sort_values(["fold", "split_order"])
        .assign(fold_split=lambda frame: frame["fold"] + "\n" + frame["split"])["fold_split"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=data_long, x="fold_split", y="clips", hue="label", order=ordered, ax=ax, palette=[COLORS["blue"], COLORS["orange"]], errorbar=None)
    ax.set_title("DOLOS Label Distribution by Fold and Split")
    ax.set_xlabel("")
    ax.set_ylabel("Number of clips")
    save(fig, out_dir, "graph01_dolos_label_distribution", dpi)


def graph_dolos_method_comparison(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_results_table.csv")
    metrics = ["auc_roc", "calibrated_balanced_accuracy", "calibrated_f1_lie"]
    data = df[["method_label", *metrics]].melt("method_label", var_name="metric", value_name="value")
    data["percent"] = pct(data["value"])
    labels = {"auc_roc": "AUC", "calibrated_balanced_accuracy": "Cal. BA", "calibrated_f1_lie": "Cal. F1 Lie"}
    data["metric"] = data["metric"].map(labels)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=data, x="method_label", y="percent", hue="metric", ax=ax, palette=[COLORS["blue"], COLORS["green"], COLORS["orange"]], errorbar=None)
    ax.set_title("DOLOS 3-Fold Mean: Method Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.tick_params(axis="x", rotation=18)
    save(fig, out_dir, "graph02_dolos_method_comparison", dpi)


def graph_per_fold(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_per_fold_metrics.csv")
    df = df[df["method"] == "ensemble_raw_balanced_accuracy"].copy()
    data = df[["fold", "auc_roc", "calibrated_balanced_accuracy"]].melt("fold", var_name="metric", value_name="value")
    data["percent"] = pct(data["value"])
    data["metric"] = data["metric"].map({"auc_roc": "AUC", "calibrated_balanced_accuracy": "Cal. BA"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data, x="fold", y="percent", hue="metric", marker="o", linewidth=2.5, ax=ax, palette=[COLORS["blue"], COLORS["green"]])
    ax.set_ylim(50, 72)
    ax.set_title("Final Ensemble Performance by DOLOS Fold")
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    save(fig, out_dir, "graph03_dolos_per_fold_auc_ba", dpi)


def graph_ours_vs_paper(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "dolos_paper_comparison.csv")
    keep = [
        "DOLOS paper Visual",
        "DOLOS paper Audio",
        "DOLOS paper PAVF + Multi-task",
        "Cross-attention AUC baseline",
        "Gated logits prior-KL",
        "Final ensemble raw-BA",
    ]
    df = df[df["method"].isin(keep)].copy()
    df["method"] = pd.Categorical(df["method"], categories=keep, ordered=True)
    df = df.sort_values("method")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [COLORS["gray"] if g == "paper" else COLORS["blue"] for g in df["source_group"]]
    ax.bar(df["method"].astype(str), pct(df["auc_roc"]), color=colors)
    ax.set_title("AUC Comparison with DOLOS Paper")
    ax.set_ylabel("AUC (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(64.58, color=COLORS["red"], linestyle="--", linewidth=1, label="Paper PAVF + Multi-task")
    ax.legend()
    save(fig, out_dir, "graph04_ours_vs_paper_auc", dpi)


def graph_stream_ablation(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv("outputs/metrics/fold3_stream_ablation/fold3_stream_ablation_summary.csv")
    keep = ["spatial", "flow", "audio", "all", "all_residual"]
    labels = {
        "spatial": "Spatial",
        "flow": "Flow",
        "audio": "Audio",
        "all": "Three-stream",
        "all_residual": "Residual fusion",
    }
    df = df[df["stream"].isin(keep)].copy()
    df["stream_label"] = df["stream"].map(labels)
    extra_rows = []
    summaries = [
        ("Soft temporal cross", "outputs/metrics/retrain_clean_dolos_three_stream_auc_soft_temporal_penalty/fold3_seed42_summary.json"),
        ("Gated prior-KL mask", "outputs/metrics/retrain_clean_dolos_three_stream_gated_logits_prior_kl_temporal_mask/fold3_seed42_summary.json"),
    ]
    for label, summary_path in summaries:
        path = Path(summary_path)
        if not path.exists():
            continue
        summary = pd.read_json(path, typ="series")
        test_metrics = summary["test_metrics"]
        cal_metrics = summary.get("test_metrics_calibrated_threshold", summary.get("calibrated_threshold_metrics", {}))
        extra_rows.append(
            {
                "stream_label": label,
                "best_val_auc_roc": float(summary["best_val_auc_roc"]),
                "test_auc_roc": float(test_metrics["auc_roc"]),
                "calibrated_balanced_accuracy": float(cal_metrics["balanced_accuracy"]),
            }
        )
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    data = df[["stream_label", "best_val_auc_roc", "test_auc_roc", "calibrated_balanced_accuracy"]].melt(
        "stream_label",
        var_name="metric",
        value_name="value",
    )
    data["percent"] = pct(data["value"])
    data["metric"] = data["metric"].map(
        {
            "best_val_auc_roc": "Val AUC",
            "test_auc_roc": "Test AUC",
            "calibrated_balanced_accuracy": "Cal. BA",
        }
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=data, x="stream_label", y="percent", hue="metric", ax=ax, palette=[COLORS["purple"], COLORS["blue"], COLORS["green"]], errorbar=None)
    ax.set_title("Fold3 Stream Ablation")
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.tick_params(axis="x", rotation=15)
    save(fig, out_dir, "graph05_stream_ablation", dpi)


def graph_error_by_host(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_error_by_host.csv").sort_values("balanced_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["host"], pct(df["balanced_accuracy"]), color=COLORS["orange"])
    ax.set_title("Final Ensemble: Balanced Accuracy by Host")
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_ylabel("Host")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    save(fig, out_dir, "graph06_error_by_host", dpi)


def graph_episode_heatmap(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_error_by_episode.csv").head(20).copy()
    matrix = df.set_index("episode_group_id")[["balanced_accuracy"]] * 100.0
    labels = [f"n={int(n)}" for n in df["n"]]
    fig, ax = plt.subplots(figsize=(5, 9))
    sns.heatmap(matrix, annot=np.asarray(labels).reshape(-1, 1), fmt="", cmap="YlOrRd_r", vmin=0, vmax=70, cbar_kws={"label": "BA (%)"}, ax=ax)
    ax.set_title("Worst DOLOS Episodes by Balanced Accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, out_dir, "graph07_episode_error_heatmap", dpi)


def graph_confusion_matrices(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_per_fold_metrics.csv")
    df = df[df["method"] == "ensemble_raw_balanced_accuracy"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (_, row) in zip(axes, df.iterrows(), strict=True):
        cm = np.asarray(ast.literal_eval(row["calibrated_confusion_matrix"]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Truth", "Lie"], yticklabels=["Truth", "Lie"])
        ax.set_title(f"{row['fold']} @ thr={float(row['threshold']):.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Final Ensemble Confusion Matrices", y=1.02, fontsize=14, weight="bold")
    save(fig, out_dir, "graph08_confusion_matrices", dpi)


def load_dolos_final_predictions() -> pd.DataFrame:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_ensemble_predictions_with_errors.csv")
    df["label_name"] = df["label"].astype(int).map({0: "Truth", 1: "Lie"})
    return df


def graph_dolos_score_distribution(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="score_lie",
        hue="label_name",
        bins=32,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.28,
        palette={"Truth": COLORS["blue"], "Lie": COLORS["orange"]},
        ax=ax,
    )
    ax.axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="0.5")
    ax.axvline(float(df["threshold"].mean()), color=COLORS["red"], linestyle="-.", linewidth=1.2, label="Mean calibrated threshold")
    ax.set_title("DOLOS Final Ensemble Score Distribution")
    ax.set_xlabel("Predicted P(lie)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, out_dir, "graph11_dolos_score_distribution", dpi)


def graph_dolos_roc_curve(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    fig, ax = plt.subplots(figsize=(7, 6))
    y = df["label"].astype(int).to_numpy()
    s = df["score_lie"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=COLORS["red"], linewidth=2.5, label=f"All folds AUC={roc_auc_score(y, s) * 100:.2f}")
    for fold, group in df.groupby("fold", sort=True):
        y_fold = group["label"].astype(int).to_numpy()
        s_fold = group["score_lie"].astype(float).to_numpy()
        fpr_fold, tpr_fold, _ = roc_curve(y_fold, s_fold)
        ax.plot(fpr_fold, tpr_fold, linewidth=1.4, alpha=0.75, label=f"{fold} AUC={roc_auc_score(y_fold, s_fold) * 100:.1f}")
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_title("DOLOS Final Ensemble ROC Curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save(fig, out_dir, "graph12_dolos_roc_curve", dpi)


def graph_dolos_pr_curve(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    y = df["label"].astype(int).to_numpy()
    s = df["score_lie"].astype(float).to_numpy()
    precision, recall, _ = precision_recall_curve(y, s)
    baseline = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=COLORS["orange"], linewidth=2.5, label=f"AP={average_precision_score(y, s) * 100:.2f}")
    ax.axhline(baseline, color=COLORS["gray"], linestyle="--", linewidth=1, label=f"Lie prior={baseline * 100:.1f}%")
    ax.set_title("DOLOS Final Ensemble Precision-Recall Curve")
    ax.set_xlabel("Recall (Lie)")
    ax.set_ylabel("Precision (Lie)")
    ax.legend(loc="lower left")
    save(fig, out_dir, "graph13_dolos_pr_curve", dpi)


def graph_dolos_threshold_sweep(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    y = df["label"].astype(int).to_numpy()
    s = df["score_lie"].astype(float).to_numpy()
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "Balanced accuracy": balanced_accuracy_score(y, pred),
                "F1 Lie": f1_score(y, pred, zero_division=0),
            }
        )
    sweep = pd.DataFrame(rows).melt("threshold", var_name="metric", value_name="value")
    sweep["percent"] = pct(sweep["value"])
    best = sweep[sweep["metric"] == "Balanced accuracy"].sort_values("value", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=sweep, x="threshold", y="percent", hue="metric", ax=ax, palette=[COLORS["green"], COLORS["orange"]], linewidth=2.2)
    ax.axvline(float(best["threshold"]), color=COLORS["red"], linestyle="--", linewidth=1.2, label=f"Best BA thr={float(best['threshold']):.2f}")
    ax.set_title("DOLOS Final Ensemble Threshold Sweep")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (%)")
    ax.legend()
    save(fig, out_dir, "graph14_dolos_threshold_sweep", dpi)


def graph_dolos_error_type_counts(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    order = ["TN", "TP", "FP", "FN"]
    counts = df["error_type"].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["error_type", "count"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts["error_type"], counts["count"], color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]])
    ax.set_title("DOLOS Final Ensemble Prediction Outcomes")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of clips")
    ax.bar_label(bars, padding=3, fontsize=9)
    save(fig, out_dir, "graph15_dolos_error_type_counts", dpi)


def graph_contamination_metric_comparison(out_dir: Path, dpi: int) -> None:
    df = pd.read_csv(FINAL_REPORT_DIR / "final_error_by_contamination.csv")
    df["contamination"] = df["clip_suspect_contamination"].map({False: "Clean heuristic", True: "Suspect"})
    data = df[["contamination", "balanced_accuracy", "auc_roc", "error_rate"]].melt("contamination", var_name="metric", value_name="value")
    data["metric"] = data["metric"].map({"balanced_accuracy": "BA", "auc_roc": "AUC", "error_rate": "Error rate"})
    data["percent"] = pct(data["value"])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x="contamination", y="percent", hue="metric", ax=ax, palette=[COLORS["green"], COLORS["blue"], COLORS["red"]], errorbar=None)
    ax.set_title("Effect of Suspected Face Contamination")
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    save(fig, out_dir, "graph16_contamination_metric_comparison", dpi)


def graph_dolos_score_by_host(out_dir: Path, dpi: int) -> None:
    df = load_dolos_final_predictions()
    order = df.groupby("host")["score_lie"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="host", y="score_lie", hue="label_name", order=order, ax=ax, palette={"Truth": COLORS["blue"], "Lie": COLORS["orange"]})
    ax.set_title("DOLOS Final Ensemble Score by Host")
    ax.set_xlabel("Host")
    ax.set_ylabel("Predicted P(lie)")
    ax.legend(title="")
    save(fig, out_dir, "graph17_dolos_score_by_host", dpi)


def write_index(out_dir: Path) -> None:
    lines = ["# Generated Report Figures", ""]
    for name, title in FIGURE_SPECS:
        lines.append(f"- `{name}.png` / `{name}.svg`: {title}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    dpi = int(args.dpi)
    setup_style()

    simple_pipeline(out_dir, dpi)
    micro_expression_regions(out_dir, dpi)
    dolos_internal_variation(out_dir, dpi)
    preprocessing_pipeline(out_dir, dpi)
    face_valid_timeline(out_dir, dpi)
    optical_flow_schematic(out_dir, dpi)
    window_aggregation(out_dir, dpi)
    three_stream_architecture(out_dir, dpi)
    cross_attention_block(out_dir, dpi)
    gated_logit_fusion(out_dir, dpi)
    prediction_level_ensemble(out_dir, dpi)
    dolos_protocol(out_dir, dpi)

    graph_label_distribution(out_dir, dpi)
    graph_dolos_method_comparison(out_dir, dpi)
    graph_per_fold(out_dir, dpi)
    graph_ours_vs_paper(out_dir, dpi)
    graph_stream_ablation(out_dir, dpi)
    graph_error_by_host(out_dir, dpi)
    graph_episode_heatmap(out_dir, dpi)
    graph_confusion_matrices(out_dir, dpi)
    graph_dolos_score_distribution(out_dir, dpi)
    graph_dolos_roc_curve(out_dir, dpi)
    graph_dolos_pr_curve(out_dir, dpi)
    graph_dolos_threshold_sweep(out_dir, dpi)
    graph_dolos_error_type_counts(out_dir, dpi)
    graph_contamination_metric_comparison(out_dir, dpi)
    graph_dolos_score_by_host(out_dir, dpi)
    write_index(out_dir)

    print(f"Wrote {len(FIGURE_SPECS)} figures to {out_dir}")
    for name, title in FIGURE_SPECS:
        print(f"{name}.png\t{title}")


if __name__ == "__main__":
    main()
