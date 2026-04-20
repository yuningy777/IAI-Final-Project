"""
Week 5: regenerate presentation-ready figures from saved logs (no training required).

Reads metrics under outputs/logs and writes PNGs to outputs/figures/week5/.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
FIG_W5 = ROOT / "outputs" / "figures" / "week5"
WEEK3_DIR = LOGS / "week3-logs-mobilenetVSresnet"


def _ensure_dirs() -> None:
    FIG_W5.mkdir(parents=True, exist_ok=True)


def _dedupe_week4(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best test macro-F1 row per experiment_name (CSV may contain duplicate runs)."""
    if df.empty:
        return df
    return (
        df.sort_values("test_f1_macro", ascending=False)
        .groupby("experiment_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def load_week4_table() -> pd.DataFrame:
    path = LOGS / "experiment_results_week4.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _dedupe_week4(df)


def plot_week3_comparison() -> Path | None:
    r_path = WEEK3_DIR / "resnet18_bs32_ep10_lr0.001_adam_4class_test_metrics.json"
    m_path = WEEK3_DIR / "mobilenet_v2_bs32_ep10_lr0.001_adam_4class_test_metrics.json"
    if not r_path.exists() or not m_path.exists():
        return None

    with open(r_path, encoding="utf-8") as f:
        rj = json.load(f)
    with open(m_path, encoding="utf-8") as f:
        mj = json.load(f)

    labels = ["ResNet18", "MobileNetV2"]
    acc = [rj["test_accuracy"], mj["test_accuracy"]]
    f1 = [rj["test_f1_macro"], mj["test_f1_macro"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, acc, width=w, label="Test accuracy")
    ax.bar(x + w / 2, f1, width=w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Week 3 - same setting (frozen backbone, 4-class)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    out = FIG_W5 / "week5_week3_resnet_vs_mobilenet.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_week4_overview(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None

    def short_label(name: str) -> str:
        m = re.search(r"freeze(\d)_aug\d_wloss(\d)", name)
        if not m:
            return name
        freeze, wloss = m.group(1), m.group(2)
        ft = "freeze" if freeze == "1" else "finetune"
        wl = "+wCE" if wloss == "1" else "+CE"
        return f"{ft}\n{wl}"

    df = df.copy()
    df["label"] = df["experiment_name"].map(short_label)

    order = [
        "freeze\n+CE",
        "freeze\n+wCE",
        "finetune\n+CE",
        "finetune\n+wCE",
    ]
    df["sort_key"] = df["label"].map({lab: i for i, lab in enumerate(order)})
    df = df.sort_values("sort_key")

    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, df["test_acc"], width=w, label="Test accuracy")
    ax.bar(x + w / 2, df["test_f1_macro"], width=w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Week 4 - MobileNetV2 ablations (E0-E3)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    out = FIG_W5 / "week5_week4_ablation_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_freeze_vs_finetune(df: pd.DataFrame) -> Path | None:
    """Week 5 emphasis: frozen-backbone vs full fine-tuning under standard CE."""
    if df.empty:
        return None
    e0 = df[df["experiment_name"].str.contains("freeze1_aug1_wloss0", regex=False)]
    e1 = df[df["experiment_name"].str.contains("freeze0_aug1_wloss0", regex=False)]
    if e0.empty or e1.empty:
        return None
    e0 = e0.iloc[0]
    e1 = e1.iloc[0]

    labels = ["E0: frozen backbone\n(lr=1e-3, CE)", "E1: full fine-tune\n(lr=1e-4, CE)"]
    metrics = ["test_acc", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
    titles = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    for ax, m, t in zip(axes, metrics, titles):
        vals = [float(e0[m]), float(e1[m])]
        ax.bar(labels, vals, color=["#4C72B0", "#55A868"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(t)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Week 5 - freeze experiment: backbone frozen vs all layers trained (CE only)")
    out = FIG_W5 / "week5_freeze_vs_finetune_ce.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_weighted_loss_effect(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None

    def pick(sub: str) -> pd.Series | None:
        hit = df[df["experiment_name"].str.contains(sub, regex=False)]
        return None if hit.empty else hit.iloc[0]

    pairs = [
        ("freeze1_aug1_wloss0", "freeze1_aug1_wloss1", "Frozen backbone: CE vs weighted CE"),
        ("freeze0_aug1_wloss0", "freeze0_aug1_wloss1", "Full fine-tune: CE vs weighted CE"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (a_sub, b_sub, title) in zip(axes, pairs):
        ra = pick(a_sub)
        rb = pick(b_sub)
        if ra is None or rb is None:
            ax.axis("off")
            continue
        x = np.arange(2)
        w = 0.2
        ax.bar(x - 1.5 * w, [ra["test_acc"], rb["test_acc"]], width=w, label="Accuracy")
        ax.bar(x - 0.5 * w, [ra["test_precision_macro"], rb["test_precision_macro"]], width=w, label="Macro P")
        ax.bar(x + 0.5 * w, [ra["test_recall_macro"], rb["test_recall_macro"]], width=w, label="Macro R")
        ax.bar(x + 1.5 * w, [ra["test_f1_macro"], rb["test_f1_macro"]], width=w, label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(["CE", "Weighted CE"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Week 4/5 - effect of class-weighted loss (same freeze setting)")
    out = FIG_W5 / "week5_weighted_loss_effect.png"
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_train_split_distribution() -> Path | None:
    split_path = WEEK3_DIR / "split_stats_4class.csv"
    if not split_path.exists():
        split_path = LOGS / "split_stats_4class.csv"
    if not split_path.exists():
        return None

    rows = []
    with open(split_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None

    classes = [r["class"] for r in rows]
    train = [int(r["train"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(classes, train, color="#8172B2")
    ax.set_title("Training split size per class (4-class TrashNet merge)")
    ax.set_ylabel("Images")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(train):
        ax.text(i, v + max(train) * 0.01, str(v), ha="center", fontsize=9)
    out = FIG_W5 / "week5_train_class_counts.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_final_per_class_f1() -> Path | None:
    report_path = (
        LOGS / "mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_classification_report.txt"
    )
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8")
    classes = []
    f1s = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("accuracy") or line.startswith("macro") or line.startswith("weighted"):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0] in {"glass_metal", "paper", "plastic", "trash"}:
            cls = parts[0]
            f1 = float(parts[3])
            classes.append(cls)
            f1s.append(f1)

    if len(classes) != 4:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(classes, f1s, color="#CCB974")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Final best model (E1) - per-class F1 on test set")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    out = FIG_W5 / "week5_final_model_per_class_f1.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_e1_learning_curves() -> Path | None:
    log_path = LOGS / "mobilenet_v2_img224_bs32_ep10_lr0.0001_adam_freeze0_aug1_wloss0_4class_train_log.csv"
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    if df.empty or "epoch" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_acc"], label="train")
    axes[0].plot(df["epoch"], df["val_acc"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].grid(linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_f1"], label="train")
    axes[1].plot(df["epoch"], df["val_f1"], label="val")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.suptitle("Week 5 - E1 training curves (best model)")
    out = FIG_W5 / "week5_e1_train_val_curves.png"
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_manifest(paths: list[Path | None]) -> Path:
    lines = ["Week 5 figure outputs", f"Root: {FIG_W5.as_posix()}", ""]
    for p in paths:
        if p is None:
            lines.append("(skipped - missing inputs)")
        else:
            lines.append(p.as_posix())
    out = FIG_W5 / "week5_figure_manifest.txt"
    FIG_W5.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    _ensure_dirs()
    df4 = load_week4_table()

    outs = [
        plot_week3_comparison(),
        plot_week4_overview(df4),
        plot_freeze_vs_finetune(df4),
        plot_weighted_loss_effect(df4),
        plot_train_split_distribution(),
        plot_final_per_class_f1(),
        plot_e1_learning_curves(),
    ]
    manifest = write_manifest(outs)
    print("Wrote:")
    for p in outs + [manifest]:
        if p is not None:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
