from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig: plt.Figure, stem: str | Path, dpi: int = 300) -> Tuple[Path, Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def ascii_flowchart() -> str:
    return """
[Data Ingestion]
 -> [Audit + Cleaning + Leakage Checks]
 -> [Physics-motivated Features]
 -> [Splits: Stratified + Grouped + Leave-Level-Out]
 -> [Tier A Baselines]
 -> [Tier B Ensembles]
 -> [Tier C Physics-aware/Hybrid]
 -> [Calibration + Threshold Policy]
 -> [Scenario Evaluation (Step 1..Step 5)]
 -> [Explainability + Stability Maps]
 -> [Decision Support: Counterfactuals]
 -> [API + Tests]
 -> [Monitoring/Drift + Retraining]
 -> [Export: Tables, Figures, Docs, Manifest]
""".strip("\n")


def print_ascii_flowchart() -> None:
    print("\n=== WORKFLOW FLOWCHART (ASCII) ===")
    print(ascii_flowchart())
    print("==================================\n")


def plot_flowchart_figure(output_dir: str | Path) -> Tuple[Path, Path]:
    steps = [
        "Data Ingestion",
        "Audit + Cleaning\n+ Leakage Checks",
        "Physics-motivated\nFeatures",
        "Split Strategy\n(Stratified/Group/Leave-Level)",
        "Tier A\nBaselines",
        "Tier B\nEnsembles",
        "Tier C\nPhysics-aware/Hybrid",
        "Calibration +\nThreshold Policy",
        "Scenario Evaluation (1..5)",
        "Explainability +\nStability Maps",
        "Counterfactual\nDecision Support",
        "API + Tests",
        "Monitoring + Drift\n+ Retraining",
        "Export Tables/Figures/\nDocs/Manifest",
    ]

    fig, ax = plt.subplots(figsize=(9, 16))
    ax.axis("off")
    y_positions = np.linspace(0.95, 0.05, len(steps))

    for i, (y, text) in enumerate(zip(y_positions, steps)):
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#EAF2F8", ec="#1F618D", lw=1.2),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.03),
                xytext=(0.5, y - 0.03),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#1B4F72"),
            )

    ax.set_title("OOS GR1 End-to-End ML + Physics + Deployment Workflow", fontsize=13, pad=16)
    out_stem = ensure_dir(output_dir) / "flowchart_oos_pipeline"
    return save_figure(fig, out_stem)


def plot_feature_distributions(df: pd.DataFrame, target_col: str, output_dir: str | Path) -> Tuple[Path, Path]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    n = len(numeric)
    cols = 3
    rows = int(np.ceil(n / cols)) if n > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, col in enumerate(numeric):
        ax = axes[i]
        ax.axis("on")
        x0 = df.loc[df[target_col] == 0, col].values
        x1 = df.loc[df[target_col] == 1, col].values
        bins = 30
        ax.hist(x0, bins=bins, alpha=0.6, label="Stable (0)", density=True, color="#2E86C1")
        ax.hist(x1, bins=bins, alpha=0.6, label="OOS (1)", density=True, color="#C0392B")
        ax.set_title(col)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend()

    fig.suptitle("Feature Distributions by Class", fontsize=14)
    out_stem = ensure_dir(output_dir) / "feature_distributions"
    return save_figure(fig, out_stem)


def _build_grid(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    fixed_values: Dict[str, float],
    n_grid: int = 120,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_min, x_max = float(df[x_feature].min()), float(df[x_feature].max())
    y_min, y_max = float(df[y_feature].min()), float(df[y_feature].max())
    xv = np.linspace(x_min, x_max, n_grid)
    yv = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(xv, yv)

    grid = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
    for c in df.columns:
        if c in (x_feature, y_feature):
            continue
        if c in fixed_values:
            grid[c] = fixed_values[c]
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                grid[c] = float(df[c].median())
            else:
                grid[c] = str(df[c].mode(dropna=True).iloc[0]) if len(df[c].mode(dropna=True)) else "GR1"

    return xx, yy, grid


def plot_stability_map(
    model,
    df: pd.DataFrame,
    target_col: str,
    x_feature: str,
    y_feature: str,
    output_stem: str | Path,
    fixed_values: Dict[str, float] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    fixed = fixed_values or {}
    xx, yy, grid = _build_grid(df.drop(columns=[target_col]), x_feature, y_feature, fixed)
    p = model.predict_proba(grid)[:, 1]
    zz = p.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(xx, yy, zz, levels=30, cmap="coolwarm", alpha=0.85)
    cs = ax.contour(xx, yy, zz, levels=[threshold], colors="k", linewidths=1.8)
    ax.clabel(cs, fmt={threshold: f"tau={threshold:.2f}"}, inline=True, fontsize=8)

    d0 = df[df[target_col] == 0]
    d1 = df[df[target_col] == 1]
    ax.scatter(d0[x_feature], d0[y_feature], s=12, color="#1B4F72", alpha=0.4, label="Stable (0)")
    ax.scatter(d1[x_feature], d1[y_feature], s=12, color="#7B241C", alpha=0.5, label="OOS (1)")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Stability Map: {x_feature} vs {y_feature}")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax, label="P(Out_of_step=1)")
    save_figure(fig, output_stem)
    return zz


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_stem: str | Path,
    n_bins: int = 10,
) -> Tuple[Path, Path]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1

    conf, acc = [], []
    for b in range(n_bins):
        mask = idx == b
        if np.any(mask):
            conf.append(np.mean(y_prob[mask]))
            acc.append(np.mean(y_true[mask]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax.plot(conf, acc, marker="o", color="#117A65", lw=2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.grid(alpha=0.25)
    return save_figure(fig, output_stem)


def plot_feature_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    output_stem: str | Path,
    n_repeats: int = 8,
    random_state: int = 42,
) -> Tuple[Path, Path]:
    imp = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]

    labels = [X.columns[i] for i in order]
    vals = imp.importances_mean[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(vals)), vals, color="#2471A3")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Permutation importance")
    ax.set_title("Model Feature Importance")
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, output_stem)


def plot_pdp(
    model,
    X: pd.DataFrame,
    features: List[str],
    output_stem: str | Path,
) -> Tuple[Path, Path]:
    valid = [f for f in features if f in X.columns]
    fig, ax = plt.subplots(figsize=(6 * max(len(valid), 1), 4))
    if len(valid) == 1:
        ax = [ax]
    PartialDependenceDisplay.from_estimator(model, X, features=valid, ax=ax)
    fig.suptitle("Partial Dependence", y=1.02)
    return save_figure(fig, output_stem)


def plot_boundary_comparison(
    model_a,
    model_b,
    df: pd.DataFrame,
    target_col: str,
    x_feature: str,
    y_feature: str,
    output_stem: str | Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model, title in zip(axes, [model_a, model_b], ["Unconstrained", "Physics-aware"]):
        xx, yy, grid = _build_grid(df.drop(columns=[target_col]), x_feature, y_feature, fixed_values={})
        p = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        im = ax.contourf(xx, yy, p, levels=30, cmap="coolwarm", alpha=0.85)
        ax.contour(xx, yy, p, levels=[0.5], colors="k", linewidths=1.6)
        ax.set_title(title)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Boundary Comparison: Unconstrained vs Physics-aware")
    return save_figure(fig, output_stem)


def plot_drift_monitoring(
    timeline: pd.DataFrame,
    output_stem: str | Path,
    metric_cols: Iterable[str],
) -> Tuple[Path, Path]:
    cols = [c for c in metric_cols if c in timeline.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3 * max(len(cols), 1)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(timeline.index, timeline[c].values, color="#AF601A", lw=1.8)
        ax.set_ylabel(c)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Drift Monitoring Timeline")
    return save_figure(fig, output_stem)


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    output_stem: str | Path,
    color_col: str | None = None,
) -> Tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col and color_col in df.columns:
        uniq = sorted(df[color_col].astype(str).unique())
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        color_map = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
        for u in uniq:
            d = df[df[color_col].astype(str) == u]
            ax.scatter(d[x_col], d[y_col], s=70, alpha=0.9, label=u, color=color_map[u])
            for _, r in d.iterrows():
                ax.text(r[x_col], r[y_col], str(r[label_col]), fontsize=8, ha="left", va="bottom")
        ax.legend(title=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], s=70, alpha=0.9, color="#1f77b4")
        for _, r in df.iterrows():
            ax.text(r[x_col], r[y_col], str(r[label_col]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return save_figure(fig, output_stem)


def plot_noise_robustness(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    output_stem: str | Path,
) -> Tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in df.groupby(group_col):
        ds = d.sort_values(x_col)
        ax.plot(ds[x_col], ds[y_col], marker="o", lw=1.8, label=str(name))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, output_stem)
