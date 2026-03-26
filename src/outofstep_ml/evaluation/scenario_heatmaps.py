from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid(
    frame: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    n_grid: int,
    fixed: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xv = np.linspace(float(frame[x_feature].min()), float(frame[x_feature].max()), n_grid)
    yv = np.linspace(float(frame[y_feature].min()), float(frame[y_feature].max()), n_grid)
    xx, yy = np.meshgrid(xv, yv)
    g = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
    for c in frame.columns:
        if c in g.columns:
            continue
        if c in fixed:
            g[c] = fixed[c]
        else:
            if pd.api.types.is_numeric_dtype(frame[c]):
                g[c] = float(frame[c].median())
            else:
                g[c] = str(frame[c].mode(dropna=True).iloc[0]) if len(frame[c].mode(dropna=True)) else "GR1"
    return xx, yy, g


def generate_interaction_heatmaps(
    model,
    X_reference: pd.DataFrame,
    output_dir: str | Path,
    feature_pairs: Iterable[Tuple[str, str]] = (("H_s", "Ikssmin_kA"), ("H_s", "Sgn_eff_MVA"), ("Ikssmin_kA", "Sgn_eff_MVA")),
    n_grid: int = 100,
    fixed: Dict[str, float] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = fixed or {}
    rows: List[Dict] = []

    for x_feature, y_feature in feature_pairs:
        if x_feature not in X_reference.columns or y_feature not in X_reference.columns:
            continue
        xx, yy, grid = _grid(X_reference, x_feature, y_feature, n_grid=n_grid, fixed=fixed)
        p = np.clip(model.predict_proba(grid)[:, 1], 1e-6, 1 - 1e-6)
        zz = p.reshape(xx.shape)

        csv_path = out_dir / f"heatmap_{x_feature}_vs_{y_feature}.csv"
        pd.DataFrame(
            {
                x_feature: xx.ravel(),
                y_feature: yy.ravel(),
                "p_oos": zz.ravel(),
            }
        ).to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.contourf(xx, yy, zz, levels=30, cmap="coolwarm")
        cs = ax.contour(xx, yy, zz, levels=[threshold], colors="k", linewidths=1.4)
        ax.clabel(cs, inline=True, fontsize=8, fmt={threshold: f"tau={threshold:.2f}"})
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"Risk Heatmap: {x_feature} vs {y_feature}")
        fig.colorbar(im, ax=ax, label="P(OOS=1)")
        png_path = out_dir / f"heatmap_{x_feature}_vs_{y_feature}.png"
        pdf_path = out_dir / f"heatmap_{x_feature}_vs_{y_feature}.pdf"
        fig.tight_layout()
        fig.savefig(png_path, dpi=260)
        fig.savefig(pdf_path, dpi=260)
        plt.close(fig)

        rows.append(
            {
                "x_feature": x_feature,
                "y_feature": y_feature,
                "grid_size": int(n_grid),
                "csv_path": str(csv_path),
                "png_path": str(png_path),
                "pdf_path": str(pdf_path),
                "mean_risk": float(np.mean(zz)),
                "max_risk": float(np.max(zz)),
                "min_risk": float(np.min(zz)),
            }
        )
    return pd.DataFrame(rows)

