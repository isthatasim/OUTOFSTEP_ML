from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import textwrap
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def _esc(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=False)


def _p(text: str = "", style: str | None = None, bold: bool = False) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    b = "<w:b/>" if bold else ""
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:p>{ppr}<w:r><w:rPr>{b}</w:rPr><w:t{preserve}>{_esc(text)}</w:t></w:r></w:p>"


def _heading(text: str, level: int = 1) -> str:
    return _p(text, style=f"Heading{level}")


def _bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>'
        f"<w:r><w:t>{_esc('- ' + text)}</w:t></w:r></w:p>"
    )


def _page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def _table(headers: list[str], rows: list[list[object]]) -> str:
    cell_width = max(950, int(10000 / max(1, len(headers))))

    def cell(value: object, header: bool = False) -> str:
        bold = "<w:b/>" if header else ""
        shading = '<w:shd w:fill="D9EAF7"/>' if header else ""
        return (
            f"<w:tc><w:tcPr>{shading}<w:tcW w:w=\"{cell_width}\" w:type=\"dxa\"/>"
            '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
            '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
            "</w:tcPr>"
            f"<w:p><w:r><w:rPr>{bold}<w:sz w:val=\"18\"/></w:rPr><w:t>{_esc(value)}</w:t></w:r></w:p></w:tc>"
        )

    xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="10000" w:type="dxa"/><w:tblLayout w:type="fixed"/></w:tblPr>']
    xml.append("<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>")
    for row in rows:
        xml.append("<w:tr>" + "".join(cell(v) for v in row) + "</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def _image(rid: str, width_in: float, height_in: float, name: str) -> str:
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    doc_pr_id = abs(hash(rid)) % 100000
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{doc_pr_id}" name="{_esc(name)}"/>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="0" name="{_esc(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>
"""


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out: Path, color: str = "#2F6F8F") -> None:
    plt.figure(figsize=(8, 4.2))
    vals = pd.to_numeric(df[y], errors="coerce")
    labels = df[x].astype(str)
    plt.bar(labels, vals, color=color)
    plt.title(title)
    plt.ylabel(y)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def _plot_flowchart(out: Path) -> None:
    steps = [
        "Data",
        "Features",
        "Static ML",
        "Calibration",
        "Thresholds",
        "Robustness",
        "Conformal",
        "Decision",
    ]
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="#E8F3F7", ec="#2F6F8F"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.045, 0.55), xytext=(x + 0.045, 0.55), arrowprops=dict(arrowstyle="->", color="#334E5C"))
    ax.set_title("OOS Risk Screening Workflow", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def _equation_image(text: str, out: Path, width: float = 9.0, height: float = 1.0) -> None:
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.02, 0.5, f"${text}$", fontsize=18, va="center", ha="left")
    plt.tight_layout(pad=0.4)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def _fmt(x: object, nd: int = 4) -> str:
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return ""
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def generate_assets(root: Path) -> dict[str, Path]:
    tables = root / "results" / "logic_ladder" / "tables"
    assets = root / "outputs" / "word" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    scenario = pd.read_csv(tables / "logic_ladder_scenario_comparison.csv")
    combo = pd.read_csv(tables / "logic_ladder_combination_comparison.csv")
    robust = pd.read_csv(tables / "logic_ladder_robustness.csv")
    conformal = pd.read_csv(tables / "logic_ladder_conformal_summary.csv")

    _plot_flowchart(assets / "workflow.png")
    _plot_bar(scenario.dropna(subset=["PR_AUC"]), "scenario_id", "PR_AUC", "Scenario PR-AUC Comparison", assets / "scenario_pr_auc.png")
    _plot_bar(combo.head(8), "combo_logic", "composite_score", "Top Combination Composite Scores", assets / "combo_scores.png", color="#6B8E3D")
    _plot_bar(robust, "shift", "PR_AUC_drop_vs_clean", "Robustness: PR-AUC Drop vs Clean", assets / "robustness_drop.png", color="#B65F3A")
    _plot_bar(conformal, "alpha", "coverage", "Conformal Coverage", assets / "conformal_coverage.png", color="#4A6FA5")

    equations = {
        "eq_data": r"\mathbf{x}^{(i)}=[T^{(i)}, I^{(i)}, S^{(i)}, H^{(i)}], \quad y^{(i)}\in\{0,1\}",
        "eq_features": r"\mathbf{z}^{(i)}=\left[\frac{1}{H^{(i)}},\frac{S^{(i)}}{H^{(i)}},\frac{S^{(i)}}{I^{(i)}},\frac{I^{(i)}}{H^{(i)}}\right]",
        "eq_swing1": r"\dot{\delta}=\omega,\quad M\dot{\omega}=P_m-P_e(\delta,V,\mathrm{network})-D\omega,\quad M=\frac{2H}{\omega_s}",
        "eq_model": r"p^{(i)}=f_{\theta}(\mathbf{x}^{(i)},\mathbf{z}^{(i)})\approx P(y^{(i)}=1|\mathbf{x}^{(i)})",
        "eq_decision": r"\hat{y}^{(i)}(\tau)=1[p^{(i)}\geq\tau]",
        "eq_bce": r"\mathcal{L}_{CE}=-\sum_i[w_1 y_i\log(p_i)+w_0(1-y_i)\log(1-p_i)]",
        "eq_priors": r"\frac{\partial f}{\partial H}\leq0,\quad \frac{\partial f}{\partial I}\leq0,\quad \frac{\partial f}{\partial S}\geq0",
        "eq_rphys": r"\mathcal{R}_{phys}=\lambda_H E[\max(0,\Delta f/\Delta H)]+\lambda_I E[\max(0,\Delta f/\Delta I)]+\lambda_S E[\max(0,-\Delta f/\Delta S)]",
        "eq_total": r"\min_{\theta}\ \mathcal{L}(\theta)=\mathcal{L}_{CE}(\theta)+\mathcal{R}_{phys}(\theta)",
        "eq_threshold": r"\tau^*=\arg\min_{\tau}\left[C_{FN}FN(\tau)+C_{FP}FP(\tau)\right]",
        "eq_cf": r"\min_{\Delta x}\|W\Delta x\|_1\quad s.t.\quad f_{\theta}(x+\Delta x)\leq\tau_{stable}",
        "eq_composite": r"\mathrm{Score}=\mathrm{PR\!-\!AUC}-0.35\,\mathrm{FNR}-0.10\,\mathrm{ECE}",
        "eq_metrics": r"\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN},\quad \mathrm{FNR}=\frac{FN}{TP+FN}",
        "eq_calib": r"\mathrm{ECE}=\sum_b\frac{n_b}{N}\left|\mathrm{acc}(b)-\mathrm{conf}(b)\right|",
    }
    for name, eq in equations.items():
        _equation_image(eq, assets / f"{name}.png")

    return {p.stem: p for p in assets.glob("*.png")}


def build_docx(root: Path, output: Path) -> None:
    tables = root / "results" / "logic_ladder" / "tables"
    scenario = pd.read_csv(tables / "logic_ladder_scenario_comparison.csv")
    combo = pd.read_csv(tables / "logic_ladder_combination_comparison.csv")
    robust = pd.read_csv(tables / "logic_ladder_robustness.csv")
    conformal = pd.read_csv(tables / "logic_ladder_conformal_summary.csv")
    thresholds = pd.read_csv(tables / "logic_ladder_threshold_policies.csv")
    with open(tables / "logic_ladder_best_combination.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    with open(tables / "logic_ladder_data_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)

    images = generate_assets(root)
    media = list(images.values())
    rid_map = {path: f"rId{i+1}" for i, path in enumerate(media)}

    body: list[str] = []
    body.append(_p("Out-of-Step (OOS) Risk Prediction for GR1", style="Title"))
    body.append(_p("Paper-style research report with mathematical model, case-study design, statistical comparison, and deployment recommendation."))

    body.append(_heading("Abstract"))
    body.append(_p("This report develops and evaluates a static operating-point machine-learning framework for out-of-step (OOS) risk screening of generator GR1. The study uses raw operating descriptors T, I, S, and H together with physics-aware ratios 1/H, S/H, S/I, and I/H. The framework is evaluated as a progressive capability ladder: raw prediction, engineered physics ratios, monotonic physical priors, imbalance-aware cost-sensitive thresholding, calibration, robustness, counterfactual correction, and drift monitoring. Current results show that engineered physics ratios are the dominant predictive improvement, while calibration and cost-sensitive thresholding make the model more suitable for operational deployment. The best pure static predictor is the 1+2 combination, whereas the best practical deployment approach is the calibrated physics-aware stack with cost-sensitive thresholding and monitoring."))

    body.append(_heading("1. Introduction"))
    body.append(_p("Out-of-step instability is a loss-of-synchronism condition in which a generator rotor angle no longer remains bounded with respect to the rest of the system after a disturbance. Conventional transient-stability assessment relies on simulation studies or protection logic, but operators also need fast screening tools that can score many operating points before detailed dynamic studies are performed."))
    body.append(_p("The proposed workflow treats GR1 OOS prediction as a static risk-screening problem. It does not claim to replace full transient simulation; instead, it learns a surrogate stability boundary from labeled study data and returns calibrated probability, a class decision, robustness indicators, drift alerts, and counterfactual guidance. This is useful for planning studies, contingency screening, and model-assisted operational review."))
    body.append(_p("The main research question is: which combination of raw inputs, physics-aware features, physical priors, imbalance handling, calibration, and deployment safeguards gives the best OOS screening behavior? Because this is a safety-critical problem, the conclusion is not based on PR-AUC alone. Recall, false-negative rate, calibration error, robustness drop, and actionability are also considered."))

    body.append(_heading("2. Dataset and Case Study"))
    body.append(_p("The case study uses the repository dataset at data/raw/Dataset_output.csv. Extra non-core columns are retained during ingestion but ignored by the core feature pipeline unless explicitly selected. The leakage check found Tag_rate AUC close to 0.5, indicating that Tag_rate alone is not a direct label leak."))
    class_counts = audit.get("class_counts", {})
    body.append(_table(["Item", "Value"], [
        ["Raw rows", audit.get("n_rows_raw", "")],
        ["Clean rows", audit.get("n_rows_clean", "")],
        ["Raw columns", audit.get("n_columns_raw", "")],
        ["Clean columns", audit.get("n_columns_clean", "")],
        ["Stable samples y=0", class_counts.get("0", "")],
        ["OOS samples y=1", class_counts.get("1", "")],
        ["OOS class ratio", _fmt(audit.get("class_ratio_positive", ""))],
        ["Duplicates removed", audit.get("duplicate_rows_removed", "")],
        ["Constant columns", ", ".join(audit.get("constant_columns", []))],
    ]))

    body.append(_heading("3. Problem Definition"))
    body.append(_p("The objective is to predict whether a static operating point for generator GR1 is stable or out-of-step. The target is binary: 0 means stable and 1 means OOS/unstable."))
    body.append(_table(["Symbol", "Column", "Meaning"], [["T", "Tag_rate", "disturbance/acceleration proxy"], ["I", "Ikssmin_kA", "grid strength proxy"], ["S", "Sgn_eff_MVA", "stress/loading proxy"], ["H", "H_s", "inertia constant"], ["y", "Out_of_step", "binary target"]]))
    body.append(_image(rid_map[images["workflow"]], 6.7, 1.8, "workflow"))

    body.append(_heading("4. Mathematical Model"))
    for key, caption in [
        ("eq_data", "Dataset definition"),
        ("eq_features", "Physics-aware engineered ratios"),
        ("eq_swing1", "Swing-equation motivation"),
        ("eq_model", "Probabilistic classifier"),
        ("eq_decision", "Probability-to-class decision rule"),
        ("eq_bce", "Imbalance-aware binary cross-entropy"),
        ("eq_priors", "Monotonic physical priors"),
        ("eq_rphys", "Finite-difference monotonic penalty"),
        ("eq_total", "Total objective"),
        ("eq_threshold", "Cost-sensitive threshold"),
        ("eq_cf", "Counterfactual correction"),
        ("eq_metrics", "Key threshold metrics"),
        ("eq_calib", "Expected calibration error"),
    ]:
        body.append(_p(caption, bold=True))
        body.append(_image(rid_map[images[key]], 6.5, 0.75, key))

    body.append(_heading("5. Scenario Design"))
    body.append(_p("S1-S5 build the model step by step. S6-S8 evaluate operational readiness. S9 is the compact final row that carries the predictive result from S5 and adds robustness, counterfactual, drift, and conformal uncertainty summaries."))
    body.append(_table(["Scenario", "Meaning"], [["S1", "raw baseline"], ["S2", "raw + engineered physics ratios"], ["S3", "S2 + monotonic priors"], ["S4", "S2 + imbalance handling + cost threshold"], ["S5", "S4 + calibration"], ["S6", "robustness evaluation"], ["S7", "counterfactual evaluation"], ["S8", "deployment/drift evaluation"], ["S9", "compact final summary"]]))
    body.append(_image(rid_map[images["scenario_pr_auc"]], 6.4, 3.2, "scenario_pr_auc"))

    body.append(_heading("6. Current Scenario Results"))
    rows = []
    for _, r in scenario[scenario["scenario_id"].isin(["S1", "S2", "S3", "S4", "S5", "S9"])].iterrows():
        rows.append([r["scenario_id"], _fmt(r["PR_AUC"]), _fmt(r["ROC_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["ECE"])])
    body.append(_table(["Scenario", "PR-AUC", "ROC-AUC", "Recall", "FNR", "ECE"], rows))
    body.append(_p("Interpretation: S2 is the strongest pure static classifier because adding physics ratios increases PR-AUC from 0.9895 to 0.9996 and reduces ECE from 0.0534 to 0.0075. S5 and S9 are better for deployment interpretation because they include calibration and operational summaries. S3 currently underperforms, which means the current monotonic implementation should be improved with a more flexible monotonic boosting approach rather than treated as a final physics-constrained model."))

    body.append(_heading("7. Detailed Statistical Comparison"))
    s1 = scenario[scenario["scenario_id"] == "S1"].iloc[0]
    s2 = scenario[scenario["scenario_id"] == "S2"].iloc[0]
    s5 = scenario[scenario["scenario_id"] == "S5"].iloc[0]
    s9 = scenario[scenario["scenario_id"] == "S9"].iloc[0]
    body.append(_table(["Comparison", "Delta PR-AUC", "Delta Recall", "Delta FNR", "Delta ECE", "Interpretation"], [
        ["S2 - S1", _fmt(s2["PR_AUC"] - s1["PR_AUC"]), _fmt(s2["Recall"] - s1["Recall"]), _fmt(s2["FNR"] - s1["FNR"]), _fmt(s2["ECE"] - s1["ECE"]), "physics ratios improve discrimination and reliability"],
        ["S5 - S2", _fmt(s5["PR_AUC"] - s2["PR_AUC"]), _fmt(s5["Recall"] - s2["Recall"]), _fmt(s5["FNR"] - s2["FNR"]), _fmt(s5["ECE"] - s2["ECE"]), "calibration improves probability reliability with small safety trade-off"],
        ["S9 - S5", _fmt(s9["PR_AUC"] - s5["PR_AUC"]), _fmt(s9["Recall"] - s5["Recall"]), _fmt(s9["FNR"] - s5["FNR"]), _fmt(s9["ECE"] - s5["ECE"]), "S9 adds operational summaries without changing predictive core"],
    ]))
    body.append(_p("The most important safety statistic is FNR. S1 and S2 have zero FNR in this split, while S5/S9 miss one OOS case in the test view (FNR=0.0017) but provide much stronger calibration than S1. For a published case study, this should be reported as a trade-off rather than hidden: the best pure detector is not necessarily the best deployable risk service."))

    body.append(_heading("8. Combination Search Results"))
    top = combo.head(6)
    rows = [[int(r["rank"]), r["combo_logic"], r["combo_name"], _fmt(r["PR_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["ECE"])] for _, r in top.iterrows()]
    body.append(_table(["Rank", "Combo", "Meaning", "PR-AUC", "Recall", "FNR", "ECE"], rows))
    body.append(_image(rid_map[images["combo_scores"]], 6.4, 3.2, "combo_scores"))
    best_combo = best.get("best_composite", {})
    body.append(_p(f"Best composite combination: {best_combo.get('combo_logic', '')} ({best_combo.get('combo_name', '')})."))
    body.append(_p("Best algorithm/approach by purpose: for pure static discrimination choose 1+2 (raw + engineered ratios); for calibrated probability reporting choose 1+2+5; for conservative deployment choose 1+2+4+5 because it includes cost-sensitive thresholding that explicitly penalizes missed OOS cases. The proposed research contribution should therefore be described as a deployment framework, not merely a single classifier."))

    body.append(_heading("9. Threshold Policy Comparison"))
    rows = [[r["policy"], _fmt(r["tau"]), _fmt(r["Precision"]), _fmt(r["Recall"]), _fmt(r["F1"]), int(r["FN"]), int(r["FP"]), _fmt(r["Cost"])] for _, r in thresholds.iterrows()]
    body.append(_table(["Policy", "Tau", "Precision", "Recall", "F1", "FN", "FP", "Cost"], rows))
    body.append(_p("The cost-sensitive threshold tau_cost lowers the decision threshold from 0.5 to 0.072. This increases alarms slightly (FP from 39 to 46) but reduces missed OOS events from 2 to 1 and lowers the defined cost from 59 to 56. This is why tau_cost is preferred for a protection-oriented screening service."))

    body.append(_heading("10. Robustness and Drift"))
    rows = [[r["shift"], _fmt(r["PR_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["PR_AUC_drop_vs_clean"])] for _, r in robust.iterrows()]
    body.append(_table(["Shift", "PR-AUC", "Recall", "FNR", "PR-AUC drop"], rows))
    body.append(_image(rid_map[images["robustness_drop"]], 6.4, 3.1, "robustness_drop"))
    body.append(_p(f"S9 drift summary: max PSI = {_fmt(s9.get('drift_max_psi'))}. A high PSI suggests the incoming operating distribution has shifted and should trigger review or retraining policy."))

    body.append(_heading("11. Conformal Uncertainty"))
    c = conformal.iloc[0]
    body.append(_table(["Metric", "Value"], [["coverage", _fmt(c["coverage"])], ["OOS coverage", _fmt(c["oos_coverage"])], ["average set size", _fmt(c["average_set_size"])], ["ambiguous rate", _fmt(c["ambiguous_rate"])]]))
    body.append(_image(rid_map[images["conformal_coverage"]], 5.4, 2.8, "conformal_coverage"))
    body.append(_p("Conformal prediction is a safety layer: uncertain cases can be routed to operator review or future dynamic PMU-window refinement."))

    body.append(_heading("12. Discussion: What Is the Best Approach and Why?"))
    body.append(_p("The best overall approach is a physics-aware calibrated risk-screening framework: raw features plus engineered physical ratios, validation-fitted calibration, cost-sensitive thresholding, robustness checks, conformal uncertainty, and drift monitoring. This is stronger than selecting the highest single metric because OOS screening is safety-critical. A model that ranks points well but gives unreliable probabilities or misses unstable cases is not operationally ideal."))
    body.append(_p("The strongest empirical finding is that engineered ratios are highly valuable. They expose physically meaningful stress relationships that raw variables alone do not express directly. The ratio S/H represents loading relative to inertia support, S/I represents loading relative to grid strength, and 1/H makes low-inertia conditions easier for the model to separate."))
    body.append(_p("The current monotonic-prior implementation should not be considered final because S3 performs poorly. The correct conclusion is not that physics is harmful; rather, the imposed monotonic mechanism is too restrictive for the current data/model implementation. A better next algorithm is MonoGBM-OOS, where monotonic constraints are enforced inside a flexible gradient-boosted tree model."))
    body.append(_p("For case-study reporting, the final recommendation is: S2 for best pure static classification, S5 for best calibrated static probability model, S9 for compact paper reporting, and 1+2+4+5 as the safest deployment-style stack when missed OOS events are more costly than false alarms."))

    body.append(_heading("13. Evaluation Matrix"))
    body.append(_table(["Axis", "Metric", "Direction", "Reason"], [["Discrimination", "PR-AUC", "higher", "primary for imbalanced OOS data"], ["Discrimination", "ROC-AUC", "higher", "general separability"], ["Safety", "FNR", "lower", "missed OOS is dangerous"], ["Calibration", "ECE / Brier", "lower", "risk probabilities should be reliable"], ["Robustness", "PR-AUC drop", "lower", "survives noise and regime shift"], ["Actionability", "counterfactual success", "higher", "operator mitigation support"], ["Uncertainty", "conformal coverage", "near target", "safer decisions under uncertainty"]]))
    body.append(_p("Deployment-oriented score used in the combination table:"))
    body.append(_image(rid_map[images["eq_composite"]], 6.0, 0.65, "eq_composite"))

    body.append(_heading("14. Outcomes and Conclusion"))
    body.append(_p("Outcome 1: The model can forecast the binary OOS label from static associated parameters with very high discrimination on the current dataset. Outcome 2: Physics-aware engineered ratios are the most effective addition. Outcome 3: Calibration is necessary when the output is used as a probability risk value rather than only a class label. Outcome 4: Cost-sensitive thresholding gives a defensible operating policy for reducing missed instability. Outcome 5: Robustness tests show small PR-AUC drops under noise and missing features, with the largest degradation under unseen-regime tests. Outcome 6: Counterfactual success is currently zero, so this module is a clear target for improvement."))
    body.append(_p("Conclusion: The most defensible final approach is not a plain classifier. It is PhysiScreen-OOS as a calibrated, physics-aware, deployment-oriented static OOS risk screener. The current best empirical combination is raw + engineered ratios for pure prediction, while the most operationally complete approach combines engineered ratios, calibration, cost-sensitive thresholding, robustness evaluation, uncertainty handling, and monitoring."))

    body.append(_heading("15. Bottom Line"))
    body.append(_bullet("Best pure static screening: 1+2 = raw + engineered physics ratios."))
    body.append(_bullet("Best calibrated static model: 1+2+5 = raw + engineered + calibration."))
    body.append(_bullet("Best conservative deployment-style stack: 1+2+4+5."))
    body.append(_bullet("Best compact reporting scenario: S9."))
    body.append(_bullet("Main research target: improve monotonic modeling with MonoGBM-OOS and improve physics-constrained counterfactual recourse."))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS['w']}" xmlns:r="{NS['r']}" xmlns:wp="{NS['wp']}" xmlns:a="{NS['a']}" xmlns:pic="{NS['pic']}">
<w:body>
{''.join(body)}
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>
</w:body></w:document>"""

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for path, rid in rid_map.items():
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>')
    rels.append("</Relationships>")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", _styles_xml())
        zf.writestr("word/_rels/document.xml.rels", "".join(rels))
        for path in media:
            zf.write(path, f"word/media/{path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/word/OOS_GR1_research_report.docx")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_docx(root, root / args.output)
    print(root / args.output)


if __name__ == "__main__":
    main()
