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
        "eq_bce": r"\mathcal{L}_{CE}=-\sum_i[w_1 y_i\log(p_i)+w_0(1-y_i)\log(1-p_i)]",
        "eq_priors": r"\frac{\partial f}{\partial H}\leq0,\quad \frac{\partial f}{\partial I}\leq0,\quad \frac{\partial f}{\partial S}\geq0",
        "eq_total": r"\min_{\theta}\ \mathcal{L}(\theta)=\mathcal{L}_{CE}(\theta)+\mathcal{R}_{phys}(\theta)",
        "eq_threshold": r"\tau^*=\arg\min_{\tau}\left[C_{FN}FN(\tau)+C_{FP}FP(\tau)\right]",
        "eq_cf": r"\min_{\Delta x}\|W\Delta x\|_1\quad s.t.\quad f_{\theta}(x+\Delta x)\leq\tau_{stable}",
        "eq_composite": r"\mathrm{Score}=\mathrm{PR\!-\!AUC}-0.35\,\mathrm{FNR}-0.10\,\mathrm{ECE}",
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
    with open(tables / "logic_ladder_best_combination.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    images = generate_assets(root)
    media = list(images.values())
    rid_map = {path: f"rId{i+1}" for i, path in enumerate(media)}

    body: list[str] = []
    body.append(_p("Out-of-Step (OOS) Risk Prediction for GR1", style="Title"))
    body.append(_p("Comprehensive mathematical model, scenario interpretation, and current results."))
    body.append(_heading("1. Problem Definition"))
    body.append(_p("The objective is to predict whether a static operating point for generator GR1 is stable or out-of-step. The target is binary: 0 means stable and 1 means OOS/unstable."))
    body.append(_table(["Symbol", "Column", "Meaning"], [["T", "Tag_rate", "disturbance/acceleration proxy"], ["I", "Ikssmin_kA", "grid strength proxy"], ["S", "Sgn_eff_MVA", "stress/loading proxy"], ["H", "H_s", "inertia constant"], ["y", "Out_of_step", "binary target"]]))
    body.append(_image(rid_map[images["workflow"]], 6.7, 1.8, "workflow"))

    body.append(_heading("2. Mathematical Model"))
    for key, caption in [
        ("eq_data", "Dataset definition"),
        ("eq_features", "Physics-aware engineered ratios"),
        ("eq_swing1", "Swing-equation motivation"),
        ("eq_model", "Probabilistic classifier"),
        ("eq_bce", "Imbalance-aware binary cross-entropy"),
        ("eq_priors", "Monotonic physical priors"),
        ("eq_total", "Total objective"),
        ("eq_threshold", "Cost-sensitive threshold"),
        ("eq_cf", "Counterfactual correction"),
    ]:
        body.append(_p(caption, bold=True))
        body.append(_image(rid_map[images[key]], 6.5, 0.75, key))

    body.append(_heading("3. Scenario Design"))
    body.append(_p("S1-S5 build the model step by step. S6-S8 evaluate operational readiness. S9 is the compact final row that carries the predictive result from S5 and adds robustness, counterfactual, drift, and conformal uncertainty summaries."))
    body.append(_table(["Scenario", "Meaning"], [["S1", "raw baseline"], ["S2", "raw + engineered physics ratios"], ["S3", "S2 + monotonic priors"], ["S4", "S2 + imbalance handling + cost threshold"], ["S5", "S4 + calibration"], ["S6", "robustness evaluation"], ["S7", "counterfactual evaluation"], ["S8", "deployment/drift evaluation"], ["S9", "compact final summary"]]))
    body.append(_image(rid_map[images["scenario_pr_auc"]], 6.4, 3.2, "scenario_pr_auc"))

    body.append(_heading("4. Current Scenario Results"))
    rows = []
    for _, r in scenario[scenario["scenario_id"].isin(["S1", "S2", "S3", "S4", "S5", "S9"])].iterrows():
        rows.append([r["scenario_id"], _fmt(r["PR_AUC"]), _fmt(r["ROC_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["ECE"])])
    body.append(_table(["Scenario", "PR-AUC", "ROC-AUC", "Recall", "FNR", "ECE"], rows))
    body.append(_p("Interpretation: S2 is the strongest pure static classifier. S5 and S9 are better for deployment interpretation because they include calibration and operational summaries. S3 currently underperforms, which means the current monotonic implementation should be improved with a more flexible monotonic boosting approach."))

    body.append(_heading("5. Combination Search Results"))
    top = combo.head(6)
    rows = [[int(r["rank"]), r["combo_logic"], r["combo_name"], _fmt(r["PR_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["ECE"])] for _, r in top.iterrows()]
    body.append(_table(["Rank", "Combo", "Meaning", "PR-AUC", "Recall", "FNR", "ECE"], rows))
    body.append(_image(rid_map[images["combo_scores"]], 6.4, 3.2, "combo_scores"))
    best_combo = best.get("best_composite", {})
    body.append(_p(f"Best composite combination: {best_combo.get('combo_logic', '')} ({best_combo.get('combo_name', '')})."))

    body.append(_heading("6. Robustness and Drift"))
    rows = [[r["shift"], _fmt(r["PR_AUC"]), _fmt(r["Recall"]), _fmt(r["FNR"]), _fmt(r["PR_AUC_drop_vs_clean"])] for _, r in robust.iterrows()]
    body.append(_table(["Shift", "PR-AUC", "Recall", "FNR", "PR-AUC drop"], rows))
    body.append(_image(rid_map[images["robustness_drop"]], 6.4, 3.1, "robustness_drop"))
    s9 = scenario[scenario["scenario_id"] == "S9"].iloc[0]
    body.append(_p(f"S9 drift summary: max PSI = {_fmt(s9.get('drift_max_psi'))}. A high PSI suggests the incoming operating distribution has shifted and should trigger review or retraining policy."))

    body.append(_heading("7. Conformal Uncertainty"))
    c = conformal.iloc[0]
    body.append(_table(["Metric", "Value"], [["coverage", _fmt(c["coverage"])], ["OOS coverage", _fmt(c["oos_coverage"])], ["average set size", _fmt(c["average_set_size"])], ["ambiguous rate", _fmt(c["ambiguous_rate"])]]))
    body.append(_image(rid_map[images["conformal_coverage"]], 5.4, 2.8, "conformal_coverage"))
    body.append(_p("Conformal prediction is a safety layer: uncertain cases can be routed to operator review or future dynamic PMU-window refinement."))

    body.append(_heading("8. Evaluation Matrix"))
    body.append(_table(["Axis", "Metric", "Direction", "Reason"], [["Discrimination", "PR-AUC", "higher", "primary for imbalanced OOS data"], ["Discrimination", "ROC-AUC", "higher", "general separability"], ["Safety", "FNR", "lower", "missed OOS is dangerous"], ["Calibration", "ECE / Brier", "lower", "risk probabilities should be reliable"], ["Robustness", "PR-AUC drop", "lower", "survives noise and regime shift"], ["Actionability", "counterfactual success", "higher", "operator mitigation support"], ["Uncertainty", "conformal coverage", "near target", "safer decisions under uncertainty"]]))
    body.append(_p("Deployment-oriented score used in the combination table:"))
    body.append(_image(rid_map[images["eq_composite"]], 6.0, 0.65, "eq_composite"))

    body.append(_heading("9. Bottom Line"))
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
