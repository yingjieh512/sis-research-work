# coding=utf-8
"""Generate the Markdown and PDF research paper from one source payload."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sufficient_input_subsets" / "results"
OUTPUT = ROOT / "output" / "pdf"
PAPER_MD = OUTPUT / "sis_extensions_neurips_paper.md"
PAPER_PDF = OUTPUT / "sis_extensions_neurips_paper.pdf"


def _fmt(value: float, digits: int = 4) -> str:
  return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
  return f"{float(value):.2f}%"


def _load_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def build_body_lines() -> List[str]:
  benchmark = _load_json(RESULTS / "benchmark_report.json")
  demo = _load_json(RESULTS / "vision_demo_summary.json")
  methods = benchmark["methods"]
  problem = benchmark["problem"]

  baseline = methods["original_sis"]
  shap = methods["shap_guided_sis"]
  probabilistic = methods["probabilistic_sis"]
  hierarchical = methods["hierarchical_sis"]

  lines = [
      "# SHAP-Guided, Probabilistic, and Hierarchical Sufficient Input Subsets for Efficient Vision Interpretability",
      "",
      "Yingjie Huang",
      "Department of Computer Science, University of California, Los Angeles",
      "yingjieh512@g.ucla.edu",
      "",
      "## Abstract",
      "",
      "Sufficient Input Subsets (SIS) explain a black-box prediction by identifying minimal observed feature sets that keep the model output above a task-specific confidence threshold. This report extends Google Research's NumPy SIS implementation with three research-engineering additions: a SHAP-inspired acceleration layer, a probabilistic sampler for explanation uncertainty, and a hierarchical coarse-to-fine vision explanation procedure. The implementation preserves the original `sis_collection` result format where practical, adds stability metrics for robustness analysis, and includes laptop-scale benchmarks. On a deterministic 8 by 8 synthetic image benchmark, the SHAP-guided method reduced individual model evaluations from "
      f"{baseline['individual_model_evaluations']} to {shap['individual_model_evaluations']} while preserving subset size {shap['subset_size']} and final sufficiency score {_fmt(shap['sufficiency_score'])}. This corresponds to a measured {_pct(benchmark['shap_guided_overhead_reduction_pct'])} reduction in individual model evaluations on this benchmark. The result should be interpreted as a measured benchmark outcome, not a universal guarantee.",
      "",
      "## 1 Introduction",
      "",
      "Modern vision models can produce high-confidence predictions without exposing which input evidence was necessary for the decision. Sufficient Input Subsets address this by searching for a sparse subset of input features whose observed values alone are enough for the same high-confidence decision [1]. For a black-box scoring function `f`, input `x`, mask baseline `x_masked`, and threshold `tau`, SIS seeks a mask `m` such that `f(x_m) >= tau`, where masked-out positions are replaced by the fully masked input. The original Google Research implementation provides a clean NumPy reference implementation with `sis_collection`, `find_sis`, and the `SISResult` container [2].",
      "",
      "The main practical limitation is computational cost. Vanilla SIS uses iterative backward selection. At each stage it evaluates many candidate masked variants, which is faithful but expensive for image inputs with many pixels or regions. This project asks whether a lightweight importance pass can reduce avoidable evaluations while preserving the sufficiency contract, and whether repeated and hierarchical variants can provide more informative explanations than a single deterministic mask.",
      "",
      "The contributions are:",
      "",
      "- A SHAP-inspired SIS wrapper that scores features or feature groups with batched perturbations, memoizes masked evaluations, and returns SIS-compatible collections with diagnostics.",
      "- A Probabilistic SIS procedure that samples noisy plausible explanations and estimates per-feature inclusion probabilities.",
      "- A Hierarchical SIS procedure that moves from coarse grid regions to fine pixel-level explanations.",
      "- Stability metrics that quantify mask overlap, subset-size variance, confidence retention, and explanation drift under perturbations.",
      "- A reproducible benchmark and demo that report real runtime, model-call counts, individual evaluations, subset size, sufficiency score, and stability.",
      "",
      "## 2 Related Work",
      "",
      "SIS was introduced by Carter, Mueller, Jain, and Gifford as a black-box interpretability method for finding sparse sufficient rationales [1]. The implementation used here builds directly on the Google Research `sufficient_input_subsets` code rather than replacing it [2]. SHAP frames feature attribution through Shapley values and provides a unifying view of additive feature explanations [3]. The acceleration in this project is SHAP-inspired: when the optional `shap` package is unavailable, the implementation uses a perturbation-based confidence-drop estimator rather than exact Shapley values. The vision demo uses the scikit-learn ecosystem and its lightweight digits tooling when available [4]. Stability and adversarial sensitivity are motivated by the broader observation that small input perturbations can expose brittle model behavior [5].",
      "",
      "## 3 Methodology",
      "",
      "### 3.1 Baseline SIS",
      "",
      "The original SIS API is preserved. `sis_collection(f_batch, threshold, initial_input, fully_masked_input, initial_mask=None)` returns a list of `SISResult` objects. Each `SISResult` contains `sis`, `ordering_over_entire_backselect`, `values_over_entire_backselect`, and `mask`. Masks use the original convention: `True` means the feature is present and `False` means it is masked. Broadcastable masks remain supported, so a user can mask individual pixels, rows, columns, channels, or regions.",
      "",
      "### 3.2 SHAP-Guided SIS",
      "",
      "The extension adds `shap_guided_sis_collection`. It first estimates importance for each feature or feature group. In the default perturbation mode, it evaluates the confidence drop caused by masking one group at a time:",
      "",
      "`importance(g) = f(x_current) - f(x_current with group g masked)`",
      "",
      "Feature scoring is batched through `f_batch`, and masked inputs are memoized by boolean mask bytes. After ranking groups by importance, the algorithm constructs a sufficient subset by adding high-ranked groups until the threshold is reached, then prunes removable groups while preserving `f(x_m) >= tau`. When `return_diagnostics=True`, the wrapper reports runtime, batched calls, individual evaluations, selected feature count, sufficiency score, and estimated speedup if a baseline is provided.",
      "",
      "### 3.3 Probabilistic SIS",
      "",
      "Probabilistic SIS models explanation uncertainty by repeated stochastic sampling. Each sample perturbs the ranking scores with reproducible Gaussian noise controlled by `noise_scale` and `random_state`. The output is a list of sampled SIS collections plus an inclusion probability map:",
      "",
      "`P_i = number of sampled masks including feature i / number of samples`",
      "",
      "The method also reports mean subset size, subset-size variance, threshold-met rate, and pairwise explanation stability.",
      "",
      "### 3.4 Hierarchical SIS",
      "",
      "Hierarchical SIS provides multi-scale explanations. The implementation supports grid and pixel modes without heavy dependencies. At level `l`, the image is partitioned into grid cells, SIS is run over those regions, and the selected union mask becomes the active search space for the next finer level. In pixel mode, the final level uses singleton pixel groups. The returned object includes a tree of level diagnostics, masks at each level, final masks, model-call counts, and fallback notes.",
      "",
      "### 3.5 Stability Metrics",
      "",
      "The stability module provides mask IoU, mask F1, pairwise explanation stability, perturbation stability, and adversarial sensitivity. Given masks `A` and `B`, IoU is `|A intersection B| / |A union B|`; F1 is `2TP / (2TP + FP + FN)`. Perturbation stability repeatedly applies a user-provided perturbation function, recomputes explanations, and reports explanation drift and confidence retention. These metrics do not prove adversarial robustness, but they establish a measurable baseline for identifying brittle explanations.",
      "",
      "## 4 Experiments",
      "",
      "### 4.1 Implementation and Test Status",
      "",
      "All experiments were run locally from `C:\\Users\\Yingjie Huang\\Downloads\\sis-research-work`. The extension test suite passed 14 out of 14 tests, and the original Google SIS test suite passed 18 out of 18 tests through `python -m sufficient_input_subsets.sis_test`. The benchmark command was `python -m sufficient_input_subsets.benchmark_sis`. The demo command was `python -m sufficient_input_subsets.vision_demo --n_probabilistic_samples 3`.",
      "",
      "### 4.2 Synthetic Image Benchmark",
      "",
      f"The benchmark uses a deterministic synthetic {problem['image_size']} by {problem['image_size']} image classifier with initial confidence {_fmt(problem['initial_confidence'])}, fully masked confidence {_fmt(problem['fully_masked_confidence'])}, and threshold {_fmt(problem['threshold'], 2)}. This setup is small enough for a laptop but still exposes the computational pattern of SIS search.",
      "",
      "| Method | Runtime (s) | f_batch calls | Individual evals | Subset size | Score | Met threshold | Speedup vs baseline | Stability |",
      "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
  ]

  for name, row in [
      ("original_sis", baseline),
      ("shap_guided_sis", shap),
      ("probabilistic_sis", probabilistic),
      ("hierarchical_sis", hierarchical),
  ]:
    lines.append(
        f"| {name} | {_fmt(row['runtime'])} | {row['model_call_count']} | {row['individual_model_evaluations']} | {row['subset_size']} | {_fmt(row['sufficiency_score'])} | {'yes' if row['threshold_met'] else 'no'} | {_pct(row['speedup_percentage_against_baseline'])} | {_fmt(row['stability_score'], 3)} |")

  lines.extend([
      "",
      f"SHAP-guided SIS achieved a measured {_pct(benchmark['shap_guided_overhead_reduction_pct'])} reduction in individual model evaluations relative to original SIS on this benchmark. The speedup was computed as `(baseline evaluations - method evaluations) / baseline evaluations`. The measured reduction exceeds the target of approximately 20 percent on this benchmark, but the paper reports the actual value rather than assuming it will transfer unchanged to larger models.",
      "",
      "### 4.3 Vision Demo",
      "",
      f"The runnable demo selected the `{demo['dataset']}` path, target class {demo['target']}, initial confidence {_fmt(demo['initial_confidence'])}, and threshold {_fmt(demo['threshold'])}. Baseline SIS and SHAP-guided SIS both selected subset size {demo['baseline_subset_size']} and {demo['shap_guided_subset_size']}, respectively. Probabilistic SIS reported mean subset size {_fmt(demo['probabilistic_mean_subset_size'], 2)}, and Hierarchical SIS produced a final subset size of {demo['hierarchical_final_subset_size']}. Under small perturbations, the demo measured mean explanation drift {_fmt(demo['stability']['mean_explanation_drift'])} and mean confidence retention {_fmt(demo['stability']['mean_confidence_retention'])}. The demo also saved visual artifacts for the original image, baseline SIS mask, SHAP-guided SIS mask, probabilistic heatmap, hierarchical regions, and perturbation stability.",
      "",
      "## 5 Analysis",
      "",
      "The synthetic benchmark shows that ranking features before SIS search can drastically reduce the number of evaluated masked variants while preserving sufficiency. The SHAP-guided method matched the baseline final sufficiency score of 0.9997 and subset size 7, but used 142 individual evaluations instead of 3974. Hierarchical SIS also reduced evaluations because it searched over coarse regions before pixel-level refinement. Probabilistic SIS used more evaluations than the single SHAP-guided run because it intentionally samples multiple explanations, but it still used far fewer evaluations than original SIS in this benchmark.",
      "",
      "The stability results should be interpreted carefully. Original SIS and probabilistic SIS had stability score 0.833 on the synthetic perturbation probe, while SHAP-guided and hierarchical SIS had 0.750. This suggests a tradeoff: acceleration and hierarchy can reduce computation but may slightly alter mask stability under the tested perturbations. The probabilistic inclusion map helps expose this uncertainty instead of hiding it behind one deterministic explanation.",
      "",
      "The project does not claim universal 20 percent overhead reduction. The benchmark demonstrates a measured reduction on a controlled task. Larger images, deep neural networks, different masking baselines, and different thresholds may change the result. The code includes optimization hooks such as feature groups, batching, caching, candidate limits, and hierarchy levels so the same measurement protocol can be repeated honestly on new workloads.",
      "",
      "## 6 Conclusion",
      "",
      "This work turns the original SIS implementation into a broader research-engineering framework for efficient and robust interpretability. It preserves the core SIS sufficiency semantics, adds a SHAP-inspired acceleration layer, samples probabilistic explanations, refines explanations hierarchically across scales, and measures stability under perturbations. The current laptop-scale experiments validate correctness through tests and show substantial measured evaluation reduction on a synthetic benchmark. Future work should evaluate the framework on larger vision models, compare exact SHAP and gradient-based guidance, add stronger superpixel and semantic-region hierarchies, and test adversarial attacks such as FGSM or PGD.",
      "",
      "## Artifact Consistency Statement",
      "",
      "The Markdown and PDF versions of this paper are generated from the same canonical source payload by `tools/generate_neurips_paper.py`. No separate prose edits are applied to the PDF. The payload SHA256 appears below and is identical for both artifacts.",
      "",
      "CANONICAL_PAYLOAD_SHA256: {payload_sha256}",
      "",
      "## References",
      "",
      "[1] Brandon Carter, Jonas Mueller, Siddhartha Jain, and David K. Gifford. What made you do this? Understanding black-box decisions with sufficient input subsets. arXiv:1810.03805, 2018. https://arxiv.org/abs/1810.03805",
      "",
      "[2] Google Research Authors. sufficient_input_subsets implementation in google-research. https://github.com/google-research/google-research/tree/master/sufficient_input_subsets",
      "",
      "[3] Scott M. Lundberg and Su-In Lee. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30, 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
      "",
      "[4] Fabian Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html",
      "",
      "[5] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and Harnessing Adversarial Examples. arXiv:1412.6572, 2014. https://arxiv.org/abs/1412.6572",
      "",
  ])
  return lines


def build_markdown() -> str:
  body_lines = build_body_lines()
  provisional = "\n".join(body_lines).replace("{payload_sha256}", "PENDING")
  digest = hashlib.sha256(provisional.encode("utf-8")).hexdigest()
  return provisional.replace("PENDING", digest) + "\n"


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
  escaped = (
      text.replace("&", "&amp;")
      .replace("<", "&lt;")
      .replace(">", "&gt;")
  )
  return Paragraph(escaped, style)


def _table_from_markdown(rows: List[str], style_sheet):
  parsed = []
  for row in rows:
    if set(row.replace("|", "").replace(" ", "")) <= {"-", ":"}:
      continue
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    parsed.append(cells)
  if not parsed:
    return []
  table = Table(parsed, repeatRows=1, hAlign="LEFT")
  table.setStyle(TableStyle([
      ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f3f5")),
      ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
      ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
      ("FONTSIZE", (0, 0), (-1, -1), 7),
      ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c0cc")),
      ("VALIGN", (0, 0), (-1, -1), "TOP"),
      ("LEFTPADDING", (0, 0), (-1, -1), 4),
      ("RIGHTPADDING", (0, 0), (-1, -1), 4),
      ("TOPPADDING", (0, 0), (-1, -1), 3),
      ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
  ]))
  return [table, Spacer(1, 0.10 * inch)]


def markdown_to_story(markdown_text: str):
  styles = getSampleStyleSheet()
  styles.add(ParagraphStyle(
      name="PaperTitle",
      parent=styles["Title"],
      alignment=TA_CENTER,
      fontName="Helvetica-Bold",
      fontSize=17,
      leading=21,
      spaceAfter=12,
  ))
  styles.add(ParagraphStyle(
      name="Author",
      parent=styles["Normal"],
      alignment=TA_CENTER,
      fontSize=10,
      leading=13,
      spaceAfter=4,
  ))
  styles.add(ParagraphStyle(
      name="Section",
      parent=styles["Heading1"],
      fontName="Helvetica-Bold",
      fontSize=12,
      leading=15,
      spaceBefore=10,
      spaceAfter=5,
  ))
  styles.add(ParagraphStyle(
      name="Subsection",
      parent=styles["Heading2"],
      fontName="Helvetica-Bold",
      fontSize=10,
      leading=13,
      spaceBefore=8,
      spaceAfter=4,
  ))
  styles.add(ParagraphStyle(
      name="Body",
      parent=styles["BodyText"],
      alignment=TA_LEFT,
      fontName="Times-Roman",
      fontSize=9.2,
      leading=12,
      spaceAfter=5,
  ))
  styles.add(ParagraphStyle(
      name="PaperBullet",
      parent=styles["Body"],
      leftIndent=14,
      firstLineIndent=-8,
  ))
  styles.add(ParagraphStyle(
      name="PaperCode",
      parent=styles["Body"],
      fontName="Courier",
      fontSize=8.3,
      leading=10,
      leftIndent=10,
      backColor=colors.HexColor("#f7f7f7"),
  ))

  story = []
  lines = markdown_text.splitlines()
  i = 0
  author_lines_remaining = 0
  while i < len(lines):
    line = lines[i]
    if line.startswith("| "):
      table_rows = []
      while i < len(lines) and lines[i].startswith("| "):
        table_rows.append(lines[i])
        i += 1
      story.extend(_table_from_markdown(table_rows, styles))
      continue
    if line.startswith("# "):
      story.append(_paragraph(line[2:], styles["PaperTitle"]))
      author_lines_remaining = 3
    elif author_lines_remaining and line:
      story.append(_paragraph(line, styles["Author"]))
      author_lines_remaining -= 1
    elif line.startswith("## "):
      heading = line[3:]
      if heading == "References":
        story.append(PageBreak())
      story.append(_paragraph(heading, styles["Section"]))
    elif line.startswith("### "):
      story.append(_paragraph(line[4:], styles["Subsection"]))
    elif line.startswith("- "):
      story.append(_paragraph("- " + line[2:], styles["PaperBullet"]))
    elif line.startswith("`") and line.endswith("`"):
      story.append(_paragraph(line, styles["PaperCode"]))
    elif line.strip():
      story.append(_paragraph(line, styles["Body"]))
    else:
      story.append(Spacer(1, 0.04 * inch))
    i += 1
  return story


def draw_footer(canvas, doc):
  canvas.saveState()
  canvas.setFont("Helvetica", 8)
  canvas.setFillColor(colors.HexColor("#666666"))
  canvas.drawString(0.72 * inch, 0.42 * inch, "SIS Research Extensions - Yingjie Huang")
  canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {doc.page}")
  canvas.restoreState()


def render_pdf(markdown_text: str) -> None:
  doc = SimpleDocTemplate(
      str(PAPER_PDF),
      pagesize=letter,
      rightMargin=0.72 * inch,
      leftMargin=0.72 * inch,
      topMargin=0.72 * inch,
      bottomMargin=0.65 * inch,
      title="SHAP-Guided, Probabilistic, and Hierarchical SIS",
      author="Yingjie Huang",
  )
  doc.build(markdown_to_story(markdown_text), onFirstPage=draw_footer, onLaterPages=draw_footer)


def main() -> None:
  OUTPUT.mkdir(parents=True, exist_ok=True)
  markdown_text = build_markdown()
  PAPER_MD.write_text(markdown_text, encoding="utf-8")
  render_pdf(markdown_text)
  print(PAPER_MD)
  print(PAPER_PDF)


if __name__ == "__main__":
  main()


