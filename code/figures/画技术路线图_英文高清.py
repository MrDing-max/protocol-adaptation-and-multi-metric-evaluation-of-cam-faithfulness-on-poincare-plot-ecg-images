# -*- coding: utf-8 -*-
"""Render a high-resolution English version of the technical roadmap."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

EDGE = "#1A1A1A"
TITLE = "#111111"
TXT = "#1F2937"
SUB = "#6B7280"
FILL = "#FFFFFF"
FILL_SOFT = "#F5F5F5"
BLUE = "#2B6CB0"
RED = "#C0392B"
DARK = "#2C3E50"

fig, ax = plt.subplots(figsize=(15, 24))
ax.set_xlim(0, 15)
ax.set_ylim(0, 24)
ax.axis("off")


def box(cx, cy, w, h, title=None, lines=None, ec=EDGE, fc=FILL, tfs=15,
        tcolor=TITLE, lfs=11, lcolor=TXT, title_offset=0.42,
        line_top_offset=0.95, line_bottom_offset=0.28, line_step=0.52):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2,
    ))
    if title:
        ax.text(cx, cy + h / 2 - title_offset, title, ha="center", va="center",
                fontsize=tfs, color=tcolor, fontweight="bold", zorder=3)
    if lines:
        top = cy + h / 2 - line_top_offset if title else cy + h / 2 - 0.40
        bottom = cy - h / 2 + line_bottom_offset
        step = min((top - bottom) / max(len(lines) - 1, 1), line_step)
        for i, text in enumerate(lines):
            ax.text(cx, top - i * step, text, ha="center", va="center",
                    fontsize=lfs, color=lcolor, zorder=3)


def arrow(x1, y1, x2, y2, color=BLUE, lw=2.0, dashed=False,
          style="-|>", scale=20):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=scale,
        linewidth=lw, color=color, linestyle="--" if dashed else "-", zorder=1,
    ))


def mini_box(cx, cy, w, h, text, fs=9):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, edgecolor="#555555", facecolor=FILL_SOFT, zorder=3,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=TXT, zorder=4)


# Research review
box(7.5, 21.8, 12.5, 2.0, title="Research Review", tfs=16, lfs=10.5,
    lines=[
        "Clinical needs and missed-diagnosis risks in automated AF detection  •  Current state of deep-learning explainability (XAI)",
        "Review of CAM methods: Grad-CAM / Grad-CAM++ / Score-CAM / DFO-CAM / RISE  •  Advances in Poincaré-plot + CNN AF classification",
    ], line_top_offset=0.88, line_step=0.42)

# Detection scenario and challenges
box(4.75, 18.95, 8.0, 2.3, title="Detection Scenario Construction",
    tfs=14, lfs=10.5, lines=[
        "Model AF detection as a binary Poincaré-plot image classification task",
        "Input: 10-s lead-II ECG signal (100 Hz)",
        "Output: binary classification—normal vs AF",
    ])
box(12.0, 18.95, 4.5, 2.5, title="Detection Challenges", ec=RED,
    tcolor=RED, tfs=14, lfs=9.0, lines=[
        "Irregular RR intervals and scattered points",
        "Record segmentation can cause data leakage",
        "Concerns about CAM faithfulness (false negatives)",
        "Contradictory results across CAM mechanisms",
        "Lack of unified statistical testing",
    ], line_step=0.36)

# Data acquisition and preparation
box(7.5, 15.25, 13.5, 3.1, title="Data Acquisition and Preparation",
    tfs=15, title_offset=0.35)
mini = [
    ("PTB-XL\n597 records", 9.0),
    ("Chapman\n600 records", 9.0),
    ("Lead II\n10 s", 9.0),
    ("R-peak detection\nNeuroKit2", 8.0),
    ("RR intervals\nms", 9.0),
    ("Remove outliers\n[300, 2000] ms", 8.0),
    ("Adjacent pairing\n(RRᵢ, RRᵢ₊₁)", 7.8),
    ("224 × 224\nblack points on white", 7.2),
]
mw, mg, my = 1.5, 0.18, 15.2
total_w = len(mini) * mw + (len(mini) - 1) * mg
mx0 = 7.5 - total_w / 2 + mw / 2
for i, (text, fs) in enumerate(mini):
    cx = mx0 + i * (mw + mg)
    mini_box(cx, my, mw, 1.0, text, fs)
    if i < len(mini) - 1:
        arrow(cx + mw / 2 + 0.02, my, cx + mw / 2 + mg - 0.02, my,
              color="#555555", lw=1.2, scale=14)
ax.text(7.5, 14.1,
        "Split protocol: patient-wise 80/20 train–validation split  •  5 independent random seeds",
        ha="center", va="center", fontsize=10.5, color=SUB, zorder=3)

# Data, training, and results modules
box(3.0, 11.75, 3.8, 2.4, title="Detection Data", tfs=14, lfs=10,
    lines=["Training/validation set (PTB-XL)", "External test set (Chapman)",
           "5 random seeds"])
box(7.5, 11.75, 3.8, 2.4, title="Model Training", tfs=14, lfs=9.5,
    lines=["ResNet18 (ImageNet pretrained)", "Adam 1e−4  •  batch 32  •  8 epochs",
           "Mean ± SD over 5 seeds", "Baseline: ViT-B/16"], line_step=0.38)
box(12.0, 11.75, 3.8, 2.4, title="Detection Results", tfs=14, lfs=10,
    lines=["PTB-XL 92.6% ± 1.1%", "Chapman 98.0% ± 0.7%",
           "ViT baseline 86–88%"])

# CAM analysis
box(7.5, 8.75, 11.5, 2.5,
    title="CAM Explainability Analysis (5 methods  •  4 mechanisms)",
    tfs=15, title_offset=0.35)
cam = [
    ("Gradient-based\nGrad-CAM / Grad-CAM++", 8.6),
    ("Non-gradient-based\nScore-CAM", 8.6),
    ("Optimization-based\nDFO-CAM", 8.6),
    ("Black-box\nRISE", 9.0),
]
cw, cg, cy = 2.4, 0.3, 8.6
ctotal = len(cam) * cw + (len(cam) - 1) * cg
cx0 = 7.5 - ctotal / 2 + cw / 2
for i, (text, fs) in enumerate(cam):
    mini_box(cx0 + i * (cw + cg), cy, cw, 0.95, text, fs)
ax.text(7.5, 7.85,
        "Same-stage comparison  •  single GPU  •  time per heatmap: 0.008–3.28 s",
        ha="center", va="center", fontsize=10, color=SUB, zorder=3)

# Three-level faithfulness evaluation
ax.text(7.5, 6.45, "Three-Level Faithfulness Evaluation", ha="center",
        va="center", fontsize=16, color=TITLE, fontweight="bold", zorder=3)
items = [
    ("Adapt\nProtocol Adaptation", "4 base maps →\nunified white background", 9.2),
    ("Assess\nStatistical Tests", "5 × 300  •  t + Wilcoxon\n+ dz + Bootstrap + McNemar", 8.7),
    ("Audit\nRobustness Tests", "Cascading randomization\nSSIM / Spearman\nFunctional test", 8.5),
    ("Conclusion", "Protocol adaptation +\nmulti-metric cross-validation", 9.2),
]
aw, ag, ay = 3.1, 0.35, 5.0
atotal = len(items) * aw + (len(items) - 1) * ag
ax0 = 7.5 - atotal / 2 + aw / 2
for i, (title, body, fs) in enumerate(items):
    cx = ax0 + i * (aw + ag)
    dark = i == len(items) - 1
    ec = DARK if dark else "#555555"
    fc = DARK if dark else FILL
    tc = "white" if dark else TXT
    ax.add_patch(FancyBboxPatch(
        (cx - aw / 2, ay - 1.1), aw, 2.2,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3,
    ))
    ax.text(cx, ay + 0.48, title, ha="center", va="center",
            fontsize=11.5, color=tc, fontweight="bold", zorder=4)
    ax.text(cx, ay - 0.32, body, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4)
    if i < len(items) - 1:
        arrow(cx + aw / 2 + 0.05, ay, cx + aw / 2 + ag - 0.05, ay,
              color=BLUE, lw=1.5, scale=16)

# Main blue flow arrows
arrow(7.5, 20.8, 7.5, 20.25)
arrow(5.25, 17.8, 5.25, 17.25)
arrow(3.0, 13.7, 3.0, 13.2)
arrow(7.5, 13.7, 7.5, 13.2)
arrow(4.95, 11.75, 5.55, 11.75)
arrow(9.45, 11.75, 10.05, 11.75)
arrow(7.5, 10.55, 7.5, 10.05)
arrow(7.5, 7.5, 7.5, 6.95)

# Red feedback arrows
arrow(9.7, 18.95, 8.8, 18.95, color=RED, lw=1.6, dashed=True)
ax.add_patch(FancyArrowPatch((0.5, 4.0), (0.5, 14.3), arrowstyle="-",
                             linewidth=1.5, color=RED, linestyle="--", zorder=1))
arrow(0.5, 14.3, 1.1, 14.3, color=RED, lw=1.5, dashed=True, scale=15)
ax.text(0.22, 9.2, "Evaluation Feedback", ha="center", va="center",
        fontsize=9.5, color=RED, rotation=90, zorder=4)

plt.savefig(r"图片/技术路线图_英文高清.png", dpi=600,
            bbox_inches="tight", pad_inches=0.15, facecolor="white")
plt.close(fig)
print(r"Saved 图片/技术路线图_英文高清.png")
