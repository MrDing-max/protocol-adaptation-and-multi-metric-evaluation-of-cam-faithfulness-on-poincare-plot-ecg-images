# -*- coding: utf-8 -*-
# Fig 1 method pipeline: ECG -> R-peak -> Poincaré plot -> CNN -> CAM -> faithfulness evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

C_MAIN = '#2563EB'
C_ASSESS = '#E8590C'
C_TXT = '#1F2937'
C_SUB = '#6B7280'

fig, ax = plt.subplots(figsize=(14, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

modules = [
    ('ECG signal', 'lead II · 10 s\nPTB-XL + Chapman'),
    ('R-peak detection', 'NeuroKit2\nRR intervals'),
    ('Poincaré plot', 'RR scatter\n224×224 white'),
    ('CNN classifier', 'ResNet18\n92.6% / 98.0%'),
    ('CAM explanation', '5 methods\nincl. DFO-CAM, RISE'),
]
n = len(modules)
box_w = 0.165
box_h = 0.34
xs = [0.055 + i * 0.185 for i in range(n)]

for i, (title, sub) in enumerate(modules):
    x0, y0 = xs[i], 0.56
    ax.add_patch(FancyBboxPatch((x0, y0), box_w, box_h,
                                boxstyle='round,pad=0.008,rounding_size=0.02',
                                linewidth=1.6, edgecolor=C_MAIN, facecolor='white'))
    ax.text(x0 + box_w/2, y0 + box_h - 0.085, title, ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_MAIN)
    ax.text(x0 + box_w/2, y0 + box_h - 0.195, sub, ha='center', va='center',
            fontsize=9, color=C_SUB, linespacing=1.5)

for i in range(n - 1):
    ax.add_patch(FancyArrowPatch((xs[i] + box_w + 0.004, 0.73),
                                 (xs[i+1] - 0.004, 0.73),
                                 arrowstyle='-|>', mutation_scale=16,
                                 linewidth=1.6, color=C_TXT))

y0_b, box_b_h = 0.12, 0.30
ax.add_patch(FancyBboxPatch((0.10, y0_b), 0.80, box_b_h,
                            boxstyle='round,pad=0.008,rounding_size=0.02',
                            linewidth=1.6, edgecolor=C_ASSESS, facecolor='#FFF4EC'))
ax.text(0.5, y0_b + box_b_h - 0.085, 'Faithfulness evaluation (three-layer evidence chain)',
        ha='center', va='center', fontsize=12.5, fontweight='bold', color=C_ASSESS)
ax.text(0.5, y0_b + 0.095,
        '1. Protocol adaptation (white)    2. Statistical testing (paired t + Wilcoxon + effect size + McNemar)    3. Sanity checking (cascade randomization + functional check)',
        ha='center', va='center', fontsize=9.5, color=C_TXT)

ax.add_patch(FancyArrowPatch((0.5, 0.55), (0.5, y0_b + box_b_h + 0.01),
                             arrowstyle='-|>', mutation_scale=16,
                             linewidth=1.6, color=C_SUB, linestyle='--'))

plt.tight_layout()
plt.savefig('图片/Fig1_method_pipeline.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('已生成 图片/Fig1_method_pipeline.png')
