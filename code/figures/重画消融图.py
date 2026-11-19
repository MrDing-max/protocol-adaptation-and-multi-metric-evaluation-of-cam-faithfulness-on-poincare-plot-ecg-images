# -*- coding: utf-8 -*-
"""用本地结果数据重画英文版 SanityCheck 曲线 + 初始化消融图。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ Fig4 SanityCheck（级联随机化 Pearson 相关） ============
stages = ['Original', 'fc', '+layer4', '+layer3', '+layer2', '+layer1', '+conv1']
gradcam = [1.0000, 0.9886, 0.3357, 0.0844, -0.0348, -0.1027, 0.4664]
dfocam  = [1.0000, -0.2414, 0.1523, -0.0526, -0.2539, -0.0913, 0.4372]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(stages, gradcam, marker='o', ms=5, lw=1.6, color='#5185C0', label='Grad-CAM')
ax.plot(stages, dfocam, marker='s', ms=5, lw=1.6, color='#C96144', label='DFO-CAM')
ax.axhline(0, color='#94A3B8', lw=0.8, linestyle=':')
# 标注 DFO-CAM 在 fc 层即崩坏
ax.annotate('DFO-CAM collapses\nat fc (r = −0.24)', xy=(1, -0.2414),
            xytext=(2.1, -0.42), fontsize=9, color='#C96144',
            arrowprops=dict(arrowstyle='->', color='#C96144', lw=1.1))
ax.set_ylabel('Pearson correlation', fontsize=10.5)
ax.set_xlabel('Randomization stage', fontsize=10.5)
ax.set_title('Cascade randomization (16 images)', fontsize=11.5, fontweight='bold')
ax.legend(frameon=False, fontsize=9.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-0.55, 1.15)
plt.tight_layout()
plt.savefig('图片/Fig4_sanity_check.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('SanityCheck 曲线已生成', flush=True)

# ============ 初始化消融图（柱状：AI + 标注） ============
methods = ['Grad-CAM\n(ref.)', 'DFO-CAM\nshapleycam', 'DFO-CAM\ngradcam',
           'DFO-CAM\nzerofive', 'DFO-CAM\nrandom']
ai = [35.0, 43.0, 40.0, 37.0, 21.0]
colors = ['#5185C0', '#55966B', '#99C290', '#C0BEDC', '#C96144']

fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(range(len(methods)), ai, color=colors, width=0.6,
              edgecolor='white')
for b, v in zip(bars, ai):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.0f}', ha='center',
            fontsize=9.5, fontweight='bold')
ax.axhline(35.0, color='#94A3B8', lw=1, linestyle='--')
ax.text(4.5, 36, 'Grad-CAM AI (reference)', fontsize=8.5, color='#4A5568')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel('AI (%)', fontsize=10.5)
ax.set_title('Initialization ablation (100-image subset)', fontsize=11.5,
             fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 52)
plt.tight_layout()
plt.savefig('图片/fig_init_ablation.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('初始化消融图已生成', flush=True)
