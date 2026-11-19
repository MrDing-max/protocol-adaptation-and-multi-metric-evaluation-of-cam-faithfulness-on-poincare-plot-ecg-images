# -*- coding: utf-8 -*-
"""用本地结果重画英文版交叉验证图（三种相似度 + 功能性检验）。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 左图：三种相似度 ============
metrics = ['Pearson', 'Spearman', 'SSIM']
gradcam = [0.4664, 0.6392, 0.4782]
dfocam  = [0.4372, 0.5756, 0.3172]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
x = np.arange(3)
w = 0.35
ax1.bar(x - w/2, gradcam, w, color='#5185C0', label='Grad-CAM')
ax1.bar(x + w/2, dfocam, w, color='#C96144', label='DFO-CAM')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=9.5)
ax1.set_ylabel('Similarity (original vs fully-random CAM)', fontsize=9.5)
ax1.set_title('Similarity after full randomization', fontsize=10.5, fontweight='bold')
ax1.legend(frameon=False, fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, 0.75)

# ============ 右图：功能性检验（Deletion AUC） ============
methods = ['Grad-CAM', 'DFO-CAM']
train_cam = [0.3765, 0.3669]
rand_cam  = [0.4255, 0.4921]
rand_ord  = [0.4796, 0.4796]
x2 = np.arange(2)
w2 = 0.26
ax2.bar(x2 - w2, train_cam, w2, color='#55966B', label='Trained CAM')
ax2.bar(x2, rand_cam, w2, color='#E99D4E', label='Random-model CAM')
ax2.bar(x2 + w2, rand_ord, w2, color='#99AABB', label='Random order')
ax2.axhline(0.4796, color='#94A3B8', lw=0.8, linestyle=':')
ax2.text(0.5, 0.486, 'chance (random order)', fontsize=8, color='#4A5568')
ax2.set_xticks(x2)
ax2.set_xticklabels(methods, fontsize=9.5)
ax2.set_ylabel('Deletion AUC (lower = more faithful)', fontsize=9.5)
ax2.set_title('Functional check: random-model CAM fed to trained model',
              fontsize=10.5, fontweight='bold')
ax2.legend(frameon=False, fontsize=8.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, 0.55)

plt.tight_layout()
plt.savefig('图片/fig_cross_validation.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('交叉验证图已生成', flush=True)
