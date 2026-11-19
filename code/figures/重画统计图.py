# -*- coding: utf-8 -*-
"""用本地 CSV 重画英文版箱线图 + 配对散点图（不重跑实验）。"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 读数据 ============
rows4 = {}
with open('结果/第十四步_per_image.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows4[r['file']] = {k: (float(v) if k not in ('file', 'label') else v)
                            for k, v in r.items()}
rows_r = {}
with open('结果/第十四步_RISE_per_image.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows_r[r['file']] = {k: (float(v) if k not in ('file', 'label') else v)
                             for k, v in r.items()}
common = [f for f in rows4 if f in rows_r]

METHODS = [('grad', 'Grad-CAM'), ('gradpp', 'Grad-CAM++'), ('score', 'Score-CAM'),
           ('dfo', 'DFO-CAM'), ('rise', 'RISE')]
COLORS = {'Grad-CAM': '#5185C0', 'Grad-CAM++': '#8EA9D4', 'Score-CAM': '#99AABB',
          'DFO-CAM': '#55966B', 'RISE': '#C96144'}


def get(m, k):
    src = rows_r if m == 'rise' else rows4
    return np.array([src[f][f'{m}_{k}'] for f in common])


# ============ 图1 箱线图 ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, metric, cname in [(axes[0], 'del', 'Deletion AUC\n(lower = more faithful)'),
                          (axes[1], 'ins', 'Insertion AUC\n(higher = more faithful)')]:
    data = [get(m, metric) for m, _ in METHODS]
    bp = ax.boxplot(data, tick_labels=[n for _, n in METHODS],
                    patch_artist=True, widths=0.6, showfliers=False)
    for patch, (_, name) in zip(bp['boxes'], METHODS):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.75)
    ax.set_ylabel(cname, fontsize=10)
    ax.set_xticklabels([n for _, n in METHODS], rotation=20, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', color='#E2E8F0')
axes[0].set_title('Ranking metrics across 300 images', fontsize=11.5, fontweight='bold')
axes[1].set_title('', fontsize=11)
plt.tight_layout()
plt.savefig('图片/fig_significance_boxplot.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('箱线图已生成', flush=True)

# ============ 图2 配对散点图（DFO-CAM vs 各基线，AI 逐图） ============
baselines = [m for m, _ in METHODS if m != 'dfo']
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
for ax, b in zip(axes, baselines):
    a = get('dfo', 'ai')
    bb = get(b, 'ai')
    name = dict(METHODS)[b]
    jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(a))
    ax.scatter(bb + jitter, a + jitter, s=18, alpha=0.6, color=COLORS[name],
               edgecolor='white', linewidth=0.4)
    ax.plot([0, 1], [0, 1], '--', color='#94A3B8', lw=1)
    ax.set_title(f'DFO-CAM vs {name}', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{name} AI', fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['0', '1'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['0', '1'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('DFO-CAM AI', fontsize=9)
plt.tight_layout()
plt.savefig('图片/fig_paired_scatter.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('配对散点图已生成', flush=True)
