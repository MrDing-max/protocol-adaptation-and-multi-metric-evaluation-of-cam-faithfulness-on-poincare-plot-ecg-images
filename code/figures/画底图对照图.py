# -*- coding: utf-8 -*-
"""专业版：四底图 Deletion AUC 对照（模型排序 vs 随机基线，两数据集）。
核心发现：gray 底图让随机基线 AUC 异常降低(0.209)，模型排序反而"高于"随机 → 假阴性。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

bg = ['White\n(in-distribution)', 'Gray\n(mean fill)', 'Black', 'Noise']
C_MODEL = '#5185C0'   # 模型排序 AUC
C_RAND = '#99AABB'    # 随机基线 AUC

data = {
    'PTB-XL (300)': {
        'model': [0.3713, 0.3901, 0.2847, 0.2678],
        'rand':  [0.4877, 0.2094, 0.3629, 0.4008],
        'mstd':  [0.169, 0.224, 0.092, 0.121],
        'rstd':  [0.142, 0.110, 0.139, 0.072],
    },
    'Chapman-Shaoxing (300)': {
        'model': [0.3923, 0.3319, 0.3158, 0.2562],
        'rand':  [0.4786, 0.1914, 0.3181, 0.4151],
        'mstd':  [0.140, 0.163, 0.215, 0.097],
        'rstd':  [0.097, 0.056, 0.191, 0.054],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(4)
w = 0.36

for ax, (name, d) in zip(axes, data.items()):
    ax.bar(x - w/2, d['model'], w, yerr=d['mstd'], capsize=3, color=C_MODEL,
           label='Model ranking AUC', edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, d['rand'], w, yerr=d['rstd'], capsize=3, color=C_RAND,
           label='Random baseline AUC', edgecolor='white', linewidth=0.5)

    # 高亮 gray 底图（随机基线被异常压低 → 反常）
    ax.annotate('false negative:\nrandom AUC\nabnormally low', xy=(1, 0.21),
                xytext=(1.35, 0.34), fontsize=8.5, color='#C0392B',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(bg, fontsize=9)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylabel('Deletion AUC (lower = more faithful)', fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')

fig.suptitle('Background fill changes the random baseline, not just the model ranking',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('图片/底图对照_专业版.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('已保存 图片/底图对照_专业版.png')
