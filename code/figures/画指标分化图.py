# -*- coding: utf-8 -*-
"""专业版：指标分化图。x=Deletion AUC(排序，越低越忠实)，y=AI%(内容，越高越好)。
核心发现：RISE 排序最优(左下)却内容最差(几乎为零)；DFO-CAM 内容最优；两者分化。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# (名称, Deletion AUC, AI%)
methods = [
    ('Grad-CAM',   0.3713, 38.0),
    ('Grad-CAM++', 0.3713, 37.7),
    ('Score-CAM',  0.3683, 36.7),
    ('DFO-CAM',    0.3641, 44.7),
    ('RISE',       0.3343, 2.7),
]

colors = {
    'RISE':     '#C96144',  # terracotta（黑盒，异常点）
    'DFO-CAM':  '#55966B',  # green（内容最优）
    'Grad-CAM': '#5185C0', 'Grad-CAM++': '#8EA9D4', 'Score-CAM': '#99AABB',
}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, delauc, ai in methods:
    ax.scatter(delauc, ai, s=140, color=colors[name], zorder=3,
               edgecolor='white', linewidth=1.2)
    # 标签偏移（避免重叠）
    dx = 0.0022
    dy = 1.6
    if name == 'RISE':
        ax.annotate(name + '\n(black-box)', (delauc, ai),
                    xytext=(delauc - 0.004, ai + 2.2),
                    fontsize=10, fontweight='bold', color=colors[name],
                    ha='right', zorder=4)
    elif name == 'DFO-CAM':
        ax.annotate(name + '\n(optimization-based)', (delauc, ai),
                    xytext=(delauc + dx, ai + dy),
                    fontsize=10, fontweight='bold', color=colors[name],
                    ha='left', zorder=4)
    else:
        ax.annotate(name, (delauc, ai), xytext=(delauc + dx, ai + dy),
                    fontsize=9, color=colors[name], ha='left', zorder=4)

# 突出"分化"：RISE 和 DFO-CAM 之间画一条虚线箭头
ax.annotate('', xy=(0.3641, 44.7), xytext=(0.3343, 2.7),
            arrowprops=dict(arrowstyle='-|>', color='#8A94A6', lw=1.2,
                            linestyle='--'))
ax.text(0.353, 21, 'ranking vs. content\ndiverge', fontsize=9.5,
        color='#4A5568', ha='center', va='center', style='italic')

ax.set_xlabel('Deletion AUC  (lower = better ranking)', fontsize=10.5)
ax.set_ylabel('AI %  (higher = better content)', fontsize=10.5)
ax.set_title('Ranking quality and content quality diverge across methods',
             fontsize=12, fontweight='bold')
ax.invert_xaxis()          # Deletion 越低越好，反轴让"左边=更好"
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle=':', color='#E2E8F0', zorder=0)
ax.set_ylim(-2, 52)

plt.tight_layout()
plt.savefig('图片/指标分化_专业版.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('已保存 图片/指标分化_专业版.png')
