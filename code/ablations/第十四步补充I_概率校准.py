# -*- coding: utf-8 -*-
# 第十四步补充 I：概率校准（ECE + 可靠性图）
# 审稿意见：遮挡实验暴露过度自信（49.7% 准确率却 99.8% 置信度）；应报概率校准。
# 本脚本在 PTB-XL 300 张评估子集（每类 150）上计算 ECE（10 个等宽 bin）并画可靠性图。
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['normal', 'af']:
    for f in sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:150]:
        files.append((f'data/ptbxl/images/{lab}/{f}', {'af': 0, 'normal': 1}[lab]))

probs, trues = [], []
for path, gt in files:
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    probs.append(p)
    trues.append(gt)

probs = np.array(probs)
trues = np.array(trues)
conf = probs.max(1)
pred = probs.argmax(1)
acc = (pred == trues).astype(float)

n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
ece = 0.0
lines = [
    '=== 概率校准（ECE + 可靠性）===',
    f'样本 {len(files)} 张（评估子集，每类 150）；置信度 = softmax 最大值',
    f'总体准确率 {acc.mean()*100:.2f}%，平均置信度 {conf.mean()*100:.2f}%',
    '',
    f'{"置信度区间":<14} {"样本数":<6} {"准确率%":<9} {"平均置信度%":<12} {"gap%":<8}',
]
for b in range(n_bins):
    lo, hi = bins[b], bins[b + 1]
    sel = (conf >= lo) & (conf < hi)
    if b == n_bins - 1:
        sel = (conf >= lo) & (conf <= hi)
    n = int(sel.sum())
    if n == 0:
        continue
    bin_acc = acc[sel].mean()
    bin_conf = conf[sel].mean()
    gap = bin_conf - bin_acc
    ece += (n / len(files)) * abs(gap)
    lines.append(f'[{lo:.1f},{hi:.1f})    {n:<6} {bin_acc*100:<9.2f} '
                 f'{bin_conf*100:<12.2f} {gap*100:+.2f}')
lines.append('')
lines.append(f'ECE（期望校准误差）= {ece*100:.2f}%')

report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/概率校准.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4.5))
bc, ba, bn = [], [], []
for b in range(n_bins):
    lo, hi = bins[b], bins[b + 1]
    sel = (conf >= lo) & (conf < hi)
    if b == n_bins - 1:
        sel = (conf >= lo) & (conf <= hi)
    if sel.sum() > 0:
        bc.append(conf[sel].mean())
        ba.append(acc[sel].mean())
        bn.append(int(sel.sum()))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
ax.plot(bc, ba, 'o-', color='#2563EB', label='Model')
for x, y, n in zip(bc, ba, bn):
    ax.annotate(str(n), (x, y), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=8)
ax.set_xlabel('Confidence')
ax.set_ylabel('Accuracy')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('图片/可靠性图.png', dpi=150, bbox_inches='tight')
print('done：结果见 结果/概率校准.txt，图见 图片/可靠性图.png', flush=True)
