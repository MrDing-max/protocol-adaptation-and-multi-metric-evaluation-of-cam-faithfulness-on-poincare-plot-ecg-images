# -*- coding: utf-8 -*-
# 错误分析可视化（审稿意见 M-4）：挑 6 个代表性案例，画热力图对比
#
# 案例三类（各 2 张）：
#   ① DFO-CAM 赢（dfo_ai=1 且 score_ai=0，房颤）—— 核心卖点：DFO 找的区域质量更高
#   ② 模型误判（pred != gt，正常被误判成房颤）—— 看误判时热力图指哪里
#   ③ 判对但保留高亮区域后错（pred=gt 且 dfo_ai=0）—— 热力图高亮区"不对劲"
#
# 每张图算 Grad-CAM / Score-CAM / DFO-CAM 三个热力图，与原始庞加莱图并列。
import csv
import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

# ============ 模型与 DFO-CAM（与第十四步一致） ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    def get_cam_weights(self, input_tensor, target_layer, targets,
                        activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)
        w = super().get_cam_weights(input_tensor, target_layer, targets,
                                    activations, grads)
        return np.asarray(w, dtype=np.float32)


EXTRACTORS = {
    'grad': GradCAMElementWise(model=model, target_layers=[model.layer4]),
    'score': ScoreCAM(model=model, target_layers=[model.layer4]),
    'dfo': DFOCAMCompat(model=model, target_layers=[model.layer4]),
}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============ 挑案例 ============
rows = list(csv.DictReader(open('结果/第十四步_per_image.csv', encoding='utf-8')))
for r in rows:
    for k in r:
        if k not in ('file', 'label'):
            r[k] = float(r[k])

dfo_win = [r for r in rows if r['dfo_ai'] == 1 and r['score_ai'] == 0]
# DFO 的 Deletion 比 Score 好最多的（排序质量优势最明显）
dfo_win.sort(key=lambda r: r['score_del'] - r['dfo_del'], reverse=True)
mis = [r for r in rows if r['pred'] != r['gt']]
# 判对但保留高亮后错，挑 DFO 的 Insertion 最差的（高亮区域最可疑）
susp = [r for r in rows if r['pred'] == r['gt'] and r['dfo_ai'] == 0]
susp.sort(key=lambda r: r['dfo_ins'])

cases = []
for r in dfo_win[:2]:
    cases.append((f"DFO wins (AF): ScoreDEL {r['score_del']:.2f} -> DFO {r['dfo_del']:.2f}",
                  r['file'], r['gt']))
for r in mis[:2]:
    cases.append((f"Misclassified (true {'Normal' if r['gt'] == 1 else 'AF'} -> pred {'Normal' if r['pred'] == 1 else 'AF'})",
                  r['file'], r['gt']))
for r in susp[:2]:
    cases.append((f"Correct but AI=0 (DFO_ins {r['dfo_ins']:.2f})",
                  r['file'], r['gt']))

print('选中的 6 个案例：')
for title, f, _ in cases:
    print(f'  {title}  <- {f}', flush=True)

# ============ 画图 ============
LABELS = ['Grad-CAM', 'Score-CAM', 'DFO-CAM']
fig, axes = plt.subplots(6, 4, figsize=(15, 22))
for i, (title, path, gt) in enumerate(cases):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    # 原图
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f'Original\nGT:{gt}  Pred:{pred}', fontsize=9)
    axes[i, 0].axis('off')

    for j, (m, mname) in enumerate([('grad', 'Grad-CAM'), ('score', 'Score-CAM'),
                                    ('dfo', 'DFO-CAM')]):
        cam = EXTRACTORS[m](input_tensor=x,
                            targets=[ClassifierOutputTarget(pred)])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        axes[i, j + 1].imshow(img)
        axes[i, j + 1].imshow(cam, cmap='jet', alpha=0.55)
        axes[i, j + 1].set_title(mname, fontsize=10)
        axes[i, j + 1].axis('off')

    axes[i, 0].set_ylabel(title, fontsize=9, rotation=0, labelpad=120)

plt.suptitle('Error analysis: CAM heatmaps of 6 representative cases',
             fontsize=13)
plt.tight_layout(rect=[0.08, 0, 1, 0.98])
out = '图片/fig_error_cases.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'done：图已存 {out}', flush=True)
