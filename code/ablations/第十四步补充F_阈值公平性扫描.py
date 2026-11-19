# -*- coding: utf-8 -*-
# 第十四步补充 F：阈值公平性扫描（Content-AUC / 保留比例对齐）
#
# 背景：审稿人质疑"RISE 的 AI=2.7% 是固定 τ=0.5 对 RISE 不公平的伪影"
#   （RISE 值分布接近均匀，τ=0.5 只保留极小区域）。本脚本按【保留像素比例】
#   对齐各方法（而非固定 τ），扫描保留比例 p，看"RISE 内容指标差"是否在
#   公平阈值下仍成立。
#
# 协议：
#   - 300 张（每类 150，与第十四步文件顺序一致），白底填充
#   - 三个机制代表：RISE（黑盒）/ Grad-CAM（梯度）/ DFO-CAM（优化）
#   - 每张图每方法生成 saliency，保存 .npy 供复用（断点续跑）
#   - 扫描保留比例 p ∈ P_LIST，用 saliency 排序取最高 p 比例像素
#   - 对每个 p：mask 掉其余像素、填白底、前向算 y_keep → AI/AG
#
# 输出：结果/阈值公平性扫描结果.txt + 图片/阈值公平性扫描曲线.png
import os
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

N_MASKS = 4000
GRID = 8
MASK_P = 0.5
MASK_BATCH = 128
H = W = 224
N_PER_CLASS = 150
P_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SAL_DIR = 'saliency'          # 保存 saliency .npy
RESULT_TXT = '结果/阈值公平性扫描结果.txt'
RESULT_PNG = '图片/阈值公平性扫描曲线.png'

os.makedirs(SAL_DIR, exist_ok=True)
os.makedirs('结果', exist_ok=True)
os.makedirs('图片', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

# ============ 模型（与第十四步同一权重） ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32, device=device).view(1, 3, 1, 1)

# ============ 加载DFO-CAM DFO-CAM ============
spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)
        w = super().get_cam_weights(input_tensor, target_layer, targets,
                                    activations, grads)
        return np.asarray(w, dtype=np.float32)


dfo_cam = DFOCAMCompat(model=model, target_layers=[model.layer4])
grad_cam = GradCAMElementWise(model=model, target_layers=[model.layer4])


def rise_saliency(x, target):
    """RISE 显著性图（与第十四步补充一致，白底填充，固定种子）。"""
    rng = np.random.RandomState(0)
    masks = (rng.random((N_MASKS, 1, GRID, GRID)) < MASK_P).astype(np.float32)
    masks = torch.from_numpy(masks)
    masks_hi = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
    scores = []
    with torch.no_grad():
        for i in range(0, N_MASKS, MASK_BATCH):
            m = masks_hi[i:i + MASK_BATCH].to(device)
            xm = x * m + BASE_WHITE * (1 - m)
            out = model(xm)
            scores.append(torch.softmax(out, dim=1)[:, target].cpu())
    scores = torch.cat(scores)
    sal = (masks_hi * scores.view(N_MASKS, 1, 1, 1)).sum(0) / N_MASKS
    return sal.squeeze().cpu().numpy()


def grad_saliency(x, target):
    cam = grad_cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    return cam


def dfo_saliency(x, target):
    cam = dfo_cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    return cam


# ============ 文件清单（与第十四步完全一致） ============
files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

print(f'共 {len(files)} 张图', flush=True)

# ============ 阶段1：生成并保存 saliency（断点续跑） ============
METHODS = ['rise', 'grad', 'dfo']
SAL_FN = {m: (rise_saliency, grad_saliency, dfo_saliency)[i]
          for i, m in enumerate(METHODS)}

t_start = time.time()
n_gen = 0
for i, (path, gt, lab) in enumerate(files):
    npz = os.path.join(SAL_DIR, f'{i}.npz')
    if os.path.exists(npz):
        continue
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    sals = {}
    for m in METHODS:
        s = SAL_FN[m](x, pred)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        sals[m] = s.astype(np.float32)
    np.savez_compressed(npz, **sals)
    n_gen += 1
    if n_gen % 10 == 0:
        el = time.time() - t_start
        print(f'saliency 生成 {n_gen} 张 | 单张 {el/n_gen:.1f}s', flush=True)

print(f'saliency 生成完成（新增 {n_gen} 张）', flush=True)

# ============ 阶段2：保留比例扫描 ============
# acc[m][p] = {'ai': [...], 'ag': [...]}
acc = {m: {p: {'ai': [], 'ag': []} for p in P_LIST} for m in METHODS}

t_start = time.time()
for i, (path, gt, lab) in enumerate(files):
    npz = np.load(os.path.join(SAL_DIR, f'{i}.npz'))
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()

    for m in METHODS:
        sal = npz[m]
        order = np.argsort(-sal.flatten())          # 从高到低
        for p in P_LIST:
            k = int(p * H * W)
            mask = torch.zeros(1, H, W, device=device)
            rc = np.unravel_index(order[:k], (H, W))
            mask[0, rc[0], rc[1]] = 1.0
            kept = x * mask + BASE_WHITE * (1 - mask)
            with torch.no_grad():
                y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
            acc[m][p]['ai'].append(1.0 if y_keep > y_orig else 0.0)
            acc[m][p]['ag'].append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)

    if (i + 1) % 25 == 0:
        el = time.time() - t_start
        print(f'扫描进度 {i+1}/{len(files)} | 已用 {el:.0f}s', flush=True)

print('扫描完成，汇总 ...', flush=True)

# ============ 汇总 + 输出 ============
PRETTY = {'rise': 'RISE', 'grad': 'Grad-CAM', 'dfo': 'DFO-CAM'}
lines = [
    '=== 阈值公平性扫描（保留比例对齐，Content-AUC）===',
    f'方法：RISE（黑盒）/ Grad-CAM（梯度）/ DFO-CAM（优化）',
    f'样本 {len(files)} 张；白底填充；扫描保留比例 p ∈ {P_LIST}',
    '读法：对每个保留比例 p，看三个方法的 AI/AG 谁高。',
    '若 RISE 的 AI 在所有 p 下都显著低于 Grad-CAM/DFO-CAM，则"排序好但内容差"',
    '不是 τ=0.5 的伪影；反之则需收敛论文"指标分化"的结论。',
    '',
    f'{"保留比例p":<10} {"RISE AI%":<10} {"Grad-CAM AI%":<14} {"DFO-CAM AI%":<13} '
    f'{"RISE AG":<9} {"Grad AG":<9} {"DFO AG":<9}',
]
ai_curves, ag_curves = {}, {}
for m in METHODS:
    ai_curves[m] = [np.mean(acc[m][p]['ai']) * 100 for p in P_LIST]
    ag_curves[m] = [np.mean(acc[m][p]['ag']) for p in P_LIST]

for j, p in enumerate(P_LIST):
    lines.append(
        f'{p:<10} {ai_curves["rise"][j]:<10.1f} {ai_curves["grad"][j]:<14.1f} '
        f'{ai_curves["dfo"][j]:<13.1f} {ag_curves["rise"][j]:<9.2f} '
        f'{ag_curves["grad"][j]:<9.2f} {ag_curves["dfo"][j]:<9.2f}')

# Content-AUC：AI 对保留比例积分的曲线下面积（梯形近似）
def content_auc(curve, plist):
    # curve 已经以百分数（0–100）记录；除以扫描区间长度后仍为 0–100。
    return float(np.trapezoid(curve, plist) / (plist[-1] - plist[0]))

lines.append('')
lines.append('--- Content-AUC（AI 曲线对保留比例积分，0-100）---')
for m in METHODS:
    lines.append(f'{PRETTY[m]:<10} AI 归一化面积 {content_auc(ai_curves[m], P_LIST):.2f}%'
                 f' | AG 归一化面积 {content_auc(ag_curves[m], P_LIST):.2f}%')

report = '\n'.join(lines)
with open(RESULT_TXT, 'w', encoding='utf-8') as f:
    f.write(report)
print(report, flush=True)

# ============ 画图 ============
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'rise': '#C96144', 'grad': '#5185C0', 'dfo': '#55966B'}
for m in METHODS:
    ax1.plot(P_LIST, ai_curves[m], marker='o', ms=4, label=PRETTY[m],
             color=colors[m])
    ax2.plot(P_LIST, ag_curves[m], marker='o', ms=4, label=PRETTY[m],
             color=colors[m])
ax1.set_xlabel('Kept pixel fraction p'); ax1.set_ylabel('AI (%)')
ax1.set_title('AI vs kept fraction (threshold-free)')
ax1.legend(frameon=False)
ax2.set_xlabel('Kept pixel fraction p'); ax2.set_ylabel('AG (%)')
ax2.set_title('AG vs kept fraction (threshold-free)')
ax2.legend(frameon=False)
plt.tight_layout()
plt.savefig(RESULT_PNG, dpi=200)
print(f'\ndone：结果见 {RESULT_TXT}，曲线见 {RESULT_PNG}', flush=True)
