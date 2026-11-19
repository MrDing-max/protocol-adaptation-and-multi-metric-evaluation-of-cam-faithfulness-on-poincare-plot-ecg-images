# -*- coding: utf-8 -*-
# 第十四步补充 K：RISE 配置消融（网格分辨率 / 掩码数 N）
# Major 3：RISE 的 AI/AG 低可能受 8×8 网格、N=4000 配置影响。本脚本在 100 张子集上
# 扫 grid ∈ {8,16,28} × N ∈ {1000,4000}，报 Deletion/Insertion AUC + AI/AD/AG。
# 输出：结果/RISE消融.txt
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

H = W = 224
N_PER_CLASS = 50
NUM_STEPS = 40
CAM_THRESHOLD = 0.5
MASK_P = 0.5
MASK_BATCH = 128
GRIDS = [8, 16, 28]
NS = [1000, 4000]
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32, device=device).view(1, 3, 1, 1)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def auc_of_curve(logits):
    lo = logits.min()
    denom = logits.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits - lo) / denom, np.linspace(0, 1, NUM_STEPS + 1)))


def rise_saliency(x, target, grid, n_masks):
    rng = np.random.RandomState(0)
    masks = (rng.random((n_masks, 1, grid, grid)) < MASK_P).astype(np.float32)
    masks = torch.from_numpy(masks)
    masks_hi = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
    scores = []
    with torch.no_grad():
        for i in range(0, n_masks, MASK_BATCH):
            m = masks_hi[i:i + MASK_BATCH].to(device)
            xm = x * m + BASE_WHITE * (1 - m)
            scores.append(torch.softmax(model(xm), dim=1)[:, target].cpu())
    scores = torch.cat(scores)
    return ((masks_hi * scores.view(n_masks, 1, 1, 1)).sum(0) / n_masks).squeeze().cpu().numpy()


def deletion_insertion(x, order):
    """返回 (del_auc, ins_auc)。"""
    curves = {'del': [], 'ins': []}
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * H * W)
        mask = torch.zeros(1, H, W, device=device)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        curves['del'].append(x * (1 - mask) + BASE_WHITE * mask)
        curves['ins'].append(BASE_WHITE * (1 - mask) + x * mask)
    outs = {}
    for mode in ['del', 'ins']:
        big = torch.cat(curves[mode], dim=0)
        ls = []
        with torch.no_grad():
            for i in range(0, big.size(0), 82):
                ls.append(model(big[i:i + 82]).cpu().numpy())
        outs[mode] = np.concatenate(ls, axis=0)
    return outs['del'], outs['ins']


files = []
for lab in ['normal', 'af']:
    for f in sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]:
        files.append((f'data/ptbxl/images/{lab}/{f}', {'af': 0, 'normal': 1}[lab], lab))

# 逐张先算好 y_orig / pred（与配置无关），缓存
meta = []
for path, gt, lab in files:
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()
    meta.append((path, gt, lab, x, pred, y_orig))

results = []
for grid in GRIDS:
    for n_masks in NS:
        del_, ins_, ai_, ad_, ag_ = [], [], [], [], []
        t0 = time.time()
        for (path, gt, lab, x, pred, y_orig) in meta:
            sal = rise_saliency(x, pred, grid, n_masks)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            order = np.argsort(-sal.flatten())
            d, ins = deletion_insertion(x, order)
            a_d = auc_of_curve(d[:, pred])
            a_i = auc_of_curve(ins[:, pred])
            if a_d is not None and a_i is not None:
                del_.append(a_d)
                ins_.append(a_i)
            mask = torch.from_numpy((sal > CAM_THRESHOLD).astype(np.float32)).to(device)
            kept = x * mask
            with torch.no_grad():
                y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
            ai_.append(1.0 if y_keep > y_orig else 0.0)
            ad_.append(max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100)
            ag_.append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)
        results.append((grid, n_masks, del_, ins_, ai_, ad_, ag_))
        print(f'grid={grid} N={n_masks}: {time.time()-t0:.0f}s', flush=True)

lines = ['=== RISE 配置消融（grid × N，100 张子集，白底协议）===',
         f'{"grid":<5} {"N":<6} {"Deletion↓":<10} {"Insertion↑":<10} '
         f'{"AI%":<7} {"AD↓":<8} {"AG↑":<8}']
for grid, n_masks, del_, ins_, ai_, ad_, ag_ in results:
    lines.append(f'{grid:<5} {n_masks:<6} {np.mean(del_):.4f}±{np.std(del_):.2f}  '
                 f'{np.mean(ins_):.4f}±{np.std(ins_):.2f}  '
                 f'{np.mean(ai_)*100:.1f}  {np.mean(ad_):.2f}  {np.mean(ag_):.2f}')
lines.append('')
lines.append('对照（表3 主配置 grid=8,N=4000）：RISE Deletion 0.334 / Insertion 0.671 / '
             'AI 2.7% / AD 47.55 / AG 0.60。若各配置下 AI/AG 仍接近 0，说明"内容指标低"'
             '不是配置伪影。')

report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/RISE消融.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)
print('done：结果见 结果/RISE消融.txt', flush=True)
