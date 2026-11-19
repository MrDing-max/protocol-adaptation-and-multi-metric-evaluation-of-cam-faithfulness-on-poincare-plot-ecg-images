# -*- coding: utf-8 -*-
# 第十七步补充：Chapman-Shaoxing 五方法忠实性（Deletion/Insertion + AI/AD/AG，白底协议）
# 补审稿意见 Major 7：原第十七步只在 Chapman 上复现了 Grad-CAM；本脚本扩展到五方法
# （Grad-CAM / Grad-CAM++ / Score-CAM / DFO-CAM / RISE），并加内容指标 AI/AD/AG。
# 输出：结果/Chapman五方法忠实性.txt
import os
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

NUM_STEPS = 40
H = W = 224
N_PER_CLASS = 150
CAM_THRESHOLD = 0.5
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
N_MASKS = 4000
GRID = 8
MASK_P = 0.5
MASK_BATCH = 128

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_chapman.pth', map_location='cpu'))
model.eval()
model = model.to(device)

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


EXTRACTORS = {
    'grad': GradCAMElementWise(model=model, target_layers=[model.layer4]),
    'gradpp': GradCAMPlusPlus(model=model, target_layers=[model.layer4]),
    'score': ScoreCAM(model=model, target_layers=[model.layer4]),
    'dfo': DFOCAMCompat(model=model, target_layers=[model.layer4]),
}
METHODS = ['grad', 'gradpp', 'score', 'dfo', 'rise']
PRETTY = {'grad': 'GradCAM', 'gradpp': 'GradCAM++', 'score': 'ScoreCAM',
          'dfo': 'DFOCAM', 'rise': 'RISE'}

white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32, device=device).view(1, 3, 1, 1)


def rise_saliency(x, target):
    rng = np.random.RandomState(0)
    masks = (rng.random((N_MASKS, 1, GRID, GRID)) < MASK_P).astype(np.float32)
    masks = torch.from_numpy(masks)
    masks_hi = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
    scores = []
    with torch.no_grad():
        for i in range(0, N_MASKS, MASK_BATCH):
            m = masks_hi[i:i + MASK_BATCH].to(device)
            xm = x * m + BASE_WHITE * (1 - m)
            scores.append(torch.softmax(model(xm), dim=1)[:, target].cpu())
    scores = torch.cat(scores)
    return ((masks_hi * scores.view(N_MASKS, 1, 1, 1)).sum(0) / N_MASKS).squeeze().cpu().numpy()


def auc_of_curve(logits):
    lo = logits.min()
    denom = logits.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits - lo) / denom, np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_multi(x, orders, mode):
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    for order in orders:
        for step in range(NUM_STEPS + 1):
            k = int(step / NUM_STEPS * H * W)
            mask = torch.zeros(1, H, W, device=x.device)
            rc = np.unravel_index(order[:k], (H, W))
            mask[0, rc[0], rc[1]] = 1.0
            if mode == 'insertion':
                xs.append(base * (1 - mask) + x * mask)
            else:
                xs.append(x * (1 - mask) + base * mask)
    big = torch.cat(xs, dim=0)
    chunks = []
    with torch.no_grad():
        for i in range(0, big.size(0), 82):
            chunks.append(model(big[i:i + 82]).cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    n = NUM_STEPS + 1
    return [logits[j * n:(j + 1) * n] for j in range(len(orders))]


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['sr', 'afib']:
    for f in sorted(os.listdir(f'data/chapman/images/{lab}'))[:N_PER_CLASS]:
        files.append((f'data/chapman/images/{lab}/{f}', {'afib': 0, 'sr': 1}[lab], lab))

rows = []
t0 = time.time()
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits0 = model(x)
        pred = logits0.argmax(1).item()
        y_orig = torch.softmax(logits0, dim=1)[0, gt].item()

    cams = {}
    for m in ['grad', 'gradpp', 'score', 'dfo']:
        c = EXTRACTORS[m](input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cams[m] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    cams['rise'] = rise_saliency(x, pred)
    cams['rise'] = (cams['rise'] - cams['rise'].min()) / \
        (cams['rise'].max() - cams['rise'].min() + 1e-8)

    rng = np.random.RandomState(1000 + i)
    rand_orders = [rng.permutation(H * W) for _ in range(3)]
    orders = [np.argsort(-cams[m].flatten()) for m in METHODS] + rand_orders
    del_curves = perturbed_multi(x, orders, 'deletion')
    ins_curves = perturbed_multi(x, orders, 'insertion')
    a_del = [auc_of_curve(c[:, pred]) for c in del_curves]
    a_ins = [auc_of_curve(c[:, pred]) for c in ins_curves]
    if None in a_del or None in a_ins:
        continue

    row = {}
    for j, m in enumerate(METHODS):
        row[f'{m}_del'] = a_del[j]
        row[f'{m}_ins'] = a_ins[j]
    row['rand_del'] = float(np.mean(a_del[len(METHODS):]))
    row['rand_ins'] = float(np.mean(a_ins[len(METHODS):]))

    kept = torch.cat([
        x * torch.from_numpy((cams[m] > CAM_THRESHOLD).astype(np.float32)).to(device)
        for m in METHODS], dim=0)
    with torch.no_grad():
        y_keep = torch.softmax(model(kept), dim=1)[:, gt].cpu().numpy()
    for j, m in enumerate(METHODS):
        row[f'{m}_ad'] = max(0.0, y_orig - y_keep[j]) / (y_orig + 1e-8) * 100
        row[f'{m}_ai'] = 1.0 if y_keep[j] > y_orig else 0.0
        row[f'{m}_ag'] = max(0.0, y_keep[j] - y_orig) / (1 - y_orig + 1e-8) * 100

    rows.append(row)
    if (i + 1) % 25 == 0:
        print(f'progress {i+1}/{len(files)} | 单张 {(time.time()-t0)/(i+1):.1f}s',
              flush=True)

n = len(rows)
lines = ['=== Chapman-Shaoxing 五方法忠实性（白底协议，每类 150）===',
         f'有效样本 {n} 张', '',
         f'{"方法":<11} {"Deletion↓":<16} {"Insertion↑":<16} {"AI%":<7} {"AD↓":<8} {"AG↑":<8}']
for m in METHODS:
    lines.append(
        f'{PRETTY[m]:<11} {np.mean([r[f"{m}_del"] for r in rows]):.4f}±'
        f'{np.std([r[f"{m}_del"] for r in rows]):.2f}  '
        f'{np.mean([r[f"{m}_ins"] for r in rows]):.4f}±'
        f'{np.std([r[f"{m}_ins"] for r in rows]):.2f}  '
        f'{np.mean([r[f"{m}_ai"] for r in rows])*100:.1f}  '
        f'{np.mean([r[f"{m}_ad"] for r in rows]):.2f}  '
        f'{np.mean([r[f"{m}_ag"] for r in rows]):.2f}')
lines.append(f'{"随机基线":<11} {np.mean([r["rand_del"] for r in rows]):.4f}  '
             f'{np.mean([r["rand_ins"] for r in rows]):.4f}  {"-":<7} {"-":<8} {"-":<8}')
lines.append('')
lines.append('--- 各方法 Deletion/Insertion vs 随机基线（配对 t 检验）---')
for m in METHODS:
    for metric, rkey, better in [('del', 'rand_del', 'low'), ('ins', 'rand_ins', 'high')]:
        a = np.array([r[f'{m}_{metric}'] for r in rows])
        rr = np.array([r[rkey] for r in rows])
        t, p = stats.ttest_rel(a, rr)
        ok = (a.mean() < rr.mean()) if better == 'low' else (a.mean() > rr.mean())
        lines.append(f'{PRETTY[m]:<11} {metric}: {a.mean():.4f} vs 随机 {rr.mean():.4f} '
                     f'p={p:.2e} {"显著优于随机" if ok and p < 0.05 else "异常/不显著"}')

report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/Chapman五方法忠实性.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)
print('done：结果见 结果/Chapman五方法忠实性.txt', flush=True)
