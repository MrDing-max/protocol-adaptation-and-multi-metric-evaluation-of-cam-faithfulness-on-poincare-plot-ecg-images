# -*- coding: utf-8 -*-
# 第十四步补充 J：五方法 × 四底图 Deletion 对照（加强 Major 4）
# 原补充E 只对 Grad-CAM 做四底图对照；审稿意见要求五方法。本脚本在 PTB-XL 300 张上，
# 对五种方法（Grad-CAM / Grad-CAM++ / Score-CAM / DFO-CAM / RISE）做白/灰/黑/噪声
# 四种底图的 Deletion AUC 对照（模型排序 vs 随机基线，配对 t 检验）。
# 输出：结果/五方法四底图.txt
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
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
N_MASKS = 4000
GRID = 8
MASK_P = 0.5
MASK_BATCH = 128


def pixel_to_norm(px):
    return (np.array(px, dtype=np.float32) - MEAN) / STD


def make_bases(device):
    rng = np.random.RandomState(0)
    noise_px = rng.normal(0.5, 0.2, size=(3,)).astype(np.float32)
    return {
        'white(训练内)': torch.tensor(pixel_to_norm([1, 1, 1]), dtype=torch.float32,
                                      device=device).view(1, 3, 1, 1),
        'gray(均值灰)': torch.tensor(pixel_to_norm([0.5, 0.5, 0.5]), dtype=torch.float32,
                                     device=device).view(1, 3, 1, 1),
        'black(黑)': torch.tensor(pixel_to_norm([0, 0, 0]), dtype=torch.float32,
                                  device=device).view(1, 3, 1, 1),
        'noise(高斯)': torch.tensor(pixel_to_norm(noise_px), dtype=torch.float32,
                                   device=device).view(1, 3, 1, 1),
    }


def auc_of_curve(logits):
    lo = logits.min()
    denom = logits.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits - lo) / denom, np.linspace(0, 1, NUM_STEPS + 1)))


def deletion_curves(model, x, orders, base, pred):
    base = base.expand(-1, -1, H, W).clone()
    xs = []
    for order in orders:
        for step in range(NUM_STEPS + 1):
            k = int(step / NUM_STEPS * H * W)
            mask = torch.zeros(1, H, W, device=x.device)
            rc = np.unravel_index(order[:k], (H, W))
            mask[0, rc[0], rc[1]] = 1.0
            xs.append(x * (1 - mask) + base * mask)
    big = torch.cat(xs, dim=0)
    chunks = []
    with torch.no_grad():
        for i in range(0, big.size(0), 82):
            chunks.append(model(big[i:i + 82]).cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    n = NUM_STEPS + 1
    return [logits[j * n:(j + 1) * n] for j in range(len(orders))]


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
    return (masks_hi * scores.view(N_MASKS, 1, 1, 1)).sum(0) / N_MASKS


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['normal', 'af']:
    for f in sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]:
        files.append((f'data/ptbxl/images/{lab}/{f}', {'af': 0, 'normal': 1}[lab], lab))

bases = make_bases(device)
acc = {b: {m: [] for m in METHODS + ['rand']} for b in bases}

t0 = time.time()
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    cams = {}
    for m in ['grad', 'gradpp', 'score', 'dfo']:
        c = EXTRACTORS[m](input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cams[m] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    cams['rise'] = rise_saliency(x, pred)
    cams['rise'] = (cams['rise'] - cams['rise'].min()) / \
        (cams['rise'].max() - cams['rise'].min() + 1e-8)

    orders = [np.argsort(-cams[m].flatten()) for m in METHODS]
    rng = np.random.RandomState(1000 + i)
    rand_order = rng.permutation(H * W)

    for bname, base in bases.items():
        curves = deletion_curves(model, x, orders + [rand_order], base, pred)
        for j, m in enumerate(METHODS):
            a = auc_of_curve(curves[j][:, pred])
            if a is not None:
                acc[bname][m].append(a)
        ra = auc_of_curve(curves[-1][:, pred])
        if ra is not None:
            acc[bname]['rand'].append(ra)

    if (i + 1) % 25 == 0:
        print(f'progress {i+1}/{len(files)} | 单张 {(time.time()-t0)/(i+1):.1f}s',
              flush=True)

lines = ['=== 五方法 × 四底图 Deletion AUC 对照（PTB-XL 300 张）===',
         '每行：模型排序 AUC ± std  vs  随机基线 AUC（配对 t 检验，模型<随机=忠实）']
for bname in bases:
    lines.append('')
    lines.append(f'--- {bname} ---')
    for m in METHODS:
        a = np.array(acc[bname][m])
        r = np.array(acc[bname]['rand'])
        if len(a) < 10:
            continue
        t, p = stats.ttest_rel(a, r)
        faithful = a.mean() < r.mean() and p < 0.05
        lines.append(f'{PRETTY[m]:<11} 模型 {a.mean():.4f}±{a.std():.3f} vs 随机 '
                     f'{r.mean():.4f}  p={p:.2e}  {"忠实" if faithful else "反常/不显著"}')

report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/五方法四底图.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)
print('done：结果见 结果/五方法四底图.txt', flush=True)
