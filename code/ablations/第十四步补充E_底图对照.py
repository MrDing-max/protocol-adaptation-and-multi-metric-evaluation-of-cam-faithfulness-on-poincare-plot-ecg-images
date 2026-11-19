# -*- coding: utf-8 -*-
# 第十四步补充 E：底图填充系统对照（加强贡献 1「协议适配」的证据）
# ①② 扩大样本 + 跨数据集：PTB-XL 300 张（每类 150）+ Chapman 300 张（每类 150）
# ③ 系统扫底图：白（训练分布内）vs 灰（标准 Deletion 均值灰）vs 黑 vs 高斯噪声
# 验证 3.6 的「经验观察」：底图填充 g 若不在训练分布内（≠ 白色），Deletion 协议
#   引入分布外扰动，导致「忠实方法被判为不如随机」的假阴性；仅训练分布内的白底保持中性。
# 指标：每种底图下，Grad-CAM 排序的 Deletion AUC vs 随机基线，配对 t 检验判断「忠实 or 反常」。
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
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


def pixel_to_norm(px):
    """像素空间颜色值 (3,) -> 归一化空间 (3,)。"""
    return (np.array(px, dtype=np.float32) - MEAN) / STD


def make_bases(device):
    """四种底图（归一化空间，每张 (1,3,1,1)）。噪声用固定种子可复现。"""
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
    """对多条删除顺序，用给定底图 base 算 Deletion 曲线（logit 数组）。
    返回 list[np.array]，每个是对应 order 的 41 步 logit 曲线。"""
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


def run_dataset(name, data_dir, model_path, label_map, classes):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model = model.to(device)
    extractor = GradCAMElementWise(model=model, target_layers=[model.layer4])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    files = []
    for lab in classes:
        fs = sorted(os.listdir(f'{data_dir}/{lab}'))[:N_PER_CLASS]
        for f in fs:
            files.append((f'{data_dir}/{lab}/{f}', label_map[lab], lab))

    bases = make_bases(device)
    # 累积：base_name -> {model_auc: [], rand_auc: []}
    acc = {b: {'model': [], 'rand': []} for b in bases}

    for i, (path, gt, lab) in enumerate(files):
        img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
        x = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).item()

        cam = extractor(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        model_order = np.argsort(-cam.flatten())
        rng = np.random.RandomState(1000 + i)
        rand_order = rng.permutation(H * W)

        for bname, base in bases.items():
            curves = deletion_curves(model, x, [model_order, rand_order], base, pred)
            m_auc = auc_of_curve(curves[0][:, pred])
            r_auc = auc_of_curve(curves[1][:, pred])
            if m_auc is not None and r_auc is not None:
                acc[bname]['model'].append(m_auc)
                acc[bname]['rand'].append(r_auc)
        if (i + 1) % 50 == 0:
            print(f'  [{name}] progress {i + 1}/{len(files)}', flush=True)

    return acc


label_map_ptbxl = {'af': 0, 'normal': 1}
label_map_chapman = {'afib': 0, 'sr': 1}

lines = ['=== 底图填充系统对照（加强贡献 1）===',
         'Grad-CAM Deletion AUC（越小越忠实）；每种底图下模型排序 vs 随机基线（配对 t 检验）',
         '预期：白底（训练内）模型显著优于随机；灰/黑/噪声（分布外）模型不再显著优于随机甚至反常。']

for name, data_dir, model_path, label_map, classes in [
        ('PTB-XL(300)', 'data/ptbxl/images', '模型/resnet18_ptbxl.pth',
         label_map_ptbxl, ['normal', 'af']),
        ('Chapman(300)', 'data/chapman/images', '模型/resnet18_chapman.pth',
         label_map_chapman, ['sr', 'afib'])]:
    print(f'\n=== 跑 {name} ===', flush=True)
    acc = run_dataset(name, data_dir, model_path, label_map, classes)
    lines.append('')
    lines.append(f'--- {name} ---')
    lines.append(f'{"底图":<14} {"模型排序AUC":<22} {"随机AUC":<22} {"配对t检验p":<12} 判断')
    for bname in acc:
        m = np.array(acc[bname]['model'])
        r = np.array(acc[bname]['rand'])
        if len(m) < 10:
            continue
        t, p = stats.ttest_rel(m, r)
        faithful = m.mean() < r.mean() and p < 0.05
        judge = '忠实（模型<随机）' if faithful else '反常/不显著'
        lines.append(f'{bname:<14} {m.mean():.4f}±{m.std():.3f}   '
                     f'{r.mean():.4f}±{r.std():.3f}   p={p:.2e}   {judge}')

report = '\n'.join(lines)
print('\n' + report, flush=True)
open('结果/底图对照结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/底图对照结果.txt', flush=True)
