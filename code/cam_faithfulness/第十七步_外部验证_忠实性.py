# -*- coding: utf-8 -*-
# 第十七步（外部验证）：Chapman 模型 GradCAM 忠实性（Deletion/Insertion，白底协议）
# 复用第十二步协议：40 步逐像素、logit 单图归一化、白色底、随机顺序基线
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

NUM_STEPS = 40
H = W = 224
N_PER_CLASS = 100          # 忠实性取子集（每类 100，共 200），省时间

print('=== 第十七步 V1：Chapman 忠实性（GradCAM Deletion/Insertion 白底协议）===',
      flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_chapman.pth', map_location='cpu'))
model.eval()
model = model.to(device)

extractor = GradCAMElementWise(model=model, target_layers=[model.layer4])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def auc_of_curve(logits):
    lo = logits.min()
    denom = logits.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits - lo) / denom, np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_batch(x, order, mode):
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    total = H * W
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * total)
        mask = torch.zeros(1, H, W, device=device)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        pert = base * (1 - mask) + x * mask if mode == 'insertion' else \
            x * (1 - mask) + base * mask
        xs.append(pert)
    return torch.cat(xs, dim=0)


rng = np.random.RandomState(0)
del_model, del_rand, ins_model, ins_rand = [], [], [], []
n_used = 0

files = []
for lab in ['sr', 'afib']:
    fs = sorted(os.listdir(f'data/chapman/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/chapman/images/{lab}/{f}',
                      {'afib': 0, 'sr': 1}[lab]))

for i, (path, gt) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    cam = extractor(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    sorted_idx = np.argsort(-cam_norm.flatten())
    rand_idx = rng.permutation(sorted_idx.size)

    with torch.no_grad():
        lm = model(perturbed_batch(x, sorted_idx, 'deletion'))[:, pred].cpu().numpy()
        lr_ = model(perturbed_batch(x, rand_idx, 'deletion'))[:, pred].cpu().numpy()
        li = model(perturbed_batch(x, sorted_idx, 'insertion'))[:, pred].cpu().numpy()
        l2 = model(perturbed_batch(x, rand_idx, 'insertion'))[:, pred].cpu().numpy()
    a, r, b, c = (auc_of_curve(lm), auc_of_curve(lr_),
                  auc_of_curve(li), auc_of_curve(l2))
    if None in (a, r, b, c):
        continue
    del_model.append(a); del_rand.append(r)
    ins_model.append(b); ins_rand.append(c)
    n_used += 1
    if (i + 1) % 50 == 0:
        print(f'progress {i+1}/{len(files)}', flush=True)

lines = [
    '=== 第十七步：Chapman 忠实性（GradCAM Deletion/Insertion 白底协议）===',
    f'有效样本 {n_used} 张（每类 {N_PER_CLASS}，白底 40 步）',
    '',
    f'Deletion AUC（越小越忠实）: 按热力图 {np.mean(del_model):.4f} ± {np.std(del_model):.4f}'
    f'  vs  随机 {np.mean(del_rand):.4f}',
    f'Insertion AUC（越大越忠实）: 按热力图 {np.mean(ins_model):.4f} ± {np.std(ins_model):.4f}'
    f'  vs  随机 {np.mean(ins_rand):.4f}',
    '',
    f'结论: Deletion 按热力图{"<" if np.mean(del_model) < np.mean(del_rand) else ">="}随机'
    f'（{"忠实" if np.mean(del_model) < np.mean(del_rand) else "异常"}）；'
    f'Insertion 按热力图{">" if np.mean(ins_model) > np.mean(ins_rand) else "<="}随机'
    f'（{"忠实" if np.mean(ins_model) > np.mean(ins_rand) else "异常"}）',
]
print('\n'.join(lines), flush=True)
with open('结果/第十七步_Chapman忠实性结果.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('done', flush=True)
