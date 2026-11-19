# -*- coding: utf-8 -*-
# 第十四步补充 G：随机基线的内容指标（AI/AD/AG）
# 补审稿人缺口：表 3 只有 Deletion/Insertion 的随机基线，缺内容指标（AI/AD/AG）的随机参照。
# 协议：与第十四步一致——300 张（每类 150），每张图随机保留 50% 像素（3 个随机种子平均），
#   遮掉其余、填白底，算 y_keep，再算 AI/AD/AG（GT 类）。
# 输出：结果/随机基线内容指标.txt
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

H = W = 224
N_PER_CLASS = 150
RAND_SEEDS = 3
KEEP_FRAC = 0.5

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

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}', {'af': 0, 'normal': 1}[lab], lab))

k = int(KEEP_FRAC * H * W)
acc = {'ai': [], 'ad': [], 'ag': []}
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()

    rng = np.random.RandomState(2000 + i)
    ai_vals, ad_vals, ag_vals = [], [], []
    for _ in range(RAND_SEEDS):
        order = rng.permutation(H * W)[:k]
        mask = torch.zeros(1, H, W, device=device)
        rc = np.unravel_index(order, (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        kept = x * mask + BASE_WHITE * (1 - mask)
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        ai_vals.append(1.0 if y_keep > y_orig else 0.0)
        ad_vals.append(max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100)
        ag_vals.append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)

    acc['ai'].append(float(np.mean(ai_vals)))
    acc['ad'].append(float(np.mean(ad_vals)))
    acc['ag'].append(float(np.mean(ag_vals)))
    if (i + 1) % 50 == 0:
        print(f'progress {i+1}/{len(files)}', flush=True)

lines = [
    '=== 随机基线的内容指标（AI/AD/AG，白底协议）===',
    f'样本 {len(files)} 张（正常/房颤各半）；每张图随机保留 {KEEP_FRAC*100:.0f}% 像素，'
    f'{RAND_SEEDS} 个随机种子平均',
    '',
    f'随机基线 AI = {np.mean(acc["ai"])*100:.2f}% ± {np.std(acc["ai"])*100:.2f}%',
    f'随机基线 AD = {np.mean(acc["ad"]):.2f} ± {np.std(acc["ad"]):.2f}',
    f'随机基线 AG = {np.mean(acc["ag"]):.2f} ± {np.std(acc["ag"]):.2f}',
    '',
    '对照（表 3）：DFOCAM AI=44.7 / RISE AI=2.7 / CAM AI=36.7~38.0；'
    '各方法 AD≈47~49；AG: DFOCAM 26.74 / CAM 23~26 / RISE 0.60',
]
report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/随机基线内容指标.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)
print('done：结果见 结果/随机基线内容指标.txt', flush=True)
