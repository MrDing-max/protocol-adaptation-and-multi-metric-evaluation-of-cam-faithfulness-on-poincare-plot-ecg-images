# -*- coding: utf-8 -*-
# 第十一步（诊断）：Deletion 反常的排查——底图颜色的影响
# 疑似原因：DFO-CAM 协议底图=均值灰（对自然图像中性），但我们的图是白底黑点，
#           灰色是分布外颜色；"随机涂灰"可能比"把点涂灰"更扰乱模型。
# 方法：同一批图、同一排序，分别用 灰底(DFO-CAM原版) 和 白底(适配本数据) 跑 Deletion，
#       各配模型排序与随机顺序，对比 AUC。
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise

NUM_STEPS = 40
H = W = 224
N_PER_CLASS = 30                       # 诊断用小样本，每类 30 张

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
extractor = GradCAMElementWise(model=model, target_layers=[model.layer4])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 两种底图（归一化空间）：灰色 = 全零；白色 = 白色 RGB 反归一化后的值
base_gray = torch.zeros(3, H, W)
white_c = [(1.0 - m) / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
base_white = torch.tensor(white_c).view(3, 1, 1).repeat(1, H, W)


def auc_of_curve(logits_arr):
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    return float(np.trapezoid((logits_arr - lo) / denom, np.linspace(0, 1, NUM_STEPS + 1)))


def deletion_curve(x, order, base, target):
    xs = []
    total = H * W
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * total)
        mask = torch.zeros(1, H, W)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        xs.append(x * (1 - mask) + base * mask)
    with torch.no_grad():
        return model(torch.cat(xs, dim=0))[:, target].numpy()


rng = np.random.RandomState(0)
aucs = {f'{b}_{o}': [] for b in ['gray', 'white'] for o in ['model', 'rand']}

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    files += [(f'data/ptbxl/images/{lab}/{f}') for f in fs]

for i, path in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    cam = extractor(input_tensor=x, targets=None)[0]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    sorted_idx = np.argsort(-cam_norm.flatten())
    rand_idx = rng.permutation(sorted_idx.size)

    for base_name, base in [('gray', base_gray), ('white', base_white)]:
        a = auc_of_curve(deletion_curve(x, sorted_idx, base, pred))
        r = auc_of_curve(deletion_curve(x, rand_idx, base, pred))
        if a is not None:
            aucs[f'{base_name}_model'].append(a)
        if r is not None:
            aucs[f'{base_name}_rand'].append(r)
    if (i + 1) % 20 == 0:
        print(f'progress {i+1}/{len(files)}', flush=True)

lines = ['=== 诊断：Deletion 底图颜色影响（每类 30 张）===']
for b in ['gray', 'white']:
    m = np.mean(aucs[f'{b}_model']); r = np.mean(aucs[f'{b}_rand'])
    verdict = '模型排序更忠实' if m < r else '模型排序反而更差'
    lines.append(f'{b}底: 按热力图 {m:.4f} vs 随机 {r:.4f}  -> {verdict}')
open('结果/诊断_底色影响.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines), flush=True)
print('done', flush=True)
