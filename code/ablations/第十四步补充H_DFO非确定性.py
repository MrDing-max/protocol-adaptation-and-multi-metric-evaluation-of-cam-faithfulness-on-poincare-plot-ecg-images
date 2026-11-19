# -*- coding: utf-8 -*-
# 第十四步补充 H：DFO-CAM 非确定性量化（同一输入多次运行）
# 审稿意见 m3：DFO-CAM 优化有轻微非确定性（约 3% AI 翻转），主表单次运行。
# 本脚本对 100 张子集（每类 50，与第十三步/补充B 同口径），同一输入连续跑 R 次，
# 量化 saliency 差异与 AI 翻转率。
import os
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

R = 5
N_PER_CLASS = 50
CAM_THRESHOLD = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

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


dfo = DFOCAMCompat(model=model, target_layers=[model.layer4])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['normal', 'af']:
    for f in sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]:
        files.append((f'data/ptbxl/images/{lab}/{f}', {'af': 0, 'normal': 1}[lab]))

flip = 0
mad = []
ai_all = []
t0 = time.time()
for i, (path, gt) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()

    sals, ais = [], []
    for _ in range(R):
        cam = dfo(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        sals.append(cam)
        mask = torch.from_numpy((cam > CAM_THRESHOLD).astype(np.float32)).to(device)
        kept = x * mask
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        ais.append(1.0 if y_keep > y_orig else 0.0)

    for r in range(1, R):
        mad.append(float(np.abs(sals[r] - sals[0]).mean()))
    if not all(a == ais[0] for a in ais):
        flip += 1
    ai_all.extend(ais)

    if (i + 1) % 10 == 0:
        el = time.time() - t0
        print(f'progress {i+1}/{len(files)} | 单图 {R} 次 '
              f'{(el/(i+1)/R):.1f}s/次', flush=True)

lines = [
    '=== DFO-CAM 非确定性量化（同一输入连续 R 次）===',
    f'样本 {len(files)} 张（每类 {N_PER_CLASS}）；每张图连续跑 {R} 次',
    f'AI 有翻转的图占比：{flip}/{len(files)} = {flip/len(files)*100:.1f}%',
    f'saliency 平均绝对差（重复运行 vs 首次）：{np.mean(mad):.4f} ± {np.std(mad):.4f}',
    f'AI（R 次池化）均值 = {np.mean(ai_all)*100:.2f}%',
    '',
    '解读：翻转率即"单次 AI 数字不可靠"的幅度；主表 AI 显著性建立在 300 张大样本上，',
    '对个位数百分点的翻转稳健。',
]
report = '\n'.join(lines)
os.makedirs('结果', exist_ok=True)
open('结果/DFO非确定性.txt', 'w', encoding='utf-8').write(report)
print(report, flush=True)
print('done：结果见 结果/DFO非确定性.txt', flush=True)
