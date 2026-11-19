# -*- coding: utf-8 -*-
# B-5 效率对比：实测五种解释方法的单张耗时（审稿意见 M-5）
# 背景：论文只写了"消融总耗时 20 分钟"，缺各方法单张耗时对比。
#   审稿人会问"DFO-CAM 慢 1000 倍"这个 trade-off，需一张效率对比表。
# 方法：对同一张图，分别计时 Grad-CAM / Grad-CAM++ / Score-CAM / DFO-CAM / RISE，
#   各计时若干次取平均，输出绝对耗时 + 相对 Grad-CAM 的倍率。
# 注意：本脚本在当前设备（CPU/GPU 自动跟随）上计时；绝对秒数依赖设备，
#   相对倍率由各方法的计算机制决定（反向传播次数 vs 前向传播次数），跨设备稳定。
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计时设备: {device}', flush=True)

# RISE 参数与实验一致
N_MASKS, GRID, MASK_P, MASK_BATCH = 4000, 8, 0.5, 128
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

# DFO-CAM（与第十四步一致）
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
        return np.asarray(super().get_cam_weights(
            input_tensor, target_layer, targets, activations, grads),
            dtype=np.float32)


EXTRACTORS = {
    'grad': GradCAMElementWise(model=model, target_layers=[model.layer4]),
    'gradpp': GradCAMPlusPlus(model=model, target_layers=[model.layer4]),
    'score': ScoreCAM(model=model, target_layers=[model.layer4]),
    'dfo': DFOCAMCompat(model=model, target_layers=[model.layer4]),
}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def rise_saliency(x, target):
    rng = np.random.RandomState(0)
    masks = (rng.random((N_MASKS, 1, GRID, GRID)) < MASK_P).astype(np.float32)
    masks = torch.from_numpy(masks)
    masks_hi = F.interpolate(masks, size=(224, 224), mode='bilinear',
                             align_corners=False)
    scores = []
    with torch.no_grad():
        for i in range(0, N_MASKS, MASK_BATCH):
            m = masks_hi[i:i + MASK_BATCH].to(device)
            xm = x * m + BASE_WHITE * (1 - m)
            scores.append(torch.softmax(model(xm), dim=1)[:, target].cpu())
    scores = torch.cat(scores)
    return (masks_hi * scores.view(N_MASKS, 1, 1, 1)).sum(0) / N_MASKS


# 取一张图
f = sorted(os.listdir('data/ptbxl/images/af'))[0]
img = Image.open(f'data/ptbxl/images/af/{f}').convert('RGB').resize((224, 224))
x = to_tensor(img).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(x).argmax(1).item()
print(f'目标类 pred = {pred}，图像 {f}', flush=True)

# 预热（首次调用含模型 hook 建立等开销，不计入）
EXTRACTORS['grad'](input_tensor=x, targets=[ClassifierOutputTarget(pred)])

results = {}
# 各方法计时 3 次取平均（DFO/RISE 较慢，各 1~2 次）
for name, n_rep in [('grad', 3), ('gradpp', 3), ('score', 2), ('dfo', 2)]:
    ts = []
    for _ in range(n_rep):
        t0 = time.time()
        EXTRACTORS[name](input_tensor=x, targets=[ClassifierOutputTarget(pred)])
        ts.append(time.time() - t0)
    results[name] = float(np.mean(ts))
    print(f'{name:<8} 单张 {results[name]:.3f}s（{n_rep} 次平均）', flush=True)

# RISE
ts = []
for _ in range(2):
    t0 = time.time()
    rise_saliency(x, pred)
    ts.append(time.time() - t0)
results['rise'] = float(np.mean(ts))
print(f'rise    单张 {results["rise"]:.3f}s（2 次平均）', flush=True)

base = results['grad']
print('\n=== 效率对比（相对 Grad-CAM 倍率）===')
for name in ['grad', 'gradpp', 'score', 'dfo', 'rise']:
    print(f'{name:<8} {results[name]:.3f}s  ->  {results[name] / base:.1f}×')

report = '\n'.join([
    f'=== 五方法单张解释耗时（设备：{device}）===',
    f'Grad-CAM    {results["grad"]:.3f}s  1.0×',
    f'Grad-CAM++  {results["gradpp"]:.3f}s  {results["gradpp"] / base:.1f}×',
    f'Score-CAM   {results["score"]:.3f}s  {results["score"] / base:.1f}×',
    f'DFO-CAM     {results["dfo"]:.3f}s  {results["dfo"] / base:.1f}×',
    f'RISE        {results["rise"]:.3f}s  {results["rise"] / base:.1f}×',
    '',
    '说明：绝对秒数依赖设备（GPU 上快 1~2 个数量级），相对倍率由机制决定、跨设备稳定。',
])
open('结果/效率对比计时结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/效率对比计时结果.txt')
