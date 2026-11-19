# -*- coding: utf-8 -*-
# 第十四步补充 B：阈值 τ 消融（审稿意见 M-3）
# 问题：AI/AD/AG 指标用 CAM>0.5 的阈值截断。审稿人会问"换 0.3 或 0.7，
#   DFO-CAM 的 AI 优势还显著吗？"本脚本验证 AI 指标的显著性不依赖阈值选择。
# 协议：100 张子集（每类 50，与第十二步重合可对照）、白底、4 个 CAM 方法、
#   阈值 τ ∈ {0.3, 0.5, 0.7}，每个 τ 下算 AI，DFO-CAM vs 三个基线做 McNemar。
import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

N_PER_CLASS = 50       # 100 张子集（消融用小样本即可，协议与第十二步一致）
TAUS = [0.3, 0.5, 0.7]

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
PRETTY = {'grad': 'GradCAM', 'gradpp': 'GradCAM++',
          'score': 'ScoreCAM', 'dfo': 'DFOCAM'}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

# 每张图算 4 个 CAM，存 {method: sal}；AI 对每个 τ 算
rows = []
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits0 = model(x)
        pred = logits0.argmax(1).item()
        y_orig = torch.softmax(logits0, dim=1)[0, gt].item()

    cams = {}
    for m in EXTRACTORS:
        cam = EXTRACTORS[m](input_tensor=x,
                            targets=[ClassifierOutputTarget(pred)])[0]
        cams[m] = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    row = {'gt': gt}
    for tau in TAUS:
        for m in EXTRACTORS:
            kept = x * torch.from_numpy(
                (cams[m] > tau).astype(np.float32)).to(device)
            with torch.no_grad():
                y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
            row[f'ai_{tau}_{m}'] = 1.0 if y_keep > y_orig else 0.0
    rows.append(row)
    if (i + 1) % 10 == 0:
        print(f'progress {i + 1}/{len(files)}', flush=True)

# 统计：每个 τ 下 DFO-CAM vs 基线的 AI + McNemar
def mcnemar(a, b):
    b01 = int(np.sum((a == 1) & (b == 0)))
    c01 = int(np.sum((a == 0) & (b == 1)))
    n01 = b01 + c01
    if n01 == 0:
        return 1.0, 0, 0
    return stats.binomtest(min(b01, c01), n01, 0.5).pvalue, b01, c01

lines = ['=== 阈值 τ 消融：AI 指标的显著性是否依赖阈值（100 张子集）===',
         f'阈值 τ ∈ {TAUS}；样本 {len(rows)} 张；DFO-CAM vs 三个基线（McNemar）']
for tau in TAUS:
    lines.append('')
    lines.append(f'--- τ = {tau} ---')
    dfo = np.array([r[f'ai_{tau}_dfo'] for r in rows])
    for b in ['grad', 'gradpp', 'score']:
        bb = np.array([r[f'ai_{tau}_{b}'] for r in rows])
        p, b01, c01 = mcnemar(dfo, bb)
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        lines.append(f'  DFO-CAM {dfo.mean() * 100:.1f}% vs {PRETTY[b]:<10} '
                     f'{bb.mean() * 100:.1f}% | McNemar p={p:.4f} {star} '
                     f'（DFO对/{PRETTY[b]}错 {b01}，反 {c01}）')
    lines.append(f'  -> DFO-CAM 的 AI 优势在 τ={tau} 下'
                 f'{"仍显著" if all(mcnemar(dfo, np.array([r[f"ai_{tau}_{b}"] for r in rows]))[0] < 0.05 for b in ["grad", "gradpp", "score"]) else "不再显著"}')

report = '\n'.join(lines)
print('\n' + report, flush=True)
open('结果/阈值消融结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/阈值消融结果.txt', flush=True)
