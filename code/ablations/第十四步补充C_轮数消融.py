# -*- coding: utf-8 -*-
# 第十四步补充 C：优化轮数消融（审稿意见 M-3）
# 问题：DFO-CAM 默认 50 轮优化，审稿人会问"50 轮是不是随手选的？是否已收敛？"
# 本脚本对 epochs ∈ {10, 20, 50, 100} 分别算 AI 指标，验证 50 轮已收敛（不敏感）。
# 实现：自定义 DFOCAM_Epochs 类，准确复制DFO-CAM dfo-cam.py 的 get_cam_weights 优化
#   循环，仅把硬编码的 epochs 参数化；其余（shapleycam 初始化、AdamW lr=0.01、
#   smooth_l1 双流损失、取最优）与DFO-CAM实现逐行一致。
import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

N_PER_CLASS = 50       # 100 张子集
EPOCHS_LIST = [10, 20, 50, 100]
TAU = 0.5

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAM_Epochs(dfo_mod.DFOCAM):
    """参数化优化轮数的 DFO-CAM。get_cam_weights 与DFO-CAM逐行一致，仅 epochs 可配。"""
    def __init__(self, model, target_layers, opt_epochs=50, **kwargs):
        super().__init__(model=model, target_layers=target_layers, **kwargs)
        self.opt_epochs = opt_epochs

    def get_cam_weights(self, input_tensor, target_layer, targets,
                        activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)

        device_t = input_tensor.device
        grads_t = torch.from_numpy(grads).to(device_t)
        activations_t = torch.from_numpy(activations).to(device_t)

        init_weights = dfo_mod.init_weights_shapleycam(activations_t, grads_t)
        weights = init_weights.clone().detach().requires_grad_(True)

        epochs = self.opt_epochs
        lr = 0.01
        optimizer = torch.optim.AdamW([weights], lr=lr)
        best_loss = float("inf")
        best_weights = weights.detach().clone()

        for _ in range(epochs):
            optimizer.zero_grad()
            saliency_map = torch.einsum('bc,bchw->bhw', weights, activations_t)
            saliency_map = F.relu(saliency_map)
            eps = 1e-10
            saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min() + eps)
            saliency_map = saliency_map.unsqueeze(1)
            saliency_map_upsampled = F.interpolate(
                saliency_map, size=input_tensor.shape[2:],
                mode='bilinear', align_corners=False)
            saliency_map_upsampled = saliency_map_upsampled.expand(-1, 3, -1, -1)

            pos_input = input_tensor * saliency_map_upsampled
            neg_input = input_tensor * (1 - saliency_map_upsampled)
            pos_logits = self.model(pos_input)
            neg_logits = self.model(neg_input)
            target_idx = torch.tensor([t.category for t in targets],
                                      device=pos_logits.device)
            pos_probs = F.softmax(pos_logits, dim=1)
            neg_probs = F.softmax(neg_logits, dim=1)
            with torch.no_grad():
                original_logits = self.model(input_tensor)
                original_probs = F.softmax(original_logits, dim=1)
                original_score = original_probs[range(len(target_idx)), target_idx]
            pos_score = pos_probs[range(len(target_idx)), target_idx]
            neg_score = neg_probs[range(len(target_idx)), target_idx]
            loss_forward = F.smooth_l1_loss(
                pos_score, torch.ones_like(original_score))
            loss_backward = F.smooth_l1_loss(
                neg_score, torch.zeros_like(neg_score))
            loss = loss_forward + loss_backward
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = weights.detach().clone()

        return np.asarray(best_weights.detach().cpu().numpy(), dtype=np.float32)


EXTRACTORS = {e: DFOCAM_Epochs(model=model, target_layers=[model.layer4],
                               opt_epochs=e) for e in EPOCHS_LIST}

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

rows = []
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits0 = model(x)
        pred = logits0.argmax(1).item()
        y_orig = torch.softmax(logits0, dim=1)[0, gt].item()

    row = {'gt': gt}
    for e in EPOCHS_LIST:
        cam = EXTRACTORS[e](input_tensor=x,
                            targets=[ClassifierOutputTarget(pred)])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        kept = x * torch.from_numpy(
            (cam > TAU).astype(np.float32)).to(device)
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        row[f'ai_{e}'] = 1.0 if y_keep > y_orig else 0.0
    rows.append(row)
    if (i + 1) % 10 == 0:
        print(f'progress {i + 1}/{len(files)}', flush=True)

lines = ['=== 优化轮数消融：AI 指标 vs 轮数（100 张子集，τ=0.5）===',
         f'轮数 {EPOCHS_LIST}；样本 {len(rows)} 张']
for e in EPOCHS_LIST:
    ai = np.array([r[f'ai_{e}'] for r in rows])
    lines.append(f'  epochs={e:<3}  AI = {ai.mean() * 100:.1f}%')

# 相邻轮数两两 McNemar，看 AI 是否稳定
ai50 = np.array([r['ai_50'] for r in rows])
for e in [10, 20, 100]:
    a = np.array([r[f'ai_{e}'] for r in rows])
    b01 = int(np.sum((a == 1) & (ai50 == 0)))
    c01 = int(np.sum((a == 0) & (ai50 == 1)))
    n01 = b01 + c01
    p = stats.binomtest(min(b01, c01), n01, 0.5).pvalue if n01 else 1.0
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    lines.append(f'  epochs={e} vs 50: McNemar p={p:.4f} {star} '
                 f'（{e}轮对/50轮错 {b01}，反 {c01}）')
lines.append('')
lines.append('解读：若 20/50/100 轮的 AI 差异不显著（p>0.05 或效应量小），'
             '说明 50 轮已收敛，选 50 轮不是随手选的；'
             '若 10 轮显著更差，则说明轮数太少未收敛。')

report = '\n'.join(lines)
print('\n' + report, flush=True)
open('结果/轮数消融结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/轮数消融结果.txt', flush=True)
