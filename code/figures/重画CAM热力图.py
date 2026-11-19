# -*- coding: utf-8 -*-
"""本地重画 Fig2 CAM 热力图：一张房颤图 + 4 种 CAM 方法叠加（英文标签）。"""
import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

# DFO-CAM（DFO-CAM 官方代码）
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
        return np.asarray(super().get_cam_weights(
            input_tensor, target_layer, targets, activations, grads), dtype=np.float32)


extractors = {
    'Grad-CAM': GradCAMElementWise(model=model, target_layers=[model.layer4]),
    'Grad-CAM++': GradCAMPlusPlus(model=model, target_layers=[model.layer4]),
    'Score-CAM': ScoreCAM(model=model, target_layers=[model.layer4]),
    'DFO-CAM': DFOCAMCompat(model=model, target_layers=[model.layer4]),
}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 取一张房颤图
af_f = sorted(os.listdir('data/ptbxl/images/af'))[0]
img = Image.open(f'data/ptbxl/images/af/{af_f}').convert('RGB').resize((224, 224), Image.BILINEAR)
x = to_tensor(img).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(x).argmax(1).item()
print(f'房颤图 pred={pred}', flush=True)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Poincaré plot\n(AF)', fontsize=10)
for ax, (name, ext) in zip(axes[1:], extractors.items()):
    print(f'生成 {name} CAM ...', flush=True)
    cam = ext(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    ax.imshow(img, cmap='gray')
    ax.imshow(cam, cmap='jet', alpha=0.45)
    ax.set_title(name, fontsize=10)
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('图片/Fig2_cam_heatmaps.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('Fig2 CAM 热力图已生成', flush=True)
