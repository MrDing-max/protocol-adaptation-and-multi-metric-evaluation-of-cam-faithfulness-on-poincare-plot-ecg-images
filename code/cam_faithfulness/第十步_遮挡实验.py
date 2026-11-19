# -*- coding: utf-8 -*-
# 第十步（1/2）：PTB-XL 上的遮挡实验——涂掉散点，验证模型是否靠"点"做判断
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.ndimage import binary_dilation

# 加载 PTB-XL 训练好的模型
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(img_pil):
    t = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(1).item()
    return pred, prob[pred].item()


def occlude_points(img_pil, expand=6):
    """把散点（暗像素）涂成白色背景"""
    arr = np.array(img_pil).astype(np.float32)
    dark = arr.mean(axis=2) < 128
    mask = binary_dilation(dark, iterations=expand)
    arr[mask] = 255.0
    return Image.fromarray(arr.astype(np.uint8))


DATA = 'data/ptbxl/images'
total_orig = total_occ = 0
conf_orig, conf_occ = [], []
n = 0
for label, cls_idx in [('normal', 1), ('af', 0)]:
    for f in os.listdir(f'{DATA}/{label}'):
        img = Image.open(f'{DATA}/{label}/{f}').convert('RGB')
        occ = occlude_points(img)
        pred_orig, c_orig = predict(img)
        pred_occ, c_occ = predict(occ)
        total_orig += (pred_orig == cls_idx)
        total_occ += (pred_occ == cls_idx)
        conf_orig.append(c_orig)
        conf_occ.append(c_occ)
        n += 1

acc_orig = total_orig / n * 100
acc_occ = total_occ / n * 100
print(f'总样本: {n} 张')
print(f'原图      准确率: {acc_orig:.1f}%   平均置信度 {np.mean(conf_orig)*100:.1f}%')
print(f'涂掉散点  准确率: {acc_occ:.1f}%   平均置信度 {np.mean(conf_occ)*100:.1f}%')
print(f'-> 准确率变化: {acc_occ - acc_orig:+.1f} 个百分点')

msg = (f'PTB-XL 遮挡实验：原图 {acc_orig:.1f}%，涂掉散点后 {acc_occ:.1f}%\n'
       f'平均置信度：原图 {np.mean(conf_orig)*100:.1f}% -> 涂掉后 {np.mean(conf_occ)*100:.1f}%')
open('结果/PTBXL遮挡实验结果.txt', 'w', encoding='utf-8').write(msg)
print('完成！结果已写入 结果/PTBXL遮挡实验结果.txt')
