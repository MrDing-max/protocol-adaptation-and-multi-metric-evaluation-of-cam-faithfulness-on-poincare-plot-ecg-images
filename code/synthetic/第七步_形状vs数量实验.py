# -*- coding: utf-8 -*-
# 第七步：形状 vs 数量 实验（合成数据版）
# 核心：让"正常"和"房颤"的点数量完全相同，只保留分布形状差异，看模型还能不能分对
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ============ 1. 生成合成数据 ============
N_RR = 50          # 每张图固定 50 个 RR 间期（49 个点）——两类完全一致
N_IMG = 120        # 每类 120 张
OUT = 'data/synth'
for label in ['normal', 'af']:
    os.makedirs(f'{OUT}/{label}', exist_ok=True)

rng = np.random.RandomState(42)


def gen_rr(kind):
    if kind == 'normal':
        # 正常：心跳间隔很规律，集中在 860ms 左右
        return np.clip(rng.normal(860, 30, N_RR), 700, 1050)
    else:
        # 房颤：心跳间隔高度不规则，在 300~1500ms 间乱跳
        return rng.uniform(300, 1500, N_RR)


for label, kind in [('normal', 'normal'), ('af', 'af')]:
    for i in range(N_IMG):
        rr = gen_rr(kind)
        fig = plt.figure(figsize=(3, 3), dpi=100)
        plt.scatter(rr[:-1], rr[1:], s=14, alpha=0.75, color='k')
        plt.xlim(0, 2000)
        plt.ylim(0, 2000)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.2)
        fig.savefig(f'{OUT}/{label}/{label}_{i:04d}.png')
        plt.close(fig)
print('合成数据生成完成：正常 120 张 + 房颤 120 张，每张都是 49 个点')

# ============ 2. 训练 ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder(OUT, transform=transform)
val_size = int(len(ds) * 0.2)
train_ds, val_ds = random_split(
    ds, [len(ds) - val_size, val_size],
    generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

acc = 0
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f'  第 {epoch+1}/5 轮  验证准确率 {acc:.1f}%')

result = f'固定点数后验证准确率: {acc:.1f}%'
open('结果/形状数量实验结果.txt', 'w', encoding='utf-8').write(result)
print('完成！' + result)
print('解读：若准确率高 -> 点数量相同仅靠形状也能区分，说明"分布形状"是有效特征。')
