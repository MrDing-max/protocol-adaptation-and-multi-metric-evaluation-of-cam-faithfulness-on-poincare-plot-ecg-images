# -*- coding: utf-8 -*-
# 第七步（补）：三组对照——拆解"形状"和"数量"各自的区分能力
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def gen_rr(rng, kind):
    if kind == 'compact':       # 紧凑（像正常心律）
        return np.clip(rng.normal(860, 30, N), 700, 1050)
    else:                        # 散开（像房颤）
        return rng.uniform(300, 1500, N)


def run_one(name, normal_kind, af_kind, n_normal, n_af):
    global N
    N = max(n_normal, n_af)
    OUT = f'data/ctrl_{name}'
    for label in ['normal', 'af']:
        os.makedirs(f'{OUT}/{label}', exist_ok=True)
    rng = np.random.RandomState(42)

    def make(kind, n, label):
        for i in range(120):
            rr = gen_rr(rng, kind)[:n]
            fig = plt.figure(figsize=(3, 3), dpi=100)
            plt.scatter(rr[:-1], rr[1:], s=14, alpha=0.75, color='k')
            plt.xlim(0, 2000)
            plt.ylim(0, 2000)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout(pad=0.2)
            fig.savefig(f'{OUT}/{label}/{label}_{i:04d}.png')
            plt.close(fig)

    make(normal_kind, n_normal, 'normal')
    make(af_kind, n_af, 'af')

    ds = datasets.ImageFolder(OUT, transform=transform)
    val_size = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    acc = 0
    for _ in range(5):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                c += (pred == y).sum().item()
                t += y.size(0)
        acc = c / t * 100
    return acc


print('=== 三组对照实验 ===')
a = run_one('shape_only', 'compact', 'scatter', 50, 50)
print(f'[组A] 只有形状差异（点数都50）: {a:.1f}%')

b = run_one('count_only', 'compact', 'compact', 20, 60)
print(f'[组B] 只有数量差异（形状都紧凑，点数20 vs 60）: {b:.1f}%')

c = run_one('both', 'compact', 'scatter', 30, 60)
print(f'[组C] 形状+数量都有差异: {c:.1f}%')

msg = f'组A(只有形状) {a:.1f}% / 组B(只有数量) {b:.1f}% / 组C(两者都有) {c:.1f}%'
open('结果/三组对照结果.txt', 'w', encoding='utf-8').write(msg)
print('\n解读：组A高 -> 形状够用；组B高 -> 数量也能被利用；组C最高 -> 两者叠加。')
