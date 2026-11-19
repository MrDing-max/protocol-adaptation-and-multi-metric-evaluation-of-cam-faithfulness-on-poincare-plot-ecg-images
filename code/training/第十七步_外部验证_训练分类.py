# -*- coding: utf-8 -*-
# 第十七步（外部验证）：Chapman-Shaoxing 上重新训练 ResNet18
# 协议与第九步（PTB-XL）完全一致：80/20 随机切分、Adam 1e-4、batch 32、8 轮、
# ImageNet 预训练微调、5 种子平均。唯一区别：数据换成 Chapman-Shaoxing。
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

EPOCHS = 8
SEEDS = [0, 1, 2, 3, 4]

print('=== 第十七步 V1：Chapman-Shaoxing ResNet18 重新训练 ===', flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {"GPU " + torch.cuda.get_device_name(0) if device == "cuda" else "CPU"}',
      flush=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder('data/chapman/images', transform=transform)
print('类别:', ds.classes, '  总样本:', len(ds), flush=True)


def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    val_size = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                c += (pred == y).sum().item()
                t += y.size(0)
        acc = c / t * 100
        if seed == 0:
            print(f'  [seed 0] 第 {epoch + 1}/{EPOCHS} 轮 验证准确率 {acc:.1f}%',
                  flush=True)
    if seed == 0:
        torch.save(model.state_dict(), '模型/resnet18_chapman.pth')
        print('  [seed 0] 模型已存 模型/resnet18_chapman.pth', flush=True)
    return acc


t_start = time.time()
results = []
for s in SEEDS:
    t0 = time.time()
    a = run_once(s)
    results.append(a)
    print(f'seed {s} 完成: {a:.1f}%（本轮 {time.time() - t0:.0f}s）', flush=True)

results = np.array(results)
msg = '\n'.join([
    '=== 第十七步：Chapman-Shaoxing 外部验证（ResNet18 重新训练）===',
    '协议：与第九步 PTB-XL 完全一致（80/20、Adam 1e-4、batch 32、8 轮、'
    'ImageNet 预训练、5 种子平均）',
    '',
    f'Chapman 5 次验证准确率: {[f"{x:.1f}" for x in results]}',
    f'Chapman 平均: {results.mean():.1f}% ± {results.std():.1f}%',
    f'PTB-XL 对照（第九步）: 92.6% ± 1.1%',
    '',
    f'差值: {results.mean() - 92.6:+.1f} 个百分点',
    '解读：若 Chapman 准确率也到 85%+，则"庞加莱图 + 房颤分类"这条路在',
    '独立数据集上同样成立，堵住"单数据集"审稿意见；若明显偏低，如实报告',
    '并分析（设备/导联/采样率差异等）。',
])
print(msg, flush=True)
with open('结果/第十七步_Chapman分类结果.txt', 'w', encoding='utf-8') as f:
    f.write(msg)
print(f'done（总用时 {(time.time() - t_start) / 60:.0f} 分钟）', flush=True)
