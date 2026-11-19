# -*- coding: utf-8 -*-
# 第九步（补）：PTB-XL 训练 5 次（不同随机种子）取平均，排除"训练随机性"
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
ds = datasets.ImageFolder('data/ptbxl/images', transform=transform)


def run_once(seed):
    val_size = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc = 0
    for _ in range(8):
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


print('PTB-XL 多次训练取平均（5 次 × 8 轮，CPU，约 15-20 分钟）...')
results = [run_once(i) for i in range(5)]
results = np.array(results)
msg = (f'5 次验证准确率: {[f"{x:.1f}" for x in results]}\n'
       f'平均 {results.mean():.1f}% ± {results.std():.1f}%')
open('结果/PTBXL多次平均.txt', 'w', encoding='utf-8').write(msg)
print(msg)
print('完成！结果已写入 结果/PTBXL多次平均.txt')
