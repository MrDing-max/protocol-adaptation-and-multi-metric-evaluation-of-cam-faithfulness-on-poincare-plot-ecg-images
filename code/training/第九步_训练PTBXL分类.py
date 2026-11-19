# -*- coding: utf-8 -*-
# 第九步（4/4）：在 PTB-XL 庞加莱图上训练 ResNet 二分类
# 关键：选数据时已保证"1 病人 = 1 样本"，所以随机切分即按病人切分，无数据泄漏
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
print('类别:', ds.classes, '  总样本:', len(ds))

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

EPOCHS = 8
acc = 0
print(f'开始训练（{EPOCHS} 轮，CPU）...')
for epoch in range(EPOCHS):
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
    print(f'  第 {epoch+1}/{EPOCHS} 轮  验证准确率 {acc:.1f}%')

torch.save(model.state_dict(), '模型/resnet18_ptbxl.pth')
result = f'PTB-XL 验证准确率: {acc:.1f}%（{EPOCHS} 轮，597 张图）'
open('结果/PTBXL分类结果.txt', 'w', encoding='utf-8').write(result)
print('完成！' + result)
