# -*- coding: utf-8 -*-
# 第十六步：ViT 对比实验 —— 结论是否依赖骨干网络？
#
# 动机：目前所有结论都建立在 ResNet18 上，审稿人会问"换个架构还成立吗"。
#   本步用 ViT-B/16（timm 预训练）在完全相同的数据和协议上训练，
#   与第九步 ResNet18 的 92.6%±1.1% 对比。后续还将在 ViT 上做忠实性对比
#   （第十七步），证明"DFO-CAM 的 AI 优势"不挑骨干。
#
# 协议与第九步严格一致（逐项对照）：
#   ✓ 同一数据集：data/ptbxl/images（597 病人，1 病人 = 1 样本，随机切分=按病人切分）
#   ✓ 同一切分：80/20，每轮种子独立（seed 0~4）
#   ✓ 同一超参：Adam lr=1e-4、batch 32、8 轮、交叉熵、ImageNet 预训练微调
#   ✓ 同一统计：5 次训练取平均±标准差
#   唯一区别：骨干网络 ResNet18 -> ViT-B/16（timm vit_base_patch16_224）
#
# 产出：
#   - 结果/第十六步_ViT分类结果.txt（5 次数字 + 均值±std + 与 ResNet18 对照）
#   - 模型/vit_ptbxl.pth（seed 0 的模型，第十七步忠实性实验要用，别删！
#     对应关系：resnet18_ptbxl.pth 也是第九步 seed 0 单跑保存的）
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

EPOCHS = 8
SEEDS = [0, 1, 2, 3, 4]

print('=== 第十六步 V1：ViT-B/16 对比训练（协议与第九步一致）===', flush=True)

import os
os.makedirs('结果', exist_ok=True)
os.makedirs('模型', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {"GPU " + torch.cuda.get_device_name(0) if device == "cuda" else "CPU（ViT 在 CPU 上很慢，强烈建议 AutoDL）"}',
      flush=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder('data/ptbxl/images', transform=transform)
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

    model = timm.create_model('vit_base_patch16_224', pretrained=True,
                              num_classes=2)
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
        torch.save(model.state_dict(), '模型/vit_ptbxl.pth')
        print('  [seed 0] 模型已存 模型/vit_ptbxl.pth（第十七步要用）', flush=True)
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
    '=== 第十六步：ViT-B/16 vs ResNet18 骨干网络对比（PTB-XL 597 病人）===',
    '协议：与第九步完全一致（80/20 按病人切分、Adam 1e-4、batch 32、8 轮、'
    'ImageNet 预训练、5 种子平均）',
    '',
    f'ViT-B/16 5 次验证准确率: {[f"{x:.1f}" for x in results]}',
    f'ViT-B/16 平均: {results.mean():.1f}% ± {results.std():.1f}%',
    f'ResNet18 对照（第九步）: 92.6% ± 1.1%',
    '',
    f'差值: {results.mean() - 92.6:+.1f} 个百分点',
    '解读：两骨干准确率接近（±2 个点内）-> "庞加莱图包含房颤判别信息"的命题',
    '不依赖特定骨干网络，可写进论文 5.7 节；若 ViT 明显更高/更低，如实报告并',
    '在 Discussion 分析（ViT 全局注意力 vs CNN 局部纹理先验对点阵图的适配差异）。',
])
print(msg, flush=True)
with open('结果/第十六步_ViT分类结果.txt', 'w', encoding='utf-8') as f:
    f.write(msg)
print(f'done（总用时 {(time.time() - t_start) / 60:.0f} 分钟）', flush=True)
