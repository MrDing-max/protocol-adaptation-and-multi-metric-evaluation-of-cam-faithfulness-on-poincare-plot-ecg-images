# -*- coding: utf-8 -*-
# 补算：PTB-XL 与 Chapman 的 ROC-AUC / 敏感度 / 特异度 / 混淆矩阵
# 背景：审稿意见 m-3，论文 4.2 节三个【待补充】。医学分类论文这三项是标配。
#
# 方法：用 seed 0 模型（模型/resnet18_ptbxl.pth、模型/resnet18_chapman.pth），
#   复现与训练脚本完全一致的 seed 0 的 80/20 split（val 集即测试集），
#   取 softmax 概率后算 ROC-AUC（Mann-Whitney 秩，正确处理并列）与
#   阈值 0.5 下的敏感度/特异度/准确率/混淆矩阵。
#
# 阳性类定义：房颤（af）为阳性。ImageFolder 按字母序，classes=['af','normal']，
#   即 af=0、normal=1；房颤概率 = softmax[:, 0]。
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from scipy.stats import rankdata

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def roc_auc(y_true, score):
    """ROC-AUC = 随机抽一个阳性样本分数高于阴性样本的概率（Mann-Whitney U）。"""
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = rankdata(np.concatenate([pos, neg]), method='average')
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def evaluate(name, data_dir, model_path):
    ds = datasets.ImageFolder(data_dir, transform=transform)
    val_size = int(len(ds) * 0.2)
    _, val_ds = random_split(
        ds, [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_ds, batch_size=32)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs.append(torch.softmax(model(x), dim=1).numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # 阳性=房颤；目录名可能是 'af'（PTB-XL）或 'afib'（Chapman）
    af_cls = next(i for i, c in enumerate(ds.classes) if c.startswith('af'))
    pos_score = probs[:, af_cls]
    y_true = (labels == af_cls).astype(int)  # 1=房颤

    auc = roc_auc(y_true, pos_score)
    pred_af = (pos_score > 0.5).astype(int)
    TP = int(((pred_af == 1) & (y_true == 1)).sum())
    TN = int(((pred_af == 0) & (y_true == 0)).sum())
    FP = int(((pred_af == 1) & (y_true == 0)).sum())
    FN = int(((pred_af == 0) & (y_true == 1)).sum())
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    acc = (TP + TN) / len(y_true)

    # 约登指数最大处的最优阈值（供参考）
    order = np.argsort(pos_score)
    ys = y_true[order]
    ss = pos_score[order]
    tp_cum = np.cumsum(ys[::-1])
    fp_cum = np.cumsum(1 - ys[::-1])
    n_pos, n_neg = ys.sum(), (1 - ys).sum()
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    best = int(np.argmax(tpr - fpr))
    best_thr = float(ss[::-1][best])

    print(f'\n=== {name} ===')
    print(f'  测试集 {len(y_true)} 张（房颤 {n_pos} / 正常 {n_neg}）')
    print(f'  ROC-AUC: {auc:.4f}')
    print(f'  阈值 0.5：敏感度 {sens:.4f} | 特异度 {spec:.4f} | 准确率 {acc:.4f}')
    print(f'  混淆矩阵（行=真实, 列=预测; 1=房颤 0=正常）: '
          f'TP={TP} FP={FP} FN={FN} TN={TN}')
    print(f'  约登指数最优阈值 ≈ {best_thr:.3f}（此时敏感度 {tpr[best]:.4f} '
          f'/ 特异度 {1 - fpr[best]:.4f}）')
    return auc, sens, spec, acc, (TP, FP, FN, TN)


lines = ['=== 补算：ROC-AUC / 敏感度 / 特异度（seed 0 模型，80/20 split）===',
         '阳性类 = 房颤（af）；敏感度=房颤被正确识别比例，特异度=正常被正确识别比例']
res = {}
for name, data_dir, model_path in [
        ('PTB-XL', 'data/ptbxl/images', '模型/resnet18_ptbxl.pth'),
        ('Chapman-Shaoxing', 'data/chapman/images', '模型/resnet18_chapman.pth')]:
    auc, sens, spec, acc, cm = evaluate(name, data_dir, model_path)
    res[name] = (auc, sens, spec, acc)
    lines.append(f'{name}: ROC-AUC={auc:.4f} 敏感度={sens:.4f} '
                 f'特异度={spec:.4f} 准确率={acc:.4f}')

lines.append('')
lines.append('写进论文 4.2 节表格时的注意：ROC-AUC/敏感度/特异度基于 seed 0 模型'
             '（PTB-XL seed0 准确率 91.6%，Chapman seed0 见第十七步），'
             '而正文分类准确率报 5 种子平均（92.6%/98.0%），需在表注说明口径。')
report = '\n'.join(lines)
print('\n' + report)
open('结果/补算ROC指标结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/补算ROC指标结果.txt')
