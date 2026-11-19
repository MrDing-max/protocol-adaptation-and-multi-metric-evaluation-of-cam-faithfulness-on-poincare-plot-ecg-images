# -*- coding: utf-8 -*-
# 第十一步：正规忠实性评估——严格对齐DFO-CAM 官方协议
#
# 协议来源（DFO-CAM 官方仓库）：
#   third_party/evaluation_metrics/Del_Ins_AUC.py
#     -> 逐像素排序、40 步、底图=全零(均值灰)、logit 单图 min-max 归一后 AUC=trapz
#   third_party/evaluation_metrics/Ad_Ai_AG.py
#     -> AD/AI/AG：只保留 CAM>0.5 区域，算目标类概率的 平均下降/升高占比/平均增益
# 提取器：GradCAMElementWise（与DFO-CAM 官方代码一致；本环境已可用）
# 补充（本脚本自加）：随机像素顺序基线——"按热力图排序"必须优于随机才算数
#
# 参考：Petsiuk et al., RISE, BMVC 2018（Deletion/Insertion 指标出处）
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

NUM_STEPS = 40          # 与DFO-CAM 官方代码一致
CAM_THRESHOLD = 0.5     # 与DFO-CAM Ad_Ai_AG 代码一致
H = W = 224

# ============ 模型与提取器 ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()

extractor = GradCAMElementWise(model=model, target_layers=[model.layer4])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def auc_of_curve(logits_arr):
    """DFO-CAM 协议：单图 logit min-max 归一后 trapz；logit 几乎不变则返回 None（跳过）"""
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    norm = (logits_arr - lo) / denom
    return float(np.trapezoid(norm, np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_batch(x, order, mode):
    """按给定像素顺序构造全部 41 个扰动帧，拼成一个批量 (41,3,H,W)，一次前向算完"""
    base = torch.zeros_like(x)               # 全零（归一化下 = 均值灰）
    xs = []
    total = H * W
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * total)
        mask = torch.zeros(1, H, W)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        if mode == 'insertion':
            pert = base * (1 - mask) + x * mask   # 只亮出前 k 个最重要像素
        else:                                     # deletion
            pert = x * (1 - mask) + base * mask   # 把前 k 个最重要像素涂成均值灰
        xs.append(pert)
    return torch.cat(xs, dim=0)


# ============ 遍历全部图片 ============
DATA = 'data/ptbxl/images'
rng = np.random.RandomState(0)
CLS_IDX = {'af': 0, 'normal': 1}                 # ImageFolder 字母序

curves = {k: [] for k in ['del_model', 'del_rand', 'ins_model', 'ins_rand']}
ad_list, ai_list, ag_list = [], [], []
n_used = n_skip = 0
example = None                                   # (x, sorted_idx, path) 房颤示意

files = [(f'{DATA}/{lab}/{f}', CLS_IDX[lab], lab)
         for lab in ['normal', 'af']
         for f in sorted(os.listdir(f'{DATA}/{lab}'))]

for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(1).item()
    y_orig = torch.softmax(logits, dim=1)[0, gt].item()   # GT 类概率（AD/AI/AG 用）

    cam = extractor(input_tensor=x, targets=None)[0]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    sorted_idx = np.argsort(-cam_norm.flatten())
    rand_idx = rng.permutation(sorted_idx.size)

    # 四条曲线，各 41 个点；DFO-CAM 协议按"预测类 logit"评估
    cm = perturbed_batch(x, sorted_idx, 'deletion')
    cr = perturbed_batch(x, rand_idx, 'deletion')
    im = perturbed_batch(x, sorted_idx, 'insertion')
    ir = perturbed_batch(x, rand_idx, 'insertion')
    with torch.no_grad():
        lm = model(cm)[:, pred].numpy()
        lr_ = model(cr)[:, pred].numpy()
        li = model(im)[:, pred].numpy()
        lr2 = model(ir)[:, pred].numpy()

    a, r = auc_of_curve(lm), auc_of_curve(lr_)
    b, c = auc_of_curve(li), auc_of_curve(lr2)
    if None in (a, r, b, c):
        n_skip += 1
        continue
    curves['del_model'].append(lm); curves['del_rand'].append(lr_)
    curves['ins_model'].append(li); curves['ins_rand'].append(lr2)

    # ---- AD / AI / AG（DFO-CAM 协议：GT 类、保留 CAM>0.5 区域、其余置零）----
    binm = torch.from_numpy((cam_norm > CAM_THRESHOLD).astype(np.float32))
    kept = x * binm
    with torch.no_grad():
        y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
    ad_list.append(max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100)
    ai_list.append(1.0 if y_keep > y_orig else 0.0)
    ag_list.append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)

    if example is None and lab == 'af':
        example = (x, sorted_idx)

    if (i + 1) % 50 == 0:
        print(f'progress: {i+1}/{len(files)}')

n_used = len(curves['del_model'])
print(f'used={n_used} skipped={n_skip}')

# ============ 汇总 ============
S = {k: (float(np.mean(v)), float(np.std(v))) for k, v in
     {'del_model': [auc_of_curve(c) for c in curves['del_model']],
      'del_rand': [auc_of_curve(c) for c in curves['del_rand']],
      'ins_model': [auc_of_curve(c) for c in curves['ins_model']],
      'ins_rand': [auc_of_curve(c) for c in curves['ins_rand']]}.items()}
AD = float(np.mean(ad_list))
AI = float(np.mean(ai_list)) * 100
AG = float(np.mean(ag_list))
mean_curves = {k: np.mean(np.stack(v), axis=0) for k, v in curves.items()}

lines = []
lines.append('=== PTB-XL 忠实性指标（严格对齐DFO-CAM 官方协议）===')
lines.append(f'有效样本 {n_used} 张（跳过 {n_skip} 张：logit 无变化）')
lines.append('提取器 GradCAMElementWise，逐像素 40 步，logit 单图 min-max 归一后 AUC')
lines.append('')
lines.append('--- Deletion / Insertion AUC（平均 ± 标准差）---')
lines.append(f'Deletion  按热力图: {S["del_model"][0]:.4f} ± {S["del_model"][1]:.4f}   （越小越忠实）')
lines.append(f'Deletion  随机顺序: {S["del_rand"][0]:.4f} ± {S["del_rand"][1]:.4f}')
lines.append(f'Insertion 按热力图: {S["ins_model"][0]:.4f} ± {S["ins_model"][1]:.4f}   （越大越忠实）')
lines.append(f'Insertion 随机顺序: {S["ins_rand"][0]:.4f} ± {S["ins_rand"][1]:.4f}')
lines.append('')
lines.append('--- AD / AI / AG（DFO-CAM 协议：保留 CAM>0.5 区域，GT 类）---')
lines.append(f'AD  (Average Drop)    = {AD:.2f}%   越小越好')
lines.append(f'AI  (Average Increase)= {AI:.2f}%   越大越好')
lines.append(f'AG  (Average Gain)    = {AG:.2f}%   越大越好')
lines.append('')
ok_del = S['del_model'][0] < S['del_rand'][0]
ok_ins = S['ins_model'][0] > S['ins_rand'][0]
lines.append(f'结论：Deletion {"优于随机 -> 热力图忠实" if ok_del else "未优于随机 -> 需警惕"}；'
            f'Insertion {"优于随机 -> 热力图忠实" if ok_ins else "未优于随机 -> 需警惕"}')
open('结果/PTBXL忠实性指标结果.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('results saved')

# ============ 画图：平均曲线面板 ============
C_MODEL, C_RAND = '#2563EB', '#6B7280'
grid = np.linspace(0, 1, NUM_STEPS + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
ax.plot(grid, mean_curves['del_model'], color=C_MODEL, lw=2, label='按热力图排序')
ax.plot(grid, mean_curves['del_rand'], color=C_RAND, lw=2, ls='--', label='随机顺序')
ax.set_xlabel('已删除像素的比例')
ax.set_ylabel('目标类 logit（单图归一化）')
ax.set_title(f'Deletion 删除实验（AUC 越小越忠实）\n'
             f'模型 {S["del_model"][0]:.3f} vs 随机 {S["del_rand"][0]:.3f}')
ax.set_xlim(0, 1)
ax.legend(loc='upper right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
ax.plot(grid, mean_curves['ins_model'], color=C_MODEL, lw=2, label='按热力图排序')
ax.plot(grid, mean_curves['ins_rand'], color=C_RAND, lw=2, ls='--', label='随机顺序')
ax.set_xlabel('已放回像素的比例')
ax.set_ylabel('目标类 logit（单图归一化）')
ax.set_title(f'Insertion 插入实验（AUC 越大越忠实）\n'
             f'模型 {S["ins_model"][0]:.3f} vs 随机 {S["ins_rand"][0]:.3f}')
ax.set_xlim(0, 1)
ax.legend(loc='lower right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)

plt.tight_layout()
plt.savefig('图片/第十一步_忠实性曲线.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ============ 示意图：一张房颤图的删除过程（均值灰 = 被删掉的部分）============
if example is not None:
    x, sorted_idx = example
    fig2, axes2 = plt.subplots(1, 4, figsize=(10.4, 2.9))
    steps = [0, 13, 27, 40]
    titles = ['原图', '删 1/3 像素', '删 2/3 像素', '全部删除']
    for j, (st, tt) in enumerate(zip(steps, titles)):
        batch = perturbed_batch(x, sorted_idx, 'deletion')
        arr = batch[st].squeeze(0).numpy().transpose(1, 2, 0) * STD + MEAN
        axes2[j].imshow(np.clip(arr, 0, 1))
        axes2[j].set_title(tt, fontsize=10)
        axes2[j].axis('off')
    plt.tight_layout()
    plt.savefig('图片/第十一步_删除过程示例.png', dpi=120, bbox_inches='tight')
    plt.close(fig2)

print('done')
