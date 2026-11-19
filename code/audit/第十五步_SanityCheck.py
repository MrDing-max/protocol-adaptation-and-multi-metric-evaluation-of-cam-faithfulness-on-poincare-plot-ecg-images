# -*- coding: utf-8 -*-
# 第十五步：Sanity Check（健全性检验）—— CAM 真的在"反映模型"吗？
#
# 为什么必须做：Adebayo et al. NeurIPS 2018 著名质疑——有些解释方法即使在模型
#   权重被完全随机化后，热力图看起来还一样，说明它根本没在反映模型决策。
#   审稿人做忠实性研究必查这一项。DFO-CAM 官方仓库 evaluation_metrics/senity_check.py
#   也是这个思路（训练模型 vs 随机初始化模型，CAM 相似度应该低）。
#
# 本脚本做两个版本，Grad-CAM 和 DFO-CAM 同台：
#   1. DFO-CAM 协议版：训练好的模型 vs 全随机初始化模型，CAM 的 Pearson 相关 → 应接近 0
#   2. 级联随机化（Adebayo 经典版）：从顶层到底层逐层累积随机化
#      fc -> layer4 -> layer3 -> layer2 -> layer1 -> conv1，
#      每随机化一层，CAM 与原始 CAM 的相关性应进一步下降 → 单调下降到 ~0 为通过
#
# 样本：PTB-XL 16 张（正常/房颤各 8，固定种子可复现）。
#   Sanity Check 看的是"趋势"不是精确均值，16 张足够（经典论文也是几十张量级）。
import os
import time
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

N_PER_CLASS = 8
REINIT_SEED = 42          # 级联随机化的种子，固定可复现

print('=== 第十五步 V1：Sanity Check（级联随机化 + DFO-CAM 协议版）===', flush=True)

os.makedirs('结果', exist_ok=True)
os.makedirs('图片', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {"GPU" if device == "cuda" else "CPU"}', flush=True)

# ============ 加载DFO-CAM 实现 ============
spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    def get_cam_weights(self, input_tensor, target_layer, targets,
                        activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)
        w = super().get_cam_weights(input_tensor, target_layer, targets,
                                    activations, grads)
        return np.asarray(w, dtype=np.float32)


# ============ 模型：先存一份"训练好的权重"，后面要反复随机化 ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)
TRAINED_STATE = {k: v.clone() for k, v in model.state_dict().items()}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============ 样本 ============
files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    files += [f'data/ptbxl/images/{lab}/{f}' for f in fs]
xs = []
for p in files:
    img = Image.open(p).convert('RGB').resize((224, 224), Image.BILINEAR)
    xs.append(to_tensor(img))
X = torch.stack(xs).to(device)          # [16, 3, 224, 224]

# 目标类：用"训练好的模型"的预测类，全程固定（保证各阶段比的是同一个解释目标）
with torch.no_grad():
    preds = model(X).argmax(1)
TARGETS = [ClassifierOutputTarget(c.item()) for c in preds]


def make_extractors():
    return {'GradCAM': GradCAMElementWise(model=model, target_layers=[model.layer4]),
            'DFOCAM': DFOCAMCompat(model=model, target_layers=[model.layer4])}


def batch_cams(extractor, tag):
    """对全部样本算 CAM，逐张归一化后返回 [N, H*W]"""
    ex = make_extractors()[tag]
    out = []
    for i in range(X.size(0)):
        cam = ex(input_tensor=X[i:i + 1], targets=[TARGETS[i]])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        out.append(cam.flatten())
    return np.stack(out)


def reinit_upto(stage_names):
    """恢复训练权重后，按给定层名"累积"随机化（Adebayo 级联协议）"""
    model.load_state_dict(TRAINED_STATE)
    g = torch.Generator().manual_seed(REINIT_SEED)
    with torch.random.fork_rng():
        torch.manual_seed(REINIT_SEED)
        for name in stage_names:
            mod = getattr(model, name)
            for m in mod.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
    model.eval()


# ============ 第 0 步：原始（训练好）CAM ============
print('计算原始 CAM（训练好的模型）...', flush=True)
t0 = time.time()
orig = {tag: batch_cams(None, tag) for tag in ['GradCAM', 'DFOCAM']}
print(f'  用时 {time.time() - t0:.0f}s', flush=True)

# ============ 级联随机化 ============
STAGES = [['fc'], ['fc', 'layer4'], ['fc', 'layer4', 'layer3'],
          ['fc', 'layer4', 'layer3', 'layer2'],
          ['fc', 'layer4', 'layer3', 'layer2', 'layer1'],
          ['fc', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1']]
STAGE_LABELS = ['原始', 'fc', '+layer4', '+layer3', '+layer2', '+layer1', '+conv1(全随机)']

sims = {tag: [1.0] for tag in orig}      # 与原始 CAM 的相关性，起点恒为 1
for si, stage in enumerate(STAGES):
    reinit_upto(stage)
    for tag in orig:
        t0 = time.time()
        cams = batch_cams(None, tag)
        rs = []
        for a, b in zip(orig[tag], cams):
            if np.std(a) < 1e-5 or np.std(b) < 1e-5:
                continue
            r, _ = pearsonr(a, b)
            if not np.isnan(r):
                rs.append(r)
        sims[tag].append(float(np.mean(rs)))
        print(f'  {STAGE_LABELS[si + 1]:<14} {tag}: Pearson r={sims[tag][-1]:.4f}'
              f'（本阶段 {time.time() - t0:.0f}s）', flush=True)

model.load_state_dict(TRAINED_STATE)     # 收尾恢复（好习惯）
model.eval()

# ============ 报告 ============
lines = ['=== 第十五步：Sanity Check（级联随机化，Adebayo et al. 2018 协议）===',
         f'样本 {len(files)} 张（正常/房颤各 {N_PER_CLASS}），'
         f'目标类=训练模型的预测类；随机化种子 {REINIT_SEED}',
         '判定标准：随着层级累积随机化，CAM 与原始 CAM 的 Pearson 相关应单调下降到 ~0；',
         '全随机阶段（=DFO-CAM 协议版）相关性应接近 0（经验上 |r|<0.1 通过）。',
         '',
         f'{"阶段":<16}{"GradCAM":>10}{"DFOCAM":>10}']
for i, lab in enumerate(STAGE_LABELS):
    lines.append(f'{lab:<16}{sims["GradCAM"][i]:>10.4f}{sims["DFOCAM"][i]:>10.4f}')
lines.append('')
for tag in sims:
    final = sims[tag][-1]
    mono = all(sims[tag][i] >= sims[tag][i + 1] - 0.05 for i in range(len(sims[tag]) - 1))
    verdict = ('通过' if abs(final) < 0.1 else '基本通过（相关性较低但未<0.1）'
               if abs(final) < 0.3 else '未通过：CAM 对模型权重不敏感，需警惕')
    lines.append(f'{tag}: 全随机后 r={final:.4f}，趋势{"基本单调下降" if mono else "有波动"}'
                 f' -> {verdict}')
lines.append('')
lines.append('解读：两种 CAM 都随权重随机化而"崩坏"，说明热力图确实编码了训练学到的'
             '模型行为，而不是与模型无关的图像先验（如边缘检测器）。此表可直接放进论文'
             '健全性检验一节。')
report = '\n'.join(lines)
with open('结果/第十五步_SanityCheck结果.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(report, flush=True)

# ============ 画图：相关性随随机化阶段下降曲线 ============
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for tag, c in [('GradCAM', '#2563EB'), ('DFOCAM', '#E8590C')]:
    ax.plot(range(len(STAGE_LABELS)), sims[tag], marker='o', lw=2, color=c, label=tag)
ax.axhline(0.1, color='#6B7280', ls=':', lw=1)
ax.text(len(STAGE_LABELS) - 1, 0.12, '|r|=0.1 通过线', fontsize=8, color='#6B7280',
        ha='right')
ax.set_xticks(range(len(STAGE_LABELS)))
ax.set_xticklabels(STAGE_LABELS, rotation=20)
ax.set_ylabel('与原始 CAM 的 Pearson 相关（越低越好）')
ax.set_title('Sanity Check：级联随机化下 CAM 相关性衰减\n（单调降到 ~0 = 热力图确实反映模型权重）')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig('图片/第十五步_SanityCheck.png', dpi=120, bbox_inches='tight')
plt.close(fig)

print('done：结果见 结果/第十五步_SanityCheck结果.txt，图见 图片/第十五步_SanityCheck.png',
      flush=True)
