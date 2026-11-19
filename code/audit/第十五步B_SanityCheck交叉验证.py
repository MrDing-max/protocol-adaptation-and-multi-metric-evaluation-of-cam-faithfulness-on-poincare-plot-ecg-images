# -*- coding: utf-8 -*-
# 第十五步B：Sanity Check 交叉验证 —— 解释"全随机后相关性反弹"之谜
#
# 背景：第十五步发现，级联随机化走到"全部打乱"时，CAM 与原始 CAM 的
#   Pearson 相关不降反升（0.44~0.47）。两种可能：
#     假说A（坏消息）：CAM 不依赖模型权重，跟谁都能对上 -> 方法失效
#     假说B（好消息）：白底稀疏点阵图上，任何梯度类方法的 CAM 都"自动对准
#                      散点"（白底梯度≈0，只有黑点处有信号），所以两张 CAM
#                      只是都长得像"散点图"，但随机模型的 CAM 并不携带
#                      训练模型的决策信息 -> 方法没问题，是图像先验撞车
#   本脚本用三层证据区分 A/B：
#     1. SSIM 结构相似度（Adebayo 2018 原论文指标）：看"结构"上像不像
#     2. Spearman 等级相关：看"像素重要性排序"像不像
#     3. 功能性检验（最硬）：把"全随机模型的 CAM"拿去给【训练好的模型】做
#        Deletion——若它携带训练模型决策信息，AUC 应接近 0.36（训练 CAM 水平）；
#        若只是散点假象，应接近随机基线 ~0.49。用效果说话，不看长相。
#
# 样本与随机化种子与第十五步完全一致（16 张，seed=42），可直接对照。
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
from scipy.stats import pearsonr, spearmanr, ttest_rel
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

N_PER_CLASS = 8
REINIT_SEED = 42
NUM_STEPS = 40
H = W = 224

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

print('=== 第十五步B V1：全随机反弹交叉验证（SSIM+Spearman+功能性检验）===', flush=True)

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


# ============ 模型与样本（与第十五步完全一致）============
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
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    files += [f'data/ptbxl/images/{lab}/{f}' for f in fs]
xs = [to_tensor(Image.open(p).convert('RGB').resize((224, 224), Image.BILINEAR))
      for p in files]
X = torch.stack(xs).to(device)

with torch.no_grad():
    preds = model(X).argmax(1)
TARGETS = [ClassifierOutputTarget(c.item()) for c in preds]


def batch_cams(tag):
    ex = {'GradCAM': GradCAMElementWise(model=model, target_layers=[model.layer4]),
          'DFOCAM': DFOCAMCompat(model=model, target_layers=[model.layer4])}[tag]
    out = []
    for i in range(X.size(0)):
        cam = ex(input_tensor=X[i:i + 1], targets=[TARGETS[i]])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        out.append(cam)
    return out


def ssim(a, b):
    """标准 SSIM（7x7 均值窗，数据范围 L=1），不依赖 skimage。"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = uniform_filter(a, 7), uniform_filter(b, 7)
    va = uniform_filter(a * a, 7) - mu_a ** 2
    vb = uniform_filter(b * b, 7) - mu_b ** 2
    cov = uniform_filter(a * b, 7) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)
         / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))
    return float(s.mean())


def auc_of_curve(logits_arr):
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits_arr - lo) / denom,
                        np.linspace(0, 1, NUM_STEPS + 1)))


def deletion_auc_on_trained(x, cam, pred):
    """功能性检验：用给定 CAM 的像素顺序，在【训练好的模型】上做白底 Deletion。"""
    order = np.argsort(-cam.flatten())
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs_ = []
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * H * W)
        mask = torch.zeros(1, H, W, device=device)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        xs_.append(x * (1 - mask) + base * mask)
    with torch.no_grad():
        logits = model(torch.cat(xs_, dim=0))[:, pred].cpu().numpy()
    return auc_of_curve(logits)


# ============ 第 1 步：原始 CAM（训练好的模型）============
print('计算原始 CAM（训练好的模型）...', flush=True)
t0 = time.time()
orig = {tag: batch_cams(tag) for tag in ['GradCAM', 'DFOCAM']}
print(f'  用时 {time.time() - t0:.0f}s', flush=True)

# ============ 第 2 步：全随机模型的 CAM（与十五步"全随机"阶段同种子）============
print('全随机化模型，计算随机 CAM ...', flush=True)
t0 = time.time()
with torch.random.fork_rng():
    torch.manual_seed(REINIT_SEED)
    for name in ['fc', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1']:
        for m in getattr(model, name).modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
model.eval()
rand_cams = {tag: batch_cams(tag) for tag in ['GradCAM', 'DFOCAM']}
print(f'  用时 {time.time() - t0:.0f}s', flush=True)

# 恢复训练权重，后面功能性检验要用"训练好的模型"
model.load_state_dict(TRAINED_STATE)
model.eval()

# ============ 第 3 步：三种相似度 ============
print('计算 Pearson / Spearman / SSIM ...', flush=True)
sims = {tag: {'pearson': [], 'spearman': [], 'ssim': []} for tag in orig}
for tag in orig:
    for a, b in zip(orig[tag], rand_cams[tag]):
        fa, fb = a.flatten(), b.flatten()
        if np.std(fa) > 1e-5 and np.std(fb) > 1e-5:
            sims[tag]['pearson'].append(pearsonr(fa, fb)[0])
            sims[tag]['spearman'].append(spearmanr(fa, fb)[0])
            sims[tag]['ssim'].append(ssim(a, b))

# ============ 第 4 步：功能性检验（全随机 CAM 给训练模型做 Deletion）============
print('功能性检验：随机模型的 CAM 在训练模型上做 Deletion ...', flush=True)
func = {tag: {'trained_cam': [], 'random_cam': [], 'random_order': []}
        for tag in orig}
for i in range(X.size(0)):
    x1 = X[i:i + 1]
    p = preds[i].item()
    rng = np.random.RandomState(5000 + i)
    fake_cam = rng.rand(H, W)                      # 纯随机顺序基线
    for tag in orig:
        a1 = deletion_auc_on_trained(x1, orig[tag][i], p)
        a2 = deletion_auc_on_trained(x1, rand_cams[tag][i], p)
        a3 = deletion_auc_on_trained(x1, fake_cam, p)
        if None not in (a1, a2, a3):
            func[tag]['trained_cam'].append(a1)
            func[tag]['random_cam'].append(a2)
            func[tag]['random_order'].append(a3)
    if (i + 1) % 4 == 0:
        print(f'  {i + 1}/{X.size(0)}', flush=True)

# ============ 报告 ============
lines = ['=== 第十五步B：Sanity Check 全随机反弹 交叉验证 ===',
         f'样本 {len(files)} 张，随机化种子 {REINIT_SEED}（与第十五步一致）',
         '目的：区分"CAM 不依赖模型权重（方法失效）"还是"白底点阵图的散点先验撞车"',
         '',
         '--- 第一层 & 第二层：三种相似度（原始 CAM vs 全随机模型 CAM）---',
         f'{"指标":<12}{"GradCAM":>10}{"DFOCAM":>10}']
for met in ['pearson', 'spearman', 'ssim']:
    g = float(np.mean(sims['GradCAM'][met]))
    d = float(np.mean(sims['DFOCAM'][met]))
    lines.append(f'{met:<12}{g:>10.4f}{d:>10.4f}')
lines.append('（Spearman/SSIM 若明显低于 Pearson，说明"排序和结构"其实并不像，')
lines.append(' Pearson 反弹主要是像素值分布的假象）')
lines.append('')
lines.append('--- 第三层：功能性检验（Deletion AUC，在训练好的模型上，越小越好）---')
verdicts = {}
for tag in orig:
    tc = np.array(func[tag]['trained_cam'])
    rc = np.array(func[tag]['random_cam'])
    ro = np.array(func[tag]['random_order'])
    t1, p1 = ttest_rel(rc, ro)     # 随机CAM vs 随机顺序：是否一样差
    t2, p2 = ttest_rel(rc, tc)     # 随机CAM vs 训练CAM：是否明显更差
    lines.append(f'{tag}: 训练CAM {tc.mean():.4f} | 随机模型CAM {rc.mean():.4f} | '
                 f'纯随机顺序 {ro.mean():.4f}')
    lines.append(f'   随机CAM vs 纯随机顺序: p={p1:.4f}；'
                 f'随机CAM vs 训练CAM: p={p2:.4f}')
    # 判定：随机CAM 显著差于训练CAM，且与纯随机顺序无显著差异 -> 功能崩坏 = 通过
    broken = (rc.mean() > tc.mean() + 0.03) and (p2 < 0.05)
    like_random = (abs(rc.mean() - ro.mean()) < 0.05) or (p1 > 0.05)
    verdicts[tag] = broken and like_random
    lines.append(f'   -> 随机模型的 CAM 功能性{"已崩坏（接近瞎蒙）" if broken and like_random else "未完全崩坏（仍携带信息）"}')
lines.append('')
lines.append('--- 综合判定 ---')
for tag in orig:
    sp = float(np.mean(sims[tag]['spearman']))
    ss = float(np.mean(sims[tag]['ssim']))
    if verdicts[tag]:
        concl = ('通过：虽然 Pearson 反弹，但全随机模型的 CAM 在功能上已崩坏'
                 '（Deletion 接近瞎蒙），支持"假说B：散点先验撞车"，'
                 'CAM 对模型权重的依赖性成立。')
    else:
        concl = ('未通过：全随机模型的 CAM 在训练模型上仍有效，'
                 '提示 CAM 可能不依赖权重（假说A），需进一步排查。')
    lines.append(f'{tag}: Spearman {sp:.3f} / SSIM {ss:.3f} / 功能检验'
                 f'{"崩坏" if verdicts[tag] else "未崩坏"} -> {concl}')
lines.append('')
lines.append('论文写法建议：报告"级联随机化中间层相关性单调降至 ~0"作为主证据；'
             '全随机阶段的 Pearson 反弹如实报告，并用本功能性检验说明其源于'
             '白底点阵图的输入先验而非方法失效——这种诚实+深挖的写法反而加分。')

report = '\n'.join(lines)
with open('结果/第十五步B_交叉验证结果.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(report, flush=True)

# ============ 画图：左=三种相似度，右=功能性检验 ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
mets = ['pearson', 'spearman', 'ssim']
xpos = np.arange(len(mets))
for j, (tag, c) in enumerate([('GradCAM', '#2563EB'), ('DFOCAM', '#E8590C')]):
    vals = [float(np.mean(sims[tag][m])) for m in mets]
    ax.bar(xpos + (j - 0.5) * 0.32, vals, width=0.32, color=c, alpha=0.75, label=tag)
ax.axhline(0.1, color='#6B7280', ls=':', lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels(['Pearson\n(十五步反弹处)', 'Spearman\n(排序相似度)', 'SSIM\n(结构相似度)'],
                   fontsize=9)
ax.set_ylabel('原始CAM vs 全随机CAM 相似度（越低越好）')
ax.set_title('第一/二层证据：三种相似度', fontsize=11)
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5, axis='y')

ax = axes[1]
data, names = [], []
for tag in ['GradCAM', 'DFOCAM']:
    data += [func[tag]['trained_cam'], func[tag]['random_cam']]
    names += [f'{tag}\n训练CAM', f'{tag}\n随机模型CAM']
data.append(func['GradCAM']['random_order'])
names.append('纯随机顺序\n(瞎蒙基线)')
colors = ['#2563EB', '#93C5FD', '#E8590C', '#FDBA74', '#6B7280']
bp = ax.boxplot(data, showmeans=True, patch_artist=True,
                medianprops=dict(color='black', lw=1.5),
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=5))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.4)
ax.set_xticks(range(1, len(names) + 1))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('Deletion AUC（越小越好）')
ax.set_title('第三层证据：随机模型的CAM给训练模型用\n（接近"瞎蒙基线"= 功能崩坏 = 检验通过）', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5, axis='y')

plt.tight_layout()
plt.savefig('图片/第十五步B_交叉验证.png', dpi=120, bbox_inches='tight')
plt.close(fig)

print('done：结果见 结果/第十五步B_交叉验证结果.txt，图见 图片/第十五步B_交叉验证.png',
      flush=True)
