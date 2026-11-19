# -*- coding: utf-8 -*-
# 第十三步（GPU 版）：DFO-CAM 初始化消融——忠实性优势来自"优化"还是"初始化"？
#
# 与 CPU 版（第十三步_DFOCAM初始化消融.py）协议完全一致：
#   同样 100 张（每类 50）、白底 Deletion/Insertion（40 步）、AD/AI/AG，
#   仅把计算搬到 GPU（AutoDL 云服务器上跑）。
# 关键点：DFO-CAM DFO-CAM 代码第 74-77 行自带设备跟随（device = input_tensor.device），
#   优化循环会自动在 GPU 上跑，无需改动DFO-CAM 官方代码。
import os
import time
import importlib.util
import traceback
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Windows 用微软雅黑；Linux(AutoDL) 用文泉驿等开源中文字体（装法见 AutoDL上跑指南.md）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

torch.manual_seed(0)
np.random.seed(0)

NUM_STEPS = 40
CAM_THRESHOLD = 0.5
H = W = 224
N_PER_CLASS = 50        # GPU 上很快，一般不用改
INITS = ['shapleycam', 'gradcam', 'zerofive', 'random']   # DFO-CAM 官方代码支持的 4 种
METHODS = ['GradCAM'] + [f'DFOCAM_{n}' for n in INITS]    # 5 个同台方法

# numpy 2.0 把 trapz 改名 trapezoid；新旧版本都能跑的兼容写法
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

print('=== 脚本版本 V1-GPU：初始化消融（GradCAM参照 + 4种DFOCAM初始化）===', flush=True)

# ============ 计算设备：必须正常模式开机（无卡模式没有 GPU） ============
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    raise SystemExit('没检测到 GPU！请确认：① 实例是"正常开机"而不是"无卡模式"；'
                     '② 租实例时镜像选了 PyTorch。')
print(f'计算设备: GPU（{torch.cuda.get_device_name(0)}）', flush=True)

# ============ 加载DFO-CAM 实现（文件名带连字符，用 importlib 按路径加载）============
spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    """兼容层：不同版本 BaseCAM 传给 get_cam_weights 的类型/精度可能不同，
    DFO-CAM 官方代码按"单精度 numpy"处理，这里统一转成单精度 numpy 再进，出去再转回来。
    DFO-CAM 官方代码内部会把 numpy 搬回 input_tensor 所在设备，所以 GPU/CPU 都能跑。"""
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


# ============ 模型与提取器（先搬 GPU，再建提取器，设备才会对齐） ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

grad_ex = GradCAMElementWise(model=model, target_layers=[model.layer4])
dfo_exs = {n: DFOCAMCompat(model=model, target_layers=[model.layer4],
                           init_method=n) for n in INITS}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# 白色底（归一化空间）：第十一步诊断确认它适配白底点阵图
# dtype=torch.float32 必须显式声明（否则 Double/Float 报错，见踩坑 8）
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)


def auc_of_curve(logits_arr):
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits_arr - lo) / denom,
                        np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_batch(x, order, mode):
    device_ = x.device
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    total = H * W
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * total)
        mask = torch.zeros(1, H, W, device=device_)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        if mode == 'insertion':
            pert = base * (1 - mask) + x * mask
        else:
            pert = x * (1 - mask) + base * mask
        xs.append(pert)
    return torch.cat(xs, dim=0)


# ============ 自检：第一张图把 4 种初始化全部跑一遍，快速失败 ============
_first = 'data/ptbxl/images/af/' + sorted(os.listdir('data/ptbxl/images/af'))[0]
_x = to_tensor(Image.open(_first).convert('RGB').resize((224, 224))).unsqueeze(0)
_x = _x.to(device)
with torch.no_grad():
    _pred = model(_x).argmax(1).item()
for n in INITS:
    try:
        t0 = time.time()
        _ = dfo_exs[n](input_tensor=_x, targets=[ClassifierOutputTarget(_pred)])
        print(f'自检通过：DFOCAM[{n}] 单张用时 {time.time()-t0:.2f}s', flush=True)
    except Exception:
        traceback.print_exc()
        raise SystemExit(f'DFOCAM[{n}] 自检失败，终止（看上方报错）')

# ============ 主循环 ============
rng = np.random.RandomState(0)
aucs = {m: {'del': [], 'ins': []} for m in METHODS}
aai = {m: {'ad': [], 'ai': [], 'ag': []} for m in METHODS}
curves_raw = {m: {'del': [], 'ins': []} for m in METHODS}   # 画平均曲线用
rand_del_all, rand_ins_all = [], []                          # 共享随机基线
rand_curves_raw = {'del': [], 'ins': []}
n_used = 0
demo = {}          # 留 1 张房颤图：各方法 CAM 叠加图
t_start = time.time()

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()

    # 共享随机顺序（与热力图无关，全方法共用一条随机基线）
    rand_idx = rng.permutation(H * W)
    with torch.no_grad():
        lr_ = model(perturbed_batch(x, rand_idx, 'deletion'))[:, pred].cpu().numpy()
        l2 = model(perturbed_batch(x, rand_idx, 'insertion'))[:, pred].cpu().numpy()
    r_del, r_ins = auc_of_curve(lr_), auc_of_curve(l2)

    ok = r_del is not None and r_ins is not None
    batch_curves = {}
    for m in METHODS:
        if m == 'GradCAM':
            cam = grad_ex(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        else:
            cam = dfo_exs[m[len('DFOCAM_'):]](
                input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        sorted_idx = np.argsort(-cam_norm.flatten())
        with torch.no_grad():
            lm = model(perturbed_batch(x, sorted_idx, 'deletion'))[:, pred].cpu().numpy()
            li = model(perturbed_batch(x, sorted_idx, 'insertion'))[:, pred].cpu().numpy()
        a, b = auc_of_curve(lm), auc_of_curve(li)
        batch_curves[m] = (cam_norm, lm, li, a, b)
        ok = ok and a is not None and b is not None

    if not ok:      # 有任何一条曲线失效则整张跳过，保证各方法逐图配对
        continue
    rand_del_all.append(r_del); rand_ins_all.append(r_ins)
    rand_curves_raw['del'].append(lr_); rand_curves_raw['ins'].append(l2)
    for m in METHODS:
        cam_norm, lm, li, a, b = batch_curves[m]
        aucs[m]['del'].append(a); aucs[m]['ins'].append(b)
        curves_raw[m]['del'].append(lm); curves_raw[m]['ins'].append(li)

        binm = torch.from_numpy((cam_norm > CAM_THRESHOLD).astype(np.float32)).to(device)
        kept = x * binm
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        aai[m]['ad'].append(max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100)
        aai[m]['ai'].append(1.0 if y_keep > y_orig else 0.0)
        aai[m]['ag'].append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)

    if 'af_demo' not in demo and lab == 'af':
        for m in METHODS:
            demo[f'cam_{m}'] = batch_curves[m][0]
        demo['af_demo'] = np.array(img) / 255.0

    n_used += 1
    if (i + 1) % 10 == 0:
        el = time.time() - t_start
        print(f'progress {i+1}/{len(files)}  已用 {el/60:.1f} 分钟', flush=True)

print(f'used={n_used}  总用时 {(time.time()-t_start)/60:.1f} 分钟', flush=True)

# ============ 汇总 ============
S = {m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in aucs[m].items()}
     for m in METHODS}
AAI = {m: {k: float(np.mean(v)) for k, v in aai[m].items()} for m in METHODS}
rand_del = float(np.mean(rand_del_all))
rand_ins = float(np.mean(rand_ins_all))

# 逐图配对胜场：变体 vs 默认初始化；DFOCAM_gradcam vs 纯 GradCAM
wins = {f'DFOCAM_{n}': {'del': 0, 'ins': 0} for n in INITS}
wins['DFOCAM_gradcam_vs_GradCAM'] = {'del': 0, 'ins': 0}
for j in range(n_used):
    d_def = aucs['DFOCAM_shapleycam']['del'][j]
    i_def = aucs['DFOCAM_shapleycam']['ins'][j]
    for n in INITS:
        if n == 'shapleycam':
            continue
        if aucs[f'DFOCAM_{n}']['del'][j] < d_def:
            wins[f'DFOCAM_{n}']['del'] += 1
        if aucs[f'DFOCAM_{n}']['ins'][j] > i_def:
            wins[f'DFOCAM_{n}']['ins'] += 1
    if aucs['DFOCAM_gradcam']['del'][j] < aucs['GradCAM']['del'][j]:
        wins['DFOCAM_gradcam_vs_GradCAM']['del'] += 1
    if aucs['DFOCAM_gradcam']['ins'][j] > aucs['GradCAM']['ins'][j]:
        wins['DFOCAM_gradcam_vs_GradCAM']['ins'] += 1

name_cn = {'GradCAM': 'GradCAM(无优化参照)', 'DFOCAM_shapleycam': 'DFOCAM_shapleycam(默认)',
           'DFOCAM_gradcam': 'DFOCAM_gradcam', 'DFOCAM_zerofive': 'DFOCAM_zerofive',
           'DFOCAM_random': 'DFOCAM_random'}
lines = ['=== 第十三步：DFO-CAM 初始化消融（PTB-XL 子集，白底协议，GPU）===']
lines.append(f'有效样本 {n_used} 张；DFOCAM 优化轮数 50（DFO-CAM 默认）；种子=0；'
             f'设备 {torch.cuda.get_device_name(0)}')
lines.append('问题：DFO-CAM 的忠实性优势来自"双向流优化"还是"好的初始化"？')
lines.append('')
lines.append('--- Deletion / Insertion AUC（越小/越大越忠实）---')
lines.append(f'随机顺序基线          : del {rand_del:.4f} / ins {rand_ins:.4f}')
for m in METHODS:
    s = S[m]
    lines.append(f'{name_cn[m]:<24}: del {s["del"][0]:.4f} ± {s["del"][1]:.4f} / '
                 f'ins {s["ins"][0]:.4f} ± {s["ins"][1]:.4f}')
lines.append('')
lines.append('--- AD / AI / AG（DFO-CAM 协议）---')
for m in METHODS:
    a = AAI[m]
    lines.append(f'{name_cn[m]:<24}: AD {a["ad"]:.2f}%  AI {a["ai"]*100:.2f}%  AG {a["ag"]:.2f}%')
lines.append('')
lines.append('--- 逐图配对胜场（n 张中更好的张数）---')
for m in ['DFOCAM_gradcam', 'DFOCAM_zerofive', 'DFOCAM_random']:
    lines.append(f'{name_cn[m]:<24}: del 胜默认 {wins[m]["del"]}/{n_used}  ins 胜默认 {wins[m]["ins"]}/{n_used}')
lines.append(f'{"DFOCAM_gradcam vs GradCAM":<24}: del 胜 {wins["DFOCAM_gradcam_vs_GradCAM"]["del"]}/{n_used}'
             f'  ins 胜 {wins["DFOCAM_gradcam_vs_GradCAM"]["ins"]}/{n_used}')
lines.append('')
lines.append('--- 读法 ---')
lines.append('1) 4 种 DFOCAM 的 AUC 接近 -> 优势主要来自优化本身，对初始化不敏感；')
lines.append('2) DFOCAM_gradcam 明显好于 GradCAM(无优化参照) -> 优化在 GradCAM 之上仍有增益；')
lines.append('3) DFOCAM_random 也不差 -> 方法鲁棒，不依赖"好起点"。')
open('结果/DFOCAM初始化消融结果.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('results saved', flush=True)

# ============ 画图 1：三面板条形图（del AUC / ins AUC / AI%） ============
C_G, C_D = '#6B7280', '#E8590C'          # 参照灰 / DFOCAM 家族橙（沿用已验证配色）
grid = np.linspace(0, 1, NUM_STEPS + 1)

mean_curves = {m: {k: np.mean(np.stack(v), axis=0) for k, v in curves_raw[m].items()}
               for m in curves_raw}
rand_del_curve = np.mean(np.stack(rand_curves_raw['del']), axis=0)
rand_ins_curve = np.mean(np.stack(rand_curves_raw['ins']), axis=0)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
panels = [('del', 'Deletion AUC（越小越忠实）'),
          ('ins', 'Insertion AUC（越大越忠实）'),
          ('ai', 'AI%（保留高亮区后仍正确的比例）')]
for ax, (key, title) in zip(axes, panels):
    vals = ([S[m]['del'][0] for m in METHODS] if key == 'del' else
            [S[m]['ins'][0] for m in METHODS] if key == 'ins' else
            [AAI[m]['ai'] * 100 for m in METHODS])
    colors = [C_G if m == 'GradCAM' else C_D for m in METHODS]
    labels = ['GradCAM\n(no-opt ref.)'] + [f'DFOCAM_{n}' + (' (default)' if n == 'shapleycam' else '')
                                        for n in INITS]
    ax.barh(range(len(METHODS)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    for j, v in enumerate(vals):
        ax.text(v, j, f' {v:.3f}' if key != 'ai' else f' {v:.1f}',
                va='center', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    ax.margins(x=0.12)
plt.tight_layout()
plt.savefig('图片/fig_init_ablation.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ============ 画图 2：平均忠实性曲线（5 方法 + 共享随机基线） ============
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, key, better, rand_c in [
        (axes2[0], 'del', ' (lower = more faithful)', rand_del_curve),
        (axes2[1], 'ins', ' (higher = more faithful)', rand_ins_curve)]:
    ax.plot(grid, mean_curves['GradCAM'][key], color=C_G, lw=2, label='GradCAM')
    for n, c in zip(INITS, ['#E8590C', '#D9480F', '#B8471B', '#A61E4D']):
        lbl = f'DFOCAM_{n}' + (' (default)' if n == 'shapleycam' else '')
        ax.plot(grid, mean_curves[f'DFOCAM_{n}'][key], color=c, lw=1.8, ls='-',
                label=lbl, alpha=0.9)
    ax.plot(grid, rand_c, color='#9CA3AF', lw=2, ls='--', label='Random order')
    ax.set_xlabel('Deleted pixel fraction' if key == 'del' else 'Inserted pixel fraction')
    ax.set_ylabel('Target-class logit (per-image normalized)')
    ax.set_title(('Deletion ' if key == 'del' else 'Insertion ') + better)
    ax.legend(loc='upper right' if key == 'del' else 'lower right',
              frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig('图片/fig_ablation_curves.png', dpi=120, bbox_inches='tight')
plt.close(fig2)

# ============ 画图 3：同一张房颤图，5 种方法的 CAM 对比 ============
if 'af_demo' in demo:
    show = ['GradCAM'] + [f'DFOCAM_{n}' for n in INITS]
    titles = ['原始庞加莱图（房颤）'] + [name_cn[m] for m in show]
    fig3, axes3 = plt.subplots(2, 3, figsize=(9.6, 6.2))
    axes3 = axes3.flatten()
    axes3[0].imshow(demo['af_demo'])
    for j, m in enumerate(show):
        ax3 = axes3[j + 1]
        cam = demo[f'cam_{m}']
        cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        ax3.imshow(demo['af_demo'])
        ax3.imshow(cam_n, cmap='jet', alpha=0.45)
        ax3.set_title(titles[j + 1], fontsize=9)
    for ax3 in axes3:
        ax3.axis('off')
    plt.tight_layout()
    plt.savefig('图片/第十三步_CAM对比示例.png', dpi=120, bbox_inches='tight')
    plt.close(fig3)

print('done', flush=True)
