# -*- coding: utf-8 -*-
# 第十四步（V3 四方法版）：大样本 + 显著性检验
#   Grad-CAM vs Grad-CAM++ vs Score-CAM vs DFO-CAM 同台对比
#
# 为什么升级为四方法：只对比 Grad-CAM 一个基线，审稿人必问"为什么不比
#   Grad-CAM++/Score-CAM"。两个方法都是 pytorch-grad-cam 库现成的，
#   一次实验全部跑完，直接把这个问题堵死。
#
# 协议（与第十二/十三步完全一致）：
#   - PTB-XL 子集 300 张（每类 150，前 50 张/类与第十二步重合，可对照）
#   - 白底 Deletion/Insertion（40 步）+ AD/AI/AG（DFO-CAM 协议，CAM>0.5）
#   - 随机基线每张图 3 个随机顺序平均
#
# 统计（回答"差异是稳定的还是运气"）：
#   - 主对比：DFO-CAM vs 每个基线，配对 t 检验 + Wilcoxon + Cohen's dz
#     + Bootstrap 95%CI；AI（0/1）用 McNemar 检验
#   - 每种方法 vs 随机基线：验证"都显著优于瞎蒙"
#   - 自动自检：前 100 张与第十二步已发表数字对照，偏差大就警告
#
# 工程要点：
#   - 逐图断点续跑（结果/第十四步_per_image.csv），中断重跑自动跳过
#   - 每张图 Deletion/Insertion 各只调 1 次大batch 前向（4方法+3随机=7条曲线合并）
#   - 设备自动跟随：本地 CPU 能跑（慢），AutoDL GPU 直接跑同一个文件（推荐）
#   - 只跑统计不跑实验：STATS_ONLY = True（读已存 csv 出报告）
import os
import csv
import time
import importlib.util
import traceback
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMElementWise, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

STATS_ONLY = False      # True = 跳过实验，直接读 csv 做统计出报告
NUM_STEPS = 40
CAM_THRESHOLD = 0.5
H = W = 224
N_PER_CLASS = 150       # 300 张；AutoDL GPU 上不用改小
RAND_SEEDS = 3          # 每张图随机基线取 3 个随机顺序平均
BOOT_N = 2000           # Bootstrap 重采样次数
CSV_PATH = '结果/第十四步_per_image.csv'
TXT_PATH = '结果/第十四步_显著性检验结果.txt'

# (内部名, 显示名)。内部名用于 csv 列名；显示名用于报告和图
METHODS = [('grad', 'GradCAM'), ('gradpp', 'GradCAM++'),
           ('score', 'ScoreCAM'), ('dfo', 'DFOCAM')]

# numpy 2.0 把 trapz 改名 trapezoid；新旧版本兼容
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

print('=== 第十四步 V3：300张 + 四方法同台 + 配对显著性检验 ===', flush=True)

os.makedirs('结果', exist_ok=True)
os.makedirs('图片', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {"GPU " + torch.cuda.get_device_name(0) if device == "cuda" else "CPU（慢，建议上 AutoDL）"}',
      flush=True)

# ============ 加载DFO-CAM 实现（文件名带连字符，用 importlib 按路径加载）============
spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    """兼容层（与第十二/十三步一致）：统一成单精度 numpy 进出，
    DFO-CAM 官方代码内部会跟随 input_tensor 的设备，GPU/CPU 都能跑。"""
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


# ============ 模型与四个提取器 ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

EXTRACTORS = {
    'grad': GradCAMElementWise(model=model, target_layers=[model.layer4]),
    'gradpp': GradCAMPlusPlus(model=model, target_layers=[model.layer4]),
    'score': ScoreCAM(model=model, target_layers=[model.layer4]),
    'dfo': DFOCAMCompat(model=model, target_layers=[model.layer4]),
}

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# 白色底（归一化空间）：第十一步诊断确认适配白底点阵图
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


def perturbed_multi(x, orders, mode):
    """把多个像素顺序的 41 步扰动图合并成一个大批次，只调一次前向。
    orders: list of np.array；返回 list[np.array]，每个是对应顺序的 logit 曲线。
    注意：装盒必须"按顺序（order）为主循环"，与下方按 41 步整条拆盒保持一致，
    否则不同顺序的曲线会交叉污染（V1 的 bug：外层循环是 step，拆盒按 order）。"""
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    for order in orders:                       # 外层=顺序：一条曲线装完再装下一条
        for step in range(NUM_STEPS + 1):
            k = int(step / NUM_STEPS * H * W)
            mask = torch.zeros(1, H, W, device=x.device)
            rc = np.unravel_index(order[:k], (H, W))
            mask[0, rc[0], rc[1]] = 1.0
            if mode == 'insertion':
                xs.append(base * (1 - mask) + x * mask)
            else:
                xs.append(x * (1 - mask) + base * mask)
    big = torch.cat(xs, dim=0)
    chunks = []
    with torch.no_grad():
        for i in range(0, big.size(0), 82):      # 再分块防爆内存（82=2条曲线）
            chunks.append(model(big[i:i + 82]).cpu().numpy())
    logits = np.concatenate(chunks, axis=0)      # [41*len(orders), 2]
    n = NUM_STEPS + 1
    return [logits[j * n:(j + 1) * n] for j in range(len(orders))]


# ============ 文件清单（前 50 张/类与第十二步重合，便于对照）============
files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

FIELDS = ['file', 'label', 'gt', 'pred', 'rand_del', 'rand_ins']
for m, _ in METHODS:
    FIELDS += [f'{m}_del', f'{m}_ins', f'{m}_ad', f'{m}_ai', f'{m}_ag']

# ============ 断点续跑：读已完成清单 ============
done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            done.add(row['file'])
    print(f'断点续跑：已完成 {len(done)} 张，自动跳过', flush=True)

if not STATS_ONLY:
    # 自检：第一张图验证四个方法都能跑通，并报告单张耗时供估计总时长
    _x = to_tensor(Image.open(files[0][0]).convert('RGB')
                   .resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(device)
    with torch.no_grad():
        _pred = model(_x).argmax(1).item()
    for m, pretty in METHODS:
        try:
            t0 = time.time()
            _cam = EXTRACTORS[m](input_tensor=_x,
                                 targets=[ClassifierOutputTarget(_pred)])
            print(f'自检通过：{pretty} 单张 CAM 用时 {time.time() - t0:.1f}s',
                  flush=True)
        except Exception:
            traceback.print_exc()
            raise SystemExit(f'{pretty} 自检失败，终止（看上方报错）')

    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        t_start = time.time()
        n_run = 0
        for i, (path, gt, lab) in enumerate(files):
            if path in done:
                continue
            img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
            x = to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits0 = model(x)
                pred = logits0.argmax(1).item()
                y_orig = torch.softmax(logits0, dim=1)[0, gt].item()

            rng = np.random.RandomState(1000 + i)     # 每张图随机基线可复现
            rand_orders = [rng.permutation(H * W) for _ in range(RAND_SEEDS)]

            # --- 四个方法的 CAM ---
            cams = {}
            for m, _ in METHODS:
                cam = EXTRACTORS[m](input_tensor=x,
                                    targets=[ClassifierOutputTarget(pred)])[0]
                cams[m] = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            row = {'file': path, 'label': lab, 'gt': gt, 'pred': pred}

            # --- Deletion / Insertion：4方法序 + 3随机序，各合并成一次前向 ---
            orders = [np.argsort(-cams[m].flatten()) for m, _ in METHODS] + rand_orders
            del_curves = perturbed_multi(x, orders, 'deletion')
            ins_curves = perturbed_multi(x, orders, 'insertion')
            a_del = [auc_of_curve(c[:, pred]) for c in del_curves]
            a_ins = [auc_of_curve(c[:, pred]) for c in ins_curves]
            if None in a_del or None in a_ins:
                print(f'跳过（曲线退化，logit 无变化）: {path}', flush=True)
                continue
            n_m = len(METHODS)
            for j, (m, _) in enumerate(METHODS):
                row[f'{m}_del'], row[f'{m}_ins'] = a_del[j], a_ins[j]
            # 随机基线与方法无关，4 个方法共用同一组随机曲线
            row['rand_del'] = float(np.mean(a_del[n_m:]))
            row['rand_ins'] = float(np.mean(a_ins[n_m:]))

            # --- AD / AI / AG（DFO-CAM 协议：保留 CAM>0.5 区域，GT 类）---
            kept = torch.cat([
                x * torch.from_numpy((cams[m] > CAM_THRESHOLD).astype(np.float32))
                      .to(device)
                for m, _ in METHODS], dim=0)
            with torch.no_grad():
                y_keep = torch.softmax(model(kept), dim=1)[:, gt].cpu().numpy()
            for j, (m, _) in enumerate(METHODS):
                row[f'{m}_ad'] = max(0.0, y_orig - y_keep[j]) / (y_orig + 1e-8) * 100
                row[f'{m}_ai'] = 1.0 if y_keep[j] > y_orig else 0.0
                row[f'{m}_ag'] = (max(0.0, y_keep[j] - y_orig)
                                  / (1 - y_orig + 1e-8) * 100)

            writer.writerow(row)
            fout.flush()                     # 立即落盘，中断不丢
            n_run += 1
            if n_run % 5 == 0:
                el = time.time() - t_start
                eta = el / n_run * (len(files) - len(done) - n_run) / 60
                print(f'progress 新跑 {n_run} 张 | 总进度 {len(done) + n_run}/{len(files)}'
                      f' | 单张 {el / n_run:.1f}s | 预计剩余 {eta:.0f} 分钟', flush=True)

    print('实验部分完成，开始统计 ...', flush=True)

# ============ 统计检验（读 csv，可独立重跑）============
from scipy import stats

rows = []
with open(CSV_PATH, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows.append({k: (float(v) if k not in ('file', 'label') else v)
                     for k, v in r.items()})
n = len(rows)
if n < 10:
    raise SystemExit(f'有效样本只有 {n} 张，太少，先跑实验再做统计。')
print(f'统计样本：{n} 张', flush=True)

M = {m: {k: np.array([r[f'{m}_{k}'] for r in rows])
         for k in ['del', 'ins', 'ad', 'ai', 'ag']} for m, _ in METHODS}
R_del = np.array([r['rand_del'] for r in rows])
R_ins = np.array([r['rand_ins'] for r in rows])
pred = np.array([r['pred'] for r in rows])
gt = np.array([r['gt'] for r in rows])
acc = float(np.mean(pred == gt))
BASELINES = [m for m, _ in METHODS if m != 'dfo']
PRETTY = dict(METHODS)


def cohen_dz(a, b):
    """配对效应量：差值的标准化均值。0.2小 / 0.5中 / 0.8大（经验阈值）"""
    d = a - b
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def boot_ci(a, b, seed=0):
    """Bootstrap 求"平均差 a-b"的 95% 置信区间（对差值重采样）"""
    rng = np.random.RandomState(seed)
    d = a - b
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(BOOT_N)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def paired_report(metric_name, a, b, name_a, name_b, better):
    """a vs b 的配对检验报告行（主对比里 a=DFOCAM）"""
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = float('nan')
    dz = cohen_dz(a, b)
    lo, hi = boot_ci(a, b)
    win = name_a if ((a.mean() < b.mean()) == (better == 'low')) else name_b
    return [
        f'【{metric_name}】{name_a} {a.mean():.4f}±{a.std():.4f} vs '
        f'{name_b} {b.mean():.4f}±{b.std():.4f} -> 均值看 {win} 更优',
        f'   配对t检验  p={p_t:.4f} {stars(p_t)} | '
        f'Wilcoxon p={p_w:.4f} {stars(p_w) if p_w == p_w else "(全相等无法算)"}',
        f'   效应量 Cohen\'s dz={dz:+.3f}（|dz| 0.2小/0.5中/0.8大）| '
        f'平均差 95%CI [{lo:+.4f}, {hi:+.4f}]',
    ]


def mcnemar(a, b, name_a, name_b):
    """AI 是 0/1 指标 -> McNemar 检验（只看两种方法结论不一致的图）"""
    b01 = int(np.sum((a == 1) & (b == 0)))
    c01 = int(np.sum((a == 0) & (b == 1)))
    n01 = b01 + c01
    if n01 == 0:
        return f'   McNemar：两方法在所有图上结论一致，无法比较（p=1.0 ns）'
    p = stats.binomtest(min(b01, c01), n01, 0.5).pvalue
    return (f'   McNemar：{name_a}对/{name_b}错 {b01} 张 vs '
            f'{name_a}错/{name_b}对 {c01} 张 -> p={p:.4f} {stars(p)}')


lines = [
    '=== 第十四步：四方法忠实性对比 + 大样本显著性检验（PTB-XL 白底协议）===',
    f'方法：{" vs ".join(PRETTY[m] for m, _ in METHODS)}',
    f'样本 {n} 张（正常/房颤各半，含第十二步的 100 张）；'
    f'随机基线每张图 {RAND_SEEDS} 个随机顺序平均；模型在本子集准确率 {acc * 100:.1f}%',
    '检验：DFOCAM vs 每个基线，配对t检验 + Wilcoxon + Cohen\'s dz + Bootstrap 95%CI；'
    'AI 用 McNemar',
    '注意：每个指标 3 组对比，Bonferroni 校正后显著性阈值 α=0.05/3≈0.017',
    '',
    '--- 随机基线（3 种子平均）---',
    f'随机顺序: del {R_del.mean():.4f}±{R_del.std():.4f} / '
    f'ins {R_ins.mean():.4f}±{R_ins.std():.4f}',
]

for metric, cname, better in [('del', 'Deletion AUC（越小越忠实）', 'low'),
                              ('ins', 'Insertion AUC（越大越忠实）', 'high'),
                              ('ad', 'AD %（越小越好）', 'low'),
                              ('ag', 'AG %（越大越好）', 'high')]:
    lines.append('')
    lines.append(f'--- {cname}：DFOCAM vs 各基线 ---')
    for b in BASELINES:
        lines += paired_report(cname, M['dfo'][metric], M[b][metric],
                               'DFOCAM', PRETTY[b], better)
        lines.append('')

lines.append('--- AI %：DFOCAM vs 各基线（McNemar）---')
for b in BASELINES:
    lines.append(f'【AI】DFOCAM {M["dfo"]["ai"].mean() * 100:.1f}% vs '
                 f'{PRETTY[b]} {M[b]["ai"].mean() * 100:.1f}%')
    lines.append(mcnemar(M['dfo']['ai'], M[b]['ai'], 'DFOCAM', PRETTY[b]))

lines.append('')
lines.append('--- 每种方法 vs 随机基线（验证"都显著优于瞎蒙"）---')
for m, pretty in METHODS:
    for metric, r, better, tag in [('del', R_del, 'low', 'del'),
                                   ('ins', R_ins, 'high', 'ins')]:
        a = M[m][metric]
        t, p = stats.ttest_rel(a, r)
        dz = cohen_dz(a, r)
        ok = (a.mean() < r.mean()) if better == 'low' else (a.mean() > r.mean())
        lines.append(f'{pretty} {tag}: {a.mean():.4f} vs 随机 {r.mean():.4f} '
                     f'p={p:.2e} {stars(p)} dz={dz:+.2f} '
                     f'{"-> 方向正确，显著优于随机" if ok and p < 0.05 else "-> 注意：方向或显著性异常"}')

lines.append('')
lines.append('--- 正确性自检：与第十二步（100 张重合子集）已发表数字对照 ---')
subset50 = set()
for lab in ['normal', 'af']:
    for f in sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:50]:
        subset50.add(f'data/ptbxl/images/{lab}/{f}')
idx50 = np.array([r['file'] in subset50 for r in rows])
if idx50.sum() > 0:
    lines.append(f'重合子集 {int(idx50.sum())} 张：'
                 f'GradCAM del {M["grad"]["del"][idx50].mean():.3f}（十二步 0.364）/ '
                 f'ins {M["grad"]["ins"][idx50].mean():.3f}（十二步 0.710）；'
                 f'DFOCAM del {M["dfo"]["del"][idx50].mean():.3f}（十二步 0.358）/ '
                 f'ins {M["dfo"]["ins"][idx50].mean():.3f}（十二步 0.689）')
    close = abs(M['grad']['del'][idx50].mean() - 0.364) < 0.06
    lines.append('自检' + ('通过：与第十二步基本吻合，本次数据可信。'
                 if close else '未通过：与第十二步偏差较大，请检查协议一致性后再用本结果！'))

lines.append('')
lines.append('--- 一句话结论（写论文可直接引用数字）---')
for metric, cname in [('del', 'Deletion'), ('ins', 'Insertion')]:
    for b in BASELINES:
        d = M['dfo'][metric].mean() - M[b][metric].mean()
        p = stats.ttest_rel(M['dfo'][metric], M[b][metric]).pvalue
        better_w = '优' if (d < 0) == (metric == 'del') else '劣'
        lines.append(f'{cname}: DFOCAM 比 {PRETTY[b]} {better_w} {abs(d):.4f}'
                     f'（p={p:.2e} {stars(p)}）。')
lines.append('解读：p<0.017（Bonferroni）且方向一致，可写"DFO-CAM 在该指标上显著优于'
             '全部基线方法"；若某对比 p 不显著且 |dz|<0.2，可写"与该基线相当（差异无'
             '统计学意义且效应量小）"。')

report = '\n'.join(lines)
with open(TXT_PATH, 'w', encoding='utf-8') as f:
    f.write(report)
print(report, flush=True)

# ============ 图 1：per-image AUC 箱线图（4 方法 + 随机基线）============
COLORS = ['#2563EB', '#7C3AED', '#0D9488', '#E8590C', '#6B7280']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (metric, title) in zip(axes, [
        ('del', 'Deletion（AUC 越小越忠实）'),
        ('ins', 'Insertion（AUC 越大越忠实）')]):
    parts = [M[m][metric] for m, _ in METHODS] + [R_del if metric == 'del' else R_ins]
    names = [PRETTY[m] for m, _ in METHODS] + ['随机顺序']
    bp = ax.boxplot(parts, showmeans=True, patch_artist=True,
                    medianprops=dict(color='black', lw=1.5),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=5))
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=9)   # set_xticklabels 新旧 matplotlib 都兼容
    ps = [stats.ttest_rel(M['dfo'][metric], M[b][metric]).pvalue for b in BASELINES]
    ax.set_title(f'{title}\nDFOCAM vs 基线 p值: '
                 + ' / '.join(f'{p:.1e}' for p in ps), fontsize=10)
    ax.set_ylabel('per-image AUC')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
plt.tight_layout()
plt.savefig('图片/第十四步_显著性箱线图.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ============ 图 2：配对散点图（DFOCAM vs 三个基线，每个点=一张图）============
fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
for row, (metric, what) in enumerate([('del', 'Deletion（对角线下方=DFOCAM赢）'),
                                      ('ins', 'Insertion（对角线上方=DFOCAM赢）')]):
    for col, b in enumerate(BASELINES):
        ax = axes[row, col]
        g, d = M[b][metric], M['dfo'][metric]
        ax.scatter(g, d, s=10, alpha=0.45, color='#E8590C', edgecolors='none')
        lim = [min(g.min(), d.min()) - 0.02, max(g.max(), d.max()) + 0.02]
        ax.plot(lim, lim, color='#6B7280', ls='--', lw=1)
        win = float(np.mean(d < g) * 100) if metric == 'del' \
            else float(np.mean(d > g) * 100)
        ax.set_xlabel(f'{PRETTY[b]} AUC', fontsize=9)
        ax.set_ylabel('DFOCAM AUC', fontsize=9)
        ax.set_title(f'{what}\nDFOCAM 胜出图占 {win:.0f}%', fontsize=9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig('图片/第十四步_配对散点图.png', dpi=120, bbox_inches='tight')
plt.close(fig)

print('done：结果见 结果/第十四步_显著性检验结果.txt，'
      '图见 图片/第十四步_显著性箱线图.png、第十四步_配对散点图.png', flush=True)
