# -*- coding: utf-8 -*-
# 第十四步补充：RISE 作为第五个基线，加入四方法忠实性对比
#
# 背景：审稿意见 M-2 —— 参考文献[4]引用了 RISE 却没跑。补跑 RISE 作为
#   第五个解释方法，与 Grad-CAM / Grad-CAM++ / Score-CAM / DFO-CAM 同台，
#   用与第十四步完全相同的协议，结果可直接并进论文主表（表2）。
#
# 协议（与第十四步逐项一致）：
#   - PTB-XL 子集 300 张（每类 150，文件顺序与第十四步完全一致）
#   - 白底 Deletion/Insertion（40 步）+ AD/AI/AG（CAM>0.5 阈值）
#   - 随机基线：直接从第十四步 csv 读 rand_del/rand_ins（同一随机基线）
#
# RISE 实现要点（自实现，不依赖额外库）：
#   - 生成 N 个 GRID×GRID 的随机二值掩码，上采样到 224×224
#   - 每个掩码做一次前向，取目标类（pred）softmax 分数
#   - saliency = Σ(掩码 × 分数) / N（加权平均，即"哪些像素常被保留且分类分数高"）
#   - 掩码为 0 处填白色（BASE_WHITE）而非黑色/置零 —— 与 3.6 节底色适配一致，
#     避免白底点阵图上的分布外扰动
#   - 掩码固定种子，可复现
#
# 参数说明：RISE 原论文 N=8000、GRID=7；本实验用 N=4000、GRID=8
#   以平衡精度与耗时（4000 >> 64 个格子，采样覆盖已足够）。可复现。
#
# 工程要点（与第十四步一致）：
#   - 逐图断点续跑（结果/第十四步_RISE_per_image.csv），中断自动跳过
#   - 设备自动跟随：AutoDL GPU 直接跑（推荐）；本地 CPU 极慢
#   - STATS_ONLY=True 只做统计（读已有 csv 出 5 方法合并报告）
import os
import csv
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

STATS_ONLY = False      # True = 跳过 RISE 实验，只读 csv 做合并统计
N_MASKS = 4000          # RISE 随机掩码数量
GRID = 8                # 掩码低分辨率网格 GRID×GRID
MASK_P = 0.5            # 每个格子保留概率
MASK_BATCH = 128        # RISE 掩码前向的批大小（防爆显存）
NUM_STEPS = 40
CAM_THRESHOLD = 0.5
H = W = 224
N_PER_CLASS = 150       # 与第十四步一致：每类 150，共 300 张
BOOT_N = 2000
CSV4 = '结果/第十四步_per_image.csv'           # 已有四方法结果（含随机基线）
CSV_RISE = '结果/第十四步_RISE_per_image.csv'  # 本次 RISE 逐图结果
TXT_OUT = '结果/第十四步_RISE显著性检验结果.txt'

# numpy 2.0 把 trapz 改名 trapezoid；新旧版本兼容
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

print('=== 第十四步补充：RISE 第五基线（300 张，白底协议） ===', flush=True)
os.makedirs('结果', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {"GPU " + torch.cuda.get_device_name(0) if device == "cuda" else "CPU（极慢，建议上 AutoDL）"}',
      flush=True)

# ============ 模型（与第十四步同一权重） ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32,
                          device=device).view(1, 3, 1, 1)


def rise_saliency(x, target):
    """RISE 显著性图。x: (1,3,H,W) 已归一化；target: 目标类。返回 (H,W) numpy。
    掩码固定种子，结果可复现。掩码为 0 处填白底（与 Deletion/Insertion 协议一致）。"""
    rng = np.random.RandomState(0)
    masks = (rng.random((N_MASKS, 1, GRID, GRID)) < MASK_P).astype(np.float32)
    masks = torch.from_numpy(masks)
    masks_hi = F.interpolate(masks, size=(H, W), mode='bilinear',
                             align_corners=False)          # (N,1,H,W)
    scores = []
    with torch.no_grad():
        for i in range(0, N_MASKS, MASK_BATCH):
            m = masks_hi[i:i + MASK_BATCH].to(device)
            xm = x * m + BASE_WHITE * (1 - m)              # 白底填充
            out = model(xm)                                # (B,2)
            scores.append(torch.softmax(out, dim=1)[:, target].cpu())
    scores = torch.cat(scores)                             # (N,)
    sal = (masks_hi * scores.view(N_MASKS, 1, 1, 1)).sum(0) / N_MASKS
    return sal.squeeze().cpu().numpy()


def auc_of_curve(logits_arr):
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    return float(_trapz((logits_arr - lo) / denom,
                        np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_multi(x, orders, mode):
    """把多个像素顺序的 41 步扰动图合并成一批前向（与第十四步同实现）。
    装盒按"顺序"为主循环，与按 41 步整条拆盒一致。"""
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    for order in orders:
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
        for i in range(0, big.size(0), 82):
            chunks.append(model(big[i:i + 82]).cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    n = NUM_STEPS + 1
    return [logits[j * n:(j + 1) * n] for j in range(len(orders))]


# ============ 文件清单（与第十四步完全一致） ============
files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

FIELDS = ['file', 'label', 'gt', 'pred',
          'rise_del', 'rise_ins', 'rise_ad', 'rise_ai', 'rise_ag']

# ============ 断点续跑 ============
done = set()
if os.path.exists(CSV_RISE):
    with open(CSV_RISE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            done.add(row['file'])
    print(f'断点续跑：已完成 {len(done)} 张，自动跳过', flush=True)

if not STATS_ONLY:
    # 自检：第一张图跑通并报告单张耗时
    _x = to_tensor(Image.open(files[0][0]).convert('RGB')
                   .resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(device)
    with torch.no_grad():
        _pred = model(_x).argmax(1).item()
    t0 = time.time()
    _s = rise_saliency(_x, _pred)
    print(f'自检通过：RISE 单张耗时 {time.time() - t0:.1f}s（{N_MASKS} 掩码），'
          f'300 张预计 {(time.time() - t0) * 300 / 60:.0f} 分钟', flush=True)

    new_file = not os.path.exists(CSV_RISE)
    with open(CSV_RISE, 'a', newline='', encoding='utf-8') as fout:
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

            rng = np.random.RandomState(1000 + i)     # 与第十四步同种子的随机基线
            rand_orders = [rng.permutation(H * W) for _ in range(3)]

            # --- RISE 显著性图 ---
            sal = rise_saliency(x, pred)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

            row = {'file': path, 'label': lab, 'gt': gt, 'pred': pred}

            # --- Deletion / Insertion：RISE 序 + 3 随机序 ---
            orders = [np.argsort(-sal.flatten())] + rand_orders
            del_curves = perturbed_multi(x, orders, 'deletion')
            ins_curves = perturbed_multi(x, orders, 'insertion')
            a_del = [auc_of_curve(c[:, pred]) for c in del_curves]
            a_ins = [auc_of_curve(c[:, pred]) for c in ins_curves]
            if None in a_del or None in a_ins:
                print(f'跳过（曲线退化）: {path}', flush=True)
                continue
            row['rise_del'], row['rise_ins'] = a_del[0], a_ins[0]

            # --- AD / AI / AG（保留 sal>0.5 区域，GT 类） ---
            kept = x * torch.from_numpy(
                (sal > CAM_THRESHOLD).astype(np.float32)).to(device)
            with torch.no_grad():
                y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
            row['rise_ad'] = max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100
            row['rise_ai'] = 1.0 if y_keep > y_orig else 0.0
            row['rise_ag'] = (max(0.0, y_keep - y_orig)
                              / (1 - y_orig + 1e-8) * 100)

            writer.writerow(row)
            fout.flush()
            n_run += 1
            if n_run % 5 == 0:
                el = time.time() - t_start
                eta = el / n_run * (len(files) - len(done) - n_run) / 60
                print(f'progress 新跑 {n_run} 张 | 总进度 {len(done) + n_run}/{len(files)}'
                      f' | 单张 {el / n_run:.1f}s | 预计剩余 {eta:.0f} 分钟', flush=True)

    print('RISE 实验部分完成，开始合并统计 ...', flush=True)

# ============ 合并统计：5 方法（读两个 csv 按 file 对齐） ============
from scipy import stats

rows4 = {}
if not os.path.exists(CSV4):
    raise SystemExit('找不到 结果/第十四步_per_image.csv，请先跑第十四步主实验。')
with open(CSV4, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows4[r['file']] = {k: (float(v) if k != 'file' and k != 'label' else v)
                            for k, v in r.items()}

rows_r = {}
with open(CSV_RISE, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows_r[r['file']] = {k: (float(v) if k != 'file' and k != 'label' else v)
                             for k, v in r.items()}

# 只保留两边都有的图（RISE 若因曲线退化跳过，则不参与统计）
common = [f for f in rows4 if f in rows_r]
if len(common) < 10:
    raise SystemExit(f'两边重合样本只有 {len(common)} 张，太少，先跑 RISE 实验。')
print(f'合并统计样本：{len(common)} 张（四方法 {len(rows4)} 张，RISE {len(rows_r)} 张）',
      flush=True)

METHODS = [('grad', 'GradCAM'), ('gradpp', 'GradCAM++'),
           ('score', 'ScoreCAM'), ('dfo', 'DFOCAM'), ('rise', 'RISE')]
PRETTY = dict(METHODS)

# 组装 5 方法逐图指标（RISE 来自 rows_r，其余来自 rows4）
M = {}
for m, _ in METHODS:
    M[m] = {k: np.array([(rows_r if m == 'rise' else rows4)[f][f'{m}_{k}']
                         for f in common])
            for k in ['del', 'ins', 'ad', 'ai', 'ag']}
R_del = np.array([rows4[f]['rand_del'] for f in common])
R_ins = np.array([rows4[f]['rand_ins'] for f in common])
pred = np.array([rows4[f]['pred'] for f in common])
gt = np.array([rows4[f]['gt'] for f in common])
acc = float(np.mean(pred == gt))
BASELINES = [m for m, _ in METHODS if m != 'dfo']


def cohen_dz(a, b):
    d = a - b
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def mcnemar(a, b, name_a, name_b):
    b01 = int(np.sum((a == 1) & (b == 0)))
    c01 = int(np.sum((a == 0) & (b == 1)))
    n01 = b01 + c01
    if n01 == 0:
        return f'   McNemar：两方法在所有图上结论一致（p=1.0 ns）'
    p = stats.binomtest(min(b01, c01), n01, 0.5).pvalue
    return (f'   McNemar：{name_a}对/{name_b}错 {b01} 张 vs '
            f'{name_a}错/{name_b}对 {c01} 张 -> p={p:.4f} {stars(p)}')


lines = [
    '=== 第十四步补充：五方法忠实性对比（含 RISE，PTB-XL 白底协议）===',
    f'方法：{" vs ".join(PRETTY[m] for m, _ in METHODS)}',
    f'样本 {len(common)} 张；随机基线同第十四步（3 顺序平均）；'
    f'模型在本子集准确率 {acc * 100:.1f}%',
    '注意：DFOCAM vs 4 个基线，多重比较 Bonferroni 校正 α=0.05/4≈0.0125',
    '',
    f'随机基线: del {R_del.mean():.4f}±{R_del.std():.4f} / '
    f'ins {R_ins.mean():.4f}±{R_ins.std():.4f}',
    '',
    '--- 5 方法均值汇总（表2 可直接引用）---',
    '方法         Deletion↓     Insertion↑    AD↓           AG↑           AI%↑',
]
for m, pretty in METHODS:
    lines.append(
        f'{pretty:<12} {M[m]["del"].mean():.4f}±{M[m]["del"].std():.4f} '
        f'{M[m]["ins"].mean():.4f}±{M[m]["ins"].std():.4f} '
        f'{M[m]["ad"].mean():.2f}±{M[m]["ad"].std():.2f} '
        f'{M[m]["ag"].mean():.2f}±{M[m]["ag"].std():.2f} '
        f'{M[m]["ai"].mean() * 100:.1f}')

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
                     f'{"-> 显著优于随机" if ok and p < 0.05 else "-> 注意：方向或显著性异常"}')

lines.append('')
lines.append('--- DFOCAM vs 各基线（含 RISE）逐指标 ---')
for metric, cname, better in [('del', 'Deletion（越小越忠实）', 'low'),
                              ('ins', 'Insertion（越大越忠实）', 'high'),
                              ('ad', 'AD（越小越好）', 'low'),
                              ('ag', 'AG（越大越好）', 'high')]:
    lines.append(f'【{cname}】')
    for b in BASELINES:
        a, bb = M['dfo'][metric], M[b][metric]
        t, p = stats.ttest_rel(a, bb)
        try:
            _, pw = stats.wilcoxon(a, bb)
        except ValueError:
            pw = float('nan')
        dz = cohen_dz(a, bb)
        win = 'DFOCAM' if ((a.mean() < bb.mean()) == (better == 'low')) else PRETTY[b]
        lines.append(f'  DFOCAM {a.mean():.4f} vs {PRETTY[b]:<10} {bb.mean():.4f} '
                     f'-> {win} 更优 | t检验 p={p:.4f} {stars(p)} | '
                     f'Wilcoxon p={pw:.4f} | dz={dz:+.3f}')

lines.append('')
lines.append('--- AI %（McNemar，DFOCAM vs 各基线）---')
for b in BASELINES:
    lines.append(f'【AI】DFOCAM {M["dfo"]["ai"].mean() * 100:.1f}% vs '
                 f'{PRETTY[b]} {M[b]["ai"].mean() * 100:.1f}%')
    lines.append(mcnemar(M['dfo']['ai'], M[b]['ai'], 'DFOCAM', PRETTY[b]))

lines.append('')
lines.append('解读：看 RISE 是否也显著优于随机（若否则说明 RISE 在白底点阵图上'
             '忠实性不成立，需如实写）；看 DFOCAM 的 AI 优势对 RISE 是否仍成立'
             '（RISE 是黑盒机制，最能检验 DFOCAM 的 AI 优势是不是 CAM 家族内部现象）。')

report = '\n'.join(lines)
with open(TXT_OUT, 'w', encoding='utf-8') as f:
    f.write(report)
print(report, flush=True)
print(f'\ndone：结果见 {TXT_OUT}，逐图数据见 {CSV_RISE}', flush=True)
