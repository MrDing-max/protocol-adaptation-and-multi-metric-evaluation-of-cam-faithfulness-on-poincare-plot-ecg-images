# -*- coding: utf-8 -*-
# 第十二步：Grad-CAM vs DFO-CAM 同台对比（基线方法 vs DFO-CAM方法）
#
# 数据：PTB-XL 子集（每类前 50 张，共 100 张）
# 指标：与第十一步一致 —— Deletion/Insertion 曲线与 AUC + AD/AI/AG
# 协议要点：
#   - Deletion/Insertion 底图统一用"白色"（第十一步已证实：灰底对白底点阵图是分布外扰动）
#   - 随机顺序基线在同一子集上重算，与两种方法同台对比
#   - AD/AI/AG 按DFO-CAM Ad_Ai_AG 协议（保留 CAM>0.5 区域，GT 类）
#   - DFOCAM 用DFO-CAM 默认初始化（init_method='shapleycam'，50 轮优化）
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
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

NUM_STEPS = 40
CAM_THRESHOLD = 0.5
H = W = 224
N_PER_CLASS = 50

print('=== 脚本版本 V3：已含底图精度修复（BASE_WHITE dtype=float32）===', flush=True)

# ============ 加载DFO-CAM 实现（文件名带连字符，用 importlib 按路径加载）============
spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    """兼容层：不同版本 BaseCAM 传给 get_cam_weights 的类型/精度可能不同，
    DFO-CAM 官方代码按"单精度 numpy"处理，这里统一转成单精度 numpy 再进，出去再转回来。"""
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


# ============ 模型与两个提取器 ============
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()

grad_ex = GradCAMElementWise(model=model, target_layers=[model.layer4])
dfo_ex = DFOCAMCompat(model=model, target_layers=[model.layer4])
EXTRACTORS = [('GradCAM', grad_ex), ('DFOCAM', dfo_ex)]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# 白色底（归一化空间）：第十一步诊断确认它适配白底点阵图
# 注意 dtype=torch.float32 必须显式声明：white_c 来自 numpy 数组是双精度，
# torch.tensor 会顺水推舟建成 Double，喂给模型直接报 "expected Double but found Float"
white_c = [(1.0 - m) / s for m, s in zip(MEAN, STD)]
BASE_WHITE = torch.tensor(white_c, dtype=torch.float32).view(1, 3, 1, 1)


def auc_of_curve(logits_arr):
    lo = logits_arr.min()
    denom = logits_arr.max() - lo
    if denom < 1e-6:
        return None
    return float(np.trapezoid((logits_arr - lo) / denom,
                              np.linspace(0, 1, NUM_STEPS + 1)))


def perturbed_batch(x, order, mode):
    base = BASE_WHITE.expand(-1, -1, H, W).clone()
    xs = []
    total = H * W
    for step in range(NUM_STEPS + 1):
        k = int(step / NUM_STEPS * total)
        mask = torch.zeros(1, H, W)
        rc = np.unravel_index(order[:k], (H, W))
        mask[0, rc[0], rc[1]] = 1.0
        if mode == 'insertion':
            pert = base * (1 - mask) + x * mask
        else:
            pert = x * (1 - mask) + base * mask
        xs.append(pert)
    return torch.cat(xs, dim=0)


# ============ 自检：第一张图先验证 DFOCAM 能跑通，不行就快速失败 ============
_first = 'data/ptbxl/images/af/' + sorted(os.listdir('data/ptbxl/images/af'))[0]
_x = to_tensor(Image.open(_first).convert('RGB').resize((224, 224))).unsqueeze(0)
with torch.no_grad():
    _pred = model(_x).argmax(1).item()
try:
    t0 = time.time()
    _cam = dfo_ex(input_tensor=_x, targets=[ClassifierOutputTarget(_pred)])
    print(f'自检通过：DFOCAM 单张用时 {time.time()-t0:.1f}s', flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit('DFOCAM 自检失败，终止（看上方报错）')

# ============ 主循环 ============
rng = np.random.RandomState(0)
aucs = {name: {'del_model': [], 'del_rand': [], 'ins_model': [], 'ins_rand': []}
        for name, _ in EXTRACTORS}
curves_raw = {name: {'del_model': [], 'del_rand': [], 'ins_model': [], 'ins_rand': []}
              for name, _ in EXTRACTORS}          # 画平均曲线用
aai = {name: {'ad': [], 'ai': [], 'ag': []} for name, _ in EXTRACTORS}
n_used = 0
demo = {}          # 留 1 张房颤图：两种方法的 CAM 叠加图

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
        y_orig = torch.softmax(model(x), dim=1)[0, gt].item()

    for name, ex in EXTRACTORS:
        cam = ex(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        sorted_idx = np.argsort(-cam_norm.flatten())
        rand_idx = rng.permutation(sorted_idx.size)

        with torch.no_grad():
            lm = model(perturbed_batch(x, sorted_idx, 'deletion'))[:, pred].numpy()
            lr_ = model(perturbed_batch(x, rand_idx, 'deletion'))[:, pred].numpy()
            li = model(perturbed_batch(x, sorted_idx, 'insertion'))[:, pred].numpy()
            l2 = model(perturbed_batch(x, rand_idx, 'insertion'))[:, pred].numpy()
        a, r = auc_of_curve(lm), auc_of_curve(lr_)
        b, c = auc_of_curve(li), auc_of_curve(l2)
        if None in (a, r, b, c):
            continue
        aucs[name]['del_model'].append(a); aucs[name]['del_rand'].append(r)
        aucs[name]['ins_model'].append(b); aucs[name]['ins_rand'].append(c)
        curves_raw[name]['del_model'].append(lm); curves_raw[name]['del_rand'].append(lr_)
        curves_raw[name]['ins_model'].append(li); curves_raw[name]['ins_rand'].append(l2)

        binm = torch.from_numpy((cam_norm > CAM_THRESHOLD).astype(np.float32))
        kept = x * binm
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        aai[name]['ad'].append(max(0.0, y_orig - y_keep) / (y_orig + 1e-8) * 100)
        aai[name]['ai'].append(1.0 if y_keep > y_orig else 0.0)
        aai[name]['ag'].append(max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100)

    if 'af_demo' not in demo and lab == 'af':
        for name, ex in EXTRACTORS:
            cam = ex(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
            demo[f'cam_{name}'] = cam
        demo['af_demo'] = np.array(img) / 255.0

    n_used += 1
    if (i + 1) % 10 == 0:
        print(f'progress {i+1}/{len(files)}', flush=True)

print(f'used={n_used}', flush=True)

# ============ 汇总 ============
S = {name: {k: (float(np.mean(v)), float(np.std(v)))
            for k, v in aucs[name].items()} for name in aucs}
AAI = {name: {k: float(np.mean(v)) for k, v in aai[name].items()} for name in aai}
rand_del = float(np.mean([aucs[n]['del_rand'] for n in aucs][0]))
rand_ins = float(np.mean([aucs[n]['ins_rand'] for n in aucs][0]))

lines = ['=== Grad-CAM vs DFO-CAM 忠实性对比（PTB-XL 子集 100 张，白底协议）===']
lines.append(f'有效样本 {n_used} 张；底图=白色（第十一步诊断后的适配协议）')
lines.append('')
lines.append('--- Deletion / Insertion AUC（越小/越大越忠实）---')
lines.append(f'随机顺序基线      : del {rand_del:.4f} / ins {rand_ins:.4f}')
for name in aucs:
    s = S[name]
    lines.append(f'{name:<8} 按热力图 : del {s["del_model"][0]:.4f} ± {s["del_model"][1]:.4f} / '
                 f'ins {s["ins_model"][0]:.4f} ± {s["ins_model"][1]:.4f}')
lines.append('')
lines.append('--- AD / AI / AG（DFO-CAM 协议）---')
for name in aai:
    a = AAI[name]
    lines.append(f'{name:<8} AD {a["ad"]:.2f}%  AI {a["ai"]*100:.2f}%  AG {a["ag"]:.2f}%')
g, d = S['GradCAM'], S['DFOCAM']
lines.append('')
lines.append(f'结论（Deletion，越小越好）: GradCAM {g["del_model"][0]:.4f} vs DFOCAM {d["del_model"][0]:.4f}'
             f' -> {"DFOCAM 更忠实" if d["del_model"][0] < g["del_model"][0] else "GradCAM 更忠实"}')
lines.append(f'结论（Insertion，越大越好）: GradCAM {g["ins_model"][0]:.4f} vs DFOCAM {d["ins_model"][0]:.4f}'
             f' -> {"DFOCAM 更忠实" if d["ins_model"][0] > g["ins_model"][0] else "GradCAM 更忠实"}')
open('结果/DFOCAM对比结果.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('results saved', flush=True)

# ============ 画图：两面板对比曲线 ============
C_G, C_D, C_R = '#2563EB', '#E8590C', '#6B7280'
grid = np.linspace(0, 1, NUM_STEPS + 1)

mean_curves = {name: {k: np.mean(np.stack(v), axis=0)
                      for k, v in curves_raw[name].items()}
               for name in curves_raw}
rand_del_curve = np.mean(np.stack(curves_raw['GradCAM']['del_rand'] +
                                  curves_raw['DFOCAM']['del_rand']), axis=0)
rand_ins_curve = np.mean(np.stack(curves_raw['GradCAM']['ins_rand'] +
                                  curves_raw['DFOCAM']['ins_rand']), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
ax.plot(grid, mean_curves['GradCAM']['del_model'], color=C_G, lw=2, label='GradCAM')
ax.plot(grid, mean_curves['DFOCAM']['del_model'], color=C_D, lw=2, label='DFOCAM（DFO-CAM方法）')
ax.plot(grid, rand_del_curve, color=C_R, lw=2, ls='--', label='随机顺序')
ax.set_xlabel('已删除像素的比例')
ax.set_ylabel('目标类 logit（单图归一化）')
ax.set_title(f'Deletion（AUC 越小越忠实）\n'
             f'GradCAM {S["GradCAM"]["del_model"][0]:.3f} | '
             f'DFOCAM {S["DFOCAM"]["del_model"][0]:.3f} | 随机 {rand_del:.3f}')
ax.legend(loc='upper right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
ax.plot(grid, mean_curves['GradCAM']['ins_model'], color=C_G, lw=2, label='GradCAM')
ax.plot(grid, mean_curves['DFOCAM']['ins_model'], color=C_D, lw=2, label='DFOCAM（DFO-CAM方法）')
ax.plot(grid, rand_ins_curve, color=C_R, lw=2, ls='--', label='随机顺序')
ax.set_xlabel('已放回像素的比例')
ax.set_ylabel('目标类 logit（单图归一化）')
ax.set_title(f'Insertion（AUC 越大越忠实）\n'
             f'GradCAM {S["GradCAM"]["ins_model"][0]:.3f} | '
             f'DFOCAM {S["DFOCAM"]["ins_model"][0]:.3f} | 随机 {rand_ins:.3f}')
ax.legend(loc='lower right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)

plt.tight_layout()
plt.savefig('图片/第十二步_对比曲线.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# ============ 示意图：同一张房颤图，两种方法的热力图 ============
if 'af_demo' in demo:
    fig2, axes2 = plt.subplots(1, 3, figsize=(9.6, 3.2))
    axes2[0].imshow(demo['af_demo'])
    axes2[0].set_title('原始庞加莱图（房颤）', fontsize=10)
    for j, name in enumerate(['GradCAM', 'DFOCAM']):
        cam = demo[f'cam_{name}']
        cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        axes2[j + 1].imshow(demo['af_demo'])
        axes2[j + 1].imshow(cam_n, cmap='jet', alpha=0.45)
        axes2[j + 1].set_title(f'{name} 热力图', fontsize=10)
    for ax2 in axes2:
        ax2.axis('off')
    plt.tight_layout()
    plt.savefig('图片/第十二步_CAM对比示例.png', dpi=120, bbox_inches='tight')
    plt.close(fig2)

print('done', flush=True)
