# -*- coding: utf-8 -*-
# 第十四步补充 D：双流 vs 单流消融（审稿意见 C3 的补救——证明双流机制的贡献）
# 问题：论文声称 DFO-CAM 的内容指标（AI）优势来自"双流优化"（正向流 + 负向流），
#   但缺一个直接证据：去掉负向流（单流）后，AI 是否下降？
# 方法：单流 DFO-CAM = DFO-CAM DFO-CAM 去掉负向流损失（loss = loss_forward，
#   等价于 Opti-CAM 的单流目标）。用 inspect 从DFO-CAM源码取方法、仅替换损失行，
#   保证其余与DFO-CAM逐行一致。在 300 张上对比双流 vs 单流的 AI/AG（内容指标）。
import os
import csv
import inspect
import textwrap
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'计算设备: {device}', flush=True)

N_PER_CLASS = 150       # 300 张，与主实验一致
TAU = 0.5

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('模型/resnet18_ptbxl.pth', map_location='cpu'))
model.eval()
model = model.to(device)

spec = importlib.util.spec_from_file_location(
    'dfocam', r'third_party/dfo-cam/dfo-cam.py')
dfo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfo_mod)


class DFOCAMCompat(dfo_mod.DFOCAM):
    """双流（DFO-CAM原版）兼容层。"""
    def get_cam_weights(self, input_tensor, target_layer, targets,
                        activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)
        return np.asarray(super().get_cam_weights(
            input_tensor, target_layer, targets, activations, grads),
            dtype=np.float32)


# 单流版本：inspect 拿DFO-CAM源码，仅替换损失行
src = textwrap.dedent(inspect.getsource(dfo_mod.DFOCAM.get_cam_weights))
assert 'loss = loss_forward + loss_backward' in src, 'DFO-CAM损失行未找到，检查源码'
src_single = src.replace('loss = loss_forward + loss_backward', 'loss = loss_forward')
ns = dict(dfo_mod.__dict__)
exec(src_single, ns)
single_get = ns['get_cam_weights']


class DFOCAM_SingleFlow(DFOCAMCompat):
    """单流（仅正向）版本：去掉负向流损失，等价于 Opti-CAM 的单流目标。"""
    def get_cam_weights(self, input_tensor, target_layer, targets,
                        activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        activations = np.ascontiguousarray(activations, dtype=np.float32)
        grads = np.ascontiguousarray(grads, dtype=np.float32)
        return single_get(self, input_tensor, target_layer, targets,
                          activations, grads)


dual = DFOCAMCompat(model=model, target_layers=[model.layer4])
single = DFOCAM_SingleFlow(model=model, target_layers=[model.layer4])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = []
for lab in ['normal', 'af']:
    fs = sorted(os.listdir(f'data/ptbxl/images/{lab}'))[:N_PER_CLASS]
    for f in fs:
        files.append((f'data/ptbxl/images/{lab}/{f}',
                      {'af': 0, 'normal': 1}[lab], lab))

rows = []
for i, (path, gt, lab) in enumerate(files):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits0 = model(x)
        pred = logits0.argmax(1).item()
        y_orig = torch.softmax(logits0, dim=1)[0, gt].item()

    row = {'gt': gt}
    for name, extractor in [('dual', dual), ('single', single)]:
        cam = extractor(input_tensor=x,
                        targets=[ClassifierOutputTarget(pred)])[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        kept = x * torch.from_numpy(
            (cam > TAU).astype(np.float32)).to(device)
        with torch.no_grad():
            y_keep = torch.softmax(model(kept), dim=1)[0, gt].item()
        row[f'{name}_ai'] = 1.0 if y_keep > y_orig else 0.0
        row[f'{name}_ag'] = max(0.0, y_keep - y_orig) / (1 - y_orig + 1e-8) * 100
    rows.append(row)
    if (i + 1) % 25 == 0:
        print(f'progress {i + 1}/{len(files)}', flush=True)

dual_ai = np.array([r['dual_ai'] for r in rows])
single_ai = np.array([r['single_ai'] for r in rows])
dual_ag = np.array([r['dual_ag'] for r in rows])
single_ag = np.array([r['single_ag'] for r in rows])


def mcnemar(a, b):
    b01 = int(np.sum((a == 1) & (b == 0)))
    c01 = int(np.sum((a == 0) & (b == 1)))
    n01 = b01 + c01
    p = stats.binomtest(min(b01, c01), n01, 0.5).pvalue if n01 else 1.0
    return p, b01, c01


p_ai, b01, c01 = mcnemar(dual_ai, single_ai)
t_ag, p_ag = stats.ttest_rel(dual_ag, single_ag)

lines = ['=== 双流 vs 单流消融（300 张，τ=0.5，内容指标）===',
         f'样本 {len(rows)} 张；双流=DFO-CAM DFO-CAM（正向+负向），单流=仅正向（等价 Opti-CAM）',
         '',
         f'AI %：双流 {dual_ai.mean() * 100:.1f}% vs 单流 {single_ai.mean() * 100:.1f}%',
         f'  McNemar：双流对/单流错 {b01} 张 vs 双流错/单流对 {c01} 张 -> p={p_ai:.4f}',
         f'AG %：双流 {dual_ag.mean():.2f} vs 单流 {single_ag.mean():.2f} '
         f'（配对 t 检验 p={p_ag:.4f}）',
         '',
         '解读：若双流的 AI/AG 显著高于单流，则证明"负向流（抑制次要区域）"'
         '是 DFO-CAM 内容指标优势的机制来源，3.4 节的机制解释有了直接证据；'
         '若二者相当，则说明优势来自优化本身而非双流结构，需修正论文表述。']
report = '\n'.join(lines)
print('\n' + report, flush=True)
open('结果/双流消融结果.txt', 'w', encoding='utf-8').write(report)
print('\n结果已写入 结果/双流消融结果.txt', flush=True)
