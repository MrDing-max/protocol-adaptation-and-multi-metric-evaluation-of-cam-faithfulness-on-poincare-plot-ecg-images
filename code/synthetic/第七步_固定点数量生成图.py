# -*- coding: utf-8 -*-
# 第七步（1/2）：固定每张图的点数量，重新生成庞加莱图（剔除"点多少"这个线索）
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import wfdb
import neurokit2 as nk

N_RR = 20            # 每张图固定用 20 个 RR 间期（19 个点）
NORMAL = ['100', '101']
AF = ['201', '202']
WINDOW_SEC = 30

OUT = 'data/fixed'
os.makedirs(f'{OUT}/normal', exist_ok=True)
os.makedirs(f'{OUT}/af', exist_ok=True)


def gen(seg, fs, idx, label, rec, out_dir):
    signals, info = nk.ecg_process(seg, sampling_rate=fs)
    rpeaks = info['ECG_R_Peaks']
    rr = np.diff(rpeaks) / fs * 1000
    if len(rr) < N_RR:               # 心拍不够就跳过
        return 0
    rr_sel = rr[:N_RR]               # 取前 20 个，保持相邻关系
    fig = plt.figure(figsize=(3, 3), dpi=100)
    plt.scatter(rr_sel[:-1], rr_sel[1:], s=14, alpha=0.75, color='k')
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0.2)
    fig.savefig(f'{out_dir}/{label}_{rec}_{idx:04d}.png')
    plt.close(fig)
    return 1


def process(rec, label, out_dir):
    record = wfdb.rdrecord(rec, pn_dir='mitdb')
    ecg = record.p_signal[:, 0]
    fs = record.fs
    win = int(WINDOW_SEC * fs)
    n = 0
    for start in range(0, len(ecg) - win + 1, win):
        n += gen(ecg[start:start + win], fs, n, label, rec, out_dir)
    print(f'  {rec} ({label}): {n} 张')
    return n


total = 0
print('生成"固定点数量"的正常图 ...')
for r in NORMAL:
    total += process(r, 'normal', f'{OUT}/normal')
print('生成"固定点数量"的房颤图 ...')
for r in AF:
    total += process(r, 'af', f'{OUT}/af')
print(f'完成，共 {total} 张。')
