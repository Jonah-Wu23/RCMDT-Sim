#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_g2_stats.py - 验证G2实验的统计数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 1. 加载G2实验结果
results_csv = PROJECT_ROOT / "data" / "experiments_v4" / "semantic_alignment" / "semantic_alignment_results.csv"
df = pd.read_csv(results_csv)

# 提取G2 A4_IES的结果
g2_ies = df[(df['group'] == 'G2') & (df['config'] == 'A4_IES')]

print("=" * 70)
print("G2 实验统计数据核查")
print("=" * 70)
print()

print("【G2 A4_IES 各seed结果】")
print(g2_ies[['seed', 'n_clean', 'n_sim', 'ks_speed', 'ks_tt']].to_string(index=False))
print()

# 2. 计算平均值
n_clean_avg = g2_ies['n_clean'].mean()
n_sim_avg = g2_ies['n_sim'].mean()
ks_speed_avg = g2_ies['ks_speed'].mean()
ks_tt_avg = g2_ies['ks_tt'].mean()

print("【平均值】")
print(f"n_real (clean): {n_clean_avg:.0f}")
print(f"n_sim: {n_sim_avg:.0f}")
print(f"KS(speed): {ks_speed_avg:.4f}")
print(f"KS(TT): {ks_tt_avg:.4f}")
print()

# 3. 验证n_real计算
df_real = pd.read_csv(PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv")
n_total = len(df_real)
flagged_pct = 0.4714
n_real_clean_calc = int(n_total * (1 - flagged_pct))

print("【n_real验证】")
print(f"link_stats_offpeak 总行数: {n_total}")
print(f"Flagged % (T*=325, v*=5): {flagged_pct*100:.2f}%")
print(f"n_real_clean = {n_total} × (1 - {flagged_pct}) = {n_real_clean_calc}")
print(f"实验结果中的 n_clean: {n_clean_avg:.0f}")
print(f"匹配: {'✓' if abs(n_real_clean_calc - n_clean_avg) < 1 else '✗'}")
print()

# 4. 计算Critical Value和p-value判断
n_real = int(n_clean_avg)
n_sim = int(n_sim_avg)
c_alpha = 1.36  # for alpha=0.05
D_crit = c_alpha * np.sqrt((n_real + n_sim) / (n_real * n_sim))

print("【统计显著性检验】")
print(f"n_real: {n_real}")
print(f"n_sim: {n_sim}")
print(f"c(α=0.05): {c_alpha}")
print(f"D_crit = {c_alpha} × √[(n_real + n_sim) / (n_real × n_sim)]")
print(f"D_crit = {c_alpha} × √[({n_real} + {n_sim}) / ({n_real} × {n_sim})]")
print(f"D_crit = {D_crit:.4f}")
print()

# 5. 判断p-value
ks_speed_pass = ks_speed_avg < D_crit
ks_tt_pass = ks_tt_avg < D_crit

print("【KS检验结果】")
print(f"KS(speed) = {ks_speed_avg:.4f}")
print(f"KS(speed) < D_crit: {ks_speed_pass} → p {'≥' if ks_speed_pass else '<'} 0.05")
print()
print(f"KS(TT) = {ks_tt_avg:.4f}")
print(f"KS(TT) < D_crit: {ks_tt_pass} → p {'≥' if ks_tt_pass else '<'} 0.05")
print()

# 6. 最终输出
print("=" * 70)
print("【最终确认数字】")
print("=" * 70)
print(f"1. n_real (clean): {n_real}")
print(f"2. n_sim: {n_sim}")
print(f"3. D_crit: {D_crit:.4f}")
print(f"4. KS(speed): {ks_speed_avg:.4f}")
print(f"5. KS(TT): {ks_tt_avg:.4f}")
print(f"6. p ≥ 0.05 判断: {ks_tt_pass} (基于 KS(TT) < D_crit)")
print()
