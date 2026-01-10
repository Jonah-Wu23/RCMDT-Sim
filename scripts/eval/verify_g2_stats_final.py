#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_g2_stats_final.py - G2实验统计数据最终核查
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval.metrics_v4 import AuditConfig, apply_audit_rule_c

print("=" * 70)
print("G2 实验统计数据最终核查")
print("=" * 70)
print()

# 1. 验证真实数据来源
print("【1. 真实数据验证】")
real_stats_path = PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
df_real = pd.read_csv(real_stats_path)
print(f"数据文件: {real_stats_path.relative_to(PROJECT_ROOT)}")
print(f"场景: PM Peak (2025-12-19 17:00-18:00 HKT)")
print(f"总样本数: {len(df_real)}")
print()

# 2. 应用 Rule C 审计
print("【2. Rule C 审计 (T*=325, v*=5)】")
config = AuditConfig(T_star=325.0, v_star=5.0)
raw_speeds, clean_speeds, raw_tt, clean_tt, stats = apply_audit_rule_c(df_real, config)
print(f"n_raw: {stats.n_raw}")
print(f"n_clean: {stats.n_clean}")
print(f"n_flagged: {stats.n_flagged}")
print(f"Flagged %: {stats.flagged_fraction*100:.2f}%")
print()

# 3. 加载 G2 实验结果
print("【3. G2 A4_IES 实验结果】")
results_csv = PROJECT_ROOT / "data" / "experiments_v4" / "semantic_alignment" / "semantic_alignment_results.csv"
df_results = pd.read_csv(results_csv)
g2_ies = df_results[(df_results['group'] == 'G2') & (df_results['config'] == 'A4_IES')]

print("各 seed 结果:")
print(g2_ies[['seed', 'n_clean', 'n_sim', 'ks_speed', 'ks_tt']].to_string(index=False))
print()

# 4. 计算平均值
n_real = int(g2_ies['n_clean'].mean())
n_sim = int(g2_ies['n_sim'].mean())
ks_speed = g2_ies['ks_speed'].mean()
ks_tt = g2_ies['ks_tt'].mean()

print("【4. 平均统计量】")
print(f"n_real (clean): {n_real}")
print(f"n_sim: {n_sim}")
print(f"KS(speed): {ks_speed:.3f}")
print(f"KS(TT): {ks_tt:.3f}")
print()

# 5. 计算 Critical Value
print("【5. 统计显著性检验】")
c_alpha = 1.36  # for alpha=0.05
D_crit = c_alpha * np.sqrt((n_real + n_sim) / (n_real * n_sim))
print(f"c(α=0.05): {c_alpha}")
print(f"D_crit = {c_alpha} × √[(n_real + n_sim) / (n_real × n_sim)]")
print(f"D_crit = {c_alpha} × √[({n_real} + {n_sim}) / ({n_real} × {n_sim})]")
print(f"D_crit = {D_crit:.4f}")
print()

# 6. 判断 p-value
ks_speed_pass = ks_speed < D_crit
ks_tt_pass = ks_tt < D_crit

print("【6. KS 检验结果】")
print(f"KS(speed) = {ks_speed:.3f}")
print(f"KS(speed) < D_crit: {ks_speed_pass}")
print(f"→ p {'≥' if ks_speed_pass else '<'} 0.05")
print()
print(f"KS(TT) = {ks_tt:.3f}")
print(f"KS(TT) < D_crit: {ks_tt_pass}")
print(f"→ p {'≥' if ks_tt_pass else '<'} 0.05")
print()

# 7. 最终确认数字
print("=" * 70)
print("【最终确认数字 - 用于论文】")
print("=" * 70)
print(f"1. n_real (clean): {n_real}")
print(f"2. n_sim: {n_sim}")
print(f"3. D_crit: {D_crit:.4f}")
print(f"4. KS(speed): {ks_speed:.3f}")
print(f"5. KS(TT): {ks_tt:.3f}")
print(f"6. p ≥ 0.05 判断: {ks_tt_pass}")
print()

# 8. 验证一致性
print("【7. 一致性验证】")
print(f"实验中 n_clean ({n_real}) 与审计结果 ({stats.n_clean}) 匹配: {'✓' if n_real == stats.n_clean else '✗'}")
print()
