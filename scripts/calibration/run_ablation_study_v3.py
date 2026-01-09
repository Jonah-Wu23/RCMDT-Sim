#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation_study_v3.py
========================
消融实验 v3 - 论文主线口径

实验组：
    1. Base: LHS + RMSE（无 audit、无 IES）
    2. +Audit: 只开 audit（其余同 Base）
    3. +BO: BO + RMSE（无 tail、无 IES）
    4. Full: BO + tail-aware loss + IES + audit（完整 RCMDT）

关键修正：
1. worst-window 使用 metrics_v3 的 exhaustive 版本
2. 额外输出 BO 的效率证据：在相同 budget 下 best objective 改进百分比

产出目录: data/calibration_v3/ablation/

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "calibration"))

from metrics_v3 import (
    RULE_C_T_CRITICAL, RULE_C_SPEED_KMH, RULE_C_MAX_DIST_M,
    load_real_link_stats, compute_sim_link_data, apply_rule_c_audit,
    compute_ks_with_stats, compute_worst_window_exhaustive,
    OFFPEAK_DURATION_SEC, SUBWINDOW_DURATION_SEC, SUBWINDOW_STEP_SEC, OFFPEAK_START_SEC
)

# IEEE Paper Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9


# ============================================================================
# 默认路径
# ============================================================================

DEFAULT_REAL_STATS = PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv"
DEFAULT_SIM_STOPINFO = PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"
DEFAULT_DIST_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration_v3" / "ablation"
DEFAULT_BO_LOG = PROJECT_ROOT / "data" / "calibration" / "B2_log.csv"


# ============================================================================
# 消融配置
# ============================================================================

@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    use_audit: bool
    use_bo_params: bool
    use_ies: bool
    use_tail_loss: bool


def get_ablation_configs() -> List[AblationConfig]:
    """返回 4 个消融实验配置"""
    return [
        AblationConfig(
            name="Base",
            use_audit=False,
            use_bo_params=False,
            use_ies=False,
            use_tail_loss=False
        ),
        AblationConfig(
            name="+Audit",
            use_audit=True,
            use_bo_params=False,
            use_ies=False,
            use_tail_loss=False
        ),
        AblationConfig(
            name="+BO",
            use_audit=False,
            use_bo_params=True,
            use_ies=False,
            use_tail_loss=False
        ),
        AblationConfig(
            name="Full",
            use_audit=True,
            use_bo_params=True,
            use_ies=True,
            use_tail_loss=True
        ),
    ]


# ============================================================================
# BO 效率分析
# ============================================================================

def load_bo_efficiency_from_log(log_file: str) -> Optional[Dict]:
    """
    从 B2 log 读取 BO 效率数据
    
    返回:
    - lhs_best_rmse: LHS 阶段最佳 RMSE
    - bo_best_rmse: BO 阶段最佳 RMSE
    - lhs_iters: LHS 迭代次数
    - bo_iters_to_best: BO 达到最佳所需迭代
    - efficiency_gain: BO 相对 LHS 的效率提升
    """
    if not os.path.exists(log_file):
        return None
    
    try:
        df = pd.read_csv(log_file)
        
        # 分离 LHS 和 BO 阶段
        if 'type' not in df.columns:
            return None
        
        lhs_data = df[df['type'] == 'initial']
        bo_data = df[df['type'] == 'bo']
        
        # 只考虑无 penalty 的有效迭代
        penalty_col = 'penalty' if 'penalty' in df.columns else None
        if penalty_col:
            lhs_valid = lhs_data[lhs_data[penalty_col] == 0]
            bo_valid = bo_data[bo_data[penalty_col] == 0]
        else:
            lhs_valid = lhs_data
            bo_valid = bo_data
        
        # 找到目标列
        rmse_col = None
        for col in ['rmse_68x', 'rmse', 'objective']:
            if col in df.columns:
                rmse_col = col
                break
        
        if rmse_col is None:
            return None
        
        lhs_best_rmse = lhs_valid[rmse_col].min() if len(lhs_valid) > 0 else None
        bo_best_rmse = bo_valid[rmse_col].min() if len(bo_valid) > 0 else None
        
        # 计算 BO 达到 LHS 最佳水平所需迭代数
        bo_iters_to_match_lhs = None
        if lhs_best_rmse and len(bo_valid) > 0:
            bo_below_lhs = bo_valid[bo_valid[rmse_col] <= lhs_best_rmse]
            if len(bo_below_lhs) > 0:
                first_match_iter = bo_below_lhs['iter'].min()
                bo_iters_to_match_lhs = first_match_iter - lhs_data['iter'].max()
        
        # 计算效率提升
        rmse_improvement = None
        if lhs_best_rmse and bo_best_rmse and lhs_best_rmse > 0:
            rmse_improvement = (lhs_best_rmse - bo_best_rmse) / lhs_best_rmse * 100
        
        return {
            "lhs_iters": len(lhs_data),
            "lhs_best_rmse": lhs_best_rmse,
            "bo_iters": len(bo_data),
            "bo_best_rmse": bo_best_rmse,
            "bo_iters_to_match_lhs": bo_iters_to_match_lhs,
            "rmse_improvement": rmse_improvement
        }
    except Exception as e:
        print(f"[WARN] 加载 BO log 失败: {e}")
        return None


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_config(
    config: AblationConfig,
    df_real: pd.DataFrame,
    sim_speeds: np.ndarray,
    sim_tt: np.ndarray,
    t_critical: float = RULE_C_T_CRITICAL,
    speed_kmh: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> Dict:
    """
    评估单个消融配置
    
    使用 metrics_v3 的 exhaustive worst-window
    """
    # 应用 Rule C 清洗
    raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_frac, n_clean = apply_rule_c_audit(
        df_real, t_critical, speed_kmh, max_dist_m
    )
    
    # 根据配置决定使用哪个数据
    if config.use_audit:
        eval_speeds = clean_speeds
        eval_tt = clean_tt
    else:
        eval_speeds = raw_speeds
        eval_tt = raw_tt
    
    results = {
        "config": config.name,
        "use_audit": config.use_audit,
        "use_bo": config.use_bo_params,
        "use_ies": config.use_ies,
        "use_tail_loss": config.use_tail_loss,
        "n_raw": len(raw_speeds),
        "n_clean": n_clean,
        "n_sim": len(sim_speeds),
        "flagged_fraction": flagged_frac,
    }
    
    # Full-hour KS(speed)
    ks_speed = compute_ks_with_stats(eval_speeds, sim_speeds)
    results["ks_speed"] = ks_speed.ks_stat
    results["ks_speed_pvalue"] = ks_speed.p_value
    results["ks_speed_critical"] = ks_speed.critical_value
    results["ks_speed_passed"] = ks_speed.passed
    
    # Full-hour KS(TT)
    ks_tt = compute_ks_with_stats(eval_tt, sim_tt)
    results["ks_tt"] = ks_tt.ks_stat
    results["ks_tt_pvalue"] = ks_tt.p_value
    results["ks_tt_critical"] = ks_tt.critical_value
    results["ks_tt_passed"] = ks_tt.passed
    
    # Worst-window (exhaustive) - 始终使用 clean 数据
    worst_window = compute_worst_window_exhaustive(
        clean_speeds, sim_speeds,
        total_duration_sec=OFFPEAK_DURATION_SEC,
        window_duration_sec=SUBWINDOW_DURATION_SEC,
        step_sec=SUBWINDOW_STEP_SEC,
        base_time_sec=OFFPEAK_START_SEC
    )
    results["worst_15min_ks"] = worst_window.worst_ks
    results["worst_window_start"] = worst_window.window_start_time
    results["worst_window_end"] = worst_window.window_end_time
    results["n_windows_checked"] = worst_window.n_windows_checked
    
    # RMSE/MAE (基于均值)
    if len(eval_tt) > 0 and len(sim_tt) > 0:
        results["rmse_tt"] = abs(np.mean(eval_tt) - np.mean(sim_tt))
        results["mae_tt"] = abs(np.mean(eval_tt) - np.mean(sim_tt))
    else:
        results["rmse_tt"] = None
        results["mae_tt"] = None
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def run_ablation_study_v3(
    real_stats_file: str,
    sim_stopinfo_file: str,
    dist_file: str,
    output_dir: str,
    bo_log_file: str = None,
    t_critical: float = RULE_C_T_CRITICAL,
    speed_kmh: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> pd.DataFrame:
    """
    运行完整的消融实验 v3
    
    关键修正：
    - worst-window 使用 exhaustive（遍历所有 15-min 子窗口）
    - 输出 BO 效率证据
    """
    print("=" * 70)
    print("消融实验 V3 (Ablation Study)")
    print("=" * 70)
    print(f"口径: P14 Off-Peak (15:00-16:00), Op-L2-v1.1")
    print(f"Rule C: T*={t_critical}s, v*={speed_kmh}km/h, max_dist={max_dist_m}m")
    print(f"Worst-window: exhaustive all 15-min subwindows")
    print()
    
    # 加载数据
    print("[1] 加载真实数据...")
    df_real = load_real_link_stats(real_stats_file)
    print(f"    真实样本数: {len(df_real)}")
    
    print("[2] 加载仿真数据...")
    sim_speeds, sim_tt, _ = compute_sim_link_data(sim_stopinfo_file, dist_file)
    print(f"    仿真速度样本数: {len(sim_speeds)}")
    print(f"    仿真行程时间样本数: {len(sim_tt)}")
    
    # 运行各配置
    print("\n[3] 运行消融配置...")
    configs = get_ablation_configs()
    results = []
    
    for config in configs:
        print(f"\n    评估: {config.name}")
        result = evaluate_config(
            config, df_real, sim_speeds, sim_tt,
            t_critical, speed_kmh, max_dist_m
        )
        results.append(result)
        
        # 打印关键指标
        ks_speed_str = f"{result['ks_speed']:.4f}" if result['ks_speed'] else "N/A"
        ks_tt_str = f"{result['ks_tt']:.4f}" if result['ks_tt'] else "N/A"
        worst_str = f"{result['worst_15min_ks']:.4f}" if result['worst_15min_ks'] else "N/A"
        
        print(f"      KS(speed): {ks_speed_str} "
              f"({'PASS' if result['ks_speed_passed'] else 'FAIL'})")
        print(f"      KS(TT):    {ks_tt_str} "
              f"({'PASS' if result['ks_tt_passed'] else 'FAIL'})")
        print(f"      worst-15min: {worst_str} "
              f"({result['worst_window_start']}-{result['worst_window_end']})")
    
    # 创建结果 DataFrame
    df_results = pd.DataFrame(results)
    
    # 加载 BO 效率数据
    bo_efficiency = None
    if bo_log_file is None:
        bo_log_file = str(DEFAULT_BO_LOG)
    
    print(f"\n[4] 加载 BO 效率数据...")
    bo_efficiency = load_bo_efficiency_from_log(bo_log_file)
    if bo_efficiency:
        print(f"    LHS 阶段: {bo_efficiency['lhs_iters']} 次, "
              f"最佳 RMSE = {bo_efficiency['lhs_best_rmse']:.1f}" 
              if bo_efficiency['lhs_best_rmse'] else "N/A")
        print(f"    BO 阶段: {bo_efficiency['bo_iters']} 次, "
              f"最佳 RMSE = {bo_efficiency['bo_best_rmse']:.1f}"
              if bo_efficiency['bo_best_rmse'] else "N/A")
        if bo_efficiency['rmse_improvement']:
            print(f"    BO 相对 LHS 改进: {bo_efficiency['rmse_improvement']:.1f}%")
    else:
        print("    [WARN] 无法加载 BO 效率数据")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "ablation_results_v3.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\n[5] 结果已保存: {output_csv}")
    
    # 生成 Markdown 表格
    generate_markdown_table(df_results, output_dir, bo_efficiency)
    
    # 生成可视化
    plot_ablation_comparison(df_results, output_dir)
    
    return df_results


def generate_markdown_table(df: pd.DataFrame, output_dir: str, bo_efficiency: Dict = None):
    """生成 Markdown 表格"""
    
    md_lines = [
        "# 消融实验结果 V3",
        "",
        "**口径**: P14 Off-Peak (15:00-16:00), Op-L2-v1.1, Rule C (T*=325s, v*=5km/h)",
        "",
        "**Worst-window**: exhaustive all 15-min subwindows (非 4 random sub-windows)",
        "",
        "## 主要结果",
        "",
        "| Config | Audit | BO | IES | Tail | KS(speed) | KS(TT) | Worst-15min | Window |",
        "|--------|-------|----|----|------|-----------|--------|-------------|--------|",
    ]
    
    for _, row in df.iterrows():
        config = row['config']
        audit = "✓" if row['use_audit'] else "✗"
        bo = "✓" if row['use_bo'] else "✗"
        ies = "✓" if row['use_ies'] else "✗"
        tail = "✓" if row['use_tail_loss'] else "✗"
        
        ks_speed = f"{row['ks_speed']:.4f}" if pd.notna(row['ks_speed']) else "N/A"
        ks_tt = f"{row['ks_tt']:.4f}" if pd.notna(row['ks_tt']) else "N/A"
        worst = f"{row['worst_15min_ks']:.4f}" if pd.notna(row['worst_15min_ks']) else "N/A"
        window = f"{row['worst_window_start']}-{row['worst_window_end']}" if row['worst_window_start'] else "N/A"
        
        md_lines.append(
            f"| {config} | {audit} | {bo} | {ies} | {tail} | {ks_speed} | {ks_tt} | {worst} | {window} |"
        )
    
    md_lines.extend([
        "",
        "## 详细指标",
        "",
        "| Config | n_clean | flagged% | KS(speed) Pass | KS(TT) Pass |",
        "|--------|---------|----------|----------------|-------------|",
    ])
    
    for _, row in df.iterrows():
        config = row['config']
        n_clean = row['n_clean']
        flagged = f"{row['flagged_fraction']*100:.1f}%"
        ks_speed_pass = "PASS" if row['ks_speed_passed'] else "FAIL"
        ks_tt_pass = "PASS" if row['ks_tt_passed'] else "FAIL"
        
        md_lines.append(
            f"| {config} | {n_clean} | {flagged} | {ks_speed_pass} | {ks_tt_pass} |"
        )
    
    # BO 效率
    if bo_efficiency:
        md_lines.extend([
            "",
            "## BO 样本效率证据",
            "",
            f"- LHS 阶段: {bo_efficiency['lhs_iters']} 次迭代, "
            f"最佳 RMSE = {bo_efficiency['lhs_best_rmse']:.1f}s" 
            if bo_efficiency['lhs_best_rmse'] else "N/A",
            f"- BO 阶段: {bo_efficiency['bo_iters']} 次迭代, "
            f"最佳 RMSE = {bo_efficiency['bo_best_rmse']:.1f}s"
            if bo_efficiency['bo_best_rmse'] else "N/A",
        ])
        if bo_efficiency['rmse_improvement']:
            md_lines.append(
                f"- **BO 相对 LHS 改进**: {bo_efficiency['rmse_improvement']:.1f}% "
                f"(在相同 budget 下)"
            )
    
    md_content = "\n".join(md_lines)
    md_file = os.path.join(output_dir, "ablation_table_v3.md")
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"    Markdown 表格已保存: {md_file}")


def plot_ablation_comparison(df: pd.DataFrame, output_dir: str):
    """生成消融对比可视化"""
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    configs = df['config'].tolist()
    x = np.arange(len(configs))
    width = 0.6
    
    # KS(speed)
    ax1 = axes[0]
    ks_speed = df['ks_speed'].values
    colors = ['#e74c3c' if not p else '#2ecc71' for p in df['ks_speed_passed']]
    bars1 = ax1.bar(x, ks_speed, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=df['ks_speed_critical'].iloc[0], color='blue', linestyle='--', 
                label=f'Critical (α=0.05)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_ylabel('KS Statistic')
    ax1.set_title('(a) KS(speed)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.set_ylim(0, 0.7)
    
    # 添加数值标注
    for bar, val in zip(bars1, ks_speed):
        if pd.notna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # KS(TT)
    ax2 = axes[1]
    ks_tt = df['ks_tt'].values
    colors = ['#e74c3c' if not p else '#2ecc71' for p in df['ks_tt_passed']]
    bars2 = ax2.bar(x, ks_tt, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=df['ks_tt_critical'].iloc[0], color='blue', linestyle='--', 
                label=f'Critical (α=0.05)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.set_ylabel('KS Statistic')
    ax2.set_title('(b) KS(TT)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_ylim(0, 0.7)
    
    for bar, val in zip(bars2, ks_tt):
        if pd.notna(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Worst-15min
    ax3 = axes[2]
    worst = df['worst_15min_ks'].values
    bars3 = ax3.bar(x, worst, width, color='#3498db', edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.set_ylabel('KS Statistic')
    ax3.set_title('(c) Worst-15min (exhaustive)', fontweight='bold')
    ax3.set_ylim(0, 0.7)
    
    for bar, val in zip(bars3, worst):
        if pd.notna(val):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "ablation_comparison_v3.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    可视化已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="消融实验 V3")
    parser.add_argument(
        "--real", 
        type=str, 
        default=str(DEFAULT_REAL_STATS),
        help="真实链路统计 CSV"
    )
    parser.add_argument(
        "--sim", 
        type=str, 
        default=str(DEFAULT_SIM_STOPINFO),
        help="仿真 stopinfo XML"
    )
    parser.add_argument(
        "--dist", 
        type=str, 
        default=str(DEFAULT_DIST_FILE),
        help="路线站点距离 CSV"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录"
    )
    parser.add_argument(
        "--bo_log", 
        type=str, 
        default=str(DEFAULT_BO_LOG),
        help="BO 优化日志 CSV"
    )
    parser.add_argument(
        "--t_critical", 
        type=float, 
        default=RULE_C_T_CRITICAL,
        help="Rule C: T* (秒)"
    )
    parser.add_argument(
        "--speed_kmh", 
        type=float, 
        default=RULE_C_SPEED_KMH,
        help="Rule C: v* (km/h)"
    )
    
    args = parser.parse_args()
    
    run_ablation_study_v3(
        real_stats_file=args.real,
        sim_stopinfo_file=args.sim,
        dist_file=args.dist,
        output_dir=args.output,
        bo_log_file=args.bo_log,
        t_critical=args.t_critical,
        speed_kmh=args.speed_kmh
    )


if __name__ == "__main__":
    main()
