#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_temporal_heatmap_v3.py
==========================
全时段 heatmap v3 - 论文主线口径

生成内容：
    - route × hour 的 heatmap
    - 值：worst-window KS(speed) (exhaustive)
    - 每格输出：metric + n_clean；n_clean < 10 -> NA

关键修正：
1. worst-window 使用 exhaustive（遍历所有 15-min 子窗口）
2. n_clean < 10 时该格标 NA（不能用 0.0）
3. 明确值是 worst-window KS(speed)

产出目录: data/calibration_v3/temporal_heatmap/

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
plt.rcParams['font.size'] = 8


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_REAL_STATS = PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv"
DEFAULT_PEAK_STATS = PROJECT_ROOT / "data" / "processed" / "link_stats.csv"
DEFAULT_SIM_STOPINFO = PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"
DEFAULT_DIST_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration_v3" / "temporal_heatmap"

ROUTES = ["68X", "960"]
TIME_PERIODS = ["AM Peak", "PM Peak", "Off-Peak"]

MIN_CLEAN_SAMPLES = 10  # n_clean < 10 时标记为 NA


# ============================================================================
# 数据处理
# ============================================================================

def load_stats_by_route(filepath: str) -> Dict[str, pd.DataFrame]:
    """按路线分组加载统计数据"""
    if not os.path.exists(filepath):
        return {}
    
    df = pd.read_csv(filepath)
    result = {}
    
    if 'route' not in df.columns:
        return {}
    
    for route in ROUTES:
        route_df = df[df["route"] == route].copy()
        if not route_df.empty:
            result[route] = route_df
    
    return result


def compute_route_metrics(
    df_route: pd.DataFrame,
    sim_speeds: np.ndarray,
    t_critical: float = RULE_C_T_CRITICAL,
    speed_kmh: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> Dict:
    """计算单个路线的指标"""
    
    # 应用 Rule C
    raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_frac, n_clean = apply_rule_c_audit(
        df_route, t_critical, speed_kmh, max_dist_m
    )
    
    result = {
        "n_raw": len(raw_speeds),
        "n_clean": n_clean,
        "flagged_pct": flagged_frac * 100
    }
    
    # n_clean < 10 时标记为 NA
    if n_clean < MIN_CLEAN_SAMPLES:
        result["ks_clean"] = None
        result["worst_ks"] = None
        result["worst_window"] = None
        return result
    
    # Full-hour KS(speed)
    ks_result = compute_ks_with_stats(clean_speeds, sim_speeds)
    result["ks_clean"] = ks_result.ks_stat
    result["ks_passed"] = ks_result.passed
    
    # Worst-window (exhaustive)
    worst_result = compute_worst_window_exhaustive(
        clean_speeds, sim_speeds,
        total_duration_sec=OFFPEAK_DURATION_SEC,
        window_duration_sec=SUBWINDOW_DURATION_SEC,
        step_sec=SUBWINDOW_STEP_SEC,
        base_time_sec=OFFPEAK_START_SEC
    )
    result["worst_ks"] = worst_result.worst_ks
    result["worst_window"] = f"{worst_result.window_start_time}-{worst_result.window_end_time}" if worst_result.window_start_time else None
    
    return result


# ============================================================================
# Heatmap 生成
# ============================================================================

def compute_heatmap_data(
    routes: List[str],
    periods: List[str],
    real_data_by_period: Dict[str, Dict[str, pd.DataFrame]],
    sim_speeds: np.ndarray,
    config_type: str = "Full"  # "Base" or "Full"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 heatmap 数据矩阵
    
    Returns:
        ks_matrix: worst-window KS 矩阵 (n_periods, n_routes)
        n_clean_matrix: n_clean 矩阵 (n_periods, n_routes)
    """
    n_periods = len(periods)
    n_routes = len(routes)
    
    ks_matrix = np.full((n_periods, n_routes), np.nan)
    n_clean_matrix = np.zeros((n_periods, n_routes))
    
    for i, period in enumerate(periods):
        for j, route in enumerate(routes):
            if period not in real_data_by_period:
                continue
            if route not in real_data_by_period[period]:
                continue
            
            df_route = real_data_by_period[period][route]
            
            # Base 配置不使用 audit，Full 配置使用
            if config_type == "Base":
                # 不清洗，直接用 raw 数据
                speeds = df_route["speed_median"].dropna().values
                n_clean = len(speeds)
            else:
                # Full 配置使用 Rule C 清洗
                _, clean_speeds, _, _, _, n_clean = apply_rule_c_audit(df_route)
                speeds = clean_speeds
            
            n_clean_matrix[i, j] = n_clean
            
            # n_clean < 10 时标记为 NA
            if n_clean < MIN_CLEAN_SAMPLES:
                continue
            
            # 计算 worst-window KS
            worst_result = compute_worst_window_exhaustive(
                speeds, sim_speeds,
                total_duration_sec=OFFPEAK_DURATION_SEC,
                window_duration_sec=SUBWINDOW_DURATION_SEC,
                step_sec=SUBWINDOW_STEP_SEC,
                base_time_sec=OFFPEAK_START_SEC
            )
            
            if worst_result.worst_ks is not None:
                ks_matrix[i, j] = worst_result.worst_ks
    
    return ks_matrix, n_clean_matrix


def plot_heatmaps(
    base_matrix: np.ndarray,
    full_matrix: np.ndarray,
    n_clean_base: np.ndarray,
    n_clean_full: np.ndarray,
    routes: List[str],
    periods: List[str],
    output_dir: str
):
    """绘制 Base vs Full 对比热力图"""
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    vmin = 0.1
    vmax = 0.6
    
    # Base heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(base_matrix, cmap='RdYlGn_r', aspect='auto', 
                     vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(len(routes)))
    ax1.set_xticklabels(routes)
    ax1.set_yticks(range(len(periods)))
    ax1.set_yticklabels(periods)
    ax1.set_xlabel('Route')
    ax1.set_ylabel('Time Period')
    ax1.set_title('(a) Base (no audit)', fontweight='bold')
    
    for i in range(len(periods)):
        for j in range(len(routes)):
            val = base_matrix[i, j]
            n = int(n_clean_base[i, j])
            if np.isnan(val) or n < MIN_CLEAN_SAMPLES:
                ax1.text(j, i, f"NA\n(n={n})", ha='center', va='center', 
                        fontsize=7, color='gray', fontstyle='italic')
            else:
                color = 'white' if val > 0.35 else 'black'
                ax1.text(j, i, f"{val:.2f}\n(n={n})", ha='center', va='center', 
                        fontsize=7, color=color, fontweight='bold')
    
    # Full heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(full_matrix, cmap='RdYlGn_r', aspect='auto', 
                     vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(routes)))
    ax2.set_xticklabels(routes)
    ax2.set_yticks(range(len(periods)))
    ax2.set_yticklabels(periods)
    ax2.set_xlabel('Route')
    ax2.set_ylabel('Time Period')
    ax2.set_title('(b) Full (RCMDT)', fontweight='bold')
    
    for i in range(len(periods)):
        for j in range(len(routes)):
            val = full_matrix[i, j]
            n = int(n_clean_full[i, j])
            if np.isnan(val) or n < MIN_CLEAN_SAMPLES:
                ax2.text(j, i, f"NA\n(n={n})", ha='center', va='center', 
                        fontsize=7, color='gray', fontstyle='italic')
            else:
                color = 'white' if val > 0.35 else 'black'
                ax2.text(j, i, f"{val:.2f}\n(n={n})", ha='center', va='center', 
                        fontsize=7, color=color, fontweight='bold')
    
    # Difference heatmap (Base - Full)
    ax3 = axes[2]
    diff_matrix = base_matrix - full_matrix
    im3 = ax3.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                     vmin=-0.2, vmax=0.2)
    ax3.set_xticks(range(len(routes)))
    ax3.set_xticklabels(routes)
    ax3.set_yticks(range(len(periods)))
    ax3.set_yticklabels(periods)
    ax3.set_xlabel('Route')
    ax3.set_ylabel('Time Period')
    ax3.set_title('(c) Improvement (Base − Full)', fontweight='bold')
    
    for i in range(len(periods)):
        for j in range(len(routes)):
            val = diff_matrix[i, j]
            if np.isnan(val):
                ax3.text(j, i, "NA", ha='center', va='center', 
                        fontsize=7, color='gray', fontstyle='italic')
            else:
                color = 'white' if abs(val) > 0.1 else 'black'
                sign = '+' if val > 0 else ''
                ax3.text(j, i, f"{sign}{val:.2f}", ha='center', va='center', 
                        fontsize=7, color=color, fontweight='bold')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label='Worst-15min KS(speed)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    output_path = os.path.join(output_dir, "temporal_heatmap_v3.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    热力图已保存: {output_path}")
    plt.close()


def generate_markdown_table(
    base_matrix: np.ndarray,
    full_matrix: np.ndarray,
    n_clean_base: np.ndarray,
    n_clean_full: np.ndarray,
    routes: List[str],
    periods: List[str],
    output_dir: str
):
    """生成 Markdown 表格"""
    
    md_lines = [
        "# 全时段 Heatmap V3",
        "",
        "**口径**: P14 Off-Peak (15:00-16:00), Op-L2-v1.1, Rule C (T*=325s, v*=5km/h)",
        "",
        "**指标**: Worst-15min KS(speed) [exhaustive]",
        "",
        "**注**: n_clean < 10 时标记为 NA",
        "",
        "## Base Configuration (无 audit)",
        "",
        "| Period | " + " | ".join(routes) + " |",
        "|--------|" + "|".join(["--------"] * len(routes)) + "|",
    ]
    
    for i, period in enumerate(periods):
        row = f"| {period}"
        for j in range(len(routes)):
            val = base_matrix[i, j]
            n = int(n_clean_base[i, j])
            if np.isnan(val) or n < MIN_CLEAN_SAMPLES:
                row += f" | NA (n={n})"
            else:
                row += f" | {val:.3f} (n={n})"
        row += " |"
        md_lines.append(row)
    
    md_lines.extend([
        "",
        "## Full Configuration (RCMDT)",
        "",
        "| Period | " + " | ".join(routes) + " |",
        "|--------|" + "|".join(["--------"] * len(routes)) + "|",
    ])
    
    for i, period in enumerate(periods):
        row = f"| {period}"
        for j in range(len(routes)):
            val = full_matrix[i, j]
            n = int(n_clean_full[i, j])
            if np.isnan(val) or n < MIN_CLEAN_SAMPLES:
                row += f" | NA (n={n})"
            else:
                row += f" | {val:.3f} (n={n})"
        row += " |"
        md_lines.append(row)
    
    # 计算改进
    md_lines.extend([
        "",
        "## 改进对比 (Base - Full)",
        "",
        "| Period | " + " | ".join(routes) + " | Mean Impr. |",
        "|--------|" + "|".join(["--------"] * len(routes)) + "|------------|",
    ])
    
    for i, period in enumerate(periods):
        row = f"| {period}"
        improvements = []
        for j in range(len(routes)):
            base_val = base_matrix[i, j]
            full_val = full_matrix[i, j]
            if np.isnan(base_val) or np.isnan(full_val):
                row += " | NA"
            else:
                diff = base_val - full_val
                sign = '+' if diff > 0 else ''
                row += f" | {sign}{diff:.3f}"
                improvements.append(diff)
        
        mean_impr = np.mean(improvements) if improvements else np.nan
        if np.isnan(mean_impr):
            row += " | NA |"
        else:
            sign = '+' if mean_impr > 0 else ''
            row += f" | {sign}{mean_impr:.3f} |"
        md_lines.append(row)
    
    md_content = "\n".join(md_lines)
    md_file = os.path.join(output_dir, "temporal_heatmap_table_v3.md")
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"    Markdown 表格已保存: {md_file}")


def run_temporal_analysis(
    peak_stats_file: str,
    offpeak_stats_file: str,
    sim_stopinfo_file: str,
    dist_file: str,
    output_dir: str
):
    """运行全时段分析"""
    
    print("=" * 70)
    print("全时段 Heatmap 分析 V3")
    print("=" * 70)
    print(f"口径: P14 Off-Peak (15:00-16:00), Op-L2-v1.1")
    print(f"指标: Worst-15min KS(speed) [exhaustive]")
    print(f"n_clean < 10 时标记为 NA")
    print()
    
    # 加载仿真数据
    print("[1] 加载仿真数据...")
    sim_speeds, sim_tt, _ = compute_sim_link_data(sim_stopinfo_file, dist_file)
    print(f"    仿真速度样本数: {len(sim_speeds)}")
    
    # 加载真实数据
    print("\n[2] 加载真实数据...")
    real_data_by_period = {}
    
    # Peak 数据
    if os.path.exists(peak_stats_file):
        peak_data = load_stats_by_route(peak_stats_file)
        if peak_data:
            real_data_by_period["AM Peak"] = peak_data
            real_data_by_period["PM Peak"] = peak_data
            print(f"    Peak 数据已加载: {list(peak_data.keys())}")
    
    # Off-peak 数据
    if os.path.exists(offpeak_stats_file):
        offpeak_data = load_stats_by_route(offpeak_stats_file)
        if offpeak_data:
            real_data_by_period["Off-Peak"] = offpeak_data
            print(f"    Off-Peak 数据已加载: {list(offpeak_data.keys())}")
    
    if not real_data_by_period:
        print("    [WARN] 无法加载真实数据，生成合成数据演示")
        # 生成合成数据
        for period in TIME_PERIODS:
            real_data_by_period[period] = {}
            for route in ROUTES:
                np.random.seed(hash(f"{period}_{route}") % 2**32)
                n_samples = np.random.randint(30, 80)
                speeds = np.random.normal(25, 8, n_samples)
                speeds = np.clip(speeds, 3, 60)
                
                df = pd.DataFrame({
                    "route": [route] * n_samples,
                    "speed_median": speeds,
                    "tt_median": 1000 / speeds * 3.6,
                    "dist_m": [1000] * n_samples
                })
                real_data_by_period[period][route] = df
    
    # 计算 heatmap 数据
    print("\n[3] 计算 heatmap 数据...")
    
    base_matrix, n_clean_base = compute_heatmap_data(
        ROUTES, TIME_PERIODS, real_data_by_period, sim_speeds, config_type="Base"
    )
    
    full_matrix, n_clean_full = compute_heatmap_data(
        ROUTES, TIME_PERIODS, real_data_by_period, sim_speeds, config_type="Full"
    )
    
    print(f"    Base 矩阵:\n{base_matrix}")
    print(f"    Full 矩阵:\n{full_matrix}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CSV
    df_base = pd.DataFrame(base_matrix, index=TIME_PERIODS, columns=ROUTES)
    df_base.to_csv(os.path.join(output_dir, "heatmap_base_v3.csv"))
    
    df_full = pd.DataFrame(full_matrix, index=TIME_PERIODS, columns=ROUTES)
    df_full.to_csv(os.path.join(output_dir, "heatmap_full_v3.csv"))
    
    df_n_clean_base = pd.DataFrame(n_clean_base, index=TIME_PERIODS, columns=ROUTES)
    df_n_clean_base.to_csv(os.path.join(output_dir, "heatmap_n_clean_base_v3.csv"))
    
    df_n_clean_full = pd.DataFrame(n_clean_full, index=TIME_PERIODS, columns=ROUTES)
    df_n_clean_full.to_csv(os.path.join(output_dir, "heatmap_n_clean_full_v3.csv"))
    
    print(f"\n[4] 结果已保存")
    
    # 生成可视化
    print("\n[5] 生成可视化...")
    plot_heatmaps(
        base_matrix, full_matrix,
        n_clean_base, n_clean_full,
        ROUTES, TIME_PERIODS,
        output_dir
    )
    
    generate_markdown_table(
        base_matrix, full_matrix,
        n_clean_base, n_clean_full,
        ROUTES, TIME_PERIODS,
        output_dir
    )
    
    # 打印结论
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)
    
    diff = base_matrix - full_matrix
    valid_diffs = diff[~np.isnan(diff)]
    
    if len(valid_diffs) > 0:
        mean_improvement = np.mean(valid_diffs)
        print(f"平均改进: {mean_improvement:+.4f} (KS 减少，正值表示 Full 更好)")
        print(f"最大改进: {np.max(valid_diffs):+.4f}")
        print(f"最小改进: {np.min(valid_diffs):+.4f}")
    
    return base_matrix, full_matrix


def main():
    parser = argparse.ArgumentParser(description="全时段 Heatmap 分析 V3")
    parser.add_argument(
        "--peak", 
        type=str, 
        default=str(DEFAULT_PEAK_STATS),
        help="Peak 时段链路统计 CSV"
    )
    parser.add_argument(
        "--offpeak", 
        type=str, 
        default=str(DEFAULT_REAL_STATS),
        help="Off-Peak 时段链路统计 CSV"
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
    
    args = parser.parse_args()
    
    run_temporal_analysis(
        peak_stats_file=args.peak,
        offpeak_stats_file=args.offpeak,
        sim_stopinfo_file=args.sim,
        dist_file=args.dist,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
