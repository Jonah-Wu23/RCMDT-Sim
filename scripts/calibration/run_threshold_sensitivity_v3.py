#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_threshold_sensitivity_v3.py
===============================
Audit 阈值敏感性实验 v3 - 论文主线口径

二维网格:
    v* ∈ {3, 4, 5, 6, 7} km/h
    T* ∈ {250, 300, 325, 350, 400} s

每个点报告:
    - flagged_pct: 被标记比例
    - n_clean: 清洗后样本数
    - KS_clean(speed): full-hour KS(speed)
    - worst_window_ks: exhaustive worst-window KS(speed)

关键修正：
1. worst-window 使用 exhaustive（遍历所有 15-min 子窗口）
2. 热力图明确标注是 KS(speed)
3. n_clean < 10 时标记为 NA

产出目录: data/calibration_v3/sensitivity/

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
from typing import Dict, List, Tuple

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
DEFAULT_SIM_STOPINFO = PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"
DEFAULT_DIST_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration_v3" / "sensitivity"

# 敏感性网格
V_STAR_GRID = [3, 4, 5, 6, 7]  # km/h
T_STAR_GRID = [250, 300, 325, 350, 400]  # seconds

MIN_CLEAN_SAMPLES = 10  # n_clean < 10 时标记为 NA


# ============================================================================
# 敏感性分析
# ============================================================================

def run_sensitivity_analysis(
    df_real: pd.DataFrame,
    sim_speeds: np.ndarray,
    v_star_grid: List[float] = V_STAR_GRID,
    t_star_grid: List[float] = T_STAR_GRID,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> pd.DataFrame:
    """
    运行二维敏感性分析
    
    使用 exhaustive worst-window
    """
    results = []
    
    for t_star in t_star_grid:
        for v_star in v_star_grid:
            # 应用 Rule C
            raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_frac, n_clean = apply_rule_c_audit(
                df_real, t_star, v_star, max_dist_m
            )
            
            # Full-hour KS(speed)
            if n_clean >= MIN_CLEAN_SAMPLES:
                ks_result = compute_ks_with_stats(clean_speeds, sim_speeds)
                ks_clean = ks_result.ks_stat
                ks_passed = ks_result.passed
                
                # Worst-window (exhaustive)
                worst_result = compute_worst_window_exhaustive(
                    clean_speeds, sim_speeds,
                    total_duration_sec=OFFPEAK_DURATION_SEC,
                    window_duration_sec=SUBWINDOW_DURATION_SEC,
                    step_sec=SUBWINDOW_STEP_SEC,
                    base_time_sec=OFFPEAK_START_SEC
                )
                worst_ks = worst_result.worst_ks
                worst_window_time = f"{worst_result.window_start_time}-{worst_result.window_end_time}" if worst_result.window_start_time else None
            else:
                ks_clean = None
                ks_passed = None
                worst_ks = None
                worst_window_time = None
            
            results.append({
                "T_star": t_star,
                "v_star": v_star,
                "flagged_pct": flagged_frac * 100,
                "n_clean": n_clean,
                "ks_clean": ks_clean,
                "ks_passed": ks_passed,
                "worst_window_ks": worst_ks,
                "worst_window_time": worst_window_time
            })
    
    return pd.DataFrame(results)


def plot_heatmaps(
    df_results: pd.DataFrame,
    output_dir: str,
    v_star_grid: List[float] = V_STAR_GRID,
    t_star_grid: List[float] = T_STAR_GRID
):
    """生成热力图 - 明确标注为 KS(speed)"""
    
    n_v = len(v_star_grid)
    n_t = len(t_star_grid)
    
    # 准备数据矩阵
    flagged_matrix = np.zeros((n_t, n_v))
    n_clean_matrix = np.zeros((n_t, n_v))
    ks_matrix = np.full((n_t, n_v), np.nan)
    worst_matrix = np.full((n_t, n_v), np.nan)
    
    for _, row in df_results.iterrows():
        t_idx = t_star_grid.index(row["T_star"])
        v_idx = v_star_grid.index(row["v_star"])
        
        flagged_matrix[t_idx, v_idx] = row["flagged_pct"]
        n_clean_matrix[t_idx, v_idx] = row["n_clean"]
        
        if row["n_clean"] >= MIN_CLEAN_SAMPLES:
            ks_matrix[t_idx, v_idx] = row["ks_clean"] if pd.notna(row["ks_clean"]) else np.nan
            worst_matrix[t_idx, v_idx] = row["worst_window_ks"] if pd.notna(row["worst_window_ks"]) else np.nan
    
    # 创建四个子图
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    
    # 热力图 1: Flagged Fraction
    ax1 = axes[0, 0]
    im1 = ax1.imshow(flagged_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax1.set_xticks(range(n_v))
    ax1.set_xticklabels([f"{v}" for v in v_star_grid])
    ax1.set_yticks(range(n_t))
    ax1.set_yticklabels([f"{t}" for t in t_star_grid])
    ax1.set_xlabel(r'$v^*$ (km/h)')
    ax1.set_ylabel(r'$T^*$ (s)')
    ax1.set_title('(a) Flagged Fraction (%)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            ax1.text(j, i, f"{flagged_matrix[i,j]:.1f}", 
                    ha='center', va='center', fontsize=7,
                    color='white' if flagged_matrix[i,j] > 40 else 'black')
    
    # 热力图 2: n_clean
    ax2 = axes[0, 1]
    im2 = ax2.imshow(n_clean_matrix, cmap='Blues', aspect='auto', origin='lower')
    ax2.set_xticks(range(n_v))
    ax2.set_xticklabels([f"{v}" for v in v_star_grid])
    ax2.set_yticks(range(n_t))
    ax2.set_yticklabels([f"{t}" for t in t_star_grid])
    ax2.set_xlabel(r'$v^*$ (km/h)')
    ax2.set_ylabel(r'$T^*$ (s)')
    ax2.set_title('(b) n_clean (samples)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            val = int(n_clean_matrix[i,j])
            color = 'red' if val < MIN_CLEAN_SAMPLES else ('white' if val > 40 else 'black')
            ax2.text(j, i, f"{val}", 
                    ha='center', va='center', fontsize=7, color=color,
                    fontweight='bold' if val < MIN_CLEAN_SAMPLES else 'normal')
    
    # 热力图 3: KS(speed) full-hour
    ax3 = axes[1, 0]
    im3 = ax3.imshow(ks_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                     vmin=0.1, vmax=0.6)
    ax3.set_xticks(range(n_v))
    ax3.set_xticklabels([f"{v}" for v in v_star_grid])
    ax3.set_yticks(range(n_t))
    ax3.set_yticklabels([f"{t}" for t in t_star_grid])
    ax3.set_xlabel(r'$v^*$ (km/h)')
    ax3.set_ylabel(r'$T^*$ (s)')
    ax3.set_title('(c) KS(speed) Full-hour', fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            val = ks_matrix[i, j]
            if np.isnan(val):
                ax3.text(j, i, "NA", ha='center', va='center', fontsize=7, 
                        color='gray', fontstyle='italic')
            else:
                ax3.text(j, i, f"{val:.2f}", 
                        ha='center', va='center', fontsize=7,
                        color='white' if val > 0.35 else 'black')
    
    # 热力图 4: worst-window KS(speed) (exhaustive)
    ax4 = axes[1, 1]
    im4 = ax4.imshow(worst_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                     vmin=0.1, vmax=0.6)
    ax4.set_xticks(range(n_v))
    ax4.set_xticklabels([f"{v}" for v in v_star_grid])
    ax4.set_yticks(range(n_t))
    ax4.set_yticklabels([f"{t}" for t in t_star_grid])
    ax4.set_xlabel(r'$v^*$ (km/h)')
    ax4.set_ylabel(r'$T^*$ (s)')
    ax4.set_title('(d) Worst-15min KS(speed) [exhaustive]', fontweight='bold')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            val = worst_matrix[i, j]
            if np.isnan(val):
                ax4.text(j, i, "NA", ha='center', va='center', fontsize=7, 
                        color='gray', fontstyle='italic')
            else:
                ax4.text(j, i, f"{val:.2f}", 
                        ha='center', va='center', fontsize=7,
                        color='white' if val > 0.35 else 'black')
    
    # 标记论文选择的阈值 (T*=325, v*=5)
    if 325 in t_star_grid and 5 in v_star_grid:
        t_idx = t_star_grid.index(325)
        v_idx = v_star_grid.index(5)
        for ax in [ax1, ax2, ax3, ax4]:
            rect = plt.Rectangle(
                (v_idx - 0.5, t_idx - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "threshold_sensitivity_heatmap_v3.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    热力图已保存: {output_path}")
    plt.close()


def generate_markdown_table(df_results: pd.DataFrame, output_dir: str):
    """生成 Markdown 表格"""
    
    md_lines = [
        "# 阈值敏感性分析 V3",
        "",
        "**口径**: P14 Off-Peak (15:00-16:00), Op-L2-v1.1",
        "",
        "**Worst-window**: exhaustive all 15-min subwindows",
        "",
        "**注**: n_clean < 10 时，KS 标记为 NA",
        "",
        "## 完整结果",
        "",
        "| T* (s) | v* (km/h) | Flagged (%) | n_clean | KS(speed) | Worst-15min KS | Window |",
        "|--------|-----------|-------------|---------|-----------|----------------|--------|",
    ]
    
    for _, row in df_results.iterrows():
        t_star = int(row['T_star'])
        v_star = int(row['v_star'])
        flagged = f"{row['flagged_pct']:.1f}"
        n_clean = int(row['n_clean'])
        
        # n_clean < 10 时标记为 NA
        if n_clean < MIN_CLEAN_SAMPLES:
            ks_clean = "NA"
            worst = "NA"
            window = "NA"
        else:
            ks_clean = f"{row['ks_clean']:.4f}" if pd.notna(row['ks_clean']) else "NA"
            worst = f"{row['worst_window_ks']:.4f}" if pd.notna(row['worst_window_ks']) else "NA"
            window = row['worst_window_time'] if row['worst_window_time'] else "NA"
        
        # 高亮论文选择
        if t_star == 325 and v_star == 5:
            md_lines.append(
                f"| **{t_star}** | **{v_star}** | **{flagged}** | **{n_clean}** | **{ks_clean}** | **{worst}** | **{window}** |"
            )
        else:
            md_lines.append(
                f"| {t_star} | {v_star} | {flagged} | {n_clean} | {ks_clean} | {worst} | {window} |"
            )
    
    # 添加论文选择的说明
    paper_config = df_results[
        (df_results["T_star"] == 325) & (df_results["v_star"] == 5)
    ]
    
    if not paper_config.empty:
        row = paper_config.iloc[0]
        md_lines.extend([
            "",
            "## 论文选择 (T*=325s, v*=5km/h)",
            "",
            f"- **Flagged**: {row['flagged_pct']:.1f}%",
            f"- **n_clean**: {int(row['n_clean'])}",
            f"- **KS(speed)**: {row['ks_clean']:.4f}" if pd.notna(row['ks_clean']) else "- **KS(speed)**: NA",
            f"- **Worst-15min**: {row['worst_window_ks']:.4f}" if pd.notna(row['worst_window_ks']) else "- **Worst-15min**: NA",
            f"- **Worst window**: {row['worst_window_time']}" if row['worst_window_time'] else "- **Worst window**: NA",
        ])
    
    md_content = "\n".join(md_lines)
    md_file = os.path.join(output_dir, "threshold_sensitivity_table_v3.md")
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"    Markdown 表格已保存: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Audit 阈值敏感性分析 V3")
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
        "--max_dist_m", 
        type=float, 
        default=RULE_C_MAX_DIST_M,
        help="Rule C: 最大距离 (米)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Audit 阈值敏感性分析 V3")
    print("=" * 70)
    print(f"口径: P14 Off-Peak (15:00-16:00), Op-L2-v1.1")
    print(f"v* 网格: {V_STAR_GRID} km/h")
    print(f"T* 网格: {T_STAR_GRID} s")
    print(f"Worst-window: exhaustive all 15-min subwindows")
    print()
    
    # 加载数据
    print("[1] 加载真实数据...")
    df_real = load_real_link_stats(args.real)
    print(f"    样本数: {len(df_real)}")
    
    print("[2] 加载仿真数据...")
    sim_speeds, sim_tt, _ = compute_sim_link_data(args.sim, args.dist)
    print(f"    仿真速度样本数: {len(sim_speeds)}")
    
    # 运行敏感性分析
    print("\n[3] 运行敏感性分析...")
    df_results = run_sensitivity_analysis(
        df_real, sim_speeds,
        v_star_grid=V_STAR_GRID,
        t_star_grid=T_STAR_GRID,
        max_dist_m=args.max_dist_m
    )
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    csv_path = os.path.join(args.output, "threshold_sensitivity_results_v3.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[4] 结果已保存: {csv_path}")
    
    # 显示结果摘要
    print("\n" + "=" * 70)
    print("敏感性分析结果摘要")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    # 生成图表和表格
    print("\n[5] 生成图表...")
    plot_heatmaps(df_results, args.output)
    generate_markdown_table(df_results, args.output)
    
    # 分析结论
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)
    
    # 找到论文选择的配置
    paper_config = df_results[
        (df_results["T_star"] == 325) & (df_results["v_star"] == 5)
    ]
    
    if not paper_config.empty:
        row = paper_config.iloc[0]
        print(f"论文选择 (T*=325s, v*=5km/h):")
        print(f"  - Flagged: {row['flagged_pct']:.1f}%")
        print(f"  - n_clean: {int(row['n_clean'])}")
        if pd.notna(row['ks_clean']):
            print(f"  - KS(speed): {row['ks_clean']:.4f}")
        if pd.notna(row['worst_window_ks']):
            print(f"  - Worst-15min: {row['worst_window_ks']:.4f}")
            print(f"  - Worst window: {row['worst_window_time']}")
    
    # 找到最优配置（最小 KS）
    valid_results = df_results[df_results['n_clean'] >= MIN_CLEAN_SAMPLES]
    if not valid_results.empty and valid_results['ks_clean'].notna().any():
        best_idx = valid_results["ks_clean"].idxmin()
        best_row = df_results.loc[best_idx]
        print(f"\n最优 KS 配置 (T*={int(best_row['T_star'])}s, v*={int(best_row['v_star'])}km/h):")
        print(f"  - Flagged: {best_row['flagged_pct']:.1f}%")
        print(f"  - n_clean: {int(best_row['n_clean'])}")
        print(f"  - KS(speed): {best_row['ks_clean']:.4f}")
        if pd.notna(best_row['worst_window_ks']):
            print(f"  - Worst-15min: {best_row['worst_window_ks']:.4f}")


if __name__ == "__main__":
    main()
