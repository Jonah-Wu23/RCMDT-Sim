#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_threshold_sensitivity_ieee.py
==================================
生成 IEEE 格式的 Rule C 阈值敏感性 3×3 Heatmap

格式要求:
- Times New Roman 8pt
- 单栏宽度 (3.5 in)
- 橙-蓝色系 (与 plot_p14_robustness.py 一致)
- 高亮论文选择点 (T*=325, v*=5)

Author: RCMDT Project
Date: 2026-01-11
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# IEEE SMC Style: Times New Roman 8pt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

# 橙-蓝色系 (与 plot_p14_robustness.py 一致)
COLOR_BLUE = '#1f77b4'
COLOR_ORANGE = '#ff7f0e'
COLOR_GRAY = '#7f7f7f'

# 3×3 网格
T_STAR_GRID = [300, 325, 350]  # seconds
V_STAR_GRID = [4, 5, 6]  # km/h

# IEEE 单栏宽度
SINGLE_COL_WIDTH = 3.5  # inches


def create_orange_blue_cmap():
    """创建橙-蓝渐变色图 (低值=蓝, 高值=橙)"""
    colors = [COLOR_BLUE, 'white', COLOR_ORANGE]
    return LinearSegmentedColormap.from_list('OrangeBlue', colors, N=256)


def load_sensitivity_data(csv_path: str) -> pd.DataFrame:
    """加载敏感性分析结果"""
    df = pd.read_csv(csv_path)
    # 筛选 3×3 网格
    df_filtered = df[
        (df['T_star'].isin(T_STAR_GRID)) & 
        (df['v_star'].isin(V_STAR_GRID))
    ].copy()
    return df_filtered


def plot_heatmap_single(
    df: pd.DataFrame,
    output_path: str,
    metric: str = 'n_clean',
    title: str = None,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    fmt: str = '.0f',
    highlight_center: bool = True
):
    """
    绘制单个热力图
    
    Parameters:
    -----------
    metric : str
        要绘制的指标列名 ('n_clean', 'flagged_pct', 'ks_clean')
    """
    n_t = len(T_STAR_GRID)
    n_v = len(V_STAR_GRID)
    
    # 构建矩阵
    matrix = np.full((n_t, n_v), np.nan)
    for _, row in df.iterrows():
        if row['T_star'] in T_STAR_GRID and row['v_star'] in V_STAR_GRID:
            t_idx = T_STAR_GRID.index(int(row['T_star']))
            v_idx = V_STAR_GRID.index(int(row['v_star']))
            matrix[t_idx, v_idx] = row[metric]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.5))
    
    # 确定色图
    if cmap is None:
        cmap = create_orange_blue_cmap()
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax)
    
    # 设置坐标轴
    ax.set_xticks(range(n_v))
    ax.set_xticklabels([f"{v}" for v in V_STAR_GRID])
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([f"{t}" for t in T_STAR_GRID])
    ax.set_xlabel(r'$v^*$ (km/h)')
    ax.set_ylabel(r'$T^*$ (s)')
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    # 添加数值标注
    for i in range(n_t):
        for j in range(n_v):
            val = matrix[i, j]
            if np.isnan(val):
                text = 'NA'
                color = 'gray'
            else:
                text = f"{val:{fmt}}"
                # 根据背景亮度选择文字颜色
                norm_val = (val - (vmin or matrix.min())) / ((vmax or matrix.max()) - (vmin or matrix.min()) + 1e-6)
                color = 'white' if 0.3 < norm_val < 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)
    
    # 高亮论文选择 (T*=325, v*=5)
    if highlight_center and 325 in T_STAR_GRID and 5 in V_STAR_GRID:
        t_idx = T_STAR_GRID.index(325)
        v_idx = V_STAR_GRID.index(5)
        rect = plt.Rectangle(
            (v_idx - 0.5, t_idx - 0.5), 1, 1,
            fill=False, edgecolor='black', linewidth=2, linestyle='-'
        )
        ax.add_patch(rect)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()


def plot_combined_heatmap(
    df: pd.DataFrame,
    output_path: str,
    highlight_center: bool = True
):
    """
    绘制组合热力图 (n_clean 和 KS(speed) 并排)
    符合 IEEE 双栏排版要求
    """
    n_t = len(T_STAR_GRID)
    n_v = len(V_STAR_GRID)
    
    # 构建矩阵
    n_clean_matrix = np.full((n_t, n_v), np.nan)
    ks_matrix = np.full((n_t, n_v), np.nan)
    
    for _, row in df.iterrows():
        if row['T_star'] in T_STAR_GRID and row['v_star'] in V_STAR_GRID:
            t_idx = T_STAR_GRID.index(int(row['T_star']))
            v_idx = V_STAR_GRID.index(int(row['v_star']))
            n_clean_matrix[t_idx, v_idx] = row['n_clean']
            if pd.notna(row.get('ks_clean')):
                ks_matrix[t_idx, v_idx] = row['ks_clean']
    
    # 创建组合图 (2行1列 - 上下排列)
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 4.0))
    
    # 左图: n_clean (蓝色系 - 越多越好)
    ax1 = axes[0]
    im1 = ax1.imshow(n_clean_matrix, cmap='Blues', aspect='auto', origin='lower',
                     vmin=30, vmax=55)
    ax1.set_xticks(range(n_v))
    ax1.set_xticklabels([f"{v}" for v in V_STAR_GRID])
    ax1.set_yticks(range(n_t))
    ax1.set_yticklabels([f"{t}" for t in T_STAR_GRID])
    ax1.set_xlabel(r'$v^*$ (km/h)')
    ax1.set_ylabel(r'$T^*$ (s)')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')
    
    for i in range(n_t):
        for j in range(n_v):
            val = n_clean_matrix[i, j]
            text = f"{val:.0f}" if not np.isnan(val) else 'NA'
            color = 'white' if val > 45 else 'black'
            ax1.text(j, i, text, ha='center', va='center', fontsize=7, color=color)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.ax.tick_params(labelsize=6)
    
    # 右图: KS(speed) (橙色系 - 越低越好, 反转色阶)
    ax2 = axes[1]
    # 使用 Oranges_r 使低值更亮
    im2 = ax2.imshow(ks_matrix, cmap='Oranges', aspect='auto', origin='lower',
                     vmin=0.15, vmax=0.45)
    ax2.set_xticks(range(n_v))
    ax2.set_xticklabels([f"{v}" for v in V_STAR_GRID])
    ax2.set_yticks(range(n_t))
    ax2.set_yticklabels([f"{t}" for t in T_STAR_GRID])
    ax2.set_xlabel(r'$v^*$ (km/h)')
    ax2.set_ylabel(r'$T^*$ (s)')
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')
    
    for i in range(n_t):
        for j in range(n_v):
            val = ks_matrix[i, j]
            if np.isnan(val):
                text = 'NA'
                color = 'gray'
            else:
                text = f"{val:.2f}"
                color = 'white' if val > 0.35 else 'black'
            ax2.text(j, i, text, ha='center', va='center', fontsize=7, color=color)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.ax.tick_params(labelsize=6)
    
    # 高亮论文选择 (T*=325, v*=5)
    if highlight_center and 325 in T_STAR_GRID and 5 in V_STAR_GRID:
        t_idx = T_STAR_GRID.index(325)
        v_idx = V_STAR_GRID.index(5)
        for ax in [ax1, ax2]:
            rect = plt.Rectangle(
                (v_idx - 0.5, t_idx - 0.5), 1, 1,
                fill=False, edgecolor='black', linewidth=2, linestyle='-'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()


def extract_ghost_jam_examples(raw_csv: str, t_critical: float = 325, speed_kmh: float = 5, max_dist_m: float = 1500) -> pd.DataFrame:
    """
    提取典型 Ghost Jam 案例
    
    返回被 Rule C 标记的 top-N 案例
    """
    df = pd.read_csv(raw_csv)
    
    # 应用 Rule C
    cond_ghost = (
        (df['tt_median'] > t_critical) & 
        (df['speed_median'] < speed_kmh) & 
        (df['dist_m'] < max_dist_m)
    )
    
    df_ghost = df[cond_ghost].copy()
    df_ghost['implied_speed'] = df_ghost['dist_m'] / df_ghost['tt_median'] * 3.6  # m/s -> km/h
    
    # 按 tt_median 降序排列，取最极端的案例
    df_ghost = df_ghost.sort_values('tt_median', ascending=False)
    
    return df_ghost.head(5)


def main():
    parser = argparse.ArgumentParser(description="生成 IEEE 格式阈值敏感性热力图")
    parser.add_argument(
        "--sensitivity-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "calibration_v3" / "sensitivity" / "threshold_sensitivity_results_v3.csv"),
        help="敏感性分析结果 CSV"
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "link_stats.csv"),
        help="原始 link_stats CSV (用于提取 Ghost Jam 案例)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plots"),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IEEE 格式阈值敏感性热力图生成")
    print("=" * 60)
    print(f"3×3 网格: T* ∈ {T_STAR_GRID}, v* ∈ {V_STAR_GRID}")
    print(f"论文选择: T*=325s, v*=5km/h")
    print()
    
    # 加载数据
    print("[1] 加载敏感性分析数据...")
    df = load_sensitivity_data(args.sensitivity_csv)
    print(f"    筛选后样本数: {len(df)}")
    print(df.to_string(index=False))
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成组合热力图
    print("[2] 生成组合热力图...")
    output_combined = os.path.join(args.output_dir, "threshold_sensitivity_3x3_ieee.png")
    plot_combined_heatmap(df, output_combined)
    
    # 生成单独热力图 (n_clean)
    print("[3] 生成 n_clean 热力图...")
    output_n_clean = os.path.join(args.output_dir, "threshold_sensitivity_n_clean_ieee.png")
    plot_heatmap_single(
        df, output_n_clean,
        metric='n_clean',
        title=r'Clean Samples ($n_{real}$)',
        cmap='Blues',
        vmin=30, vmax=55,
        fmt='.0f'
    )
    
    # 生成单独热力图 (KS)
    print("[4] 生成 KS(speed) 热力图...")
    output_ks = os.path.join(args.output_dir, "threshold_sensitivity_ks_ieee.png")
    plot_heatmap_single(
        df, output_ks,
        metric='ks_clean',
        title='KS(speed) Full-hour',
        cmap='Oranges',
        vmin=0.15, vmax=0.45,
        fmt='.2f'
    )
    
    # 提取 Ghost Jam 案例
    print("\n[5] 提取典型 Ghost Jam 案例...")
    if os.path.exists(args.raw_csv):
        df_ghost = extract_ghost_jam_examples(args.raw_csv)
        print("\n典型 Ghost Jam 案例 (Rule C: T*=325s, v*=5km/h):")
        print("-" * 80)
        for _, row in df_ghost.iterrows():
            print(f"  Route {row['route']} ({row['bound']}): "
                  f"seq {row['from_seq']}->{row['to_seq']}, "
                  f"TT={row['tt_median']:.0f}s, "
                  f"dist={row['dist_m']:.0f}m, "
                  f"implied_speed={row['implied_speed']:.1f}km/h")
        
        # 保存案例到文件
        ghost_csv = os.path.join(args.output_dir, "ghost_jam_examples.csv")
        df_ghost.to_csv(ghost_csv, index=False)
        print(f"\n    Ghost Jam 案例已保存: {ghost_csv}")
    else:
        print(f"    警告: 未找到 {args.raw_csv}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
