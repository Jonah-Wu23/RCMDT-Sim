#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_threshold_sensitivity.py
============================
Audit 阈值敏感性实验

目的：回答"阈值拍脑袋""54% flagged 是否过滤"的质疑

二维网格:
    v* ∈ {3, 4, 5, 6, 7} km/h
    T* ∈ {250, 300, 325, 350, 400} s

每个点报告:
    - flagged fraction (%)
    - KS(clean)（验证集）
    - worst-15min（验证集）

Author: RCMDT Project
Date: 2026-01-08
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
DEFAULT_SIM_STOPINFO = PROJECT_ROOT / "sumo" / "output" / "offpeak_stopinfo.xml"
DEFAULT_DIST_FILE = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"

# 敏感性网格
V_STAR_GRID = [3, 4, 5, 6, 7]  # km/h
T_STAR_GRID = [250, 300, 325, 350, 400]  # seconds

MAX_DIST_M = 1500.0


# ============================================================================
# 数据处理函数
# ============================================================================

def load_real_stats(filepath: str) -> pd.DataFrame:
    """加载真实链路统计数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    return pd.read_csv(filepath)


def load_sim_stopinfo(filepath: str) -> pd.DataFrame:
    """解析 SUMO stopinfo.xml"""
    from xml.etree import ElementTree as ET
    
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    for stop in root.findall('.//stopinfo'):
        records.append({
            'vehicle_id': stop.get('id'),
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started', 0)),
            'departure': float(stop.get('ended', 0)),
            'duration': float(stop.get('duration', 0))
        })
    
    return pd.DataFrame(records)


def compute_sim_speeds(stopinfo_xml: str, dist_csv: str) -> np.ndarray:
    """从仿真输出计算有效速度"""
    if not os.path.exists(stopinfo_xml) or not os.path.exists(dist_csv):
        return np.array([])
    
    df_dist = pd.read_csv(dist_csv)
    required = {"route", "bound", "service_type", "seq", "stop_id", "link_dist_m"}
    if not required.issubset(set(df_dist.columns)):
        return np.array([])
    
    dist_map = {}
    for _, group in df_dist.groupby(["route", "bound", "service_type"]):
        group = group.sort_values("seq")
        stops = group["stop_id"].astype(str).tolist()
        link_dists = group["link_dist_m"].tolist()
        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            d = link_dists[i + 1]
            if pd.notna(d) and d > 0:
                dist_map[(s1, s2)] = float(d)
    
    df_stops = load_sim_stopinfo(stopinfo_xml)
    if df_stops.empty:
        return np.array([])
    
    df_stops = df_stops.sort_values(["vehicle_id", "arrival"]).reset_index(drop=True)
    speeds = []
    
    for veh_id, veh_data in df_stops.groupby("vehicle_id"):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = str(veh_data.loc[i, "stop_id"])
            to_stop = str(veh_data.loc[i + 1, "stop_id"])
            departure = float(veh_data.loc[i, "departure"])
            arrival = float(veh_data.loc[i + 1, "arrival"])
            travel_time_s = arrival - departure
            
            if travel_time_s <= 0:
                continue
            
            dist_m = dist_map.get((from_stop, to_stop))
            if not dist_m:
                continue
            
            speed_kmh = (dist_m / 1000.0) / (travel_time_s / 3600.0)
            if 0.1 < speed_kmh < 120:
                speeds.append(speed_kmh)
    
    return np.array(speeds)


def apply_rule_c(
    df: pd.DataFrame,
    t_critical: float,
    speed_kmh: float,
    max_dist_m: float = MAX_DIST_M
) -> Tuple[np.ndarray, float]:
    """
    应用 Rule C 并返回 clean 样本和 flagged 比例
    
    Returns:
        clean_speeds: 清洗后速度数组
        flagged_fraction: 被标记比例
    """
    required_cols = {"tt_median", "speed_median", "dist_m"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"缺少必需列: {required_cols - set(df.columns)}")
    
    cond_ghost = (
        (df["tt_median"] > t_critical) & 
        (df["speed_median"] < speed_kmh) & 
        (df["dist_m"] < max_dist_m)
    )
    
    clean_speeds = df.loc[~cond_ghost, "speed_median"].dropna().values
    flagged_fraction = cond_ghost.sum() / len(df) if len(df) > 0 else 0.0
    
    return clean_speeds, flagged_fraction


def compute_ks(real_speeds: np.ndarray, sim_speeds: np.ndarray) -> float:
    """计算 KS 统计量"""
    if len(real_speeds) < 5 or len(sim_speeds) < 5:
        return np.nan
    ks_stat, _ = ks_2samp(real_speeds, sim_speeds)
    return ks_stat


def compute_worst_15min_ks(clean_speeds: np.ndarray, sim_speeds: np.ndarray) -> float:
    """计算最差 15 分钟窗口 KS"""
    if len(clean_speeds) < 10 or len(sim_speeds) < 10:
        return np.nan
    
    n_windows = 4
    window_size = len(clean_speeds) // n_windows
    worst_ks = 0.0
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < n_windows - 1 else len(clean_speeds)
        window_speeds = clean_speeds[start_idx:end_idx]
        
        if len(window_speeds) < 5:
            continue
        
        sim_sample = np.random.choice(
            sim_speeds, 
            min(len(window_speeds), len(sim_speeds)), 
            replace=False
        )
        ks_stat, _ = ks_2samp(window_speeds, sim_sample)
        worst_ks = max(worst_ks, ks_stat)
    
    return worst_ks


# ============================================================================
# 敏感性分析主函数
# ============================================================================

def run_sensitivity_analysis(
    df_real: pd.DataFrame,
    sim_speeds: np.ndarray,
    v_star_grid: List[float] = V_STAR_GRID,
    t_star_grid: List[float] = T_STAR_GRID,
    max_dist_m: float = MAX_DIST_M
) -> pd.DataFrame:
    """
    运行二维敏感性分析
    
    Returns:
        pd.DataFrame: 包含每个 (T*, v*) 组合的结果
    """
    results = []
    
    for t_star in t_star_grid:
        for v_star in v_star_grid:
            clean_speeds, flagged_frac = apply_rule_c(
                df_real, t_star, v_star, max_dist_m
            )
            
            ks_clean = compute_ks(clean_speeds, sim_speeds)
            worst_15min = compute_worst_15min_ks(clean_speeds, sim_speeds)
            
            results.append({
                "T_star": t_star,
                "v_star": v_star,
                "flagged_pct": flagged_frac * 100,
                "n_clean": len(clean_speeds),
                "ks_clean": ks_clean,
                "worst_15min_ks": worst_15min
            })
    
    return pd.DataFrame(results)


def plot_heatmaps(
    df_results: pd.DataFrame,
    output_dir: str,
    v_star_grid: List[float] = V_STAR_GRID,
    t_star_grid: List[float] = T_STAR_GRID
):
    """生成热力图"""
    
    # 准备数据矩阵
    n_v = len(v_star_grid)
    n_t = len(t_star_grid)
    
    flagged_matrix = np.zeros((n_t, n_v))
    ks_matrix = np.zeros((n_t, n_v))
    worst_matrix = np.zeros((n_t, n_v))
    
    for _, row in df_results.iterrows():
        t_idx = t_star_grid.index(row["T_star"])
        v_idx = v_star_grid.index(row["v_star"])
        
        flagged_matrix[t_idx, v_idx] = row["flagged_pct"]
        ks_matrix[t_idx, v_idx] = row["ks_clean"] if pd.notna(row["ks_clean"]) else 1.0
        worst_matrix[t_idx, v_idx] = row["worst_15min_ks"] if pd.notna(row["worst_15min_ks"]) else 1.0
    
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    # 热力图 1: Flagged Fraction
    ax1 = axes[0]
    im1 = ax1.imshow(flagged_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax1.set_xticks(range(n_v))
    ax1.set_xticklabels([f"{v}" for v in v_star_grid])
    ax1.set_yticks(range(n_t))
    ax1.set_yticklabels([f"{t}" for t in t_star_grid])
    ax1.set_xlabel(r'$v^*$ (km/h)')
    ax1.set_ylabel(r'$T^*$ (s)')
    ax1.set_title('(a) Flagged Fraction (%)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 添加数值标注
    for i in range(n_t):
        for j in range(n_v):
            ax1.text(j, i, f"{flagged_matrix[i,j]:.1f}", 
                    ha='center', va='center', fontsize=7,
                    color='white' if flagged_matrix[i,j] > 40 else 'black')
    
    # 热力图 2: KS(clean)
    ax2 = axes[1]
    im2 = ax2.imshow(ks_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                     vmin=0.1, vmax=0.6)
    ax2.set_xticks(range(n_v))
    ax2.set_xticklabels([f"{v}" for v in v_star_grid])
    ax2.set_yticks(range(n_t))
    ax2.set_yticklabels([f"{t}" for t in t_star_grid])
    ax2.set_xlabel(r'$v^*$ (km/h)')
    ax2.set_ylabel(r'$T^*$ (s)')
    ax2.set_title('(b) KS(clean)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            ax2.text(j, i, f"{ks_matrix[i,j]:.2f}", 
                    ha='center', va='center', fontsize=7,
                    color='white' if ks_matrix[i,j] > 0.35 else 'black')
    
    # 热力图 3: worst-15min KS
    ax3 = axes[2]
    im3 = ax3.imshow(worst_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower',
                     vmin=0.1, vmax=0.6)
    ax3.set_xticks(range(n_v))
    ax3.set_xticklabels([f"{v}" for v in v_star_grid])
    ax3.set_yticks(range(n_t))
    ax3.set_yticklabels([f"{t}" for t in t_star_grid])
    ax3.set_xlabel(r'$v^*$ (km/h)')
    ax3.set_ylabel(r'$T^*$ (s)')
    ax3.set_title('(c) Worst-15min KS', fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    for i in range(n_t):
        for j in range(n_v):
            ax3.text(j, i, f"{worst_matrix[i,j]:.2f}", 
                    ha='center', va='center', fontsize=7,
                    color='white' if worst_matrix[i,j] > 0.35 else 'black')
    
    # 标记论文选择的阈值 (T*=325, v*=5)
    if 325 in t_star_grid and 5 in v_star_grid:
        t_idx = t_star_grid.index(325)
        v_idx = v_star_grid.index(5)
        for ax in axes:
            rect = plt.Rectangle(
                (v_idx - 0.5, t_idx - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "threshold_sensitivity_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存: {output_path}")
    plt.close()


def generate_markdown_table(df_results: pd.DataFrame, output_dir: str):
    """生成敏感性分析 Markdown 表格"""
    
    md_lines = ["# Threshold Sensitivity Analysis (P14 Off-Peak)", ""]
    
    df_display = df_results[['T_star', 'v_star', 'flagged_pct', 'ks_clean', 'worst_15min_ks']].copy()
    df_display.columns = ['T* (s)', 'v* (km/h)', 'Flagged (%)', 'KS(speed)*', 'Worst-15min']
    
    md_lines.append(df_display.to_markdown(index=False, floatfmt=".3f"))
    md_lines.append("")
    md_lines.append("**Bold row (325, 5)** is the paper's chosen threshold.")
    md_lines.append("")
    md_lines.append("***KS(speed)**: full-hour KS on P14 Off-Peak 15:00-16:00 transfer.")
    md_lines.append("experiments.md 'KS ≈0.29' is hour-level KS(TT) after Rule-C.")
    md_lines.append("Paper's stress-test worst-window (15:45-16:00) KS=0.3337.")
    
    md_file = os.path.join(output_dir, "threshold_sensitivity_table.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    print(f"Markdown 表格已保存: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Audit 阈值敏感性分析")
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
        default=str(PROJECT_ROOT / "data" / "calibration" / "sensitivity"),
        help="输出目录"
    )
    parser.add_argument(
        "--max_dist_m", 
        type=float, 
        default=MAX_DIST_M,
        help="Rule C: 最大距离 (米)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Audit 阈值敏感性分析")
    print("=" * 70)
    print(f"v* 网格: {V_STAR_GRID} km/h")
    print(f"T* 网格: {T_STAR_GRID} s")
    print()
    
    # 加载数据
    print("[1] 加载真实数据...")
    df_real = load_real_stats(args.real)
    print(f"    样本数: {len(df_real)}")
    
    print("[2] 加载仿真数据...")
    sim_speeds = compute_sim_speeds(args.sim, args.dist)
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
    
    csv_path = os.path.join(args.output, "threshold_sensitivity_results.csv")
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
    generate_latex_table(df_results, args.output)
    
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
        print(f"  - KS(clean): {row['ks_clean']:.3f}")
        print(f"  - Worst-15min: {row['worst_15min_ks']:.3f}")
    
    # 找到最优配置
    best_idx = df_results["ks_clean"].idxmin()
    best_row = df_results.loc[best_idx]
    print(f"\n最优 KS 配置 (T*={int(best_row['T_star'])}s, v*={int(best_row['v_star'])}km/h):")
    print(f"  - Flagged: {best_row['flagged_pct']:.1f}%")
    print(f"  - KS(clean): {best_row['ks_clean']:.3f}")
    print(f"  - Worst-15min: {best_row['worst_15min_ks']:.3f}")


if __name__ == "__main__":
    main()
