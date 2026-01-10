#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_real_sensitivity_table.py - 真实数据敏感性分析表生成器
================================================================

针对真实数据，在不同 Rule C 阈值组合下计算 Flagged % 和 KS(TT)。

Author: RCMDT Project
Date: 2026-01-10
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval.metrics_v4 import load_real_link_stats, compute_sim_link_data


def apply_audit_filter(
    df: pd.DataFrame,
    T_star: float,
    v_star: float,
    max_dist_m: float = 1500.0
) -> Tuple[np.ndarray, float]:
    """
    应用 Rule C 审计过滤
    
    与 metrics_v4.py 中 apply_audit_rule_c 保持一致：
    Rule C: flagged if (tt >= T*) AND (speed <= v*) AND (dist < max_dist)
    
    Args:
        df: 真实数据 DataFrame，需包含 tt_median, speed_median, dist_m
        T_star: 时间阈值（秒）
        v_star: 速度阈值（km/h）
        max_dist_m: 最大距离（米）
    
    Returns:
        clean_tt: 清洗后的出行时间数组
        flagged_pct: 标记样本百分比
    """
    # Rule C: flagged if (tt >= T*) AND (speed <= v*) AND (dist < max_dist)
    cond_flagged = (
        (df["tt_median"] >= T_star) &
        (df["speed_median"] <= v_star) &
        (df["dist_m"] < max_dist_m)
    )
    
    clean_mask = ~cond_flagged
    clean_tt = df.loc[clean_mask, "tt_median"].dropna().values
    
    n_total = len(df)
    n_flagged = cond_flagged.sum()
    flagged_pct = (n_flagged / n_total * 100) if n_total > 0 else 0.0
    
    return clean_tt, flagged_pct


def compute_ks_tt(real_tt: np.ndarray, sim_tt: np.ndarray) -> float:
    """
    计算出行时间的 KS 统计量
    
    Args:
        real_tt: 真实出行时间数组
        sim_tt: 仿真出行时间数组
    
    Returns:
        ks_stat: KS 统计量
    """
    if len(real_tt) == 0 or len(sim_tt) == 0:
        return np.nan
    
    ks_stat, _ = ks_2samp(real_tt, sim_tt)
    return ks_stat


def generate_sensitivity_table(
    real_csv: str,
    sim_xml: str,
    dist_csv: str,
    T_star_values: list = [300, 325, 350],
    v_star_values: list = [4, 5, 6],
    max_dist_m: float = 1500.0,
    tt_mode: str = "door"
) -> pd.DataFrame:
    """
    生成敏感性分析表
    
    Args:
        real_csv: 真实数据 CSV 路径（聚合统计数据，需包含 tt_median, speed_median, dist_m）
        sim_xml: 仿真 stopinfo XML 路径
        dist_csv: 站点距离 CSV 路径
        T_star_values: 时间阈值列表
        v_star_values: 速度阈值列表
        max_dist_m: 最大距离（米）
        tt_mode: 出行时间模式 ('door' 或 'moving')
    
    Returns:
        包含结果的 DataFrame
    """
    # 加载真实数据（使用 metrics_v4.py 的函数，确保一致性）
    df_real = load_real_link_stats(real_csv)
    
    # 加载仿真数据（使用 metrics_v4.py 中的函数）
    sim_speeds, sim_tt, sim_timestamps = compute_sim_link_data(
        sim_xml, dist_csv, tt_mode=tt_mode
    )
    
    print(f"真实数据样本数: {len(df_real)}")
    print(f"仿真数据样本数: {len(sim_tt)}")
    print()
    
    results = []
    
    for T_star in T_star_values:
        for v_star in v_star_values:
            # 应用审计过滤
            clean_tt, flagged_pct = apply_audit_filter(
                df_real, T_star, v_star, max_dist_m
            )
            
            # 计算 KS (TT)
            ks_tt = compute_ks_tt(clean_tt, sim_tt)
            
            results.append({
                "T*": T_star,
                "v*": v_star,
                "Flagged %": flagged_pct,
                "KS (TT)": ks_tt,
                "n_clean": len(clean_tt),
                "n_sim": len(sim_tt)
            })
            
            # 验证基准结果 (325, 5)
            if T_star == 325 and v_star == 5:
                print(f"基准结果 (T*={T_star}, v*={v_star}):")
                print(f"  Flagged %: {flagged_pct:.2f}%")
                print(f"  KS (TT): {ks_tt:.4f}")
                print(f"  n_clean: {len(clean_tt)}")
                print()
    
    return pd.DataFrame(results)


def print_markdown_table(df: pd.DataFrame) -> str:
    """
    打印 Markdown 格式表格
    """
    # 选择显示列
    display_cols = ["T*", "v*", "Flagged %", "KS (TT)"]
    df_display = df[display_cols].copy()
    
    # 格式化数值
    df_display["Flagged %"] = df_display["Flagged %"].apply(lambda x: f"{x:.2f}%")
    df_display["KS (TT)"] = df_display["KS (TT)"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    
    markdown = df_display.to_markdown(index=False)
    return markdown


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成真实数据敏感性分析表"
    )
    parser.add_argument(
        "--real",
        type=str,
        default=str(PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv"),
        help="真实数据 CSV 路径（聚合统计，需包含 tt_median, speed_median, dist_m）"
    )
    parser.add_argument(
        "--sim",
        type=str,
        default=str(PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"),
        help="仿真 stopinfo XML 路径"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"),
        help="站点距离 CSV 路径"
    )
    parser.add_argument(
        "--tt-mode",
        type=str,
        default="door",
        choices=["door", "moving"],
        help="出行时间计算模式"
    )
    
    args = parser.parse_args()
    
    # 定义阈值网格
    T_star_values = [300, 325, 350]
    v_star_values = [4, 5, 6]
    
    print("=" * 70)
    print("真实数据敏感性分析")
    print("=" * 70)
    print(f"真实数据: {args.real}")
    print(f"仿真数据: {args.sim}")
    print(f"距离数据: {args.dist}")
    print(f"TT 模式: {args.tt_mode}")
    print(f"阈值网格: T* ∈ {T_star_values}, v* ∈ {v_star_values}")
    print()
    
    # 生成敏感性表
    df_results = generate_sensitivity_table(
        args.real,
        args.sim,
        args.dist,
        T_star_values=T_star_values,
        v_star_values=v_star_values,
        tt_mode=args.tt_mode
    )
    
    # 打印 Markdown 表格
    print("=" * 70)
    print("敏感性分析表")
    print("=" * 70)
    print(print_markdown_table(df_results))
    print()
    
    # 保存 CSV
    output_csv = PROJECT_ROOT / "scripts" / "eval" / "real_sensitivity_table.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")


if __name__ == "__main__":
    main()
