#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_paper_tables.py - 生成论文用 Markdown 主表
====================================================

生成 Word 友好的 Markdown 表格：
1. protocol_ablation_main.md: A0-A4 across scenarios/durations
2. scale_sweep_summary.md: Scale sweep 汇总（由 run_scale_sweep.py 生成）
3. scale_sweep_delta.md: ΔKS vs scale（由 run_scale_sweep.py 生成）

硬约束：
- 禁止生成 .tex 文件
- 所有表格输出为 Markdown (.md) 或 CSV (.csv)

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval.metrics_v4 import (
    compute_metrics_v4,
    MetricsV4Result,
    AuditConfig,
    load_real_link_stats
)


# ============================================================================
# Protocol Ablation 主表
# ============================================================================

def generate_protocol_ablation_main_table(
    results_csv: Path,
    output_path: Path
) -> None:
    """
    生成 Protocol Ablation 主表（A0-A4）
    
    Args:
        results_csv: protocol_ablation 的 results.csv 路径
        output_path: 输出 Markdown 路径
    """
    if not results_csv.exists():
        print(f"[ERROR] 结果文件不存在: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    md_lines = ["# Protocol Ablation Results (A0-A4)", ""]
    md_lines.append("完整协议消融实验：从 Zero-shot (A0) 到 Full-RCMDT (A4)")
    md_lines.append("")
    
    # 选择关键列
    cols = ['config_id', 'scenario', 'n_clean', 'n_sim', 
            'ks_speed', 'ks_tt', 'passed', 
            'worst_window_ks', 'worst_window_start', 'flagged_fraction']
    
    df_display = df[cols].copy()
    
    # 格式化列名
    df_display.columns = [
        'Config', 'Scenario', 'n_clean', 'n_sim',
        'KS(speed)', 'KS(TT)', 'Pass', 
        'Worst-15min', 'Window', 'Flagged(%)'
    ]
    
    # 格式化 Pass 列
    if 'Pass' in df_display.columns:
        df_display['Pass'] = df_display['Pass'].apply(lambda x: '✓' if x else '✗')
    
    # 格式化 Flagged(%)
    if 'Flagged(%)' in df_display.columns:
        df_display['Flagged(%)'] = (df_display['Flagged(%)'] * 100).round(1)
    
    md_lines.append(df_display.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    
    # 添加说明
    md_lines.append("**配置说明**:")
    md_lines.append("- **A0** (Zero-shot): 默认参数，无 L2，无 Audit")
    md_lines.append("- **A1** (Raw-L1): L1 用 raw D2D，无 Audit")
    md_lines.append("- **A2** (Audit-Val-Only): L1 用 raw，验证用 clean")
    md_lines.append("- **A3** (Audit-in-Cal): L1 只在 clean 集上算 loss")
    md_lines.append("- **A4** (Full-RCMDT): A3 + L2/IES")
    md_lines.append("")
    md_lines.append("**指标说明**:")
    md_lines.append("- **n_clean**: Audit 清洗后的观测数")
    md_lines.append("- **n_sim**: 仿真观测数")
    md_lines.append("- **KS(speed)**: Kolmogorov-Smirnov 距离（速度分布）")
    md_lines.append("- **KS(TT)**: Kolmogorov-Smirnov 距离（旅行时间分布）")
    md_lines.append("- **Pass**: KS < Dcrit (α=0.05)")
    md_lines.append("- **Worst-15min**: 15-min exhaustive 搜索的最差 KS(speed)")
    md_lines.append("- **Window**: 对应的时间窗口起点")
    md_lines.append("- **Flagged(%)**: Audit Rule-C 标记的异常观测比例")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✓ Protocol Ablation 主表已保存: {output_path}")


# ============================================================================
# Scale Sweep 表格（由 run_scale_sweep.py 生成，此处仅提供接口）
# ============================================================================

def check_scale_sweep_tables(tables_dir: Path) -> None:
    """检查 scale_sweep 表格是否存在"""
    summary_path = tables_dir / "scale_sweep_summary.md"
    delta_path = tables_dir / "scale_sweep_delta.md"
    
    if summary_path.exists():
        print(f"✓ Scale Sweep 汇总表已存在: {summary_path}")
    else:
        print(f"⚠ Scale Sweep 汇总表不存在: {summary_path}")
        print(f"  请运行: python scripts/experiments_v4/run_scale_sweep.py")
    
    if delta_path.exists():
        print(f"✓ Scale Sweep Delta 表已存在: {delta_path}")
    else:
        print(f"⚠ Scale Sweep Delta 表不存在: {delta_path}")
        print(f"  请运行: python scripts/experiments_v4/run_scale_sweep.py")


# ============================================================================
# L2 观测向量消融表（可选）
# ============================================================================

def generate_l2_obs_ablation_table(
    results_csv: Path,
    output_path: Path
) -> None:
    """
    生成 L2 观测向量消融表（如果后续做此实验）
    
    Args:
        results_csv: l2_obs_ablation 的 results.csv 路径
        output_path: 输出 Markdown 路径
    """
    if not results_csv.exists():
        print(f"[INFO] L2 观测向量消融实验尚未运行")
        return
    
    df = pd.read_csv(results_csv)
    
    md_lines = ["# L2 观测向量消融实验", ""]
    md_lines.append("比较不同 L2 观测向量配置的效果")
    md_lines.append("")
    
    # 选择关键列
    cols = ['obs_config', 'ks_speed', 'ks_tt', 'passed', 'worst_window_ks']
    
    if all(col in df.columns for col in cols):
        df_display = df[cols].copy()
        df_display.columns = ['观测向量配置', 'KS(speed)', 'KS(TT)', 'Pass', 'Worst-15min']
        
        if 'Pass' in df_display.columns:
            df_display['Pass'] = df_display['Pass'].apply(lambda x: '✓' if x else '✗')
        
        md_lines.append(df_display.to_markdown(index=False, floatfmt=".4f"))
        md_lines.append("")
        
        md_lines.append("**配置说明**:")
        md_lines.append("- **moving-only**: 仅使用移动速度")
        md_lines.append("- **D2D**: Door-to-Door 速度")
        md_lines.append("- **D2D+decont**: D2D + 去污染（最终版本）")
    else:
        md_lines.append("（数据格式不符合预期）")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✓ L2 观测向量消融表已保存: {output_path}")


# ============================================================================
# 批量生成所有表格
# ============================================================================

def generate_all_tables(
    data_dir: Path,
    tables_dir: Path
) -> None:
    """
    批量生成所有论文表格
    
    Args:
        data_dir: 数据目录（data/experiments_v4）
        tables_dir: 表格输出目录（tables/）
    """
    print("=" * 70)
    print("生成论文 Markdown 表格")
    print("=" * 70)
    print(f"数据目录: {data_dir}")
    print(f"表格输出目录: {tables_dir}")
    print("=" * 70)
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Protocol Ablation 主表
    print("\n[1/3] Protocol Ablation 主表...")
    protocol_results = data_dir / "protocol_ablation" / "results.csv"
    protocol_table = tables_dir / "protocol_ablation_main.md"
    generate_protocol_ablation_main_table(protocol_results, protocol_table)
    
    # 2. Scale Sweep 表格（检查是否存在）
    print("\n[2/3] Scale Sweep 表格...")
    check_scale_sweep_tables(tables_dir)
    
    # 3. L2 观测向量消融表（可选）
    print("\n[3/3] L2 观测向量消融表...")
    l2_results = data_dir / "l2_obs_ablation" / "results.csv"
    l2_table = tables_dir / "l2_obs_ablation.md"
    if l2_results.exists():
        generate_l2_obs_ablation_table(l2_results, l2_table)
    else:
        print(f"  [INFO] L2 观测向量消融实验尚未运行，跳过")
    
    print("\n" + "=" * 70)
    print("✓ 表格生成完成")
    print("=" * 70)
    print(f"输出目录: {tables_dir}")
    print("\n生成的表格：")
    for md_file in sorted(tables_dir.glob("*.md")):
        print(f"  - {md_file.name}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成论文 Markdown 表格")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "experiments_v4"),
        help="数据目录"
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=str(PROJECT_ROOT / "tables"),
        help="表格输出目录"
    )
    parser.add_argument(
        "--table",
        choices=["protocol", "scale_sweep", "l2_obs", "all"],
        default="all",
        help="生成哪个表格"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    tables_dir = Path(args.tables_dir)
    
    if args.table == "all":
        generate_all_tables(data_dir, tables_dir)
    elif args.table == "protocol":
        protocol_results = data_dir / "protocol_ablation" / "results.csv"
        protocol_table = tables_dir / "protocol_ablation_main.md"
        generate_protocol_ablation_main_table(protocol_results, protocol_table)
    elif args.table == "scale_sweep":
        check_scale_sweep_tables(tables_dir)
    elif args.table == "l2_obs":
        l2_results = data_dir / "l2_obs_ablation" / "results.csv"
        l2_table = tables_dir / "l2_obs_ablation.md"
        generate_l2_obs_ablation_table(l2_results, l2_table)


if __name__ == "__main__":
    main()
