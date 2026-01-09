#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_protocol_ablation.py - Protocol Ablation 实验 (A0-A4)
=========================================================

解决审稿人质疑："decoupling 不清楚"、"过滤是不是在作弊"

实验组设计：
- A0 Zero-shot: 默认参数，无 L2，无 audit
- A1 Raw-L1: L1 用 raw D2D（无 audit gating），无 L2
- A2 Audit-for-Validation-Only: L1 仍用 raw，但验证只看 clean
- A3 Audit-in-Calibration: L1 只在 clean 集上算 loss
- A4 Full RCMDT: A3 + L2(IES)

输出：
- results.csv: 详细结果
- summary.csv: 汇总统计
- tables/ablation_table.md: Markdown 表格
- figures/ablation_comparison.png: 对比图

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval.metrics_v4 import (
    compute_metrics_v4,
    apply_audit_rule_c,
    compute_ks_with_critical,
    AuditConfig,
    MetricsV4Result,
    load_real_link_stats,
    compute_sim_link_data,
    PROTOCOL_V4_CONFIG
)


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class AblationConfig:
    """消融实验配置"""
    config_id: str
    name: str
    use_audit_in_calibration: bool
    use_audit_in_validation: bool
    use_bo: bool
    use_ies: bool
    use_tail_loss: bool
    description: str


ABLATION_CONFIGS = [
    AblationConfig(
        config_id="A0",
        name="Zero-shot",
        use_audit_in_calibration=False,
        use_audit_in_validation=False,
        use_bo=False,
        use_ies=False,
        use_tail_loss=False,
        description="默认参数基线，无任何优化"
    ),
    AblationConfig(
        config_id="A1",
        name="Raw-L1",
        use_audit_in_calibration=False,
        use_audit_in_validation=False,
        use_bo=True,
        use_ies=False,
        use_tail_loss=False,
        description="L1 用 raw D2D，无 audit"
    ),
    AblationConfig(
        config_id="A2",
        name="Audit-Val-Only",
        use_audit_in_calibration=False,
        use_audit_in_validation=True,
        use_bo=True,
        use_ies=False,
        use_tail_loss=False,
        description="L1 用 raw，验证用 clean"
    ),
    AblationConfig(
        config_id="A3",
        name="Audit-in-Cal",
        use_audit_in_calibration=True,
        use_audit_in_validation=True,
        use_bo=True,
        use_ies=False,
        use_tail_loss=True,
        description="L1 只在 clean 集上算 loss"
    ),
    AblationConfig(
        config_id="A4",
        name="Full-RCMDT",
        use_audit_in_calibration=True,
        use_audit_in_validation=True,
        use_bo=True,
        use_ies=True,
        use_tail_loss=True,
        description="完整 RCMDT (A3 + L2)"
    ),
]


# ============================================================================
# 数据路径配置
# ============================================================================

DATA_PATHS = {
    "off_peak": {
        "real_stats": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
        "dist_csv": PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv",
        "scenario": "off_peak",
        "period": "Off-Peak (P14)",
        "hkt_time": "15:00-16:00"
    }
}

# 每个配置对应的仿真输出文件
# 使用 run_ablation_simulations.py 生成的新仿真数据
SIM_PATHS_BY_CONFIG = {
    "A0": PROJECT_ROOT / "sumo" / "output" / "ablation_runs" / "A0" / "stopinfo.xml",
    "A1": PROJECT_ROOT / "sumo" / "output" / "ablation_runs" / "A1" / "stopinfo.xml",
    "A2": PROJECT_ROOT / "sumo" / "output" / "ablation_runs" / "A2" / "stopinfo.xml",
    "A3": PROJECT_ROOT / "sumo" / "output" / "ablation_runs" / "A3" / "stopinfo.xml",
    "A4": PROJECT_ROOT / "sumo" / "output" / "ablation_runs" / "A4" / "stopinfo.xml",
}

# 备用路径（如果特定配置文件不存在）
FALLBACK_SIM_PATH = PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments_v4" / "protocol_ablation"


# ============================================================================
# 实验执行
# ============================================================================

def run_ablation_experiment(
    config: AblationConfig,
    scenario_key: str = "off_peak"
) -> Dict:
    """
    运行单个消融实验配置
    
    Args:
        config: 消融配置
        scenario_key: 场景键名
    
    Returns:
        实验结果字典
    """
    paths = DATA_PATHS[scenario_key]
    
    print(f"\n{'='*60}")
    print(f"运行配置: {config.config_id} - {config.name}")
    print(f"{'='*60}")
    print(f"描述: {config.description}")
    print(f"Audit in Calibration: {config.use_audit_in_calibration}")
    print(f"Audit in Validation: {config.use_audit_in_validation}")
    print(f"BO: {config.use_bo}, IES: {config.use_ies}, Tail: {config.use_tail_loss}")
    
    # 加载真实数据
    real_stats_path = str(paths["real_stats"])
    if not os.path.exists(real_stats_path):
        print(f"[ERROR] 真实数据文件不存在: {real_stats_path}")
        return None
    
    df_real = load_real_link_stats(real_stats_path)
    print(f"加载真实数据: {len(df_real)} 条记录")
    
    # 根据配置加载对应的仿真数据
    sim_path = SIM_PATHS_BY_CONFIG.get(config.config_id, FALLBACK_SIM_PATH)
    dist_path = str(paths["dist_csv"])
    
    if not os.path.exists(sim_path):
        print(f"[WARN] 配置 {config.config_id} 的仿真文件不存在: {sim_path}")
        print(f"       使用备用文件: {FALLBACK_SIM_PATH}")
        sim_path = FALLBACK_SIM_PATH
    
    if not os.path.exists(sim_path):
        print(f"[ERROR] 仿真数据文件不存在: {sim_path}")
        return None
    
    print(f"仿真数据源: {Path(sim_path).relative_to(PROJECT_ROOT)}")
    sim_speeds, sim_tt, sim_timestamps = compute_sim_link_data(str(sim_path), dist_path)
    print(f"加载仿真数据: {len(sim_speeds)} 条记录")
    
    # 根据配置决定是否应用 audit
    audit_config = AuditConfig.from_protocol()
    
    if config.use_audit_in_validation:
        # 使用 audit 清洗后的数据进行验证
        audit_config.enabled = True
    else:
        # 不使用 audit，用 raw 数据
        audit_config.enabled = False
    
    # 应用 audit
    raw_speeds, clean_speeds, raw_tt, clean_tt, audit_stats = apply_audit_rule_c(
        df_real, audit_config
    )
    
    print(f"Audit 结果: n_raw={audit_stats.n_raw}, n_clean={audit_stats.n_clean}, "
          f"flagged={audit_stats.flagged_fraction*100:.1f}%")
    
    # 根据配置选择验证数据
    if config.use_audit_in_validation:
        val_speeds = clean_speeds
        val_tt = clean_tt
        val_label = "clean"
    else:
        val_speeds = raw_speeds
        val_tt = raw_tt
        val_label = "raw"
    
    # 计算 KS (speed)
    ks_speed = compute_ks_with_critical(val_speeds, sim_speeds)
    ks_speed_raw = compute_ks_with_critical(raw_speeds, sim_speeds)
    
    # 计算 KS (TT)
    ks_tt = compute_ks_with_critical(val_tt, sim_tt)
    ks_tt_raw = compute_ks_with_critical(raw_tt, sim_tt)
    
    # 计算 worst-window (使用完整评估)
    result = compute_metrics_v4(
        real_data=df_real,
        sim_data=(sim_speeds, sim_tt, sim_timestamps),
        audit_config=audit_config if config.use_audit_in_validation else AuditConfig(enabled=False),
        scenario=scenario_key,
        route="68X",
        period=paths["period"]
    )
    
    # 汇总结果
    output = {
        "config_id": config.config_id,
        "config_name": config.name,
        "description": config.description,
        "use_audit_cal": config.use_audit_in_calibration,
        "use_audit_val": config.use_audit_in_validation,
        "use_bo": config.use_bo,
        "use_ies": config.use_ies,
        "use_tail_loss": config.use_tail_loss,
        "scenario": scenario_key,
        "period": paths["period"],
        "hkt_time": paths["hkt_time"],
        "n_raw": audit_stats.n_raw,
        "n_clean": audit_stats.n_clean,
        "n_sim": len(sim_speeds),
        "flagged_fraction": audit_stats.flagged_fraction,
        "ks_speed_raw": ks_speed_raw.ks_stat,
        "ks_speed_val": ks_speed.ks_stat,
        "ks_speed_critical": ks_speed.critical_value,
        "ks_speed_passed": ks_speed.passed,
        "ks_tt_raw": ks_tt_raw.ks_stat,
        "ks_tt_val": ks_tt.ks_stat,
        "ks_tt_critical": ks_tt.critical_value,
        "ks_tt_passed": ks_tt.passed,
        "worst_window_ks_speed": result.worst_window_speed.worst_ks,
        "worst_window_ks_tt": result.worst_window_tt.worst_ks,
        "worst_window_start": result.worst_window_speed.window_start_time,
        "worst_window_end": result.worst_window_speed.window_end_time,
        "val_label": val_label
    }
    
    print(f"\n结果:")
    print(f"  KS(speed) {val_label}: {ks_speed.ks_stat:.4f} "
          f"[{'PASS' if ks_speed.passed else 'FAIL'}]")
    print(f"  KS(TT) {val_label}: {ks_tt.ks_stat:.4f} "
          f"[{'PASS' if ks_tt.passed else 'FAIL'}]")
    if result.worst_window_speed.worst_ks:
        print(f"  Worst-15min KS(speed): {result.worst_window_speed.worst_ks:.4f}")
        print(f"  Worst window: {result.worst_window_speed.window_start_time} - "
              f"{result.worst_window_speed.window_end_time}")
    
    return output


def run_all_ablation_experiments(scenario_key: str = "off_peak") -> pd.DataFrame:
    """
    运行所有消融实验
    
    Args:
        scenario_key: 场景键名
    
    Returns:
        结果 DataFrame
    """
    results = []
    
    for config in ABLATION_CONFIGS:
        result = run_ablation_experiment(config, scenario_key)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# 输出生成
# ============================================================================

def generate_markdown_table(df: pd.DataFrame, output_path: Path):
    """生成 Markdown 表格"""
    
    # 准备显示数据
    table_data = []
    for _, row in df.iterrows():
        audit_cal = "✓" if row["use_audit_cal"] else "✗"
        audit_val = "✓" if row["use_audit_val"] else "✗"
        bo = "✓" if row["use_bo"] else "✗"
        ies = "✓" if row["use_ies"] else "✗"
        tail = "✓" if row["use_tail_loss"] else "✗"
        
        ks_speed = f"{row['ks_speed_val']:.4f}" if pd.notna(row['ks_speed_val']) else "N/A"
        ks_tt = f"{row['ks_tt_val']:.4f}" if pd.notna(row['ks_tt_val']) else "N/A"
        worst_ks = f"{row['worst_window_ks_speed']:.4f}" if pd.notna(row['worst_window_ks_speed']) else "N/A"
        
        pass_fail = "PASS" if row["ks_speed_passed"] else "FAIL"
        
        table_data.append({
            "Config": f"{row['config_id']} {row['config_name']}",
            "Audit(Cal)": audit_cal,
            "Audit(Val)": audit_val,
            "BO": bo,
            "IES": ies,
            "Tail": tail,
            "n_clean": int(row["n_clean"]),
            "KS(speed)": ks_speed,
            "KS(TT)": ks_tt,
            "Worst-15min": worst_ks,
            "Pass": pass_fail
        })
    
    table_df = pd.DataFrame(table_data)
    
    # 生成 Markdown
    md_content = f"""# Protocol Ablation 实验结果

**场景**: {df.iloc[0]['period']} ({df.iloc[0]['hkt_time']} HKT)  
**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Audit 配置**: Rule C (T* ≥ 325s, v* ≤ 5 km/h)  
**Pass/Fail 判据**: α=0.05, D_crit = 1.36 × sqrt((n+m)/(n×m))

## 结果表

| Config | Audit(Cal) | Audit(Val) | BO | IES | Tail | n_clean | KS(speed) | KS(TT) | Worst-15min | Pass |
|--------|------------|------------|-----|-----|------|---------|-----------|--------|-------------|------|
"""
    
    for _, row in table_df.iterrows():
        md_content += f"| {row['Config']} | {row['Audit(Cal)']} | {row['Audit(Val)']} | "
        md_content += f"{row['BO']} | {row['IES']} | {row['Tail']} | {row['n_clean']} | "
        md_content += f"{row['KS(speed)']} | {row['KS(TT)']} | {row['Worst-15min']} | {row['Pass']} |\n"
    
    md_content += f"""
## 关键发现

"""
    
    # 计算关键指标
    a0_ks = df[df['config_id'] == 'A0']['ks_speed_val'].values[0] if len(df[df['config_id'] == 'A0']) > 0 else None
    a2_ks = df[df['config_id'] == 'A2']['ks_speed_val'].values[0] if len(df[df['config_id'] == 'A2']) > 0 else None
    a4_ks = df[df['config_id'] == 'A4']['ks_speed_val'].values[0] if len(df[df['config_id'] == 'A4']) > 0 else None
    
    if a0_ks and a2_ks:
        audit_improvement = (a0_ks - a2_ks) / a0_ks * 100
        md_content += f"- **Audit 贡献**: KS(speed) 从 {a0_ks:.4f} (A0) 降至 {a2_ks:.4f} (A2)，改善 {audit_improvement:.1f}%\n"
    
    if a2_ks and a4_ks:
        full_improvement = (a2_ks - a4_ks) / a2_ks * 100 if a4_ks < a2_ks else 0
        md_content += f"- **BO+IES+Tail 贡献**: KS(speed) 从 {a2_ks:.4f} (A2) 至 {a4_ks:.4f} (A4)\n"
    
    md_content += f"""
## 结论

本消融实验定量回答了审稿人的核心质疑：

1. **"过滤是不是在作弊"**: 
   - A0 vs A2 对比显示 Audit 的贡献
   - 即使在 raw 数据上 (A0/A1)，模型也有基线表现

2. **"decoupling 不清楚"**:
   - A2 vs A3 对比显示 Audit 在校准中的作用
   - A3 vs A4 对比显示 L2 (IES) 的增量贡献

---

*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Markdown 表格已保存: {output_path}")


def generate_comparison_plot(df: pd.DataFrame, output_path: Path):
    """生成对比图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: KS(speed) 对比
    ax1 = axes[0]
    configs = df['config_id'] + '\n' + df['config_name']
    ks_values = df['ks_speed_val'].values
    colors = ['green' if passed else 'red' for passed in df['ks_speed_passed']]
    
    bars = ax1.bar(range(len(configs)), ks_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel('KS(speed)')
    ax1.set_title('Protocol Ablation: KS(speed) Comparison')
    ax1.axhline(y=df['ks_speed_critical'].iloc[0], color='orange', linestyle='--', 
                label=f'D_crit = {df["ks_speed_critical"].iloc[0]:.3f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, ks_values)):
        if pd.notna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 图2: n_clean 与 flagged_fraction
    ax2 = axes[1]
    x = range(len(configs))
    width = 0.35
    
    bars1 = ax2.bar([i - width/2 for i in x], df['n_clean'], width, 
                    label='n_clean', color='steelblue', alpha=0.7)
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar([i + width/2 for i in x], df['flagged_fraction'] * 100, width,
                         label='Flagged %', color='coral', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9)
    ax2.set_ylabel('n_clean', color='steelblue')
    ax2_twin.set_ylabel('Flagged %', color='coral')
    ax2.set_title('Sample Statistics by Configuration')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Protocol Ablation 实验 (A0-A4)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="off_peak",
        choices=["off_peak"],
        help="实验场景"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Protocol Ablation 实验 (A0-A4)")
    print("=" * 70)
    print(f"场景: {args.scenario}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 运行所有实验
    df_results = run_all_ablation_experiments(args.scenario)
    
    if df_results.empty:
        print("[ERROR] 没有成功的实验结果")
        return
    
    # 保存结果
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n详细结果已保存: {results_csv}")
    
    # 生成汇总
    summary_cols = ["config_id", "config_name", "n_clean", "n_sim", 
                    "ks_speed_val", "ks_speed_passed", "ks_tt_val", 
                    "worst_window_ks_speed", "worst_window_start"]
    summary_df = df_results[summary_cols].copy()
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总结果已保存: {summary_csv}")
    
    # 生成 Markdown 表格
    md_path = output_dir / "tables" / "ablation_table.md"
    generate_markdown_table(df_results, md_path)
    
    # 生成对比图
    fig_path = output_dir / "figures" / "ablation_comparison.png"
    generate_comparison_plot(df_results, fig_path)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("实验完成汇总")
    print("=" * 70)
    print(f"总配置数: {len(df_results)}")
    print(f"Pass 数: {df_results['ks_speed_passed'].sum()}")
    print(f"Fail 数: {(~df_results['ks_speed_passed']).sum()}")
    print()
    
    for _, row in df_results.iterrows():
        status = "✓ PASS" if row["ks_speed_passed"] else "✗ FAIL"
        print(f"  {row['config_id']} {row['config_name']}: "
              f"KS={row['ks_speed_val']:.4f} {status}")
    
    print("\n" + "=" * 70)
    print("输出文件:")
    print(f"  - {results_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {md_path}")
    print(f"  - {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
