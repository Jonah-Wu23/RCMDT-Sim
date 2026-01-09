#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_scale_sweep.py - Scale Sweep 实验（拥堵强度 → RCMDT 增益）
==============================================================

实验目标：
将 seed "看似不生效" 的现象升级为 "拥堵强度 (scale) → RCMDT/IES 增益" 的系统证据。

实验设计：
- configs: A3 (Audit-in-Cal), A4 (Full-RCMDT)
- scenarios: off_peak 1h, pm_peak 1h （可选 pm_peak 2h）
- scales: 0.00, 0.05, 0.10, 0.15, 0.20, 0.30（至少 5 档）
- seeds: 0..9（至少 10 个）

输出：
- data/experiments_v4/scale_sweep/{scenario}/{duration}/scale{X}/A{config}/seed{seed}/stopinfo.xml
- results.csv: 详细结果（长表）
- summary.csv: 汇总统计（每个 scale×config 的均值/标准差/CI）
- tables/scale_sweep_summary.md: Markdown 汇总表
- tables/scale_sweep_delta.md: ΔKS = KS(A3)-KS(A4) 表格
- README.md: seed 机制解释

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval.metrics_v4 import (
    compute_metrics_v4,
    MetricsV4Result,
    AuditConfig,
    load_real_link_stats
)
from tools.scale_background_routes import scale_background


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ScaleSweepConfig:
    """Scale Sweep 实验配置"""
    scenario: str
    duration_hours: int
    scale: float
    config_id: str
    seed: int
    
    def get_output_dir(self, base_dir: Path) -> Path:
        """获取输出目录路径"""
        return base_dir / self.scenario / f"{self.duration_hours}h" / f"scale{self.scale:.2f}" / self.config_id / f"seed{self.seed}"


SCENARIOS = {
    "off_peak": {
        "hkt_time": "15:00-16:00",
        "real_stats": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
        "base_config": PROJECT_ROOT / "sumo" / "config" / "experiment_robustness.sumocfg",
        "base_bg_routes": PROJECT_ROOT / "sumo" / "routes" / "background_offpeak.rou.xml",
        "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
        "network": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
        "additional": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
        "sim_end": 3900
    },
    "pm_peak": {
        "hkt_time": "17:35-18:35",
        "real_stats": PROJECT_ROOT / "data2" / "processed" / "link_stats_peak.csv",
        "base_config": PROJECT_ROOT / "sumo" / "config" / "experiment2_calibrated.sumocfg",
        "base_bg_routes": PROJECT_ROOT / "sumo" / "routes" / "background_corridor_source_filtered_test.rou.xml",
        "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
        "network": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
        "additional": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
        "sim_end": 3900
    }
}

# Scale 档位（至少 5 档）
SCALE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

# Seeds（至少 10 个）
SEEDS = list(range(10))

# Configs（A3 和 A4）
CONFIGS = ["A3", "A4"]

# 配置参数（从 protocol_ablation 借用）
CONFIG_PARAMS = {
    "A3": {
        "capacityFactor": 1.45,
        "minGap": 0.48,
        "impatience": 0.95,
        "use_audit": True,
        "use_ies": False
    },
    "A4": {
        "capacityFactor": 1.50,
        "minGap": 0.50,
        "impatience": 1.00,
        "use_audit": True,
        "use_ies": True
    }
}

OUTPUT_BASE = PROJECT_ROOT / "data" / "experiments_v4" / "scale_sweep"
DIST_CSV = PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"


# ============================================================================
# SUMO 仿真运行
# ============================================================================

def create_scaled_routes(base_routes: Path, scale: float, seed: int, output_path: Path) -> None:
    """创建 scaled 背景流量文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if scale == 0.0:
        # scale=0 表示无背景车，创建空文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            f.write('    <!-- No background traffic (scale=0.00) -->\n')
            f.write('</routes>\n')
    else:
        # 使用 scale_background 工具
        scale_background(
            in_xml=str(base_routes),
            out_xml=str(output_path),
            alpha=scale,
            seed_tag=f"seed{seed}"
        )


def create_sumo_config(
    scenario_info: Dict,
    bg_routes: Path,
    output_dir: Path,
    config_id: str,
    seed: int
) -> Path:
    """创建 SUMO 配置文件和 additional 文件"""
    config_path = output_dir / "experiment.sumocfg"
    additional_path = output_dir / "vtype.add.xml"
    
    params = CONFIG_PARAMS[config_id]
    
    # 创建 additional 文件定义车辆类型
    additional_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    
    <!-- Config {config_id} Vehicle Type Parameters -->
    <vType id="bus" vClass="bus" speedFactor="{params['capacityFactor']}" 
          minGap="{params['minGap']}" impatience="{params['impatience']}" />
    <vType id="car" speedFactor="{params['capacityFactor']}" 
          minGap="{params['minGap']}" impatience="{params['impatience']}" />
    <vType id="DEFAULT_VEHTYPE" speedFactor="{params['capacityFactor']}" 
          minGap="{params['minGap']}" impatience="{params['impatience']}" />
    
</additional>
'''
    
    with open(additional_path, 'w', encoding='utf-8') as f:
        f.write(additional_xml)
    
    # 创建 SUMO 配置文件
    bus_stops_add = scenario_info.get('additional', '')
    additional_files = f"{additional_path},{bus_stops_add}" if bus_stops_add else str(additional_path)
    
    config_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{scenario_info['network']}" />
        <route-files value="{bg_routes},{scenario_info['bus_routes']}" />
        <additional-files value="{additional_files}" />
    </input>

    <time>
        <begin value="0" />
        <end value="{scenario_info['sim_end']}" />
    </time>

    <processing>
        <time-to-teleport value="300" />
        <ignore-route-errors value="true" />
        <lateral-resolution value="0.8" />
    </processing>

    <random>
        <seed value="{seed}" />
    </random>

    <report>
        <verbose value="false" />
        <no-step-log value="true" />
    </report>

    <output>
        <stop-output value="{output_dir / 'stopinfo.xml'}" />
        <tripinfo-output value="{output_dir / 'tripinfo.xml'}" />
    </output>

</configuration>
'''
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_xml)
    
    return config_path


def run_sumo_simulation(config_path: Path) -> bool:
    """运行 SUMO 仿真"""
    try:
        result = subprocess.run(
            ["sumo", "-c", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"  [ERROR] SUMO 返回错误码: {result.returncode}")
            if result.stderr:
                # 查找 "Error:" 行
                stderr_lines = result.stderr.split('\n')
                error_lines = [line for line in stderr_lines if 'Error:' in line]
                if error_lines:
                    print(f"  错误信息:")
                    for line in error_lines[:10]:  # 显示前 10 个错误
                        print(f"    {line}")
                else:
                    print(f"  STDERR (前 1000 字符): {result.stderr[:1000]}")
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] SUMO 仿真失败: {e}")
        return False


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment(
    scenario: str,
    duration_hours: int,
    scale: float,
    config_id: str,
    seed: int,
    simulate_only: bool = False
) -> Optional[Dict]:
    """
    运行单次 scale sweep 实验
    
    Args:
        scenario: 场景名称
        duration_hours: 持续时长（小时）
        scale: 背景流量缩放比例
        config_id: 配置 ID（A3/A4）
        seed: 随机种子
        simulate_only: 仅模拟（不实际运行 SUMO）
    
    Returns:
        实验结果字典
    """
    config = ScaleSweepConfig(scenario, duration_hours, scale, config_id, seed)
    output_dir = config.get_output_dir(OUTPUT_BASE)
    
    print(f"\n{'='*60}")
    print(f"运行实验: {scenario} {duration_hours}h, scale={scale:.2f}, {config_id}, seed={seed}")
    print(f"{'='*60}")
    
    scenario_info = SCENARIOS[scenario]
    
    # 1. 创建 scaled 背景流量文件
    scaled_routes = output_dir / "background_scaled.rou.xml"
    create_scaled_routes(
        base_routes=scenario_info['base_bg_routes'],
        scale=scale,
        seed=seed,
        output_path=scaled_routes
    )
    print(f"  [1/4] 已创建 scaled routes (scale={scale:.2f})")
    
    # 2. 创建 SUMO 配置
    sumo_config = create_sumo_config(
        scenario_info=scenario_info,
        bg_routes=scaled_routes,
        output_dir=output_dir,
        config_id=config_id,
        seed=seed
    )
    print(f"  [2/4] 已创建 SUMO 配置")
    
    # 3. 运行 SUMO 仿真
    if not simulate_only:
        print(f"  [3/4] 运行 SUMO 仿真...")
        success = run_sumo_simulation(sumo_config)
        if not success:
            print(f"  [ERROR] SUMO 仿真失败")
            return None
        print(f"  [3/4] 仿真完成")
    else:
        print(f"  [3/4] 跳过仿真（模拟模式）")
        return None
    
    # 4. 评估结果（使用 metrics_v4）
    stopinfo_path = output_dir / "stopinfo.xml"
    if not stopinfo_path.exists():
        print(f"  [ERROR] stopinfo.xml 不存在")
        return None
    
    print(f"  [4/4] 评估结果...")
    
    # 加载真实数据
    real_stats_path = scenario_info['real_stats']
    if not real_stats_path.exists():
        print(f"  [ERROR] 真实数据不存在: {real_stats_path}")
        return None
    
    df_real = load_real_link_stats(str(real_stats_path))
    
    # 使用 metrics_v4 评估
    audit_config = AuditConfig.from_protocol()
    result = compute_metrics_v4(
        real_data=df_real,
        sim_data=str(stopinfo_path),
        dist_file=str(DIST_CSV),
        audit_config=audit_config,
        scenario=f"{scenario}_{duration_hours}h",
        route="68X"
    )
    
    if result is None:
        print(f"  [ERROR] metrics_v4 评估失败")
        return None
    
    # 整理结果
    result_dict = {
        "scenario": scenario,
        "duration_hours": duration_hours,
        "scale": scale,
        "config_id": config_id,
        "seed": seed,
        "n_clean": result.audit_stats.n_clean,
        "n_sim": result.n_sim,
        "n_events": result.audit_stats.n_raw,  # 使用 n_raw 作为事件数
        "ks_speed": result.ks_speed_clean.ks_stat,
        "ks_tt": result.ks_tt_clean.ks_stat,
        "dcrit_speed": result.ks_speed_clean.critical_value,
        "dcrit_tt": result.ks_tt_clean.critical_value,
        "passed": result.sanity_passed,
        "worst_window_ks": result.worst_window_speed.worst_ks,
        "worst_window_start": result.worst_window_speed.window_start_time,
        "flagged_fraction": result.audit_stats.flagged_fraction
    }
    
    print(f"  [完成] KS(speed)={result.ks_speed_clean.ks_stat:.4f}, "
          f"KS(TT)={result.ks_tt_clean.ks_stat:.4f}, pass={result.sanity_passed}")
    
    return result_dict


# ============================================================================
# 批量实验运行
# ============================================================================

def run_scale_sweep_batch(
    scenarios: List[str],
    duration_hours: List[int],
    scales: List[float],
    configs: List[str],
    seeds: List[int],
    simulate_only: bool = False
) -> pd.DataFrame:
    """
    批量运行 scale sweep 实验
    
    Args:
        scenarios: 场景列表
        duration_hours: 时长列表
        scales: scale 档位列表
        configs: 配置列表
        seeds: seed 列表
        simulate_only: 仅模拟
    
    Returns:
        结果 DataFrame
    """
    results = []
    total = len(scenarios) * len(duration_hours) * len(scales) * len(configs) * len(seeds)
    current = 0
    
    print(f"\n{'='*70}")
    print(f"Scale Sweep 批量实验")
    print(f"{'='*70}")
    print(f"总实验数: {total}")
    print(f"  - Scenarios: {scenarios}")
    print(f"  - Durations: {duration_hours}h")
    print(f"  - Scales: {scales}")
    print(f"  - Configs: {configs}")
    print(f"  - Seeds: {seeds}")
    print(f"{'='*70}\n")
    
    for scenario in scenarios:
        for duration in duration_hours:
            for scale in scales:
                for config_id in configs:
                    for seed in seeds:
                        current += 1
                        print(f"\n[{current}/{total}] {scenario} {duration}h scale={scale:.2f} {config_id} seed={seed}")
                        
                        result = run_single_experiment(
                            scenario=scenario,
                            duration_hours=duration,
                            scale=scale,
                            config_id=config_id,
                            seed=seed,
                            simulate_only=simulate_only
                        )
                        
                        if result:
                            results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# 统计分析
# ============================================================================

def compute_summary_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    计算汇总统计
    
    对每个 (scenario, duration, scale, config) 组合计算：
    - ks_speed_mean/std/ci95
    - ks_tt_mean/std/ci95
    - pass_rate
    """
    summary_rows = []
    
    grouped = df_results.groupby(['scenario', 'duration_hours', 'scale', 'config_id'])
    
    for (scenario, duration, scale, config_id), group in grouped:
        n = len(group)
        
        # KS(speed) 统计
        ks_speed_vals = group['ks_speed'].dropna()
        if len(ks_speed_vals) > 0:
            ks_speed_mean = ks_speed_vals.mean()
            ks_speed_std = ks_speed_vals.std()
            ks_speed_ci = stats.t.interval(0.95, len(ks_speed_vals)-1, 
                                           loc=ks_speed_mean, 
                                           scale=stats.sem(ks_speed_vals))
        else:
            ks_speed_mean = ks_speed_std = np.nan
            ks_speed_ci = (np.nan, np.nan)
        
        # KS(TT) 统计
        ks_tt_vals = group['ks_tt'].dropna()
        if len(ks_tt_vals) > 0:
            ks_tt_mean = ks_tt_vals.mean()
            ks_tt_std = ks_tt_vals.std()
            ks_tt_ci = stats.t.interval(0.95, len(ks_tt_vals)-1,
                                        loc=ks_tt_mean,
                                        scale=stats.sem(ks_tt_vals))
        else:
            ks_tt_mean = ks_tt_std = np.nan
            ks_tt_ci = (np.nan, np.nan)
        
        # Pass rate
        pass_rate = group['passed'].mean() if 'passed' in group.columns else np.nan
        
        summary_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'config_id': config_id,
            'n_runs': n,
            'ks_speed_mean': ks_speed_mean,
            'ks_speed_std': ks_speed_std,
            'ks_speed_ci_low': ks_speed_ci[0],
            'ks_speed_ci_high': ks_speed_ci[1],
            'ks_tt_mean': ks_tt_mean,
            'ks_tt_std': ks_tt_std,
            'ks_tt_ci_low': ks_tt_ci[0],
            'ks_tt_ci_high': ks_tt_ci[1],
            'pass_rate': pass_rate
        })
    
    return pd.DataFrame(summary_rows)


def compute_delta_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    计算 ΔKS = KS(A3) - KS(A4)
    
    对每个 (scenario, duration, scale) 计算 A3 与 A4 的差异
    """
    delta_rows = []
    
    grouped = df_summary.groupby(['scenario', 'duration_hours', 'scale'])
    
    for (scenario, duration, scale), group in grouped:
        a3_row = group[group['config_id'] == 'A3']
        a4_row = group[group['config_id'] == 'A4']
        
        if len(a3_row) == 0 or len(a4_row) == 0:
            continue
        
        a3 = a3_row.iloc[0]
        a4 = a4_row.iloc[0]
        
        # ΔKS(speed)
        delta_ks_speed = a3['ks_speed_mean'] - a4['ks_speed_mean']
        delta_ks_speed_ci_low = a3['ks_speed_ci_low'] - a4['ks_speed_ci_high']
        delta_ks_speed_ci_high = a3['ks_speed_ci_high'] - a4['ks_speed_ci_low']
        
        # ΔKS(TT)
        delta_ks_tt = a3['ks_tt_mean'] - a4['ks_tt_mean']
        delta_ks_tt_ci_low = a3['ks_tt_ci_low'] - a4['ks_tt_ci_high']
        delta_ks_tt_ci_high = a3['ks_tt_ci_high'] - a4['ks_tt_ci_low']
        
        delta_rows.append({
            'scenario': scenario,
            'duration_hours': duration,
            'scale': scale,
            'delta_ks_speed': delta_ks_speed,
            'delta_ks_speed_ci_low': delta_ks_speed_ci_low,
            'delta_ks_speed_ci_high': delta_ks_speed_ci_high,
            'delta_ks_tt': delta_ks_tt,
            'delta_ks_tt_ci_low': delta_ks_tt_ci_low,
            'delta_ks_tt_ci_high': delta_ks_tt_ci_high
        })
    
    return pd.DataFrame(delta_rows)


# ============================================================================
# Markdown 表格生成
# ============================================================================

def generate_summary_markdown(df_summary: pd.DataFrame, output_path: Path) -> None:
    """生成汇总 Markdown 表格"""
    md_lines = ["# Scale Sweep Summary", ""]
    md_lines.append("每个 (scenario, duration, scale, config) 的统计汇总（n=10 seeds）")
    md_lines.append("")
    
    df_display = df_summary.copy()
    df_display['ks_speed_ci'] = df_display.apply(
        lambda row: f"[{row['ks_speed_ci_low']:.4f}, {row['ks_speed_ci_high']:.4f}]", axis=1
    )
    df_display['ks_tt_ci'] = df_display.apply(
        lambda row: f"[{row['ks_tt_ci_low']:.4f}, {row['ks_tt_ci_high']:.4f}]", axis=1
    )
    
    cols = ['scenario', 'duration_hours', 'scale', 'config_id', 'n_runs',
            'ks_speed_mean', 'ks_speed_std', 'ks_speed_ci',
            'ks_tt_mean', 'ks_tt_std', 'ks_tt_ci', 'pass_rate']
    df_display = df_display[cols]
    df_display.columns = ['Scenario', 'Duration(h)', 'Scale', 'Config', 'N',
                          'KS(speed)_μ', 'KS(speed)_σ', 'KS(speed)_CI95',
                          'KS(TT)_μ', 'KS(TT)_σ', 'KS(TT)_CI95', 'Pass Rate']
    
    md_lines.append(df_display.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("**说明**:")
    md_lines.append("- A3: Audit-in-Calibration (无 L2/IES)")
    md_lines.append("- A4: Full-RCMDT (A3 + L2/IES)")
    md_lines.append("- CI95: 95% 置信区间（t-分布）")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n汇总表已保存: {output_path}")


def generate_delta_markdown(df_delta: pd.DataFrame, output_path: Path) -> None:
    """生成 ΔKS Markdown 表格"""
    md_lines = ["# Scale Sweep Delta (ΔKS = A3 - A4)", ""]
    md_lines.append("展示 RCMDT/IES 在不同拥堵强度下的改进效果")
    md_lines.append("")
    
    if len(df_delta) == 0:
        md_lines.append("（暂无数据：需要同时有 A3 和 A4 的结果才能计算 Delta）")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"\nDelta 表已保存（空）: {output_path}")
        return
    
    df_display = df_delta.copy()
    df_display['delta_ks_speed_ci'] = df_display.apply(
        lambda row: f"[{row['delta_ks_speed_ci_low']:.4f}, {row['delta_ks_speed_ci_high']:.4f}]", axis=1
    )
    df_display['delta_ks_tt_ci'] = df_display.apply(
        lambda row: f"[{row['delta_ks_tt_ci_low']:.4f}, {row['delta_ks_tt_ci_high']:.4f}]", axis=1
    )
    
    cols = ['scenario', 'duration_hours', 'scale', 
            'delta_ks_speed', 'delta_ks_speed_ci',
            'delta_ks_tt', 'delta_ks_tt_ci']
    df_display = df_display[cols]
    df_display.columns = ['Scenario', 'Duration(h)', 'Scale',
                          'ΔKS(speed)', 'ΔKS(speed)_CI95',
                          'ΔKS(TT)', 'ΔKS(TT)_CI95']
    
    md_lines.append(df_display.to_markdown(index=False, floatfmt=".4f"))
    md_lines.append("")
    md_lines.append("**说明**:")
    md_lines.append("- ΔKS = KS(A3) - KS(A4)")
    md_lines.append("- **正值** 表示 A4 (Full-RCMDT) 优于 A3 (仅 Audit)")
    md_lines.append("- Scale 越高 → 系统随机性越强 → RCMDT/IES 增益越显著（预期）")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Delta 表已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scale Sweep 实验")
    parser.add_argument("--scenarios", nargs='+', default=["off_peak"], 
                        choices=["off_peak", "pm_peak"],
                        help="场景列表")
    parser.add_argument("--durations", nargs='+', type=int, default=[1],
                        help="时长列表（小时）")
    parser.add_argument("--scales", nargs='+', type=float, 
                        default=SCALE_LEVELS,
                        help="Scale 档位")
    parser.add_argument("--configs", nargs='+', default=CONFIGS,
                        choices=["A3", "A4"],
                        help="配置列表")
    parser.add_argument("--seeds", nargs='+', type=int, default=SEEDS,
                        help="随机种子列表")
    parser.add_argument("--simulate-only", action="store_true",
                        help="仅模拟（不运行 SUMO）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_BASE),
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 运行批量实验
    df_results = run_scale_sweep_batch(
        scenarios=args.scenarios,
        duration_hours=args.durations,
        scales=args.scales,
        configs=args.configs,
        seeds=args.seeds,
        simulate_only=args.simulate_only
    )
    
    # 保存详细结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n详细结果已保存: {results_csv}")
    
    if len(df_results) == 0:
        print("\n[警告] 没有成功的实验结果")
        return
    
    # 计算汇总统计
    df_summary = compute_summary_statistics(df_results)
    summary_csv = output_dir / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"汇总统计已保存: {summary_csv}")
    
    # 计算 Delta 表
    df_delta = compute_delta_table(df_summary)
    delta_csv = output_dir / "delta.csv"
    df_delta.to_csv(delta_csv, index=False)
    print(f"Delta 表已保存: {delta_csv}")
    
    # 生成 Markdown 表格
    tables_dir = PROJECT_ROOT / "tables"
    generate_summary_markdown(df_summary, tables_dir / "scale_sweep_summary.md")
    generate_delta_markdown(df_delta, tables_dir / "scale_sweep_delta.md")
    
    print(f"\n{'='*70}")
    print("Scale Sweep 实验完成！")
    print(f"{'='*70}")
    print(f"结果目录: {output_dir}")
    print(f"表格目录: {tables_dir}")


if __name__ == "__main__":
    main()
