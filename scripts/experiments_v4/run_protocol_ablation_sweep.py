#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_protocol_ablation_sweep.py - Protocol Ablation 多场景多seed批量实验
========================================================================

扩展 A0-A4 到多场景/多seed/多时长，用更强场景与统计把审稿人质疑逐条击穿。

运行维度：
- time windows: Off-peak(15:00-16:00), PM Peak(17:00-18:00), Next-day transfer
- durations: 1h vs 2h
- seeds: [0,1,2,3,4]
- routes: 68X + 960

输出：
- long-form results.csv（逐窗口逐seed）
- summary.csv（均值/中位数/CI/Pass率）
- Figure：KS boxplots, Pass率柱状图, worst-window heatmap

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 导入评估模块
from experiments_v4.run_protocol_ablation import (
    ABLATION_CONFIGS,
    run_ablation_experiment,
    AblationConfig
)
from eval.metrics_v4 import compute_metrics_v4, AuditConfig

# ============================================================================
# 实验配置
# ============================================================================

# 场景定义
SCENARIOS = {
    "off_peak": {
        "name": "Off-Peak",
        "hkt_start": "15:00",
        "hkt_end": "16:00",
        "utc_start_sec": 25200,  # 07:00 UTC
        "utc_end_sec": 28800,    # 08:00 UTC
        "duration_sec": 3600,
        "real_stats": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
        "dist_csv": PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv",
    },
    "pm_peak": {
        "name": "PM Peak",
        "hkt_start": "17:00",
        "hkt_end": "18:00",
        "utc_start_sec": 61200,  # 09:00 UTC
        "utc_end_sec": 64800,    # 10:00 UTC
        "duration_sec": 3600,
        "real_stats": PROJECT_ROOT / "data" / "processed" / "link_stats.csv",
        "dist_csv": PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv",
    },
}

# 时长配置
DURATIONS = {
    "1h": 3600,
    "2h": 7200,
}

# Seed 集合
SEEDS = [0, 1, 2, 3, 4]

# 路线配置
ROUTES = ["68X"]  # 先只做 68X，后续可扩展到 960

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments_v4" / "protocol_ablation_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 仿真配置（基于 run_ablation_simulations.py）
# ============================================================================

SUMO_CONFIG = {
    "net_file": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
    "bus_stops": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
    "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
    "background_routes": PROJECT_ROOT / "sumo" / "routes" / "background_cropped.rou.xml",
    "time_to_teleport": 300,
}

# 各配置的参数和设置
ABLATION_SIM_CONFIGS = {
    "A0": {
        "name": "Zero-shot",
        "use_background": False,
        "scale": 0.0,
        "l1_params_source": "baseline",
        "l2_params": None,
    },
    "A1": {
        "name": "Raw-L1 (BO)",
        "use_background": True,
        "scale": 0.1,
        "l1_params_source": "best",
        "l2_params": None,
    },
    "A2": {
        "name": "Audit-Val-Only",
        "use_background": True,
        "scale": 0.1,
        "l1_params_source": "best",
        "l2_params": None,
        "same_as": "A1",
    },
    "A3": {
        "name": "Audit-in-Cal + Tail",
        "use_background": True,
        "scale": 0.12,
        "l1_params_source": "best",
        "l2_params": {
            "capacityFactor": 1.2,
            "minGap_background": 2.0,
            "impatience": 0.6,
        },
    },
    "A4": {
        "name": "Full-RCMDT (IES)",
        "use_background": True,
        "scale": 0.15,
        "l1_params_source": "best",
        "l2_params": {
            "capacityFactor": 1.5,
            "minGap_background": 0.5,
            "impatience": 1.0,
        },
    },
}

# ============================================================================
# 参数加载函数
# ============================================================================

def load_l1_params(source: str) -> Dict:
    """加载 L1 微观参数"""
    if source == "baseline":
        path = PROJECT_ROOT / "config" / "calibration" / "baseline_parameters.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bus_params = data['micro_parameters']['kmb_double_decker']
        dwell = data['dwell_time_model']
        
        return {
            't_board': dwell['mean_s'] / 10,
            't_fixed': dwell['mean_s'] * 0.1,
            'tau': bus_params['tau'],
            'sigma': bus_params['sigma'],
            'minGap': bus_params['minGap'],
            'accel': bus_params['accel'],
            'decel': bus_params['decel'],
            'maxSpeed': bus_params['maxSpeed'],
        }
    else:  # best
        path = PROJECT_ROOT / "config" / "calibration" / "best_l1_parameters.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        params = data['best_parameters'].copy()
        params['maxSpeed'] = 20.0
        return params


def create_vtype_with_params(l1_params: Dict, l2_params: Optional[Dict] = None) -> str:
    """创建带参数的 vType XML 片段"""
    
    bus_vtype = f'''    <vType id="kmb_double_decker" 
           accel="{l1_params.get('accel', 2.6):.4f}"
           decel="{l1_params.get('decel', 4.5):.4f}"
           sigma="{l1_params.get('sigma', 0.5):.4f}"
           tau="{l1_params.get('tau', 1.0):.4f}"
           minGap="{l1_params.get('minGap', 2.5):.4f}"
           maxSpeed="{l1_params.get('maxSpeed', 20.0):.2f}"
           length="12.0"
           color="1,0.5,0"
           vClass="bus"/>
'''
    
    if l2_params:
        bg_mingap = l2_params.get('minGap_background', 2.5)
        bg_impatience = l2_params.get('impatience', 0.5)
    else:
        bg_mingap = 2.5
        bg_impatience = 0.5
    
    bg_vtype = f'''    <vType id="passenger"
           accel="2.6"
           decel="4.5"
           sigma="0.5"
           tau="1.0"
           minGap="{bg_mingap:.4f}"
           impatience="{bg_impatience:.4f}"
           maxSpeed="13.89"
           length="5.0"
           vClass="passenger"/>
'''
    
    return bus_vtype + bg_vtype


def create_route_file_with_vtypes(
    config_id: str,
    l1_params: Dict,
    l2_params: Optional[Dict],
    use_background: bool,
    output_dir: Path
) -> str:
    """创建带自定义 vType 的路由文件"""
    
    output_path = output_dir / f"{config_id}_routes.rou.xml"
    
    # 读取原始公交路由
    bus_routes_path = SUMO_CONFIG["bus_routes"]
    tree = ET.parse(bus_routes_path)
    root = tree.getroot()
    
    # 移除现有的 vType（如果有）
    for vtype in root.findall('vType'):
        root.remove(vtype)
    
    # 创建新的 XML
    new_root = ET.Element('routes')
    new_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    
    # 添加 vTypes
    vtype_xml = create_vtype_with_params(l1_params, l2_params)
    
    # 直接写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
        f.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
        
        # 写入 vTypes
        f.write(vtype_xml)
        f.write('\n')
        
        # 复制原始路由的 flow/vehicle/route 元素
        for elem in root:
            if elem.tag != 'vType':
                elem_str = ET.tostring(elem, encoding='unicode')
                f.write('    ' + elem_str + '\n')
        
        f.write('</routes>\n')
    
    return str(output_path)


def create_sumocfg(
    config_id: str,
    route_file: str,
    use_background: bool,
    scale: float,
    duration_sec: int,
    seed: int,
    output_dir: Path
) -> Tuple[str, str]:
    """创建 SUMO 配置文件"""
    
    sumocfg_path = output_dir / f"{config_id}.sumocfg"
    stopinfo_path = output_dir / "stopinfo.xml"
    
    # 构建 route-files
    route_files = [route_file]
    if use_background:
        route_files.append(str(SUMO_CONFIG["background_routes"]))
    
    route_files_str = ",".join(route_files)
    
    config_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{SUMO_CONFIG['net_file']}"/>
        <route-files value="{route_files_str}"/>
        <additional-files value="{SUMO_CONFIG['bus_stops']}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{duration_sec}"/>
    </time>

    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="{SUMO_CONFIG['time_to_teleport']}"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
    
    <output>
        <stop-output value="{stopinfo_path}"/>
    </output>

</configuration>
'''
    
    with open(sumocfg_path, 'w', encoding='utf-8') as f:
        f.write(config_xml)
    
    return str(sumocfg_path), str(stopinfo_path)


def run_sumo_simulation(sumocfg_path: str, scale: float, seed: int) -> bool:
    """运行 SUMO 仿真"""
    
    cmd = [
        "sumo",
        "-c", sumocfg_path,
        "--scale", f"{scale:.3f}" if scale > 0 else "1.0",
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def run_single_ablation_simulation(
    config_id: str,
    scenario_key: str,
    duration_key: str,
    seed: int
) -> Optional[Dict]:
    """运行单个消融配置的仿真"""
    
    config = ABLATION_SIM_CONFIGS[config_id]
    scenario = SCENARIOS[scenario_key]
    duration_sec = DURATIONS[duration_key]
    
    # 检查是否与其他配置共享仿真
    if 'same_as' in config:
        ref_id = config['same_as']
        ref_output_dir = OUTPUT_DIR / f"{scenario_key}_{duration_key}" / f"seed{seed}" / ref_id
        ref_stopinfo = ref_output_dir / "stopinfo.xml"
        if ref_stopinfo.exists():
            output_dir = OUTPUT_DIR / f"{scenario_key}_{duration_key}" / f"seed{seed}" / config_id
            output_dir.mkdir(parents=True, exist_ok=True)
            dst_path = output_dir / "stopinfo.xml"
            if not dst_path.exists():
                import shutil
                shutil.copy(ref_stopinfo, dst_path)
            
            # 统计 stopinfo 事件数
            tree = ET.parse(dst_path)
            n_events = len(tree.findall('.//stopinfo'))
            
            return {
                "config_id": config_id,
                "scenario": scenario_key,
                "duration": duration_key,
                "seed": seed,
                "stopinfo_path": str(dst_path),
                "n_events": n_events,
                "success": True,
            }
        else:
            return None
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / f"{scenario_key}_{duration_key}" / f"seed{seed}" / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在 stopinfo.xml
    stopinfo_path = output_dir / "stopinfo.xml"
    if stopinfo_path.exists():
        print(f"  [SKIP] stopinfo.xml 已存在: {stopinfo_path.relative_to(PROJECT_ROOT)}")
        # 统计已有 stopinfo 事件数
        tree = ET.parse(stopinfo_path)
        n_events = len(tree.findall('.//stopinfo'))
        
        return {
            "config_id": config_id,
            "scenario": scenario_key,
            "duration": duration_key,
            "seed": seed,
            "stopinfo_path": str(stopinfo_path),
            "n_events": n_events,
            "success": True,
        }
    
    # 加载 L1 参数
    l1_params = load_l1_params(config['l1_params_source'])
    
    # L2 参数
    l2_params = config.get('l2_params')
    
    # 创建路由文件
    route_file = create_route_file_with_vtypes(
        config_id, l1_params, l2_params,
        config['use_background'], output_dir
    )
    
    # 创建 sumocfg
    sumocfg_path, stopinfo_path = create_sumocfg(
        config_id, route_file,
        config['use_background'],
        config['scale'],
        duration_sec,
        seed,
        output_dir
    )
    
    # 运行仿真
    success = run_sumo_simulation(sumocfg_path, config['scale'], seed)
    
    if success and os.path.exists(stopinfo_path):
        # 统计 stopinfo 事件数
        tree = ET.parse(stopinfo_path)
        n_events = len(tree.findall('.//stopinfo'))
        
        return {
            "config_id": config_id,
            "scenario": scenario_key,
            "duration": duration_key,
            "seed": seed,
            "stopinfo_path": str(stopinfo_path),
            "n_events": n_events,
            "success": True,
        }
    else:
        return None


def evaluate_single_result(
    result: Dict,
    scenario_key: str,
    duration_key: str
) -> Optional[Dict]:
    """评估单个仿真结果"""
    
    config_id = result["config_id"]
    stopinfo_path = result["stopinfo_path"]
    seed = result["seed"]
    
    # 获取配置
    config = ABLATION_SIM_CONFIGS[config_id]
    scenario = SCENARIOS[scenario_key]
    
    # 查找对应的 AblationConfig
    ablation_config = None
    for ac in ABLATION_CONFIGS:
        if ac.config_id == config_id:
            ablation_config = ac
            break
    
    if ablation_config is None:
        return None
    
    # 运行评估
    try:
        metrics_result = compute_metrics_v4(
            real_data=str(scenario["real_stats"]),
            sim_data=stopinfo_path,
            dist_file=str(scenario["dist_csv"]),
            audit_config=AuditConfig.from_protocol() if ablation_config.use_audit_in_validation else AuditConfig(enabled=False),
            scenario=scenario_key,
            route="68X",
            period=f"{scenario_key}_{duration_key}"
        )
        
        # 提取关键指标
        if ablation_config.use_audit_in_validation:
            ks_speed = metrics_result.ks_speed_clean.ks_stat
            ks_tt = metrics_result.ks_tt_clean.ks_stat
            ks_speed_passed = metrics_result.ks_speed_clean.passed
            ks_tt_passed = metrics_result.ks_tt_clean.passed
            n_clean = metrics_result.audit_stats.n_clean
        else:
            ks_speed = metrics_result.ks_speed_raw.ks_stat
            ks_tt = metrics_result.ks_tt_raw.ks_stat
            ks_speed_passed = metrics_result.ks_speed_raw.passed
            ks_tt_passed = metrics_result.ks_tt_raw.passed
            n_clean = metrics_result.audit_stats.n_raw
        
        return {
            "config_id": config_id,
            "scenario": scenario_key,
            "duration": duration_key,
            "seed": seed,
            "n_events": result["n_events"],
            "n_clean": n_clean,
            "n_sim": metrics_result.n_sim,
            "ks_speed": ks_speed,
            "ks_tt": ks_tt,
            "ks_speed_passed": ks_speed_passed,
            "ks_tt_passed": ks_tt_passed,
            "worst_window_ks_speed": metrics_result.worst_window_speed.worst_ks,
            "worst_window_ks_tt": metrics_result.worst_window_tt.worst_ks,
            "worst_window_start": metrics_result.worst_window_speed.window_start_time,
            "worst_window_end": metrics_result.worst_window_speed.window_end_time,
            "sanity_passed": metrics_result.sanity_passed,
        }
    except Exception as e:
        print(f"[ERROR] 评估失败: {config_id} {scenario_key} {duration_key} seed{seed}: {e}")
        return None


def compute_bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """计算 bootstrap 置信区间"""
    if len(values) < 2:
        return np.nan, np.nan
    
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrapped.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    
    return lower, upper


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Protocol Ablation 多场景多seed批量实验")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["off_peak", "pm_peak"],
        help="场景列表"
    )
    parser.add_argument(
        "--durations",
        type=str,
        nargs="+",
        default=["1h"],
        help="时长列表"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="种子列表"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["A0", "A1", "A3", "A4"],  # A2 与 A1 共享
        help="配置列表"
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="跳过仿真，直接评估已有结果"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Protocol Ablation 多场景多seed批量实验")
    print("=" * 70)
    print(f"场景: {args.scenarios}")
    print(f"时长: {args.durations}")
    print(f"Seeds: {args.seeds}")
    print(f"配置: {args.configs}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 第一阶段：运行仿真
    all_results = []
    
    if not args.skip_simulation:
        print("=" * 70)
        print("第一阶段：运行仿真")
        print("=" * 70)
        
        for scenario_key in args.scenarios:
            for duration_key in args.durations:
                for seed in args.seeds:
                    for config_id in args.configs:
                        if config_id not in ABLATION_SIM_CONFIGS:
                            continue
                        
                        print(f"\n运行: {config_id} {scenario_key} {duration_key} seed{seed}")
                        result = run_single_ablation_simulation(
                            config_id, scenario_key, duration_key, seed
                        )
                        
                        if result:
                            all_results.append(result)
                            print(f"  ✓ 成功 (n_events={result['n_events']})")
                        else:
                            print(f"  ✗ 失败")
    
    # 第二阶段：评估结果
    print("\n" + "=" * 70)
    print("第二阶段：评估结果")
    print("=" * 70)
    
    evaluation_results = []
    
    # 收集所有已有的仿真结果
    for scenario_key in args.scenarios:
        for duration_key in args.durations:
            for seed in args.seeds:
                for config_id in ["A0", "A1", "A2", "A3", "A4"]:
                    stopinfo_path = OUTPUT_DIR / f"{scenario_key}_{duration_key}" / f"seed{seed}" / config_id / "stopinfo.xml"
                    
                    if not stopinfo_path.exists():
                        continue
                    
                    # 统计事件数
                    tree = ET.parse(stopinfo_path)
                    n_events = len(tree.findall('.//stopinfo'))
                    
                    result = {
                        "config_id": config_id,
                        "scenario": scenario_key,
                        "duration": duration_key,
                        "seed": seed,
                        "stopinfo_path": str(stopinfo_path),
                        "n_events": n_events,
                        "success": True,
                    }
                    
                    # 评估
                    eval_result = evaluate_single_result(result, scenario_key, duration_key)
                    if eval_result:
                        evaluation_results.append(eval_result)
                        print(f"  {config_id} {scenario_key} {duration_key} seed{seed}: "
                              f"KS(speed)={eval_result['ks_speed']:.4f}, "
                              f"KS(TT)={eval_result['ks_tt']:.4f}")
                    else:
                        print(f"  [WARN] {config_id} {scenario_key} {duration_key} seed{seed}: 评估失败")
    
    # 保存详细结果
    if evaluation_results:
        df_results = pd.DataFrame(evaluation_results)
        results_path = OUTPUT_DIR / "results.csv"
        df_results.to_csv(results_path, index=False)
        print(f"\n详细结果已保存: {results_path}")
        
        # 第三阶段：统计汇总
        print("\n" + "=" * 70)
        print("第三阶段：统计汇总")
        print("=" * 70)
        
        # 按配置、场景、时长分组统计
        summary_rows = []
        
        for config_id in df_results["config_id"].unique():
            for scenario_key in df_results["scenario"].unique():
                for duration_key in df_results["duration"].unique():
                    subset = df_results[
                        (df_results["config_id"] == config_id) &
                        (df_results["scenario"] == scenario_key) &
                        (df_results["duration"] == duration_key)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    # KS speed 统计
                    ks_speed_values = subset["ks_speed"].values
                    ks_speed_mean = np.mean(ks_speed_values)
                    ks_speed_std = np.std(ks_speed_values)
                    ks_speed_ci_low, ks_speed_ci_high = compute_bootstrap_ci(ks_speed_values)
                    
                    # KS TT 统计
                    ks_tt_values = subset["ks_tt"].values
                    ks_tt_mean = np.mean(ks_tt_values)
                    ks_tt_std = np.std(ks_tt_values)
                    ks_tt_ci_low, ks_tt_ci_high = compute_bootstrap_ci(ks_tt_values)
                    
                    # Pass 率
                    ks_speed_pass_rate = np.mean(subset["ks_speed_passed"])
                    ks_tt_pass_rate = np.mean(subset["ks_tt_passed"])
                    
                    # 样本统计
                    n_clean_mean = np.mean(subset["n_clean"])
                    n_sim_mean = np.mean(subset["n_sim"])
                    n_events_mean = np.mean(subset["n_events"])
                    
                    summary_rows.append({
                        "config_id": config_id,
                        "scenario": scenario_key,
                        "duration": duration_key,
                        "n_runs": len(subset),
                        "n_clean_mean": n_clean_mean,
                        "n_sim_mean": n_sim_mean,
                        "n_events_mean": n_events_mean,
                        "ks_speed_mean": ks_speed_mean,
                        "ks_speed_std": ks_speed_std,
                        "ks_speed_ci_low": ks_speed_ci_low,
                        "ks_speed_ci_high": ks_speed_ci_high,
                        "ks_speed_pass_rate": ks_speed_pass_rate,
                        "ks_tt_mean": ks_tt_mean,
                        "ks_tt_std": ks_tt_std,
                        "ks_tt_ci_low": ks_tt_ci_low,
                        "ks_tt_ci_high": ks_tt_ci_high,
                        "ks_tt_pass_rate": ks_tt_pass_rate,
                    })
        
        df_summary = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / "summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"汇总结果已保存: {summary_path}")
        
        # 打印汇总表
        print("\n" + df_summary.to_string(index=False))
        
        # Seed 差异 sanity check
        print("\n" + "=" * 70)
        print("Seed 差异 Sanity Check")
        print("=" * 70)
        
        for config_id in df_results["config_id"].unique():
            for scenario_key in df_results["scenario"].unique():
                for duration_key in df_results["duration"].unique():
                    subset = df_results[
                        (df_results["config_id"] == config_id) &
                        (df_results["scenario"] == scenario_key) &
                        (df_results["duration"] == duration_key)
                    ]
                    
                    if len(subset) < 2:
                        continue
                    
                    ks_speed_values = subset["ks_speed"].values
                    ks_speed_std = np.std(ks_speed_values)
                    
                    if ks_speed_std < 1e-6:
                        print(f"[WARN] {config_id} {scenario_key} {duration_key}: "
                              f"KS(speed) std = {ks_speed_std:.2e} < 1e-6")
                        print(f"       Seeds 可能未生效！所有 seed 的 KS 值完全相同。")
                        print(f"       KS 值: {ks_speed_values}")
                    else:
                        print(f"[OK] {config_id} {scenario_key} {duration_key}: "
                              f"KS(speed) std = {ks_speed_std:.4f}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
