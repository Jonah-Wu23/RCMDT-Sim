#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation_simulations.py - 为 Protocol Ablation 各配置运行 SUMO 仿真
=======================================================================

为 A0-A4 各配置生成真实的仿真输出（stopinfo.xml）。

配置说明：
- A0 Zero-shot: 默认参数（baseline_parameters.json），无背景流
- A1 Raw-L1: BO 优化参数（best_l1_parameters.json），有背景流
- A2 Audit-Val-Only: 同 A1（仿真相同，仅验证时用 clean 数据）
- A3 Audit-in-Cal: BO + Tail Loss 参数，有背景流
- A4 Full-RCMDT: IES 优化后参数，有背景流

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ============================================================================
# 仿真配置
# ============================================================================

SUMO_CONFIG = {
    "net_file": PROJECT_ROOT / "sumo" / "net" / "hk_cropped.net.xml",
    "bus_stops": PROJECT_ROOT / "sumo" / "additional" / "bus_stops_cropped.add.xml",
    "bus_routes": PROJECT_ROOT / "sumo" / "routes" / "fixed_routes_cropped.rou.xml",
    "background_routes": PROJECT_ROOT / "sumo" / "routes" / "background_cropped.rou.xml",
    "simulation_time": 3600,  # 1 小时
    "time_to_teleport": 300,
}

# 各配置的参数和设置
ABLATION_SIM_CONFIGS = {
    "A0": {
        "name": "Zero-shot",
        "description": "默认参数，无背景流",
        "use_background": False,
        "scale": 0.0,
        "l1_params_source": "baseline",  # baseline_parameters.json
        "l2_params": None,  # 不使用 L2 参数
    },
    "A1": {
        "name": "Raw-L1 (BO)",
        "description": "BO 优化参数，有背景流",
        "use_background": True,
        "scale": 0.1,  # P6 baseline scale
        "l1_params_source": "best",  # best_l1_parameters.json
        "l2_params": None,
    },
    "A2": {
        "name": "Audit-Val-Only",
        "description": "同 A1（仿真相同）",
        "use_background": True,
        "scale": 0.1,
        "l1_params_source": "best",
        "l2_params": None,
        "same_as": "A1",  # 仿真与 A1 相同
    },
    "A3": {
        "name": "Audit-in-Cal + Tail",
        "description": "BO + Tail Loss 参数",
        "use_background": True,
        "scale": 0.12,  # 略高的 scale
        "l1_params_source": "best",
        "l2_params": {
            "capacityFactor": 1.2,
            "minGap_background": 2.0,
            "impatience": 0.6,
        },
    },
    "A4": {
        "name": "Full-RCMDT (IES)",
        "description": "IES 优化后参数",
        "use_background": True,
        "scale": 0.15,  # IES 优化的 scale (capacityFactor 效果)
        "l1_params_source": "best",
        "l2_params": {
            "capacityFactor": 1.5,  # B4_v2 最优
            "minGap_background": 0.5,
            "impatience": 1.0,
        },
    },
}

OUTPUT_DIR = PROJECT_ROOT / "sumo" / "output" / "ablation_runs"


# ============================================================================
# 参数加载
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
        params['maxSpeed'] = 20.0  # 默认最大速度
        return params


def create_vtype_with_params(l1_params: Dict, l2_params: Optional[Dict] = None) -> str:
    """创建带参数的 vType XML 片段"""
    
    # 公交车 vType
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
    
    # 背景车辆 vType
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
    
    # 解析 vType XML 并添加
    for line in vtype_xml.strip().split('\n'):
        if line.strip().startswith('<vType'):
            # 简单解析 vType
            pass
    
    # 直接写入文件（手动构建）
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
    output_dir: Path
) -> str:
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
        <end value="{SUMO_CONFIG['simulation_time']}"/>
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


def run_sumo_simulation(sumocfg_path: str, scale: float) -> bool:
    """运行 SUMO 仿真"""
    
    cmd = [
        "sumo",
        "-c", sumocfg_path,
        "--scale", f"{scale:.3f}" if scale > 0 else "1.0",
    ]
    
    print(f"  运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] 仿真失败!")
        print(f"  stderr: {e.stderr[:500] if e.stderr else 'N/A'}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] 仿真超时!")
        return False


def run_ablation_simulation(config_id: str) -> Optional[str]:
    """运行单个 ablation 配置的仿真"""
    
    config = ABLATION_SIM_CONFIGS[config_id]
    
    print(f"\n{'='*60}")
    print(f"配置: {config_id} - {config['name']}")
    print(f"{'='*60}")
    print(f"描述: {config['description']}")
    
    # 检查是否与其他配置共享仿真
    if 'same_as' in config:
        ref_id = config['same_as']
        ref_output = OUTPUT_DIR / ref_id / "stopinfo.xml"
        if ref_output.exists():
            print(f"  [INFO] 与 {ref_id} 共享仿真结果")
            # 创建符号链接或复制
            output_dir = OUTPUT_DIR / config_id
            output_dir.mkdir(parents=True, exist_ok=True)
            dst_path = output_dir / "stopinfo.xml"
            if not dst_path.exists():
                shutil.copy(ref_output, dst_path)
            return str(dst_path)
        else:
            print(f"  [WARN] 引用配置 {ref_id} 的结果不存在，需要先运行 {ref_id}")
            return None
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 L1 参数
    l1_params = load_l1_params(config['l1_params_source'])
    print(f"  L1 参数源: {config['l1_params_source']}")
    print(f"    tau={l1_params['tau']:.4f}, sigma={l1_params['sigma']:.4f}, minGap={l1_params['minGap']:.4f}")
    
    # L2 参数
    l2_params = config.get('l2_params')
    if l2_params:
        print(f"  L2 参数: capacityFactor={l2_params.get('capacityFactor', 1.0):.2f}, "
              f"minGap_bg={l2_params.get('minGap_background', 2.5):.2f}, "
              f"impatience={l2_params.get('impatience', 0.5):.2f}")
    
    # 创建路由文件
    route_file = create_route_file_with_vtypes(
        config_id, l1_params, l2_params,
        config['use_background'], output_dir
    )
    print(f"  路由文件: {Path(route_file).name}")
    
    # 创建 sumocfg
    sumocfg_path, stopinfo_path = create_sumocfg(
        config_id, route_file,
        config['use_background'],
        config['scale'],
        output_dir
    )
    print(f"  配置文件: {Path(sumocfg_path).name}")
    print(f"  背景流: {'启用' if config['use_background'] else '禁用'}, scale={config['scale']}")
    
    # 运行仿真
    print(f"\n  开始仿真...")
    start_time = datetime.now()
    
    success = run_sumo_simulation(sumocfg_path, config['scale'])
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  仿真耗时: {elapsed:.1f}s")
    
    if success and os.path.exists(stopinfo_path):
        file_size = os.path.getsize(stopinfo_path)
        print(f"  ✓ 成功! stopinfo.xml = {file_size/1024:.1f} KB")
        return stopinfo_path
    else:
        print(f"  ✗ 失败!")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="为 Protocol Ablation 运行 SUMO 仿真")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["A0", "A1", "A3", "A4"],  # A2 与 A1 共享
        help="要运行的配置 ID 列表"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新运行（即使输出已存在）"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Protocol Ablation SUMO 仿真")
    print("=" * 70)
    print(f"配置: {args.configs}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for config_id in args.configs:
        if config_id not in ABLATION_SIM_CONFIGS:
            print(f"[WARN] 未知配置: {config_id}")
            continue
        
        # 检查是否已存在
        existing_output = OUTPUT_DIR / config_id / "stopinfo.xml"
        if existing_output.exists() and not args.force:
            print(f"\n[SKIP] {config_id}: 输出已存在 ({existing_output})")
            results[config_id] = str(existing_output)
            continue
        
        # 运行仿真
        result = run_ablation_simulation(config_id)
        results[config_id] = result
    
    # 处理 A2（与 A1 共享）
    if "A2" in args.configs or "A2" not in results:
        if "A1" in results and results["A1"]:
            a2_dir = OUTPUT_DIR / "A2"
            a2_dir.mkdir(parents=True, exist_ok=True)
            a2_output = a2_dir / "stopinfo.xml"
            if not a2_output.exists():
                shutil.copy(results["A1"], a2_output)
            results["A2"] = str(a2_output)
            print(f"\n[INFO] A2: 复制自 A1")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("仿真汇总")
    print("=" * 70)
    
    for config_id, result in results.items():
        status = "✓ 成功" if result else "✗ 失败"
        print(f"  {config_id}: {status}")
        if result:
            print(f"      -> {result}")
    
    print("\n" + "=" * 70)
    print("完成! 现在可以运行 run_protocol_ablation.py 进行评估")
    print("=" * 70)


if __name__ == "__main__":
    main()
