#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_a1_c_validation.py
======================
方案 A+C 联合验证：
- A1: 300vph 插车验证（10min）→ 检查插入率 ≥ 85%
- A2: 三点压载（0/150/300 vph × 1h）
- C: 公交 TT 受影响验证（对比 bus-only vs bus+300vph）
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/calibration'))

from build_l2_sim_vector_traveltime import build_simulation_vector_tt

# 68X inbound route edges
CORRIDOR_EDGES = "105735 105735_rev 273264_rev 105528 105501 105502 106883 106884_rev 106894_rev 105511_rev 105609_rev 284967_rev 284974 284930_rev 106938 261602 106935_rev 107180_rev 105753_rev 105729_rev 106963 106955 105653 105653_rev 105653 107154 106838 106831_rev 106831 106838_rev 107154_rev 107002 105952_rev 105929 115859 105926 105922 105923 105925 105910 105819 137853 137854 285166 105817 105832 105830 105829 105827 106986_rev 106985_rev 105836_rev 105886_rev 105866_rev 105880_rev 106995 106996 106993 106991 106728 106729 106073 105770_rev 106116_rev 106062_rev 106056_rev 105785_rev 105786_rev 106028 106053_rev 106077 106285_rev 106243_rev 105334_rev 105335 105335_rev 105336_rev 105351 105343 106429 107083 106344 106365 106366_rev 106367 106272_rev 106270_rev 106320 284524 106580 106608 106627 106628 107088 107087 106628 107088 106625 106623 106624 106537_rev 105377 106535 9343 9029 272309 9623_rev 8312_rev 9639_rev 8006_rev 9640_rev 8991_rev 8998 7993 8997 8303_rev 9023_rev 9023 8303 8302 9638 8337 9357 8327 116396 8024 8044 8663_rev 7654_rev 9043_rev 8975_rev 8696_rev 9677_rev"


def create_corridor_load_file(vph: int, output_path: Path, duration: int = 3600):
    """创建走廊压载文件（优化注入参数）"""
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="bg" vClass="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" minGap="2.5" maxSpeed="33.3" color="0,1,0"/>
    <route id="r_68x_in_corridor" edges="{CORRIDOR_EDGES}"/>
'''
    if vph > 0:
        # 优化参数：random_free + free + max
        content += f'''    <flow id="bg_corridor_{vph}vph" type="bg" route="r_68x_in_corridor"
          begin="0" end="{duration}" vehsPerHour="{vph}"
          departPos="random_free" departLane="free" departSpeed="max"/>
'''
    content += '</routes>'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_simulation(label: str, vph: int, duration: int = 3600) -> Path:
    """运行隔离仿真"""
    output_dir = PROJECT_ROOT / f'sumo/output/a1c_validation/{label}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corridor_load_path = output_dir / 'corridor_load.rou.xml'
    create_corridor_load_file(vph, corridor_load_path, duration)
    
    route_files = [str(PROJECT_ROOT / 'sumo/routes/fixed_routes_cropped.rou.xml')]
    if vph > 0:
        route_files.append(str(corridor_load_path))
    
    cmd = [
        'sumo',
        '-n', str(PROJECT_ROOT / 'sumo/net/hk_cropped.net.xml'),
        '-r', ','.join(route_files),
        '-a', str(PROJECT_ROOT / 'sumo/additional/bus_stops_cropped.add.xml'),
        '-b', '0',
        '-e', str(duration),
        '--ignore-route-errors', 'true',
        '--time-to-teleport', '300',
        '--no-step-log', 'true',
        '--stop-output', str(output_dir / 'stopinfo.xml'),
        '--tripinfo-output', str(output_dir / 'tripinfo.xml'),
        '--summary-output', str(output_dir / 'summary.xml'),
        '--statistic-output', str(output_dir / 'statistics.xml'),
    ]
    
    print(f"\n[{label}] 运行仿真 (vph={vph}, duration={duration}s)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] SUMO 失败: {result.stderr[:500] if result.stderr else '(no stderr)'}")
        return None
    
    print(f"[{label}] 仿真完成")
    return output_dir


def check_insertion_metrics(output_dir: Path, expected_vph: int, duration: int) -> dict:
    """检查插入指标"""
    metrics = {'inserted': 0, 'waiting': 0, 'expected': int(expected_vph * duration / 3600),
               'bg_inserted': 0, 'bg_insertion_rate': 0.0, 'pass_threshold': False}
    
    stats_path = output_dir / 'statistics.xml'
    if stats_path.exists():
        tree = ET.parse(stats_path)
        root = tree.getroot()
        veh_elem = root.find('.//vehicles')
        if veh_elem is not None:
            metrics['inserted'] = int(veh_elem.get('inserted', 0))
            metrics['waiting'] = int(veh_elem.get('waiting', 0))
    
    # 减去公交车（约 5 辆）
    metrics['bg_inserted'] = max(0, metrics['inserted'] - 5)
    if metrics['expected'] > 0:
        metrics['bg_insertion_rate'] = metrics['bg_inserted'] / metrics['expected']
    
    # 硬门槛：插入率 ≥ 85%
    metrics['pass_threshold'] = metrics['bg_insertion_rate'] >= 0.85
    
    return metrics


def extract_bus_tt(output_dir: Path) -> pd.DataFrame:
    """提取公交 TT"""
    df = build_simulation_vector_tt(
        stopinfo_path=str(output_dir / 'stopinfo.xml'),
        observation_csv=str(PROJECT_ROOT / 'data/calibration/l2_observation_vector_corridor_M11_TT.csv'),
        route_stop_csv=str(PROJECT_ROOT / 'data/processed/kmb_route_stop_dist.csv'),
        verbose=False
    )
    return df


def run_a1_insertion_test():
    """A1: 300vph 插车验证（10min）"""
    print("\n" + "="*60)
    print("A1: 300vph 插车验证（10min）")
    print("="*60)
    
    output_dir = run_simulation('a1_300vph_10min', vph=300, duration=600)
    if not output_dir:
        return None
    
    metrics = check_insertion_metrics(output_dir, expected_vph=300, duration=600)
    
    print(f"  bg_inserted: {metrics['bg_inserted']} / {metrics['expected']} ({metrics['bg_insertion_rate']*100:.1f}%)")
    print(f"  waiting: {metrics['waiting']}")
    
    if metrics['pass_threshold']:
        print("  ✓ 通过硬门槛 (≥85%)，可进行 A2")
    else:
        print("  ⚠️ 未通过 85% 硬门槛，但可尝试继续")
    
    return metrics


def run_a2_three_point():
    """A2: 三点压载（0/150/300 vph × 1h）"""
    print("\n" + "="*60)
    print("A2: 三点压载（0/150/300 vph × 1h）")
    print("="*60)
    
    vphs = [0, 150, 300]
    results = []
    
    for vph in vphs:
        label = f'a2_vph{vph}'
        output_dir = run_simulation(label, vph=vph, duration=3600)
        if not output_dir:
            continue
        
        # 检查插入指标
        if vph > 0:
            metrics = check_insertion_metrics(output_dir, expected_vph=vph, duration=3600)
            print(f"  [{vph}vph] bg_inserted: {metrics['bg_inserted']}/{metrics['expected']} ({metrics['bg_insertion_rate']*100:.1f}%)")
        
        # 提取公交 TT
        df = extract_bus_tt(output_dir)
        
        results.append({
            'vph': vph,
            'median_TT_sim_s': df['travel_time_sim_s'].median(),
            'median_TT_obs_s': df['travel_time_obs_s'].median(),
            'median_ratio': df['ratio'].median(),
            'matched_count': df['travel_time_sim_s'].notna().sum()
        })
        
        print(f"  [{vph}vph] 公交 TT_sim = {results[-1]['median_TT_sim_s']:.0f}s, ratio = {results[-1]['median_ratio']:.3f}")
    
    # 输出对比表
    print("\n" + "-"*60)
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    # 保存结果
    output_path = PROJECT_ROOT / 'data/calibration/a2_three_point_results.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n[输出] 已保存到 {output_path}")
    
    return results


def run_c_bus_tt_impact():
    """C: 公交 TT 受影响验证"""
    print("\n" + "="*60)
    print("C: 公交 TT 受影响验证（0 vs 300 vph）")
    print("="*60)
    
    # 复用 A2 的结果
    a2_vph0_dir = PROJECT_ROOT / 'sumo/output/a1c_validation/a2_vph0'
    a2_vph300_dir = PROJECT_ROOT / 'sumo/output/a1c_validation/a2_vph300'
    
    if not a2_vph0_dir.exists() or not a2_vph300_dir.exists():
        print("  [警告] A2 结果不存在，先运行 A2")
        return None
    
    df_0 = extract_bus_tt(a2_vph0_dir)
    df_300 = extract_bus_tt(a2_vph300_dir)
    
    tt_0 = df_0['travel_time_sim_s'].median()
    tt_300 = df_300['travel_time_sim_s'].median()
    delta_pct = (tt_300 - tt_0) / tt_0 * 100 if tt_0 > 0 else 0
    
    print(f"  公交 TT_sim(0vph) = {tt_0:.0f}s")
    print(f"  公交 TT_sim(300vph) = {tt_300:.0f}s")
    print(f"  变化: {delta_pct:+.1f}%")
    
    if delta_pct > 15:
        print("  ✓ 交互成立：压载导致公交 TT 显著上升")
        return True
    elif delta_pct > 5:
        print("  △ 交互存在但较弱")
        return True
    else:
        print("  ✗ 无明显交互：走廊堵了但公交 TT 不变")
        print("    → 检查车道权限 / 公交专用道 / 共享瓶颈")
        return False


def main():
    print("="*60)
    print("方案 A+C 联合验证")
    print("="*60)
    
    # A1: 插车验证
    a1_metrics = run_a1_insertion_test()
    
    # A2: 三点试验（无论 A1 是否通过都做，只是记录插入率）
    a2_results = run_a2_three_point()
    
    # C: 公交 TT 影响验证
    c_result = run_c_bus_tt_impact()
    
    # 综合判读
    print("\n" + "="*60)
    print("综合判读")
    print("="*60)
    
    if a2_results and len(a2_results) >= 3:
        tt_0 = a2_results[0]['median_TT_sim_s']
        tt_300 = a2_results[2]['median_TT_sim_s']
        ratio_0 = a2_results[0]['median_ratio']
        ratio_300 = a2_results[2]['median_ratio']
        
        if c_result:
            print(f"✓ 交互成立，公交 TT 随压载上升")
            print(f"  TT: {tt_0:.0f}s → {tt_300:.0f}s")
            print(f"  ratio: {ratio_0:.3f} → {ratio_300:.3f}")
            if ratio_300 > ratio_0 + 0.05:
                print("  建议: 进入 P5 修复背景流 OD")
            else:
                print("  注意: ratio 变化不大，可能需要更高压载或更长统计窗口")
        else:
            print("✗ 交互缺失，需检查:")
            print("  1. 走廊 edge 是否有 bus-only lane")
            print("  2. 公交是否走旁路")
            print("  3. 公交 TT 是否被 stop dwell 主导")


if __name__ == "__main__":
    main()
