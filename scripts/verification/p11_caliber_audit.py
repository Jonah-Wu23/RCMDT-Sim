"""
P11-0 口径审计: 对比 moving-only TT vs door-to-door TT
用于诊断 4x 速度差距是"机理缺失"还是"口径不一致"
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import argparse


def parse_stopinfo_dual_tt(stopinfo_path: str) -> pd.DataFrame:
    """
    从 stopinfo.xml 提取两种 TT 定义:
    - moving_tt: next.started - curr.ended (不含停站)
    - door_to_door_tt: next.started - curr.started (含停站)
    
    Returns:
        DataFrame with link-level TT for both definitions
    """
    tree = ET.parse(stopinfo_path)
    root = tree.getroot()
    
    # 按车辆分组收集 stop 事件
    vehicle_stops = defaultdict(list)
    
    for stopinfo in root.findall('stopinfo'):
        vehicle_id = stopinfo.get('id')
        bus_stop = stopinfo.get('busStop')
        started = float(stopinfo.get('started'))
        ended = float(stopinfo.get('ended'))
        arrival = float(stopinfo.get('arrival', started))  # fallback
        
        vehicle_stops[vehicle_id].append({
            'busStop': bus_stop,
            'started': started,
            'ended': ended,
            'arrival': arrival
        })
    
    records = []
    
    for vehicle_id, stops in vehicle_stops.items():
        # 按到达时间排序
        stops_sorted = sorted(stops, key=lambda x: x['started'])
        
        # 从车辆ID解析路线和方向
        parts = vehicle_id.split('_')
        if len(parts) >= 3:
            veh_route = parts[1]
            veh_bound = parts[2].split('.')[0]
        else:
            continue
        
        # 遍历连续站点对
        for i in range(len(stops_sorted) - 1):
            curr = stops_sorted[i]
            next_s = stops_sorted[i + 1]
            
            # moving-only TT (当前实现)
            moving_tt = next_s['started'] - curr['ended']
            
            # door-to-door TT (含停站)
            d2d_tt = next_s['started'] - curr['started']
            
            # 停站时间
            dwell_time = curr['ended'] - curr['started']
            
            records.append({
                'vehicle_id': vehicle_id,
                'route': veh_route,
                'bound': veh_bound,
                'from_stop': curr['busStop'],
                'to_stop': next_s['busStop'],
                'from_seq': i + 1,
                'to_seq': i + 2,
                'moving_tt_s': moving_tt,
                'door_to_door_tt_s': d2d_tt,
                'dwell_time_s': dwell_time,
                'start_time': curr['started']
            })
    
    return pd.DataFrame(records)


def load_link_distances(mapping_csv: str) -> dict:
    """加载 link 距离映射"""
    df = pd.read_csv(mapping_csv)
    # 假设有 observation_id 和 link_length_m 列
    return dict(zip(df['observation_id'], df['obs_length_m']))


def main():
    parser = argparse.ArgumentParser(description='P11-0 口径审计')
    parser.add_argument('--stopinfo', required=True, help='stopinfo.xml 路径')
    parser.add_argument('--obs-vector', required=True, help='观测向量 CSV (含 link 距离)')
    parser.add_argument('--output', default=None, help='输出 CSV')
    args = parser.parse_args()
    
    print(f"[P11-0] 解析 stopinfo: {args.stopinfo}")
    df = parse_stopinfo_dual_tt(args.stopinfo)
    print(f"[P11-0] 提取到 {len(df)} 条 link 记录")
    
    # 加载观测向量获取距离
    obs_df = pd.read_csv(args.obs_vector)
    
    # 按 route + bound + seq 聚合
    agg = df.groupby(['route', 'bound', 'from_seq', 'to_seq']).agg({
        'moving_tt_s': 'mean',
        'door_to_door_tt_s': 'mean',
        'dwell_time_s': 'mean'
    }).reset_index()
    
    # 合并观测距离 (简化：假设 from_seq 对应 observation_id)
    # 实际需要更精确的映射
    if 'link_length_m' in obs_df.columns:
        link_dist = dict(zip(obs_df['observation_id'], obs_df['link_length_m']))
    elif 'obs_length_m' in obs_df.columns:
        link_dist = dict(zip(obs_df['observation_id'], obs_df['obs_length_m']))
    else:
        # 尝试从 cum_dist 推断
        link_dist = {}
        print("[WARN] 无法找到 link_length_m 列，使用默认 500m")
    
    # 计算等效速度
    results = []
    for _, row in agg.iterrows():
        obs_id = row['from_seq']  # 简化映射
        dist_m = link_dist.get(obs_id, 500)  # 默认 500m
        
        moving_speed = (dist_m / row['moving_tt_s'] * 3.6) if row['moving_tt_s'] > 0 else np.nan
        d2d_speed = (dist_m / row['door_to_door_tt_s'] * 3.6) if row['door_to_door_tt_s'] > 0 else np.nan
        
        results.append({
            'route': row['route'],
            'bound': row['bound'],
            'link_id': f"{row['from_seq']}->{row['to_seq']}",
            'dist_m': dist_m,
            'moving_tt_s': row['moving_tt_s'],
            'door_to_door_tt_s': row['door_to_door_tt_s'],
            'dwell_time_s': row['dwell_time_s'],
            'moving_speed_kmh': moving_speed,
            'door_to_door_speed_kmh': d2d_speed
        })
    
    result_df = pd.DataFrame(results)
    
    # 输出统计
    print(f"\n{'='*60}")
    print("[P11-0] 口径审计结果")
    print(f"{'='*60}")
    
    for (route, bound), g in result_df.groupby(['route', 'bound']):
        moving_median = g['moving_speed_kmh'].median()
        d2d_median = g['door_to_door_speed_kmh'].median()
        dwell_mean = g['dwell_time_s'].mean()
        
        print(f"\n{route} {bound}:")
        print(f"  - moving-only speed (median):    {moving_median:.2f} km/h")
        print(f"  - door-to-door speed (median):   {d2d_median:.2f} km/h")
        print(f"  - 平均停站时间:                    {dwell_mean:.1f} s")
        print(f"  - 速度下降比例:                    {(1 - d2d_median/moving_median)*100:.1f}%")
    
    # 全局汇总
    global_moving = result_df['moving_speed_kmh'].median()
    global_d2d = result_df['door_to_door_speed_kmh'].median()
    print(f"\n--- 全局汇总 ---")
    print(f"  - moving-only (median):    {global_moving:.2f} km/h")
    print(f"  - door-to-door (median):   {global_d2d:.2f} km/h")
    print(f"  - Obs 真值 (参考):           ~3.4 km/h")
    
    # 判读
    print(f"\n--- 判读 ---")
    if global_d2d < 10:
        print(f"✅ door-to-door 已下探至 {global_d2d:.1f} km/h < 10 km/h")
        print(f"   -> 主要是口径问题，应切换到 door-to-door 定义")
    else:
        print(f"❌ door-to-door 仍为 {global_d2d:.1f} km/h >= 10 km/h")
        print(f"   -> 机理缺失，需执行 P11-1 (走廊定向 Scale) 或 P11-2 (信号容量缩放)")
    
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\n[P11-0] 结果已保存: {args.output}")
    
    return result_df


if __name__ == "__main__":
    main()
