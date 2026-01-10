#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_l2_sim_vector_traveltime.py
=================================
从 SUMO stopinfo.xml 提取站-站段 travel_time（秒），与观测向量同构。

核心设计：
1. 以观测模板（l2_observation_vector_corridor_M11_TT.csv）为驱动
2. key = (route, bound, from_seq, to_seq)
3. travel_time = next_stop.started - curr_stop.ended（不含停站时间）
4. 规则 A: 同一车辆连续服务段
5. 规则 B: TT > 1800s 剔除（终点驻站污染）

输出：l2_simulation_vector_TT.csv
"""

import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_stop_seq_mapping(route_stop_csv: str) -> Dict[str, Tuple[str, str, int]]:
    """
    加载 stop_id -> (route, bound, seq) 映射表。
    
    对于共站情况，返回所有可能的 (route, bound, seq) 组合。
    
    Returns:
        Dict[stop_id, List[(route, bound, seq)]]
    """
    df = pd.read_csv(route_stop_csv)
    mapping = defaultdict(list)
    
    for _, row in df.iterrows():
        stop_id = row['stop_id']
        route = row['route']
        bound = row['bound']
        seq = int(row['seq'])
        # 尝试读取 KMB 距离 (米)
        cum_dist = float(row['cum_dist_m']) if 'cum_dist_m' in row else 0.0
        mapping[stop_id].append((route, bound, seq, cum_dist))
    
    return dict(mapping)


def parse_stopinfo_xml(stopinfo_path: str, stop_mapping: dict, 
                       max_gap_seconds: float = 1800.0,
                       tmin: Optional[float] = None,
                       tmax: Optional[float] = None,
                       tt_mode: str = 'moving') -> pd.DataFrame:
    """
    解析 stopinfo.xml，提取站-站段 travel_time。
    
    Args:
        stopinfo_path: stopinfo.xml 路径
        stop_mapping: stop_id -> List[(route, bound, seq)] 映射
        max_gap_seconds: 最大允许的 travel_time（超过视为异常）
        tmin: 最小时间过滤 (基于 next_stop.started)
        tmax: 最大时间过滤
        tt_mode: 'moving' (started-ended) 或 'door' (arrival-arrival)
    
    Returns:
        DataFrame with columns: [route, bound, from_seq, to_seq, travel_time_s, dist_m, vehicle_id]
    """
    tree = ET.parse(stopinfo_path)
    root = tree.getroot()
    
    # ... (omitted code for loading vehicle_stops) ...
    # 按车辆分组收集 stop 事件
    vehicle_stops = defaultdict(list)
    
    for stopinfo in root.findall('stopinfo'):
        vehicle_id = stopinfo.get('id')
        bus_stop = stopinfo.get('busStop')
        started = float(stopinfo.get('started'))
        ended = float(stopinfo.get('ended'))
        # 尝试获取 arrival, 若无则用 started (backward compatibility)
        arrival_str = stopinfo.get('arrival')
        arrival = float(arrival_str) if arrival_str else started
        
        # 只处理有映射的站点
        if bus_stop in stop_mapping:
            vehicle_stops[vehicle_id].append({
                'busStop': bus_stop,
                'started': started,
                'ended': ended,
                'arrival': arrival,
                'mappings': stop_mapping[bus_stop]
            })
    
    # 提取站-站段 travel_time
    records = []
    
    for vehicle_id, stops in vehicle_stops.items():
        # 按到达时间排序
        stops_sorted = sorted(stops, key=lambda x: x['started'])
        
        # 从车辆ID解析路线和方向（格式: flow_68X_inbound.0）
        parts = vehicle_id.split('_')
        if len(parts) >= 3:
            veh_route = parts[1]  # e.g., "68X"
            veh_bound = parts[2].split('.')[0]  # e.g., "inbound"
        else:
            continue
        
        # 遍历连续站点对
        for i in range(len(stops_sorted) - 1):
            curr_stop = stops_sorted[i]
            next_stop = stops_sorted[i + 1]
            
            # travel_time 计算
            if tt_mode == 'door':
                # door-to-door: 下一站离开 - 当前站离开 (含当前站停站 + 行驶 + 下一站停站)
                travel_time = next_stop['ended'] - curr_stop['ended']
            else:
                # moving-only: 下一站到达(started) - 当前站离开(ended) = 纯行驶时间
                travel_time = next_stop['started'] - curr_stop['ended']

            # 规则 C: 时间窗口过滤 (基于 next_stop.started)
            event_time = next_stop['started']
            if tmin is not None and event_time < tmin:
                continue
            if tmax is not None and event_time > tmax:
                continue
            
            # 规则 B: 过滤异常长的 travel_time（终点驻站/折返）
            
            # 规则 B: 过滤异常长的 travel_time（终点驻站/折返）
            if travel_time <= 0 or travel_time > max_gap_seconds:
                continue
            
            # 查找匹配的 (route, bound, from_seq, to_seq)
            curr_matches = [m for m in curr_stop['mappings'] 
                           if m[0] == veh_route and m[1] == veh_bound]
            next_matches = [m for m in next_stop['mappings'] 
                           if m[0] == veh_route and m[1] == veh_bound]
            
            if not curr_matches or not next_matches:
                continue
            
            # 取第一个匹配（同一路线/方向应该只有一个 seq）
            from_seq = curr_matches[0][2]
            cum_dist_curr = curr_matches[0][3]
            
            to_seq = next_matches[0][2]
            cum_dist_next = next_matches[0][3]
            
            # 确保是连续站点（to_seq = from_seq + 1）
            if to_seq != from_seq + 1:
                continue
            
            # 计算路段距离 (next - curr)
            dist_m = cum_dist_next - cum_dist_curr
            if dist_m <= 0:
                dist_m = 500.0 # fallback
            
            records.append({
                'route': veh_route,
                'bound': veh_bound,
                'from_seq': from_seq,
                'to_seq': to_seq,
                'travel_time_s': travel_time,
                'dist_m': dist_m,
                'vehicle_id': vehicle_id
            })
    
    return pd.DataFrame(records)


def build_simulation_vector_tt(
    stopinfo_path: str,
    observation_csv: str,
    route_stop_csv: str,
    output_csv: Optional[str] = None,
    max_gap_seconds: float = 1800.0,
    verbose: bool = True,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    tt_mode: str = 'moving'
) -> pd.DataFrame:
    """
    构建与观测向量同构的仿真 travel_time 向量。
    
    以观测模板为驱动，逐行匹配仿真数据。
    
    Args:
        stopinfo_path: stopinfo.xml 路径
        observation_csv: l2_observation_vector_corridor_M11_TT.csv 路径
        route_stop_csv: kmb_route_stop_dist.csv 路径
        output_csv: 输出 CSV 路径（可选）
        max_gap_seconds: 最大允许 travel_time
        verbose: 是否输出统计信息
        tmin: 最小开始时间 (可选)
        tmax: 最大开始时间 (可选)
        tt_mode: 'moving' 或 'door'
    """
    # 加载 stop_id -> seq 映射
    stop_mapping = load_stop_seq_mapping(route_stop_csv)
    if verbose:
        print(f"[INFO] 加载 {len(stop_mapping)} 个站点映射")
    
    # 解析 stopinfo.xml
    sim_df = parse_stopinfo_xml(stopinfo_path, stop_mapping, max_gap_seconds, tmin, tmax, tt_mode)
    if verbose:
        print(f"[INFO] 从 stopinfo 提取 {len(sim_df)} 条站-站段记录")
    
    # 加载观测模板
    obs_df = pd.read_csv(observation_csv)
    if verbose:
        print(f"[INFO] 观测模板有 {len(obs_df)} 个观测点")
    
    # 按 (route, bound, from_seq, to_seq) 聚合仿真数据
    if len(sim_df) > 0:
        agg_df = sim_df.groupby(['route', 'bound', 'from_seq', 'to_seq']).agg({
            'travel_time_s': ['mean', 'std', 'count'],
            'dist_m': 'first'  # 距离对同一路段应相同
        }).reset_index()
        agg_df.columns = ['route', 'bound', 'from_seq', 'to_seq', 
                         'travel_time_sim_s', 'std_travel_time_sim_s', 'sim_count', 'sim_dist_m']
    else:
        agg_df = pd.DataFrame(columns=['route', 'bound', 'from_seq', 'to_seq',
                                       'travel_time_sim_s', 'std_travel_time_sim_s', 'sim_count', 'sim_dist_m'])
    
    # 与观测模板合并（以观测为驱动）
    result_df = obs_df.merge(
        agg_df,
        on=['route', 'bound', 'from_seq', 'to_seq'],
        how='left'
    )
    
    # 计算比率 (如果有 travel_time_obs_s)
    if 'travel_time_obs_s' in result_df.columns:
        result_df['ratio'] = result_df['travel_time_sim_s'] / result_df['travel_time_obs_s']
    else:
        result_df['ratio'] = np.nan

    # 计算 sim_speed_kmh (用于 IES 接口兼容)
    # 优先使用 sim_dist_m (从 route_stop 计算), 其次 obs_length_m, 最后 link_length_m
    if 'sim_dist_m' in result_df.columns:
        result_df['sim_speed_kmh'] = (result_df['sim_dist_m'] / result_df['travel_time_sim_s']) * 3.6
    else:
        length_col = 'obs_length_m' if 'obs_length_m' in result_df.columns else ('link_length_m' if 'link_length_m' in result_df.columns else None)
        if length_col:
            result_df['sim_speed_kmh'] = (result_df[length_col] / result_df['travel_time_sim_s']) * 3.6
        else:
            result_df['sim_speed_kmh'] = np.nan
        
    # 增加 matched 字段 (用于 IES)
    result_df['matched'] = result_df['travel_time_sim_s'].notna()
    
    # 统计
    matched = result_df['travel_time_sim_s'].notna().sum()
    total = len(result_df)
    
    if verbose:
        print(f"\n[统计] 覆盖率: {matched}/{total} ({100*matched/total:.1f}%)")
        if matched > 0:
            print(f"[统计] TT_sim range: {result_df['travel_time_sim_s'].min():.1f} ~ {result_df['travel_time_sim_s'].max():.1f} s")
            if 'travel_time_obs_s' in result_df.columns:
                print(f"[统计] TT_obs range: {result_df['travel_time_obs_s'].min():.1f} ~ {result_df['travel_time_obs_s'].max():.1f} s")
                print(f"[统计] Ratio (sim/obs) median: {result_df['ratio'].median():.3f}")
            if length_col:
                print(f"[统计] Sim Speed (D2D) median: {result_df['sim_speed_kmh'].median():.2f} km/h")
    
    # 保存输出
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n[输出] 已保存到 {output_csv}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description='从 stopinfo.xml 提取站-站段 travel_time')
    parser.add_argument('--stopinfo', required=True, help='stopinfo.xml 路径')
    parser.add_argument('--observation', default=None, help='观测向量 CSV（默认使用 M11_TT）')
    parser.add_argument('--route-stop', default=None, help='路线站点距离 CSV')
    parser.add_argument('--output', default=None, help='输出 CSV 路径')
    parser.add_argument('--max-gap', type=float, default=1800.0, help='最大 travel_time 阈值（秒）')
    parser.add_argument('--tmin', type=float, default=None, help='最小时间过滤')
    parser.add_argument('--tmax', type=float, default=None, help='最大时间过滤')
    parser.add_argument('--tt-mode', default='moving', choices=['moving', 'door'], help='TT 计算口径')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    # 默认路径
    if args.observation is None:
        args.observation = str(PROJECT_ROOT / 'data/calibration/l2_observation_vector_corridor_M11_TT.csv')
    if args.route_stop is None:
        args.route_stop = str(PROJECT_ROOT / 'data/processed/kmb_route_stop_dist.csv')
    
    result = build_simulation_vector_tt(
        stopinfo_path=args.stopinfo,
        observation_csv=args.observation,
        route_stop_csv=args.route_stop,
        output_csv=args.output,
        max_gap_seconds=args.max_gap,
        verbose=not args.quiet,
        tmin=args.tmin,
        tmax=args.tmax,
        tt_mode=args.tt_mode
    )
    
    # 打印对比表
    if not args.quiet:
        print("\n[对比表]")
        print(result[['observation_id', 'route', 'bound', 'from_seq', 'to_seq',
                      'travel_time_obs_s', 'travel_time_sim_s', 'ratio']].to_string())


if __name__ == "__main__":
    main()
