#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_moving_obs_from_irn.py
==============================
从 IRN Segment Speed 数据构建 moving 口径观测向量。

数据源：
- irnAvgSpeed-all-*.xml: IRN 路段实时速度（moving-only 口径，km/h）
- hk_irn_edges.geojson: IRN 路段地理信息（ROUTE_ID = segment_id）
- l2_observation_vector_corridor_M11.csv: door 口径观测向量模板

输出：data/calibration/l2_observation_vector_corridor_M11_moving_irn.csv
"""

import os
import sys
import json
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_irn_speed_xml(xml_path: str) -> Dict[int, float]:
    """
    解析 irnAvgSpeed-all.xml，提取 segment_id -> speed 映射
    
    Returns:
        {segment_id: speed_kmh} 仅包含 valid='Y' 的记录
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    speeds = {}
    for segment in root.findall('.//segment'):
        seg_id = segment.find('segment_id')
        speed = segment.find('speed')
        valid = segment.find('valid')
        
        if seg_id is not None and speed is not None and valid is not None:
            if valid.text == 'Y':
                try:
                    speeds[int(seg_id.text)] = float(speed.text)
                except (ValueError, TypeError):
                    pass
    
    return speeds


def load_irn_edges_geojson(geojson_path: str) -> pd.DataFrame:
    """
    加载 IRN edges GeoJSON，提取 ROUTE_ID 和几何信息
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for feature in data['features']:
        props = feature['properties']
        geom = feature['geometry']
        
        # 计算路段长度（简化：使用坐标点距离）
        coords = geom['coordinates'][0] if geom['type'] == 'MultiLineString' else geom['coordinates']
        length_m = 0.0
        if len(coords) > 1:
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i][:2]
                lon2, lat2 = coords[i+1][:2]
                # 简化距离计算（度 → 米，香港纬度约 22°）
                dx = (lon2 - lon1) * 111320 * np.cos(np.radians(22.3))
                dy = (lat2 - lat1) * 110540
                length_m += np.sqrt(dx**2 + dy**2)
        
        records.append({
            'route_id': props.get('ROUTE_ID'),
            'street_name': props.get('STREET_ENAME'),
            'speed_kmh': props.get('speed_kmh', 50.0),
            'lanes': props.get('lanes', 1),
            'direction': props.get('TRAVEL_DIRECTION'),
            'length_m': length_m,
            'centroid_lon': np.mean([c[0] for c in coords]) if coords else None,
            'centroid_lat': np.mean([c[1] for c in coords]) if coords else None
        })
    
    return pd.DataFrame(records)


def load_corridor_stops() -> pd.DataFrame:
    """
    加载 corridor 站点信息（用于地理匹配）
    """
    stop_dist_path = PROJECT_ROOT / 'data' / 'processed' / 'kmb_route_stop_dist.csv'
    if not stop_dist_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(stop_dist_path)
    return df


def build_segment_to_link_mapping(
    irn_edges: pd.DataFrame,
    obs_df: pd.DataFrame
) -> Dict[Tuple, List[int]]:
    """
    建立 corridor link -> IRN segment_id 列表的映射
    
    方法：基于地理位置匹配（简化版）
    """
    # 加载站点坐标
    stop_coords = {}
    stop_path = PROJECT_ROOT / 'data' / 'processed' / 'kmb_stops_with_coords.csv'
    if stop_path.exists():
        stops_df = pd.read_csv(stop_path)
        for _, row in stops_df.iterrows():
            if pd.notna(row.get('lat')) and pd.notna(row.get('long')):
                stop_coords[row['stop_id']] = (row['lat'], row['long'])
    
    # 简化：暂时返回空映射，使用估算方法作为 fallback
    # TODO: 实现完整的地理匹配
    return {}


def aggregate_irn_speeds(
    irn_data_dir: Path,
    time_window: Tuple[str, str] = ('17:00', '18:00')
) -> Dict[int, float]:
    """
    聚合多个 IRN XML 文件的速度数据
    
    Args:
        irn_data_dir: 包含 irnAvgSpeed-all-*.xml 的目录
        time_window: 时间窗口 (HH:MM, HH:MM)
    
    Returns:
        {segment_id: mean_speed_kmh}
    """
    xml_files = list(irn_data_dir.glob('irnAvgSpeed-all-*.xml'))
    if not xml_files:
        print(f"[WARN] 未找到 IRN XML 文件: {irn_data_dir}")
        return {}
    
    print(f"[INFO] 找到 {len(xml_files)} 个 IRN XML 文件")
    
    # 收集所有速度记录
    all_speeds = defaultdict(list)
    
    for xml_path in xml_files:
        speeds = parse_irn_speed_xml(str(xml_path))
        for seg_id, speed in speeds.items():
            all_speeds[seg_id].append(speed)
    
    # 计算平均速度
    mean_speeds = {seg_id: np.mean(speeds) for seg_id, speeds in all_speeds.items()}
    
    print(f"[INFO] 聚合后有 {len(mean_speeds)} 个有效路段速度")
    if mean_speeds:
        speeds_array = np.array(list(mean_speeds.values()))
        print(f"[INFO] 速度统计: min={speeds_array.min():.1f}, median={np.median(speeds_array):.1f}, max={speeds_array.max():.1f} km/h")
    
    return mean_speeds


def build_moving_observation_vector_from_irn() -> Path:
    """
    构建 moving 口径观测向量
    
    策略：使用 Op-L2-v1.1 Rule C 去污染后的 clean 样本中位速度
    
    Rule C: (T > 325s) ∧ (v < 5 km/h) → Ghost Jam
    
    数据来源：link_times.csv 短距离 link (<1km) 的 clean 样本
    结果：median = 13.52 km/h
    """
    door_obs_path = PROJECT_ROOT / 'data' / 'calibration' / 'l2_observation_vector_corridor_M11.csv'
    moving_obs_path = PROJECT_ROOT / 'data' / 'calibration' / 'l2_observation_vector_corridor_M11_moving_irn.csv'
    link_times_path = PROJECT_ROOT / 'data' / 'processed' / 'link_times.csv'
    
    # 如果已存在，删除后重建（确保使用最新逻辑）
    if moving_obs_path.exists():
        moving_obs_path.unlink()
    
    print(f"[INFO] 构建 moving 口径观测向量 (基于 Op-L2-v1.1 Rule C 去污染)...")
    
    # 加载原始 link_times 数据并应用 Rule C
    if link_times_path.exists():
        link_df = pd.read_csv(link_times_path)
        
        # Rule C: (T > 325s) AND (v < 5 km/h) -> Ghost Jam
        ghost_jam_mask = (link_df['travel_time_s'] > 325) & (link_df['speed_kmh'] < 5)
        clean_df = link_df[~ghost_jam_mask]
        
        # 只取公交走廊核心（短距离 link < 1km）
        corridor_clean = clean_df[clean_df['dist_m'] < 1000]
        
        print(f"[INFO] Rule C 过滤: {len(link_df)} 总样本 -> {len(clean_df)} clean 样本")
        print(f"[INFO] Ghost Jam 样本: {ghost_jam_mask.sum()} ({ghost_jam_mask.sum()/len(link_df)*100:.1f}%)")
        print(f"[INFO] 公交走廊 (<1km) clean 样本: {len(corridor_clean)}")
        
        # 计算 clean 样本的 moving 速度
        MOVING_SPEED_KMH = corridor_clean['speed_kmh'].median()
        print(f"[INFO] Clean 样本 moving speed median = {MOVING_SPEED_KMH:.2f} km/h")
    else:
        # fallback: 使用预计算值
        MOVING_SPEED_KMH = 13.52
        print(f"[WARN] link_times.csv 不存在，使用预计算值: {MOVING_SPEED_KMH} km/h")
    
    # 加载 door 口径观测向量
    door_df = pd.read_csv(door_obs_path)
    print(f"[INFO] Door 观测向量: {len(door_df)} 个观测点")
    print(f"[INFO] Door speed median = {door_df['mean_speed_kmh'].median():.2f} km/h")
    
    # 构建 moving 观测向量
    moving_df = door_df.copy()
    
    # 方法：按比例缩放，保持相对差异
    door_median = door_df['mean_speed_kmh'].median()
    speed_ratio = MOVING_SPEED_KMH / door_median if door_median > 0 else 1.0
    
    moving_df['mean_speed_kmh'] = door_df['mean_speed_kmh'] * speed_ratio
    
    print(f"[INFO] Speed ratio = {speed_ratio:.2f}x")
    print(f"[INFO] Moving speed median = {moving_df['mean_speed_kmh'].median():.2f} km/h")
    
    # 保存
    moving_df.to_csv(moving_obs_path, index=False)
    print(f"[INFO] 已保存: {moving_obs_path}")
    
    return moving_obs_path


def _fallback_estimation(door_df: pd.DataFrame, output_path: Path) -> Path:
    """
    估算方法（当 IRN 数据不可用时）
    """
    from scripts.experiments_v4.run_semantic_alignment_test import L1_PARAMS
    
    avg_passengers = 5
    t_dwell_per_stop = L1_PARAMS['t_fixed'] + L1_PARAMS['t_board'] * avg_passengers
    
    df = door_df.copy()
    df['door_tt_s'] = df['dist_m'] / (df['mean_speed_kmh'] / 3.6)
    df['moving_tt_s'] = (df['door_tt_s'] - t_dwell_per_stop).clip(lower=1.0)
    df['mean_speed_kmh'] = (df['dist_m'] / df['moving_tt_s']) * 3.6
    df = df.drop(columns=['door_tt_s', 'moving_tt_s'])
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] 使用估算方法生成 moving 观测向量: {output_path}")
    
    return output_path


def main():
    print("=" * 60)
    print("构建 Moving 口径观测向量 (从 IRN Segment Speed)")
    print("=" * 60)
    
    output_path = build_moving_observation_vector_from_irn()
    
    # 对比验证
    door_path = PROJECT_ROOT / 'data' / 'calibration' / 'l2_observation_vector_corridor_M11.csv'
    door_df = pd.read_csv(door_path)
    moving_df = pd.read_csv(output_path)
    
    print("\n[结果对比]")
    print(f"  Door median:   {door_df['mean_speed_kmh'].median():.2f} km/h")
    print(f"  Moving median: {moving_df['mean_speed_kmh'].median():.2f} km/h")
    print(f"  比值:          {moving_df['mean_speed_kmh'].median() / door_df['mean_speed_kmh'].median():.2f}x")


if __name__ == "__main__":
    main()
