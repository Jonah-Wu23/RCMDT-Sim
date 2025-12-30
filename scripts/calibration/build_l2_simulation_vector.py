#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_l2_simulation_vector.py
=============================
从 SUMO edgedata.out.xml 构建与观测向量同口径的 45 维仿真向量。

核心设计（与观测完全同口径）：
1. 使用 l2_observation_vector.csv 定义的 45 个目标路段
2. 使用 link_edge_mapping.csv 的 link→edge 映射
3. 旅行时间加权调和平均：v_link = sum(L_i) / sum(L_i / v_i)
4. 过滤规则：sampledSeconds >= 10s, speed > 0, 排除 internal edges

Author: RMBC-Sim project
Date: 2025-12-25
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_net_edge_lengths(net_path: str) -> Dict[str, float]:
    """
    从 SUMO 路网文件解析每条边的长度。
    
    Args:
        net_path: hk_irn_v3.net.xml 路径
    
    Returns:
        Dict[edge_id, length_m]
    """
    print(f"[INFO] 解析路网边长度: {net_path}")
    
    edge_lengths = {}
    
    # 使用迭代解析避免内存问题
    context = ET.iterparse(net_path, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'edge':
            edge_id = elem.get('id')
            if edge_id and not edge_id.startswith(':'):  # 排除 internal edges
                # 取第一个 lane 的长度
                lane = elem.find('lane')
                if lane is not None:
                    length = lane.get('length')
                    if length:
                        edge_lengths[edge_id] = float(length)
            elem.clear()
    
    print(f"[INFO] 解析到 {len(edge_lengths)} 条边的长度")
    return edge_lengths


def parse_edgedata_xml(
    edgedata_path: str,
    min_sampled_seconds: float = 10.0
) -> Dict[str, float]:
    """
    解析 SUMO edgedata.out.xml，提取边速度。
    
    过滤规则：
    - sampledSeconds >= min_sampled_seconds
    - speed > 0
    - 排除 internal edges (以 : 开头)
    
    Args:
        edgedata_path: edgedata.out.xml 路径
        min_sampled_seconds: 最小采样时间阈值
    
    Returns:
        Dict[edge_id, speed_m_s]
    """
    print(f"[INFO] 解析 edgedata: {edgedata_path}")
    
    edge_speeds = {}
    n_filtered = 0
    
    tree = ET.parse(edgedata_path)
    root = tree.getroot()
    
    for interval in root.findall('.//interval'):
        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            
            # 排除 internal edges
            if edge_id and edge_id.startswith(':'):
                continue
            
            # 获取采样时间和速度
            sampled_seconds = float(edge.get('sampledSeconds', '0'))
            speed_str = edge.get('speed')
            
            # 过滤规则
            if sampled_seconds < min_sampled_seconds:
                n_filtered += 1
                continue
            
            if speed_str:
                speed = float(speed_str)
                if speed > 0:
                    edge_speeds[edge_id] = speed
                else:
                    n_filtered += 1
    
    print(f"[INFO] 有效边: {len(edge_speeds)}, 过滤: {n_filtered}")
    return edge_speeds


def parse_edgedata_xml_subset(
    edgedata_path: str,
    edge_subset: set,
    min_sampled_seconds: float = 10.0,
    t_min: float = 0.0,
    t_max: float = 86400.0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    只解析 mapping 子集内的边数据，使用 sampledSeconds 加权聚合跨 interval。
    并支持时间窗过滤 [t_min, t_max]。
    
    修复：
    - 保留 speed >= 0（拥堵数据很重要）
    - 跨 interval 使用 sampledSeconds 加权平均，而非覆盖写入
    - 时间窗过滤：只聚合 begin >= t_min 且 end <= t_max 的 intervals
    
    Args:
        edgedata_path: edgedata.out.xml 路径
        edge_subset: 需要解析的边 ID 子集（包括 _rev 变体）
        min_sampled_seconds: 最小采样时间阈值
        t_min: 时间窗开始 (秒)
        t_max: 时间窗结束 (秒)
    
    Returns:
        (edge_speeds, edge_sampled_seconds):
            edge_speeds: Dict[edge_id, speed_m_s] (加权平均后)
            edge_sampled_seconds: Dict[edge_id, sampledSeconds] (总和)
    """
    print(f"[INFO] 解析 edgedata (子集模式, 加权聚合): {edgedata_path}")
    print(f"[INFO] 时间窗过滤: [{t_min:.1f}, {t_max:.1f}]")
    print(f"[INFO] 目标边子集大小: {len(edge_subset)}")
    
    # 扩展子集，包含 _rev 变体
    expanded_subset = set(edge_subset)
    for eid in list(edge_subset):
        if eid.endswith('_rev'):
            expanded_subset.add(eid[:-4])
        else:
            expanded_subset.add(f"{eid}_rev")
    
    # 用于 sampledSeconds 加权聚合
    # sum_speed_weighted[edge_id] = sum(speed * sampledSeconds)
    # sum_sampled[edge_id] = sum(sampledSeconds)
    sum_speed_weighted = {}  # Dict[edge_id, float]
    sum_sampled = {}  # Dict[edge_id, float]
    n_records = 0
    
    tree = ET.parse(edgedata_path)
    root = tree.getroot()
    
    for interval in root.findall('.//interval'):
        begin = float(interval.get('begin', -1))
        end = float(interval.get('end', -1))
        
        # 时间窗过滤
        if begin < t_min:
            continue
        if end > t_max:
             continue

        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            
            # 排除 internal edges
            if edge_id and edge_id.startswith(':'):
                continue
            
            # 只处理子集内的边
            if edge_id not in expanded_subset:
                continue
            
            n_records += 1
            
            # 获取采样时间和速度
            sampled_seconds = float(edge.get('sampledSeconds', '0'))
            speed_str = edge.get('speed')
            
            if sampled_seconds <= 0:
                continue
            
            # 保留 speed >= 0（拥堵数据很重要！）
            if speed_str is not None:
                speed = float(speed_str)
                if speed >= 0:  # 包括 0（拥堵）
                    # 累加加权值
                    if edge_id not in sum_speed_weighted:
                        sum_speed_weighted[edge_id] = 0.0
                        sum_sampled[edge_id] = 0.0
                    
                    sum_speed_weighted[edge_id] += speed * sampled_seconds
                    sum_sampled[edge_id] += sampled_seconds
    
    # 计算加权平均速度
    edge_speeds = {}
    edge_sampled_seconds = {}
    n_valid = 0
    n_filtered = 0
    
    for edge_id in sum_sampled:
        total_sampled = sum_sampled[edge_id]
        edge_sampled_seconds[edge_id] = total_sampled
        
        # 采样时间门槛
        if total_sampled >= min_sampled_seconds:
            avg_speed = sum_speed_weighted[edge_id] / total_sampled
            edge_speeds[edge_id] = avg_speed
            n_valid += 1
        else:
            n_filtered += 1
    
    print(f"[INFO] 子集内记录: {n_records}, 有效边: {n_valid}, 采样不足: {n_filtered}")
    return edge_speeds, edge_sampled_seconds


def load_link_edge_mapping(mapping_path: str) -> Dict[int, List[str]]:
    """
    加载 link→edge 映射表。
    
    Returns:
        Dict[observation_id, List[edge_id]]
    """
    mapping = {}
    df = pd.read_csv(mapping_path)
    
    for _, row in df.iterrows():
        obs_id = int(row['observation_id'])
        edge_ids_str = row['edge_ids']
        
        try:
            edge_ids = json.loads(edge_ids_str)
        except (json.JSONDecodeError, TypeError):
            edge_ids = []
        
        mapping[obs_id] = edge_ids
    
    return mapping


def compute_link_speed(edge_speeds, edge_lengths, edge_ids, edge_sampled_seconds=None):
    """
    计算 link 速度（旅行时间加权调和平均）+ 覆盖率统计。
    
    公式: v_link = matched_length / total_travel_time
    
    原则：
    - total_length: 映射边总长度（每条边只加一次）
    - matched_length: 有速度数据的边长度
    - v_link 只由 matched 部分定义
    """
    SPEED_EPS = 0.1  # m/s, 保留拥堵(≈0.36km/h)

    total_length = 0.0
    matched_length = 0.0
    total_travel_time = 0.0
    sampled_seconds_sum = 0.0
    n_matched = 0
    n_total = len(edge_ids)

    for edge_id in edge_ids:
        # 候选（原 / _rev）
        candidates = [edge_id, edge_id[:-4] if edge_id.endswith('_rev') else f"{edge_id}_rev"]

        # 1) total_length：只加一次（以"能找到的长度"为准）
        L_net = 0.0
        for eid in candidates:
            if eid in edge_lengths:
                L_net = edge_lengths[eid]
                break
        total_length += L_net

        # 2) matched：找到 speed 的那条边，累加 matched_length 与 travel_time
        for eid in candidates:
            if eid in edge_speeds:
                L = edge_lengths.get(eid, L_net)
                if L <= 0:
                    continue
                v = max(edge_speeds[eid], SPEED_EPS)
                matched_length += L
                total_travel_time += L / v
                n_matched += 1
                if edge_sampled_seconds and eid in edge_sampled_seconds:
                    sampled_seconds_sum += edge_sampled_seconds[eid]
                break

    # 速度（只由 matched 部分定义）
    speed_kmh = (matched_length / total_travel_time * 3.6) if total_travel_time > 0 else float("nan")
    coverage = (matched_length / total_length) if total_length > 0 else 0.0

    return {
        "speed_kmh": speed_kmh,
        "n_matched": n_matched,
        "n_total": n_total,
        "matched_length_m": matched_length,
        "total_length_m": total_length,
        "coverage": coverage,
        "sampledSeconds_sum": sampled_seconds_sum,
    }


def build_simulation_vector(
    edgedata_path: str,
    observation_csv: str,
    mapping_csv: str,
    net_file: str,
    output_csv: Optional[str] = None,
    min_sampled_seconds: float = 10.0,
    min_coverage: float = 0.7,
    t_min: float = 0.0,
    t_max: float = 86400.0,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    构建与观测向量同口径的 45 维仿真向量。
    
    Args:
        edgedata_path: SUMO edgedata.out.xml 路径
        observation_csv: l2_observation_vector.csv 路径
        mapping_csv: link_edge_mapping.csv 路径
        net_file: hk_irn_v3.net.xml 路径
        output_csv: 输出 CSV 路径（可选）
        min_sampled_seconds: 边采样时间阈值
        min_sampled_seconds: 边采样时间阈值
        min_coverage: 最小长度覆盖率门槛 (默认 0.7)
        t_min: (可选) 统计起始时间
        t_max: (可选) 统计结束时间
        verbose: 是否打印详细信息
    
    Returns:
        DataFrame with columns:
            observation_id, route, bound, from_seq, to_seq,
            obs_speed_kmh, sim_speed_kmh, n_edges_matched, n_edges_total,
            coverage, coverage_raw, matched_length_m, total_length_m, 
            reason, matched
    """
    # 0. 分发逻辑 (支持 Travel Time)
    metric_type = kwargs.get('metric_type', 'speed')
    
    if metric_type == 'traveltime':
        # 动态导入 TT 模块
        try:
            from build_l2_sim_vector_traveltime import build_simulation_vector_tt
            
            # 从 kwargs 提取 tt_mode，默认 'moving'
            tt_mode = kwargs.get('tt_mode', 'moving')
            
            # 推断 route_stop_csv 路径
            route_stop_csv = str(Path(observation_csv).parent.parent / 'processed' / 'kmb_route_stop_dist.csv')
            
            return build_simulation_vector_tt(
                stopinfo_path=edgedata_path,  # 注意：这里edgedata_path实际传的是stopinfo路径
                observation_csv=observation_csv,
                route_stop_csv=route_stop_csv,
                output_csv=output_csv,
                verbose=False,
                tmin=t_min,
                tmax=t_max,
                tt_mode=tt_mode
            )
        except ImportError:
            print("[ERROR] 无法导入 build_l2_sim_vector_traveltime")
            return pd.DataFrame()

    # 1. 加载数据
    obs_df = pd.read_csv(observation_csv)
    link_edge_map = load_link_edge_mapping(mapping_csv)
    edge_lengths = parse_net_edge_lengths(net_file)
    
    # 2. 收集所有 mapping 中的边，只过滤这个子集
    all_mapped_edges = set()
    for edge_ids in link_edge_map.values():
        all_mapped_edges.update(edge_ids)
    
    if verbose:
        print(f"[INFO] 观测点: {len(obs_df)}, 映射表: {len(link_edge_map)}")
        print(f"[INFO] 映射边总数: {len(all_mapped_edges)}")
        print(f"[INFO] 覆盖率门槛: {min_coverage:.0%}")
    
    # 使用子集模式解析 edgedata
    edge_speeds, edge_sampled_seconds = parse_edgedata_xml_subset(
        edgedata_path, all_mapped_edges, min_sampled_seconds, t_min, t_max
    )
    
    # 3. 构建仿真向量
    results = []
    reason_counts = {}
    
    for _, row in obs_df.iterrows():
        obs_id = int(row['observation_id'])
        edge_ids = link_edge_map.get(obs_id, [])
        
        # 诊断原因枚举
        # NO_MAPPING: mapping 里找不到 edge_ids
        # NO_LENGTH: edge_ids 在 net 里无长度
        # NO_SAMPLED_EDGE: edge 在 edgedata 里基本没有记录
        # LOW_SAMPLED: 有记录但总 sampledSeconds < min_sampled_seconds
        # LOW_COVERAGE: coverage_raw < min_coverage
        # OK: 匹配成功
        
        if len(edge_ids) == 0:
            reason = "NO_MAPPING"
            stats = {"speed_kmh": float("nan"), "n_matched": 0, "n_total": 0, 
                     "matched_length_m": 0, "total_length_m": 0, "coverage": 0, "sampledSeconds_sum": 0}
        else:
            # 计算 link 速度 + 覆盖率统计
            stats = compute_link_speed(
                edge_speeds, edge_lengths, edge_ids, edge_sampled_seconds
            )
            
            # 诊断原因
            if stats["total_length_m"] == 0:
                reason = "NO_LENGTH"
            elif stats["n_matched"] == 0:
                # 进一步诊断：是没有采样还是采样不足？
                total_sampled = sum(edge_sampled_seconds.get(eid, 0) for eid in edge_ids)
                if total_sampled == 0:
                    reason = "NO_SAMPLED_EDGE"
                else:
                    reason = "LOW_SAMPLED"
            elif stats["coverage"] < min_coverage:
                reason = "LOW_COVERAGE"
            else:
                reason = "OK"
        
        sim_speed_kmh = stats["speed_kmh"]
        n_matched = stats["n_matched"]
        n_total = stats["n_total"]
        coverage_raw = stats["coverage"]  # 原始覆盖率（不做门槛）
        matched_length_m = stats["matched_length_m"]
        total_length_m = stats["total_length_m"]
        
        # 覆盖率硬门槛：coverage < min_coverage 则标记为 NaN
        if reason != "OK":
            sim_speed_kmh = float("nan")
            matched = False
        else:
            matched = True
        
        # 统计原因分布
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        results.append({
            'observation_id': obs_id,
            'route': row['route'],
            'bound': row['bound'],
            'from_seq': row['from_seq'],
            'to_seq': row['to_seq'],
            'obs_speed_kmh': row['mean_speed_kmh'],
            'sim_speed_kmh': sim_speed_kmh,
            'n_edges_matched': n_matched,
            'n_edges_total': n_total,
            'coverage': coverage_raw if matched else float("nan"),
            'coverage_raw': coverage_raw,
            'matched_length_m': matched_length_m,
            'total_length_m': total_length_m,
            'sampledSeconds_sum': stats["sampledSeconds_sum"],  # 用于 corridor mask
            'reason': reason,
            'matched': matched
        })
    
    result_df = pd.DataFrame(results)
    
    # 4. 统计
    matched_count = result_df['matched'].sum()
    total = len(result_df)
    
    if verbose:
        print(f"\n[RESULT] 仿真向量构建完成")
        print(f"  - 维度: {total}")
        print(f"  - 匹配成功: {matched_count}/{total} ({100*matched_count/total:.1f}%)")
        
        # 打印原因分布
        print(f"\n[DIAGNOSIS] Unmatched 原因分布:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  - {reason}: {count}")
        
        valid = result_df[result_df['matched']]
        if len(valid) > 0:
            print(f"\n  - 仿真速度范围: {valid['sim_speed_kmh'].min():.2f} - {valid['sim_speed_kmh'].max():.2f} km/h")
            print(f"  - 仿真速度均值: {valid['sim_speed_kmh'].mean():.2f} km/h")
            print(f"  - 覆盖率中位数: {valid['coverage'].median():.1%}")
    
    # 5. 保存
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"[INFO] 仿真向量已保存: {output_csv}")
    
    return result_df


def calculate_residual_stats(sim_df: pd.DataFrame) -> Dict[str, float]:
    """计算残差统计量"""
    valid = sim_df[sim_df['matched']].copy()
    
    if len(valid) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'n_matched': 0}
    
    residuals = valid['obs_speed_kmh'] - valid['sim_speed_kmh']
    
    return {
        'rmse': np.sqrt((residuals ** 2).mean()),
        'mae': residuals.abs().mean(),
        'bias': residuals.mean(),
        'n_matched': len(valid)
    }


def main():
    parser = argparse.ArgumentParser(
        description='构建与观测同口径的 L2 仿真速度向量'
    )
    parser.add_argument(
        '--edgedata', '-e',
        type=str,
        required=True,
        help='SUMO edgedata.out.xml 路径'
    )
    parser.add_argument(
        '--observation', '-o',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'calibration' / 'l2_observation_vector.csv'),
        help='观测向量 CSV 路径'
    )
    parser.add_argument(
        '--mapping', '-m',
        type=str,
        default=str(PROJECT_ROOT / 'config' / 'calibration' / 'link_edge_mapping.csv'),
        help='link-edge 映射表路径'
    )
    parser.add_argument(
        '--net', '-n',
        type=str,
        default=str(PROJECT_ROOT / 'sumo' / 'net' / 'hk_irn_v3.net.xml'),
        help='SUMO 路网文件路径'
    )
    parser.add_argument(
        '--output', '-O',
        type=str,
        default=None,
        help='输出仿真向量 CSV 路径'
    )
    parser.add_argument(
        '--min-sampled-seconds',
        type=float,
        default=10.0,
        help='边采样时间阈值 (默认: 10s)'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=0.7,
        help='最小长度覆盖率门槛 (默认: 0.7)'
    )
    parser.add_argument(
        '--tmin',
        type=float,
        default=0.0,
        help='统计时间窗起始 (秒), 默认 0'
    )
    parser.add_argument(
        '--tmax',
        type=float,
        default=86400.0,
        help='统计时间窗结束 (秒), 默认 86400'
    )
    parser.add_argument(
        '--debug-links',
        type=str,
        default=None,
        help='调试：打印指定 link 的详细计算过程 (逗号分隔，如 "1,10,30")'
    )
    parser.add_argument(
        '--tt-mode',
        type=str,
        default='moving',
        choices=['moving', 'door'],
        help='Travel Time 计算口径 (仅对 traveltime 指标有效)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.edgedata):
        print(f"[ERROR] edgedata 文件不存在: {args.edgedata}")
        sys.exit(1)
    
    # 构建仿真向量
    sim_df = build_simulation_vector(
        edgedata_path=args.edgedata,
        observation_csv=args.observation,
        mapping_csv=args.mapping,
        net_file=args.net,
        output_csv=args.output,
        min_sampled_seconds=args.min_sampled_seconds,
        min_coverage=args.min_coverage,
        t_min=args.tmin,
        t_max=args.tmax,
        tt_mode=args.tt_mode
    )
    
    # 计算残差统计
    stats = calculate_residual_stats(sim_df)
    print(f"\n[STATS] 残差统计:")
    print(f"  - RMSE: {stats['rmse']:.2f} km/h")
    print(f"  - MAE:  {stats['mae']:.2f} km/h")
    print(f"  - Bias: {stats['bias']:.2f} km/h")
    
    # 调试模式：打印指定 link 的详细计算（带覆盖率统计）
    if args.debug_links:
        link_ids = [int(x.strip()) for x in args.debug_links.split(',')]
        link_edge_map = load_link_edge_mapping(args.mapping)
        edge_lengths = parse_net_edge_lengths(args.net)
        
        # 收集所有映射边
        all_mapped_edges = set()
        for edge_ids in link_edge_map.values():
            all_mapped_edges.update(edge_ids)
        
        edge_speeds, edge_sampled_seconds = parse_edgedata_xml_subset(
            args.edgedata, all_mapped_edges, args.min_sampled_seconds
        )
        
        print(f"\n[DEBUG] 详细计算过程 (覆盖率门槛: {args.min_coverage:.0%}):")
        for link_id in link_ids:
            edge_ids = link_edge_map.get(link_id, [])
            
            # 使用新的 compute_link_speed 获取完整统计
            stats = compute_link_speed(
                edge_speeds, edge_lengths, edge_ids, edge_sampled_seconds
            )
            
            print(f"\n  Link {link_id}:")
            print(f"    映射边数: {stats['n_total']}, 匹配边数: {stats['n_matched']}")
            print(f"    总长度: {stats['total_length_m']:.1f}m, 匹配长度: {stats['matched_length_m']:.1f}m")
            print(f"    覆盖率: {stats['coverage']:.1%} {'✓' if stats['coverage'] >= args.min_coverage else '✗ (低于门槛)'}")
            print(f"    速度: {stats['speed_kmh']:.2f} km/h" if not np.isnan(stats['speed_kmh']) else "    速度: NaN")
            print(f"    采样秒数: {stats['sampledSeconds_sum']:.1f}s")


if __name__ == "__main__":
    main()
