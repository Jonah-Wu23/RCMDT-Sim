#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p13_signal_audit.py
===================
P13-4: 信号容量审计 (Signal Capacity Audit)

目标: 分析走廊 core edges 上游路口的信号配时：
1. 找到 core edges 的 to-node (上游路口)
2. 提取这些路口的 tlLogic 配时
3. 分析绿信比和周期
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse
import sys

# 强制使用 UTF-8 输出以避免 Windows 控制台乱码
sys.stdout.reconfigure(encoding='utf-8')


def load_core_edges(core_edges_path: str) -> set:
    """加载 core edges"""
    edges = set()
    with open(core_edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            edges.add(line)
    return edges


def parse_edge_nodes(net_path: str, core_edges: set) -> dict:
    """解析网络文件，获取 core edges 的起点/终点节点"""
    print(f"[INFO] 解析网络文件: {net_path}")
    
    edge_nodes = {}  # edge_id -> (from_node, to_node)
    
    # 使用迭代解析以避免内存问题
    for event, elem in ET.iterparse(net_path, events=['end']):
        if elem.tag == 'edge':
            edge_id = elem.get('id')
            if edge_id in core_edges:
                from_node = elem.get('from')
                to_node = elem.get('to')
                if from_node and to_node:
                    edge_nodes[edge_id] = (from_node, to_node)
            elem.clear()
    
    print(f"[INFO] 找到 {len(edge_nodes)}/{len(core_edges)} 个 core edges 的节点信息")
    return edge_nodes


def parse_tl_logics(net_path: str, target_nodes: set) -> dict:
    """解析 tlLogic 配时信息"""
    print(f"[INFO] 解析信号配时...")
    
    tl_logics = {}  # tl_id -> {cycle, phases: [{duration, state, green_count}]}
    
    for event, elem in ET.iterparse(net_path, events=['end']):
        if elem.tag == 'tlLogic':
            tl_id = elem.get('id')
            
            phases = []
            total_duration = 0
            for phase in elem.findall('phase'):
                duration = float(phase.get('duration', 0))
                state = phase.get('state', '')
                green_count = sum(1 for c in state if c in 'Gg')
                total_count = len(state)
                phases.append({
                    'duration': duration,
                    'state': state,
                    'green_count': green_count,
                    'total_lanes': total_count
                })
                total_duration += duration
            
            tl_logics[tl_id] = {
                'cycle': total_duration,
                'phases': phases,
                'num_phases': len(phases)
            }
            elem.clear()
    
    print(f"[INFO] 解析了 {len(tl_logics)} 个信号灯配时")
    return tl_logics


def parse_connections(net_path: str, core_edges: set) -> dict:
    """解析连接，找到 core edges 对应的信号灯"""
    print(f"[INFO] 解析连接信息...")
    
    edge_tls = defaultdict(set)  # edge_id -> set of tl_ids
    
    for event, elem in ET.iterparse(net_path, events=['end']):
        if elem.tag == 'connection':
            from_edge = elem.get('from')
            tl_id = elem.get('tl')
            
            if from_edge in core_edges and tl_id:
                edge_tls[from_edge].add(tl_id)
            elem.clear()
    
    print(f"[INFO] 找到 {len(edge_tls)} 个 core edges 有信号灯控制")
    return edge_tls


def analyze_tl_timing(tl_logics: dict, edge_tls: dict, core_edges: set):
    """分析信号配时"""
    print("\n" + "="*60)
    print("P13-4A 信号配时审计结果")
    print("="*60)
    
    # 收集所有相关的 tl_ids
    relevant_tls = set()
    for edge_id in core_edges:
        if edge_id in edge_tls:
            relevant_tls.update(edge_tls[edge_id])
    
    print(f"\n[INFO] Core Edges 关联的信号灯: {len(relevant_tls)} 个")
    
    # 分析周期分布
    cycles = []
    green_ratios = []
    
    for tl_id in relevant_tls:
        if tl_id not in tl_logics:
            continue
        
        tl = tl_logics[tl_id]
        cycles.append(tl['cycle'])
        
        # 计算最大绿信比 (任一相位)
        max_green_ratio = 0
        for phase in tl['phases']:
            if phase['total_lanes'] > 0:
                ratio = phase['green_count'] / phase['total_lanes']
                if phase['duration'] > 5:  # 排除黄灯相位
                    max_green_ratio = max(max_green_ratio, ratio)
        green_ratios.append(max_green_ratio)
    
    if cycles:
        print(f"\n信号周期统计:")
        print(f"  Min: {min(cycles):.0f}s")
        print(f"  Max: {max(cycles):.0f}s")
        print(f"  Mean: {sum(cycles)/len(cycles):.0f}s")
        
        # 周期分布
        short_cycle = sum(1 for c in cycles if c < 60)
        medium_cycle = sum(1 for c in cycles if 60 <= c < 120)
        long_cycle = sum(1 for c in cycles if c >= 120)
        print(f"\n周期分布:")
        print(f"  短周期 (<60s): {short_cycle} 个 ({100*short_cycle/len(cycles):.1f}%)")
        print(f"  中周期 (60-120s): {medium_cycle} 个 ({100*medium_cycle/len(cycles):.1f}%)")
        print(f"  长周期 (>=120s): {long_cycle} 个 ({100*long_cycle/len(cycles):.1f}%)")
    
    if green_ratios:
        print(f"\n绿信比统计 (相位内绿灯占比):")
        print(f"  Min: {min(green_ratios)*100:.1f}%")
        print(f"  Max: {max(green_ratios)*100:.1f}%")
        print(f"  Mean: {sum(green_ratios)/len(green_ratios)*100:.1f}%")
        
        high_green = sum(1 for r in green_ratios if r > 0.8)
        print(f"\n高绿信比 (>80%): {high_green} 个 - ", end="")
        if high_green > 0:
            print("⚠️ 可能导致容量过大")
        else:
            print("[OK] 正常")
    
    # 检查异常配时
    print("\n" + "-"*40)
    print("异常配时检测:")
    
    anomalies = []
    for tl_id in relevant_tls:
        if tl_id not in tl_logics:
            continue
        
        tl = tl_logics[tl_id]
        
        # 检测: 超短周期
        if tl['cycle'] < 30:
            anomalies.append(f"  {tl_id}: 超短周期 {tl['cycle']:.0f}s")
        
        # 检测: 单相位 (无信号控制效果)
        if tl['num_phases'] <= 2:
            # 检查是否有冲突相位
            has_conflict = False
            for phase in tl['phases']:
                if 'r' in phase['state'].lower():
                    has_conflict = True
                    break
            if not has_conflict:
                anomalies.append(f"  {tl_id}: 疑似全绿 (无红灯相位)")
        
        # 检测: 所有相位都有高绿信比
        all_high_green = all(
            phase['green_count'] / phase['total_lanes'] > 0.7 
            if phase['total_lanes'] > 0 and phase['duration'] > 5 else False
            for phase in tl['phases']
        )
        if all_high_green and tl['num_phases'] > 2:
            anomalies.append(f"  {tl_id}: 所有相位高绿信比")
    
    if anomalies:
        print(f"发现 {len(anomalies)} 个异常:")
        for a in anomalies[:10]:  # 只显示前10个
            print(a)
        if len(anomalies) > 10:
            print(f"  ... 还有 {len(anomalies)-10} 个")
    else:
        print("  未发现明显异常 ✓")
    
    # 输出一些具体的 tlLogic 示例
    print("\n" + "-"*40)
    print("Core Edges 关联信号灯配时示例 (前5个):")
    
    shown = 0
    for tl_id in list(relevant_tls)[:5]:
        if tl_id not in tl_logics:
            continue
        tl = tl_logics[tl_id]
        print(f"\n  {tl_id}: 周期={tl['cycle']:.0f}s, 相位数={tl['num_phases']}")
        for i, phase in enumerate(tl['phases']):
            print(f"    Phase {i}: {phase['duration']:.0f}s, 状态={phase['state'][:20]}..., 绿={phase['green_count']}/{phase['total_lanes']}")
        shown += 1
    
    return {
        'num_tls': len(relevant_tls),
        'cycles': cycles,
        'green_ratios': green_ratios,
        'anomalies': anomalies
    }


def main():
    parser = argparse.ArgumentParser(description='P13-4: 信号容量审计')
    parser.add_argument('--net', '-n', 
                        default='sumo/net/hk_cropped.net.xml',
                        help='网络文件路径')
    parser.add_argument('--core-edges', '-c',
                        default='config/calibration/core_edges.txt',
                        help='Core edges 文件路径')
    
    args = parser.parse_args()
    
    # 加载 core edges
    core_edges = load_core_edges(args.core_edges)
    print(f"[INFO] 加载 {len(core_edges)} 个 core edges")
    
    # 解析连接找到相关信号灯
    edge_tls = parse_connections(args.net, core_edges)
    
    # 解析信号配时
    tl_logics = parse_tl_logics(args.net, set())
    
    # 分析配时
    results = analyze_tl_timing(tl_logics, edge_tls, core_edges)
    
    print("\n" + "="*60)
    print("审计完成")
    print("="*60)


if __name__ == '__main__':
    main()
