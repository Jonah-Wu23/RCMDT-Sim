#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scale_corridor_flows.py
=======================
P13-3: 走廊定向喂流 (Corridor Flow Weighting)

对经过 core edges 的 flows 进行加权，同时保持总需求量守恒。

公式:
    q'_i = q_i * w_i * β
    其中 β = Σq_i / Σ(q_i * w_i)
    w_i = α (core flow) or 1 (non-core flow)

输入:
    background_*.rou.xml
    core_edges.txt
    alpha (权重因子, 默认 1.5)

输出:
    background_weighted_a{alpha}.rou.xml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Set, Dict, List, Tuple
import re


def load_core_edges(core_edges_path: str) -> Set[str]:
    """加载 core edges 列表"""
    edges = set()
    with open(core_edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            edges.add(line)
    print(f"[INFO] 加载 {len(edges)} 个 core edges")
    return edges


def parse_route_edges(route_elem: ET.Element) -> List[str]:
    """解析 route 元素的 edges 属性"""
    edges_str = route_elem.get('edges', '')
    return edges_str.split() if edges_str else []


def is_core_flow(flow_elem: ET.Element, root: ET.Element, core_edges: Set[str], 
                 min_intersection: int = 1) -> Tuple[bool, int]:
    """
    判断 flow 是否为 core flow
    
    通过检查 flow 引用的 route 或 routeDistribution 是否与 core edges 有交集
    
    Returns:
        (is_core, intersection_count)
    """
    def get_route_edges_by_id(route_id: str) -> List[str]:
        """通过 route id 获取边列表"""
        route = root.find(f".//route[@id='{route_id}']")
        if route is not None:
            return parse_route_edges(route)
        return []
    
    def check_route_elem(route_elem: ET.Element) -> int:
        """检查单个 route 元素并返回交集数量"""
        # 优先检查 refId (引用外部 route)
        ref_id = route_elem.get('refId')
        if ref_id:
            edges = get_route_edges_by_id(ref_id)
        else:
            # 直接使用 edges 属性
            edges = parse_route_edges(route_elem)
        return len(set(edges) & core_edges)
    
    # 检查直接的 route 属性
    route_ref = flow_elem.get('route')
    if route_ref:
        # 可能是 routeDistribution 或者直接 route
        # 首先检查是否是 routeDistribution
        route_dist = root.find(f".//routeDistribution[@id='{route_ref}']")
        if route_dist is not None:
            max_intersection = 0
            for route in route_dist.findall('route'):
                intersection = check_route_elem(route)
                max_intersection = max(max_intersection, intersection)
            return max_intersection >= min_intersection, max_intersection
        
        # 如果不是 routeDistribution，检查是否是直接 route
        edges = get_route_edges_by_id(route_ref)
        intersection = len(set(edges) & core_edges)
        return intersection >= min_intersection, intersection
    
    # 检查内嵌的 route
    inner_route = flow_elem.find('route')
    if inner_route is not None:
        intersection = check_route_elem(inner_route)
        return intersection >= min_intersection, intersection
    
    # 检查内嵌的 routeDistribution
    inner_dist = flow_elem.find('routeDistribution')
    if inner_dist is not None:
        max_intersection = 0
        for route in inner_dist.findall('route'):
            intersection = check_route_elem(route)
            max_intersection = max(max_intersection, intersection)
        return max_intersection >= min_intersection, max_intersection
    
    return False, 0


def get_flow_demand(flow_elem: ET.Element) -> float:
    """获取 flow 的需求量 (vehsPerHour, period, number, probability)"""
    # 优先使用 vehsPerHour
    vph = flow_elem.get('vehsPerHour')
    if vph:
        return float(vph)
    
    # 使用 period (间隔秒数)
    period = flow_elem.get('period')
    if period:
        return 3600.0 / float(period)  # 转换为 veh/h
    
    # 使用 number + 时间窗
    number = flow_elem.get('number')
    if number:
        begin = float(flow_elem.get('begin', 0))
        end = float(flow_elem.get('end', 3600))
        duration_h = (end - begin) / 3600.0
        return float(number) / duration_h if duration_h > 0 else float(number)
    
    # 使用 probability
    prob = flow_elem.get('probability')
    if prob:
        return float(prob) * 3600.0  # 假设每秒一次决策
    
    return 0.0


def set_flow_demand(flow_elem: ET.Element, new_demand: float):
    """设置 flow 的需求量，保持原有属性类型"""
    if 'vehsPerHour' in flow_elem.attrib:
        flow_elem.set('vehsPerHour', f'{new_demand:.2f}')
    elif 'period' in flow_elem.attrib:
        new_period = 3600.0 / new_demand if new_demand > 0 else 3600.0
        flow_elem.set('period', f'{new_period:.4f}')
    elif 'number' in flow_elem.attrib:
        begin = float(flow_elem.get('begin', 0))
        end = float(flow_elem.get('end', 3600))
        duration_h = (end - begin) / 3600.0
        new_number = int(new_demand * duration_h)
        flow_elem.set('number', str(max(1, new_number)))
    elif 'probability' in flow_elem.attrib:
        new_prob = new_demand / 3600.0
        flow_elem.set('probability', f'{new_prob:.6f}')


def scale_corridor_flows(
    input_rou: str,
    core_edges_path: str,
    output_rou: str,
    alpha: float = 1.5,
    min_intersection: int = 1,
    verbose: bool = True
) -> Dict:
    """
    主函数：对 core flows 进行加权，保持总需求守恒
    
    Args:
        input_rou: 输入路由文件
        core_edges_path: core edges 文件
        output_rou: 输出路由文件
        alpha: 加权因子 (core flows 的权重)
        min_intersection: 最小交集边数，判定为 core flow
        verbose: 是否输出详细信息
    
    Returns:
        统计字典
    """
    # 加载 core edges
    core_edges = load_core_edges(core_edges_path)
    
    # 解析 XML
    tree = ET.parse(input_rou)
    root = tree.getroot()
    
    # 收集所有 flow 元素及其需求
    flows = root.findall('.//flow')
    if verbose:
        print(f"[INFO] 发现 {len(flows)} 个 flow 元素")
    
    # 分类 flows
    core_flows = []  # (elem, demand, intersection)
    noncore_flows = []  # (elem, demand)
    
    for flow in flows:
        demand = get_flow_demand(flow)
        is_core, intersection = is_core_flow(flow, root, core_edges, min_intersection)
        
        if is_core:
            core_flows.append((flow, demand, intersection))
        else:
            noncore_flows.append((flow, demand))
    
    if verbose:
        print(f"[INFO] Core flows: {len(core_flows)}, Non-core flows: {len(noncore_flows)}")
    
    # 计算总需求
    total_original = sum(d for _, d, _ in core_flows) + sum(d for _, d in noncore_flows)
    
    # 计算加权后的总需求 (未归一化)
    weighted_sum = sum(d * alpha for _, d, _ in core_flows) + sum(d for _, d in noncore_flows)
    
    # 归一化因子
    beta = total_original / weighted_sum if weighted_sum > 0 else 1.0
    
    if verbose:
        print(f"[INFO] 原始总需求: {total_original:.1f} veh/h")
        print(f"[INFO] 加权总需求 (未归一化): {weighted_sum:.1f} veh/h")
        print(f"[INFO] 归一化因子 β: {beta:.4f}")
    
    # 应用加权
    for flow, demand, _ in core_flows:
        new_demand = demand * alpha * beta
        set_flow_demand(flow, new_demand)
    
    for flow, demand in noncore_flows:
        new_demand = demand * 1.0 * beta
        set_flow_demand(flow, new_demand)
    
    # 验证归一化后的总需求
    new_total = 0
    for flow in flows:
        new_total += get_flow_demand(flow)
    
    if verbose:
        print(f"[INFO] 归一化后总需求: {new_total:.1f} veh/h")
        print(f"[INFO] 总需求变化: {100 * (new_total / total_original - 1):.2f}%")
    
    # 计算 core flow 占比变化
    core_original = sum(d for _, d, _ in core_flows)
    core_new = sum(get_flow_demand(f) for f, _, _ in core_flows)
    
    if verbose:
        print(f"\n[INFO] Core flow 占比:")
        print(f"  原始: {100 * core_original / total_original:.1f}%")
        print(f"  加权后: {100 * core_new / new_total:.1f}%")
    
    # 保存输出
    tree.write(output_rou, encoding='utf-8', xml_declaration=True)
    if verbose:
        print(f"\n[输出] 已保存到 {output_rou}")
    
    return {
        'total_flows': len(flows),
        'core_flows': len(core_flows),
        'noncore_flows': len(noncore_flows),
        'total_original': total_original,
        'total_new': new_total,
        'core_share_original': core_original / total_original if total_original > 0 else 0,
        'core_share_new': core_new / new_total if new_total > 0 else 0,
        'beta': beta
    }


def main():
    parser = argparse.ArgumentParser(description='P13-3: 走廊定向喂流')
    parser.add_argument('input_rou', help='输入路由文件 (background_*.rou.xml)')
    parser.add_argument('--core-edges', '-c', required=True, help='core edges 文件')
    parser.add_argument('--output', '-o', help='输出路由文件 (默认: 自动命名)')
    parser.add_argument('--alpha', '-a', type=float, default=1.5, help='加权因子 (默认: 1.5)')
    parser.add_argument('--min-intersection', '-m', type=int, default=1, 
                        help='最小交集边数 (默认: 1)')
    
    args = parser.parse_args()
    
    # 自动命名输出文件
    if args.output:
        output_rou = args.output
    else:
        input_path = Path(args.input_rou)
        output_rou = str(input_path.parent / f"{input_path.stem}_weighted_a{args.alpha}{input_path.suffix}")
    
    stats = scale_corridor_flows(
        input_rou=args.input_rou,
        core_edges_path=args.core_edges,
        output_rou=output_rou,
        alpha=args.alpha,
        min_intersection=args.min_intersection
    )
    
    print("\n[统计]")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
