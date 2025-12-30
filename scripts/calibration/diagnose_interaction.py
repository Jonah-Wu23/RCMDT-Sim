#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_interaction.py
=======================
诊断压载车与公交交互缺失的原因：
1. 检查公交 route 与压载 route 的边重合情况
2. 检查重合边的车道权限
3. 分析公交 TT 来源（dwell vs travel）
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 压载 route edges
CORRIDOR_EDGES = set("105735 105735_rev 273264_rev 105528 105501 105502 106883 106884_rev 106894_rev 105511_rev 105609_rev 284967_rev 284974 284930_rev 106938 261602 106935_rev 107180_rev 105753_rev 105729_rev 106963 106955 105653 105653_rev 105653 107154 106838 106831_rev 106831 106838_rev 107154_rev 107002 105952_rev 105929 115859 105926 105922 105923 105925 105910 105819 137853 137854 285166 105817 105832 105830 105829 105827 106986_rev 106985_rev 105836_rev 105886_rev 105866_rev 105880_rev 106995 106996 106993 106991 106728 106729 106073 105770_rev 106116_rev 106062_rev 106056_rev 105785_rev 105786_rev 106028 106053_rev 106077 106285_rev 106243_rev 105334_rev 105335 105335_rev 105336_rev 105351 105343 106429 107083 106344 106365 106366_rev 106367 106272_rev 106270_rev 106320 284524 106580 106608 106627 106628 107088 107087 106628 107088 106625 106623 106624 106537_rev 105377 106535 9343 9029 272309 9623_rev 8312_rev 9639_rev 8006_rev 9640_rev 8991_rev 8998 7993 8997 8303_rev 9023_rev 9023 8303 8302 9638 8337 9357 8327 116396 8024 8044 8663_rev 7654_rev 9043_rev 8975_rev 8696_rev 9677_rev".split())


def load_bus_routes():
    """加载公交路线"""
    routes_path = PROJECT_ROOT / 'sumo/routes/fixed_routes_cropped.rou.xml'
    tree = ET.parse(routes_path)
    root = tree.getroot()
    
    bus_routes = {}
    for route in root.findall('.//route'):
        route_id = route.get('id')
        edges = route.get('edges', '').split()
        bus_routes[route_id] = set(edges)
    
    return bus_routes


def check_edge_overlap():
    """检查公交与压载 route 的边重合"""
    print("="*60)
    print("1. 公交与压载 route 边重合分析")
    print("="*60)
    
    bus_routes = load_bus_routes()
    
    for route_id, bus_edges in bus_routes.items():
        overlap = bus_edges & CORRIDOR_EDGES
        overlap_pct = len(overlap) / len(bus_edges) * 100 if bus_edges else 0
        
        print(f"\n{route_id}:")
        print(f"  公交边数: {len(bus_edges)}")
        print(f"  与压载重合: {len(overlap)} ({overlap_pct:.1f}%)")
        
        if overlap:
            print(f"  重合边示例: {list(overlap)[:5]}")


def check_lane_permissions():
    """检查重合边的车道权限"""
    print("\n" + "="*60)
    print("2. 重合边的车道权限分析")
    print("="*60)
    
    net_path = PROJECT_ROOT / 'sumo/net/hk_cropped.net.xml'
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    bus_routes = load_bus_routes()
    all_bus_edges = set()
    for edges in bus_routes.values():
        all_bus_edges.update(edges)
    
    overlap_edges = all_bus_edges & CORRIDOR_EDGES
    print(f"\n总重合边数: {len(overlap_edges)}")
    
    # 检查各重合边的 lane 权限
    bus_only_count = 0
    passenger_allowed_count = 0
    
    for edge_id in list(overlap_edges)[:10]:  # 只检查前 10 条
        edge = root.find(f".//edge[@id='{edge_id}']")
        if edge is not None:
            lanes = edge.findall('lane')
            lane_info = []
            for lane in lanes:
                allow = lane.get('allow', 'all')
                disallow = lane.get('disallow', '')
                if allow == 'bus' or 'bus' in allow:
                    lane_info.append(f"bus-only")
                    bus_only_count += 1
                elif 'passenger' in disallow:
                    lane_info.append(f"no-passenger")
                else:
                    lane_info.append(f"all")
                    passenger_allowed_count += 1
            print(f"  {edge_id}: {len(lanes)} lanes - {lane_info}")
    
    print(f"\n统计（样本）: bus-only={bus_only_count}, passenger-allowed={passenger_allowed_count}")


def analyze_bus_tt_breakdown():
    """分析公交 TT 来源（dwell vs travel）"""
    print("\n" + "="*60)
    print("3. 公交 TT 来源分析（dwell vs travel）")
    print("="*60)
    
    # 比较有/无压载时的 stopinfo
    for label in ['a2_vph0', 'a2_vph300']:
        stopinfo_path = PROJECT_ROOT / f'sumo/output/a1c_validation/{label}/stopinfo.xml'
        if not stopinfo_path.exists():
            print(f"  {label}: stopinfo 不存在")
            continue
        
        tree = ET.parse(stopinfo_path)
        root = tree.getroot()
        
        # 统计公交车的停靠信息
        by_vehicle = {}
        for stop in root.findall('.//stopinfo'):
            veh_id = stop.get('id')
            if '68X' not in veh_id and '960' not in veh_id:
                continue
            
            started = float(stop.get('started', 0))
            ended = float(stop.get('ended', 0))
            arrival = float(stop.get('arrival', 0))
            
            dwell = ended - started  # 停站时间
            
            if veh_id not in by_vehicle:
                by_vehicle[veh_id] = {'total_dwell': 0, 'stops': []}
            by_vehicle[veh_id]['total_dwell'] += dwell
            by_vehicle[veh_id]['stops'].append({
                'arrival': arrival,
                'started': started,
                'ended': ended,
                'dwell': dwell
            })
        
        print(f"\n{label}:")
        for veh_id, info in list(by_vehicle.items())[:3]:
            stops = info['stops']
            if len(stops) >= 2:
                # 计算总行程时间（最后一站 ended - 第一站 arrival）
                first_stop = min(stops, key=lambda x: x['arrival'])
                last_stop = max(stops, key=lambda x: x['ended'])
                total_time = last_stop['ended'] - first_stop['arrival']
                total_dwell = info['total_dwell']
                travel_time = total_time - total_dwell
                
                print(f"  {veh_id}: total={total_time:.0f}s, dwell={total_dwell:.0f}s ({total_dwell/total_time*100:.0f}%), travel={travel_time:.0f}s")


def main():
    check_edge_overlap()
    check_lane_permissions()
    analyze_bus_tt_breakdown()
    
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)


if __name__ == "__main__":
    main()
