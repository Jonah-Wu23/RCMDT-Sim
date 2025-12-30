#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scale_traffic_lights.py
=======================
P13-4B: 信号容量缩放试验

目标: 缩减 Core Edges 相关信号灯的绿灯时间，以制造拥堵并打破速度平台。

逻辑:
1. 识别控制 Core Edges 的信号灯 (tlLogic)。
2. 遍历这些信号灯的所有相位。
3. 如果相位是绿灯相位 (state 包含 'G' 或 'g')：
    - 缩减时长: duration *= scale
    - 保持周期恒定: 将减少的时长 (diff) 加到下一个相位 (通常是黄灯或红灯)。
4. 保存修改后的网络文件。

输入:
    - net.xml
    - core_edges.txt
    - scale (缩放因子, e.g. 0.85)

输出:
    - net_scaled.net.xml
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

def load_core_edges(core_edges_path: str) -> set:
    """加载 core edges"""
    edges = set()
    if not os.path.exists(core_edges_path):
        print(f"[ERROR] 找不到文件: {core_edges_path}")
        return edges
        
    with open(core_edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            edges.add(line)
    return edges

def find_target_tls(net_path: str, core_edges: set) -> set:
    """
    找到控制 core edges 的信号灯 ID
    (即 core edge 是 connection 的 from edge)
    """
    print(f"[INFO] 正在扫描信号灯关联...")
    target_tls = set()
    
    for event, elem in ET.iterparse(net_path, events=['end']):
        if elem.tag == 'connection':
            from_edge = elem.get('from')
            tl_id = elem.get('tl')
            
            if from_edge in core_edges and tl_id:
                target_tls.add(tl_id)
            elem.clear()
            
    print(f"[INFO] 找到 {len(target_tls)} 个目标信号灯 (控制 Core Edges 的流出)")
    return target_tls

def modify_traffic_lights(net_path: str, output_path: str, target_tls: set, scale: float):
    """
    修改信号灯配时：缩减绿灯，补偿给下一相位
    """
    print(f"[INFO] 开始修改配时 (Scale = {scale})...")
    print(f"[INFO] 策略: 绿灯缩减 -> 加到下一相位 (保持周期不变)")
    
    tree = ET.parse(net_path)
    root = tree.getroot()
    
    modified_count = 0
    total_green_reduction = 0.0
    
    for tl in root.findall('tlLogic'):
        tl_id = tl.get('id')
        if tl_id not in target_tls:
            continue
            
        phases = tl.findall('phase')
        if not phases:
            continue
            
        # 我们不能在迭代中修改列表长度，但这里只修改属性
        # 需要两遍扫描：先计算 diff，再应用
        # 但 diff 要加到 *下一个* 相位。
        
        # 逻辑：
        # iterate i from 0 to N-1
        # if phase[i] is green:
        #   old_dur = duration
        #   new_dur = old_dur * scale
        #   diff = old_dur - new_dur
        #   phase[i].duration = new_dur
        #   phase[i+1].duration += diff  (注意 i+1 可能越界回 0)
        
        # 注意：如果连续两个绿灯，第一个缩减加给第二个，第二个会变长，然后第二个再缩减?
        # 应该基于 *原始* 时长计算缩减量，还是？
        # 简单起见，按序处理。如果 P[i]给 P[i+1]加了时间，P[i+1]如果是绿灯，它也会被缩减。
        # 这样会导致 P[i+1] 实际上包含了来自 P[i] 的 diff，然后再被乘 scale。
        # 这可能导致周期漂移？
        # 验证: 
        # P0(42) -> P1(3). Scale 0.5.
        # P0 -> 21. Diff 21.
        # P1 -> 3 + 21 = 24.
        # Next loop: P1 is Yellow (no 'G'). No scale. P1 remains 24.
        # Total: 21 + 24 = 45. Correct.
        
        # Case 2: P0(G, 10) -> P1(G, 10). Scale 0.5.
        # P0 -> 5. Diff 5. 
        # P1 -> 10 + 5 = 15.
        # Next loop: P1 is Green. 
        # P1 -> 15 * 0.5 = 7.5. Diff 7.5.
        # P2 -> P2 + 7.5.
        # Total: P0(5) + P1(7.5) + P2(...) 
        # Original: 10 + 10 = 20.
        # New: 5 + 7.5 = 12.5. (Where did the rest go? To P2).
        # Correct. Cycle is preserved.
        
        # Implementation details:
        # We need to handle the ring buffer (last phase adds to first phase).
        
        num_phases = len(phases)
        # Store pending addition
        pending_add = 0.0
        
        # To handle the wrap-around correctly without double-processing the first phase if it receives from last,
        # we can just do a single pass. But the last phase adds to first.
        # If we process 0..N-1, phase 0 is modified.
        # Then phase N-1 might add to phase 0.
        # Phase 0's duration has already been scaled. Adding to it is safe (it's just delay).
        # Wait, if Phase 0 is Green, we scale it.
        # If Phase N-1 adds to Phase 0 *after* Phase 0 was scaled, Phase 0 gets longer. 
        # Should that added time also be scaled? No, it's "waste" time from N-1.
        # Ideally, waste time should assume the character of the target phase?
        # If P0 is Green, adding waste to it makes it longer Green. That defeats the purpose!
        # Critical: "Compensate to next phase" works best if next phase is NOT Green (e.g. Yellow/Red).
        # Traffic lights usually alternate Green -> Yellow.
        # So usually P_green -> P_yellow. P_yellow is not Green. So it won't be scaled. It just absorbs time.
        # This is safe.
        
        # What if P_last is Green? It adds to P_0 (Green). 
        # Then P_0 gets longer. P_0 was scaled at step 0. 
        # So P_0 = (Original * scale) + (Diff from P_last).
        # This effectively transfers time from P_last to P_0. 
        # Ideally we want to transfer to Red.
        # But we can't easily identify "Red" generally.
        # Let's stick to "Next Phase".
        
        # Use a list of durations to modify
        import math
        
        # First pass: read durations and is_green
        raw_durations = []
        is_green = []
        for p in phases:
            d = float(p.get('duration'))
            s = p.get('state')
            raw_durations.append(d)
            # 判断是否绿灯: 只要有 'G' or 'g' 就算绿灯相位 (哪怕也是黄灯混合)
            # 更严格: 'G' 数量 > 0
            is_green.append('G' in s or 'g' in s)
            
        new_durations = list(raw_durations)
        
        # Process
        for i in range(num_phases):
            if is_green[i]:
                original = raw_durations[i] # Use original duration base? 
                # Or use current value (which might have inherited from prev)?
                # If we use current value, we cascade.
                # Let's use *current* value in `new_durations` to allow cascading, 
                # BUT we only scale the *original* portion to be safe?
                # Actually, standard logic:
                # current_dur = new_durations[i]
                # scaled_dur = current_dur * scale
                # diff = current_dur - scaled_dur
                # new_durations[i] = scaled_dur
                # new_durations[(i+1)%N] += diff
                
                # Wait, if I do this:
                # P0(G, 10) -> P1(G, 10). Scale 0.5.
                # i=0: P0=10. New=5. Diff=5. P1=10+5=15.
                # i=1: P1=15. New=7.5. Diff=7.5. P2+=7.5.
                # Total Green: 5 + 7.5 = 12.5. (Original 20). Ratio 0.625 != 0.5.
                # This cascading makes ratio unpredictable for consecutive greens.
                
                # Better logic: Only scale the *Original* amount.
                # P0(G, 10) -> P1(G, 10). Scale 0.5.
                # i=0: P0=5. Diff=5. Push 5 to P1_buffer.
                # i=1: P1=5. Diff=5. Push 5 to P2_buffer.
                # Apply buffers.
                # P0 = 5 + (from last). P1 = 5 + 5 = 10.
                # Total Green = 5 + 10 = 15. Still not 10.
                
                # Correct logic for "Reducing Capacity":
                # We want to reduce Green Time. period.
                # If we push diff to next Green, we defeat ourselves.
                # We should push diff to the *nearest Non-Green* phase?
                # Or just push to *Index+1*. Most phases are G->Y.
                # If G->G, accept it.
                
                # Let's assume standard G->Y structure dominates (which we saw in audit).
                # Simple sequential processing is fine.
                
                current_val = new_durations[i]
                # Enhance: Ensure minimal duration (e.g. 5s)
                min_green = 5.0
                target_val = max(min_green, current_val * scale)
                
                # If original was already small, don't scale or scale less
                if current_val < min_green:
                    target_val = current_val
                
                diff = current_val - target_val
                
                new_durations[i] = target_val
                next_idx = (i + 1) % num_phases
                new_durations[next_idx] += diff
                
                total_green_reduction += diff
        
        # Apply back to XML elements
        for i, p in enumerate(phases):
            # Round to sensible precision
            p.set('duration', f"{new_durations[i]:.1f}")
            
        modified_count += 1
        
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"[INFO] 已修改 {modified_count} 个信号灯")
    print(f"[INFO] 总共削减绿灯时间: {total_green_reduction:.1f} 秒 (转移给黄/红灯)")
    print(f"[OUTPUT] {output_path}")


def main():
    parser = argparse.ArgumentParser(description='P13-4B: 信号缩放')
    parser.add_argument('--net', '-n', required=True, help='输入网络文件')
    parser.add_argument('--core-edges', '-c', required=True, help='Core edges 文件')
    parser.add_argument('--scale', '-s', type=float, required=True, help='绿灯缩放因子 (e.g. 0.85)')
    parser.add_argument('--output', '-o', help='输出文件 path')
    
    args = parser.parse_args()
    
    # 自动命名
    if not args.output:
        base = os.path.splitext(args.net)[0]
        args.output = f"{base}_scale{args.scale}.net.xml"
    
    core_edges = load_core_edges(args.core_edges)
    target_tls = find_target_tls(args.net, core_edges)
    
    if not target_tls:
        print("[WARN] 未找到目标信号灯，退出")
        return
        
    modify_traffic_lights(args.net, args.output, target_tls, args.scale)

if __name__ == '__main__':
    main()
