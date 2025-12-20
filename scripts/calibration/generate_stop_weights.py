import xml.etree.ElementTree as ET
import os
import json
import collections

def generate_stop_weights(route_xml_path, output_json_path):
    """
    分析路由文件中的站点频率，为每个站点生成启发式客流权重因子。
    核心逻辑：站点在不同线路中出现的次数越多，代表其客流汇聚能力（Centrality）越强。
    """
    if not os.path.exists(route_xml_path):
        print(f"[ERROR] Route file not found: {route_xml_path}")
        return

    tree = ET.parse(route_xml_path)
    root = tree.getroot()

    # 统计每个 busStop 出现的总次数
    stop_counts = collections.Counter()
    
    # 同时也记录每个站出现在多少个不同的 'vehicle' 或 'flow' 中
    # (在裁剪后的路网中，通常是多条线路共站)
    for stop in root.iter('stop'):
        stop_id = stop.get('busStop')
        if stop_id:
            stop_counts[stop_id] += 1

    if not stop_counts:
        print("[WARN] No bus stops found in the route file.")
        return

    # 计算平均频率作为基准 (Weight = 1.0)
    avg_count = sum(stop_counts.values()) / len(stop_counts)
    
    # 生成权重字典
    # 权重计算公式：W = (当前站频率 / 平均频率)
    # 并进行剪裁，防止出现极端离群值影响校准
    weights = {}
    for stop_id, count in stop_counts.items():
        weight = count / avg_count
        # 归一化限制在 [0.3, 3.0] 之间，保证物理意义合理性
        normalized_weight = round(max(0.3, min(3.0, weight)), 3)
        weights[stop_id] = {
            "weight": normalized_weight,
            "raw_count": count,
            "is_hub": normalized_weight > 1.5
        }

    # 结果保存
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=4, ensure_ascii=False)
    
    print(f"[SUCCESS] Stop weights generated for {len(weights)} stops.")
    print(f"          Output: {os.path.relpath(output_json_path)}")
    print(f"          Avg Frequency: {avg_count:.2f}")

if __name__ == "__main__":
    # 项目根目录定位
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    route_file = os.path.join(PROJECT_ROOT, 'sumo/routes/fixed_routes_cropped.rou.xml')
    output_file = os.path.join(PROJECT_ROOT, 'config/calibration/bus_stop_weights.json')
    
    generate_stop_weights(route_file, output_file)
