
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import sumolib
import math
import glob

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_PATH = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
DETECTOR_CSV = PROJECT_ROOT / "data/raw/detector_locations/traffic_speed_volume_occ_info-20251219-170220.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data/raw/detector_locations"
OUTPUT_DIR = PROJECT_ROOT / "data/p5_dfrouter"

def get_net_offset(net_path):
    for event, elem in ET.iterparse(net_path, events=('start',)):
        if elem.tag == 'location':
            offset_str = elem.get('netOffset')
            if offset_str:
                parts = offset_str.split(',')
                return float(parts[0]), float(parts[1])
            break
    return 0.0, 0.0

def get_edge_angle(edge):
    shape = edge.getShape()
    if not shape:
        return 0
    dx = shape[-1][0] - shape[0][0]
    dy = shape[-1][1] - shape[0][1]
    rad = math.atan2(dy, dx)
    deg = math.degrees(rad)
    sumo_deg = (90 - deg) % 360
    return sumo_deg

def prepare_inputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading SUMO network...")
    net = sumolib.net.readNet(str(NET_PATH))
    offset_x, offset_y = get_net_offset(NET_PATH)
    print(f"Network Offset: {offset_x}, {offset_y}")
    
    # --- 1. Map Detectors ---
    print(f"Mapping detectors from {DETECTOR_CSV.name}...")
    df_meta = pd.read_csv(DETECTOR_CSV, encoding='utf-8-sig')
    df_meta.columns = df_meta.columns.str.strip()
    if 'AID_ID_Number' in df_meta.columns[0]:
        df_meta.rename(columns={df_meta.columns[0]: 'AID_ID_Number'}, inplace=True)
        
    bbox = net.getBBoxXY()
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[1]
    
    detector_map = {} # aid_id -> {edge_id, lane_id, pos}
    
    with open(OUTPUT_DIR / "detectors.xml", "w") as f_det:
        f_det.write('<detectors>\n')
        
        for _, row in df_meta.iterrows():
            aid = row['AID_ID_Number']
            orig_x, orig_y = row['Easting'], row['Northing']
            sumo_x = orig_x + offset_x
            sumo_y = orig_y + offset_y
            target_rot = row['Rotation']
            
            if not (min_x <= sumo_x <= max_x and min_y <= sumo_y <= max_y):
                continue
                
            candidates = net.getNeighboringEdges(sumo_x, sumo_y, 50)
            best_match = None
            
            for edge, dist in candidates:
                edge_angle = get_edge_angle(edge)
                diff = abs(edge_angle - target_rot)
                diff = min(diff, 360 - diff)
                
                if diff < 45:
                    if best_match is None or dist < best_match[1]:
                        best_match = (edge, dist)
            
            if best_match:
                edge, dist = best_match
                lanes = edge.getLanes()
                
                detector_map[aid] = {
                    'edge': edge,
                    'lanes': lanes,
                    'pos': 10 
                }

    # --- Debug: Check overlap with Corridor ---
    corridor_edges = set()
    try:
        tree = ET.parse(PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml")
        root = tree.getroot()
        targets = ['flow_68X_inbound', 'flow_960_inbound']
        
        # Check flows AND vehicles
        for elem_type in ['flow', 'vehicle']:
            for elem in root.findall(elem_type):
                if any(t in elem.get('id') for t in targets):
                    rid = elem.get('route')
                    if rid: 
                        pass 
                    r = elem.find('route')
                    if r is not None:
                        edges_str = r.get('edges')
                        if edges_str:
                            corridor_edges.update(edges_str.split())
                         
    except Exception as e:
        print(f"Warning: Could not check corridor overlap: {e}")
        
    on_corridor_count = 0
    edges_with_detectors = set()
    for info in detector_map.values():
        e_obj = info['edge']
        edges_with_detectors.add(e_obj)
        if e_obj.getID() in corridor_edges:
            on_corridor_count += 1
            
    print(f"DEBUG: {on_corridor_count} detectors mapped to edges in the Bus Corridor.")
    print(f"DEBUG: Total detectors mapped: {len(detector_map)}")

    # --- Source Filtering Logic ---
    def has_upstream_detector(start_edge, detector_edge_set, max_depth=5):
        # BFS upstream
        queue = [(start_edge, 0)]
        visited = {start_edge}
        
        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            incoming = curr.getIncoming()
            for inc in incoming.keys():
                if inc in detector_edge_set and inc != start_edge:
                    return True
                
                if inc not in visited:
                    visited.add(inc)
                    queue.append((inc, depth + 1))
        return False

    def has_downstream_detector(start_edge, detector_edge_set, max_depth=5):
        # BFS downstream
        queue = [(start_edge, 0)]
        visited = {start_edge}
        
        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            outgoing = curr.getOutgoing()
            for out in outgoing.keys():
                if out in detector_edge_set and out != start_edge:
                    return True
                
                if out not in visited:
                    visited.add(out)
                    queue.append((out, depth + 1))
        return False

    print("Identifying traffic sources and sinks...")
    source_status = {} # aid -> bool
    sink_status = {}   # aid -> bool
    source_count = 0
    sink_count = 0
    
    for aid, info in detector_map.items():
        is_source = not has_upstream_detector(info['edge'], edges_with_detectors)
        is_sink = not has_downstream_detector(info['edge'], edges_with_detectors)
        
        source_status[aid] = is_source
        sink_status[aid] = is_sink
        
        if is_source: source_count += 1
        if is_sink: sink_count += 1
            
    print(f"Identified {source_count} Sources, {sink_count} Sinks out of {len(detector_map)}.")
    # -------------------------------------------

    # --- 2. Read and Aggregate Data ---
    print("Aggregating traffic data...")
    agg_data = {} 
    
    xml_files = sorted(glob.glob(str(RAW_DATA_DIR / "rawSpeedVol-all-*.xml")))
    print(f"Found {len(xml_files)} XML data files.")
    
    total_time_min = len(xml_files) * 0.5 
    print(f"Total duration approx: {total_time_min} min")
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for det in root.findall('.//detector'):
                aid = det.find('detector_id').text
                if aid not in detector_map:
                    continue
                    
                for lane in det.find('lanes').findall('lane'):
                    if lane.find('valid').text != 'Y':
                        continue
                        
                    lane_type = lane.find('lane_id').text
                    vol = int(lane.find('volume').text)
                    
                    if aid not in agg_data:
                        agg_data[aid] = {}
                    if lane_type not in agg_data[aid]:
                        agg_data[aid][lane_type] = 0
                    
                    agg_data[aid][lane_type] += vol
        except Exception as e:
            print(f"Error reading {Path(xml_file).name}: {e}")

    # --- 3. Write Output ---
    map_count = 0
    
    with open(OUTPUT_DIR / "detectors.xml", "w") as f_det:
        f_det.write('<detectors>\n')
        
        with open(OUTPUT_DIR / "measures.xml", "w") as f_meas:
            f_meas.write('<data>\n')
            f_meas.write(f'    <interval begin="0" end="3600" id="dataset_1700_1800">\n')
            
            for aid, lane_data in agg_data.items():
                info = detector_map[aid]
                edge = info['edge']
                lanes = info['lanes']
                num_sumo_lanes = len(lanes)
                
                # Logic: If Source -> type=source. If Sink (and not Source) -> type=sink. Else -> between.
                # If both? Isolated component. Treat as source (emmiter) but also sink? SUMO only allows one type.
                # 'sink' type in dfrouter implies flow leaves. 'source' implies flow enters.
                # If isolated, we want 'source' so it generates.
                
                is_src = source_status.get(aid, False)
                is_snk = sink_status.get(aid, False)
                
                if is_src:
                    det_type = "source"
                elif is_snk:
                    det_type = "sink"
                else:
                    det_type = "between"

                
                sorted_types = []
                keys = list(lane_data.keys())
                lane_indices = {}
                
                has_slow = any('Slow' in k for k in keys)
                has_fast = any('Fast' in k for k in keys)
                mids = [k for k in keys if 'Middle' in k]
                mids.sort() 
                
                current_idx = 0
                if has_slow:
                    lane_indices[next(k for k in keys if 'Slow' in k)] = 0
                    current_idx += 1
                
                for m in mids:
                    if current_idx < num_sumo_lanes:
                        lane_indices[m] = current_idx
                        current_idx += 1
                
                if has_fast:
                    fast_idx = num_sumo_lanes - 1
                    if fast_idx < current_idx: fast_idx = current_idx
                    lane_indices[next(k for k in keys if 'Fast' in k)] = fast_idx
                
                # Consolidate per lane index
                final_lane_flows = {} # sumolane_idx -> flow
                
                for l_type, total_vol in lane_data.items():
                    if l_type not in lane_indices:
                        continue
                    
                    idx = lane_indices[l_type]
                    if idx >= num_sumo_lanes:
                        idx = num_sumo_lanes - 1
                        
                    if idx not in final_lane_flows:
                        final_lane_flows[idx] = 0
                    final_lane_flows[idx] += total_vol
                
                for idx, total_vol in final_lane_flows.items():
                    lane_id = lanes[idx].getID()
                    det_id = f"{aid}_{idx}"
                    
                    # Define Detector
                    f_det.write(f'    <detectorDefinition id="{det_id}" lane="{lane_id}" pos="10.00" type="{det_type}"/>\n')
                    
                    scale = 60.0 / total_time_min
                    hourly_flow = total_vol * scale
                    f_meas.write(f'        <edgeStats id="{det_id}" qPKW="{hourly_flow:.1f}" vPKW="50" />\n')
                    map_count += 1

            f_meas.write('    </interval>\n')
            f_meas.write('</data>\n')
            
        f_det.write('</detectors>\n')
        
    print(f"Generated data for {len(agg_data)} detectors ({map_count} mapped lanes).")
    print(f"Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_inputs()
