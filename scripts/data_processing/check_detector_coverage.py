
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import sumolib
from shapely.geometry import Point

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_PATH = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
DETECTOR_CSV = PROJECT_ROOT / "data/raw/detector_locations/traffic_speed_volume_occ_info-20251219-170220.csv"
DETECTOR_XML = PROJECT_ROOT / "data/raw/detector_locations/rawSpeedVol-all-20251219-170323.xml"


def get_net_offset(net_path):
    """Parse netOffset from net.xml location element"""
    for event, elem in ET.iterparse(net_path, events=('start',)):
        if elem.tag == 'location':
            offset_str = elem.get('netOffset')
            if offset_str:
                parts = offset_str.split(',')
                return float(parts[0]), float(parts[1])
            break
    return 0.0, 0.0

def check_coverage():
    print("Loading SUMO network...")
    net = sumolib.net.readNet(str(NET_PATH))
    
    # Get Net Offset
    offset_x, offset_y = get_net_offset(NET_PATH)
    print(f"Network Offset: {offset_x}, {offset_y}")
    
    print(f"Reading detector metadata from {DETECTOR_CSV}...")
    df_meta = pd.read_csv(DETECTOR_CSV, encoding='utf-8-sig')
    df_meta.columns = df_meta.columns.str.strip()
    # rename first col if it has weird chars
    if 'AID_ID_Number' in df_meta.columns[0]:
        df_meta.rename(columns={df_meta.columns[0]: 'AID_ID_Number'}, inplace=True)
    
    print(f"Columns found: {df_meta.columns.tolist()}")
    
    # Check bounding box
    bbox = net.getBBoxXY()
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[1]
    print(f"Network BBox (SUMO coords): {bbox}")
    
    inside_count = 0
    mapped_edges = []
    
    for _, row in df_meta.iterrows():
        # Apply Offset
        orig_x = row['Easting']
        orig_y = row['Northing']
        
        sumo_x = orig_x + offset_x
        sumo_y = orig_y + offset_y
        
        # Bounding box check
        if (min_x <= sumo_x <= max_x) and (min_y <= sumo_y <= max_y):
            # Try to map to edge
            edges = net.getNeighboringEdges(sumo_x, sumo_y, 50) # 50m radius
            if edges:
                nearest_edge, dist = edges[0]
                inside_count += 1
                mapped_edges.append({
                    'detector_id': row['AID_ID_Number'],
                    'edge_id': nearest_edge.getID(),
                    'dist': dist,
                    'orig_x': orig_x,
                    'orig_y': orig_y,
                    'sumo_x': sumo_x,
                    'sumo_y': sumo_y
                })
    
    print(f"Total Detectors in CSV: {len(df_meta)}")
    print(f"Detectors inside Cropped Network BBox and mapped to edge: {inside_count}")
    
    if mapped_edges:
        print("\nSample mapped detectors:")
        for m in mapped_edges[:5]:
            print(m)
            
    # Check XML Data availability for these mapped detectors
    print(f"\nChecking Data availability in {DETECTOR_XML.name}...")
    tree = ET.parse(DETECTOR_XML)
    root = tree.getroot()
    
    available_data_count = 0
    mapped_ids = set(m['detector_id'] for m in mapped_edges)
    
    # Also collect valid volume data to see if we have flow counts
    valid_detectors = []
    
    for detector in root.findall('.//detector'):
        det_id = detector.find('detector_id').text
        if det_id in mapped_ids:
            lanes = detector.find('lanes')
            valid_vol = False
            total_vol = 0
            for lane in lanes.findall('lane'):
                if lane.find('valid').text == 'Y':
                    vol = int(lane.find('volume').text)
                    total_vol += vol
                    valid_vol = True
            
            if valid_vol:
                available_data_count += 1
                valid_detectors.append({'id': det_id, 'vol': total_vol})
                
    print(f"Mapped detectors with VALID volume data in sample XML: {available_data_count}")
    if valid_detectors:
         print(f"Sample valid volumes (30s interval): {[v['vol'] for v in valid_detectors[:5]]}")


if __name__ == "__main__":
    check_coverage()
