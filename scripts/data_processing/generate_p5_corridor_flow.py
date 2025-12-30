
import xml.etree.ElementTree as ET
import os
import argparse
from pathlib import Path

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_FILE = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"

DFROUTER_ROUTES = PROJECT_ROOT / "data/p5_dfrouter/routes.rou.xml"
DFROUTER_EMITTERS = PROJECT_ROOT / "data/p5_dfrouter/emitters.rou.xml"
DFROUTER_MEASURES = PROJECT_ROOT / "data/p5_dfrouter/measures.xml"

OUTPUT_FILE = PROJECT_ROOT / "sumo/routes/background_corridor.rou.xml"

# Bus routes to protect/corridor definition
BUS_VEHICLE_IDS = {'flow_68X_outbound', 'flow_68X_inbound', 'flow_960_outbound', 'flow_960_inbound'}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate filtered background corridor flow.")
    parser.add_argument("--bg-scale", type=float, default=1.0, help="Scaling factor for background traffic flow (default: 1.0)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Output route file path")
    return parser.parse_args()

def get_corridor_edges(rou_file):
    """
    Extract all edges used by the bus routes.
    Looking for <vehicle id="..."> -> <route edges="...">
    """
    corridor_edges = set()
    tree = ET.parse(rou_file)
    root = tree.getroot()
    
    for veh in root.findall('vehicle'):
        if any(bus_id in veh.get('id') for bus_id in BUS_VEHICLE_IDS):
            route = veh.find('route')
            if route is not None:
                edges = route.get('edges').split()
                corridor_edges.update(edges)
                
    return corridor_edges

def main():
    args = parse_args()
    bg_scale = args.bg_scale
    output_path = Path(args.output)
    
    if not FIXED_ROUTES.exists():
        print(f"Error: {FIXED_ROUTES} not found.")
        return

    print(f"Generating corridor flow with Scale={bg_scale}...")
    print("Extracting corridor edges...")
    corridor_edges = get_corridor_edges(FIXED_ROUTES)
    print(f"Identified {len(corridor_edges)} corridor edges.")
    
    # Load all routes to check their edges
    print("Loading routes...")
    routes_map = {} # id -> list of edges
    
    # Iterative parsing for large route file
    context = ET.iterparse(DFROUTER_ROUTES, events=('end',))
    for event, elem in context:
        if elem.tag == 'route':
            r_id = elem.get('id')
            edges = elem.get('edges').split()
            routes_map[r_id] = edges
            elem.clear()
            
    print(f"Loaded {len(routes_map)} routes.")

    # Load route distributions
    print("Loading route distributions...")
    route_distributions = {}
    original_dist_weights = {} # Store total weight of each distribution
    
    emitters_file = DFROUTER_EMITTERS
    context = ET.iterparse(emitters_file, events=('end',))
    for event, elem in context:
        if elem.tag == 'routeDistribution':
            dist_id = elem.get('id')
            route_probs = []
            total_weight = 0.0
            for route in elem.findall('route'):
                ref_id = route.get('refId')
                prob = float(route.get('probability'))
                route_probs.append((ref_id, prob))
                total_weight += prob
            
            route_distributions[dist_id] = route_probs
            original_dist_weights[dist_id] = total_weight
            elem.clear()
            
    print(f"Loaded {len(route_distributions)} distributions.")

    # Load flows and filter
    print("Loading flows...")
    measures_tree = ET.parse(DFROUTER_MEASURES)
    measures_root = measures_tree.getroot()
    
    final_flows = []
    output_distributions = {}
    
    count_filtered_flows = 0
    unique_routes = set()
    
    for interval in measures_root.findall('interval'):
        begin = float(interval.get('begin'))
        end = float(interval.get('end'))
        
        for edge_stats in interval.findall('edgeStats'):
            det_id = edge_stats.get('id')
            flow_val = float(edge_stats.get('qPKW', 0))
            
            if flow_val <= 0:
                continue
                
            dist_routes = route_distributions.get(det_id, [])
            if not dist_routes:
                continue
                
            valid_dist_routes = []
            total_prob = 0.0 # Sum of weights of KEPT routes
            
            # Check which routes in distribution intersect with corridor
            for ref, prob in dist_routes:
                r_edges = routes_map.get(ref)
                if not r_edges: continue
                
                # Check intersection
                if any(e in corridor_edges for e in r_edges):
                    valid_dist_routes.append((ref, prob))
                    total_prob += prob
                    unique_routes.add(ref)
            
            if valid_dist_routes:
                # Normalize probabilities for the new filtered distribution
                normalized_routes = []
                for ref, prob in valid_dist_routes:
                    new_prob = prob / total_prob if total_prob > 0 else 0
                    normalized_routes.append((ref, new_prob))
                    
                dist_id = f"dist_{det_id}"
                output_distributions[dist_id] = normalized_routes
                
                # Scale flow by the proportion of valid routes
                # Ratio = (Sum of kept weights) / (Sum of original weights)
                original_total = original_dist_weights.get(det_id, 1.0)
                if original_total == 0: original_total = 1.0
                
                scale_factor = (total_prob / original_total) * bg_scale
                adjusted_flow = flow_val * scale_factor
                
                # Add flow
                flow_def = {
                    'id': f"flow_{det_id}",
                    'route': dist_id,
                    'vehsPerHour': adjusted_flow,
                    'type': 'bg_p5',
                    'begin': begin,
                    'end': end
                }
                final_flows.append(flow_def)
                count_filtered_flows += 1

    print(f"Filtered to {count_filtered_flows} flows and {len(unique_routes)} unique routes.")
    
    # Write Output
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="bg_p5" vClass="passenger" sigma="0.5" speedFactor="normc(1,0.1)" />\n')
        
        # Write only used routes
        for r_id in unique_routes:
            edges_str = " ".join(routes_map[r_id])
            f.write(f'    <route id="{r_id}" edges="{edges_str}"/>\n')
            
        # Write new distributions
        for dist_id, routes in output_distributions.items():
            f.write(f'    <routeDistribution id="{dist_id}">\n')
            for ref, prob in routes:
                f.write(f'        <route refId="{ref}" probability="{prob:.4f}"/>\n')
            f.write('    </routeDistribution>\n')
            
        # Write flows
        for flow in final_flows:
            f.write(f'    <flow id="{flow["id"]}" type="{flow["type"]}" route="{flow["route"]}" begin="{flow["begin"]}" end="{flow["end"]}" vehsPerHour="{flow["vehsPerHour"]:.2f}" departLane="free" departSpeed="max"/>\n')
            
        f.write('</routes>\n')
        
    print(f"Successfully wrote {output_path}")

if __name__ == "__main__":
    main()
