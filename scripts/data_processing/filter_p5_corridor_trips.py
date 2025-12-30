
import xml.etree.ElementTree as ET
from pathlib import Path
import sumolib

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"
DF_ROUTES = PROJECT_ROOT / "data/p5_dfrouter/routes.rou.xml"
DF_EMITTERS = PROJECT_ROOT / "data/p5_dfrouter/emitters.rou.xml"
OUTPUT_FILE = PROJECT_ROOT / "sumo/routes/background_corridor.rou.xml"

def get_corridor_edges():
    """Extract set of edges used by 68X and 960 inbound routes"""
    edges = set()
    tree = ET.parse(FIXED_ROUTES)
    root = tree.getroot()
    
    targets = ['flow_68X_inbound', 'flow_960_inbound'] # Flow IDs
    # But flows usually ref a route.
    # Let's find the route IDs used by these flows, or check if flows have inline routes.
    
    # Check flows AND vehicles
    for elem_type in ['flow', 'vehicle']:
        for elem in root.findall(elem_type):
            eid = elem.get('id')
            if any(t in eid for t in targets):
                # Check for explicit route attribute
                rid = elem.get('route')
                if rid:
                    route_ids.add(rid)
                
                # Check for inline route child
                r = elem.find('route')
                if r is not None:
                    edges_str = r.get('edges')
                    if edges_str:
                        edges.update(edges_str.split())
                    
    # Check named routes
    for route in root.findall('route'):
        if route.get('id') in route_ids:
            edges.update(route.get('edges').split())
            
    # Also check if user defined explicit lists in python previously?
    # "CORRIDOR_EDGES" in previous scripts. 
    # But reading fresh from file is safer.
    
    print(f"Identified {len(edges)} unique corridor edges.")
    print(f"Sample corridor edges: {list(edges)[:5]}")
    if '105735' in edges:
        print("DEBUG: Edge '105735' successfully found in corridor set.")
    else:
        print("DEBUG: Edge '105735' NOT found in corridor set.")
        
    return edges

def filter_background():
    corridor_edges = get_corridor_edges()
    
    # 1. Load Background Routes
    print("Loading background routes...")
    bg_routes = {} # id -> list of edges
    
    # Iterative parsing for large file
    for event, elem in ET.iterparse(DF_ROUTES, events=('end',)):
        if elem.tag == 'route':
            rid = elem.get('id')
            eds = elem.get('edges').split()
            bg_routes[rid] = eds
            
            # Debug check
            if '105735' in eds:
                 print(f"DEBUG: Found '105735' in background route {rid}")
                 
            elem.clear()
            
    print(f"Loaded {len(bg_routes)} background route definitions.")
    
    # 2. Filter Emitters
    print("Filtering emitters...")
    
    valid_vehicles = []
    used_route_ids = set()
    
    for event, elem in ET.iterparse(DF_EMITTERS, events=('end',)):
        if elem.tag == 'vehicle':
            vid = elem.get('id')
            route_ref = elem.get('route')
            
            # Check overlap
            is_valid = False
            route_edges = []
            
            if route_ref in bg_routes:
                route_edges = bg_routes[route_ref]
            else:
                # Inline route?
                r_elem = elem.find('route')
                if r_elem is not None:
                    route_edges = r_elem.get('edges').split()
            
            # Intersection optimization: check if any edge in route is in corridor
            if any(e in corridor_edges for e in route_edges):
                valid_vehicles.append(elem) # Store element to write back? 
                # storing elem might keep memory high. Store attributes.
                if route_ref:
                    used_route_ids.add(route_ref)
                is_valid = True
            
            if not is_valid:
                elem.clear() # Free mem
                continue
    
    print(f"Filtered {len(valid_vehicles)} vehicles intersecting the corridor.")
    
    # 3. Write Output
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write('<routes>\n')
        
        # Define vTypes? 
        # dfrouter emitters usually have type="start" or similar.
        # We should copy vTypes if they exist in source.
        # DF_EMITTERS usually has vType.
        # Let's read header of emitters to find vTypes.
        
        # Simple: Define a standard background type
        f.write('    <vType id="background" vClass="passenger" sigma="0.5" speedFactor="normc(1,0.1)" />\n')
        
        # Write used routes
        for rid in used_route_ids:
            edges_str = " ".join(bg_routes[rid])
            f.write(f'    <route id="{rid}" edges="{edges_str}"/>\n')
            
        # Write vehicles
        for veh_elem in valid_vehicles:
            # Modify to use our vType or keep original?
            # dfrouter uses 'default'. 
            # Force 'background'
            veh_elem.set('type', 'background')
            
            # Convert to string
            # Removing children if they are just routes we referenced? 
            # If inline route, we need to keep it.
            # But we only gathered used_route_ids from refs.
            # If valid vehicles had inline intersection, we missed `used_route_ids` add step?
            # Re-check logic above: 
            # if `route_ref` exists, we added to set.
            # if inline, we write the vehicle which HAS the inline route child.
            
            xml_str = ET.tostring(veh_elem, encoding='unicode')
            f.write(f'    {xml_str}')
            
        f.write('</routes>\n')

if __name__ == "__main__":
    filter_background()
