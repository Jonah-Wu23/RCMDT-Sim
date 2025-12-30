
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import statistics
import time

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_FILE = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"
ADDITIONAL_FILE = PROJECT_ROOT / "sumo/additional/bus_stops_cropped.add.xml"
OUTPUT_BASE = PROJECT_ROOT / "sumo/output/p7_sensitivity"

SCENARIOS = {
    # Scale Scan
    's0.03': {'bg_scale': '0.03', 'cap': '1.0'},
    's0.05': {'bg_scale': '0.05', 'cap': '1.0'}, # Baseline
    's0.08': {'bg_scale': '0.08', 'cap': '1.0'},
    
    # Cap Scan (Base = s0.05)
    's0.05_c2': {'bg_scale': '0.05', 'cap': '2.0'},
    's0.05_c4': {'bg_scale': '0.05', 'cap': '4.0'},
}

def get_bg_file(scale):
    # Map scale string "0.05" to filename "background_corridor_s005.rou.xml"
    suffix = f"s{scale.replace('.', '')}"
    if len(suffix) == 2: suffix += "0" # Fix s0.1 -> s010 if needed, but here 0.05 -> s005
    return PROJECT_ROOT / f"sumo/routes/background_corridor_{suffix}.rou.xml"

def run_simulation(name, config):
    print(f"[{name}] Running simulation (Scale={config['bg_scale']}, Cap={config['cap']})...")
    
    out_dir = OUTPUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_file = out_dir / "tripinfo.xml"
    
    bg_file = get_bg_file(config['bg_scale'])
    if not bg_file.exists():
        print(f"Error: {bg_file} does not exist.")
        return None

    cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", f"{FIXED_ROUTES},{bg_file}",
        "-a", str(ADDITIONAL_FILE),
        "--begin", "0",
        "--end", "3600",
        "--ignore-route-errors", "true",
        "--no-step-log", "true",
        "--tripinfo-output", str(tripinfo_file),
        "--time-to-teleport", "300",
        "--device.rerouting.probability", "0",
        "--scale", str(config['cap']) # Applying capacity factor via global scale? 
        # Wait, user said "capacityFactor". Assuming this means scaling demand? 
        # Or scaling NETWORK capacity? In SUMO, usually we scale flow. 
        # If capFactor means "More Traffic", then --scale 2.0 works. 
        # If capFactor means "Road Capacity", we'd change sigmas/headways?
        # User Context "B4" context: "Capacity Factor" usually meant scaling flow. 
        # So cap=2 means DOUBLE density.
        # But wait, logic check: if cap 1.0 (s0.05) -> 89% inserted.
        # cap 2.0 (s0.05 * 2) -> This is just s0.10?
        # User separated them: "bg_scale scan use 0.03/0.05/0.08".
        # "capacityFactor scan use 1/2/4".
        # Maybe they mean global --scale argument vs internal bag_scale?
        # Internal bg_scale affects BACKGROUND only.
        # Global --scale affects BUSES too? No, usually we want to stress background.
        # Let's assume capFactor means a multiplier on TOP of bg_scale?
        # Or maybe it means modifying 'speedFactor' or 'minGap'?
        # NO. In previous logs, 'capacityFactor' was rarely used explicitly. 
        # But commonly 'scale' is used.
        # If I use --scale 2.0, it scales EVERYTHING (buses + BG). 
        # Is that what is wanted? Usually sensitivity tests specific inputs.
        # Given "bg_scale scan" changes BG volume.
        # "capFactor scan" might typically refer to Network Capacity (supply side)? 
        # But SUMO doesn't have a simple "capacity" knob.
        # WAIT. In B1/B2 experiments, "scale" was the demand multiplier.
        # "bg_scale" is MY new internal tool.
        # So "capFactor" likely refers to the GLOBAL SUMO scale option, scanning 1.0, 2.0, 4.0.
        # BUT scanning 4.0 when 0.16 (approx 3x 0.05) already crashed?
        # 4.0 * 0.05 = 0.20. It will crash.
        # But I must run it to "restore identifiability" aka show it crashes/increases TT.
        # I will use --scale to implement CapFactor.
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[{name}] Simulation failed")
        return None
        
    return tripinfo_file

def parse_bus_tt(tripinfo_file):
    if not tripinfo_file or not tripinfo_file.exists():
        return {}
    
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    
    # Collect TT for core routes
    # 68X (in/out), 960 (in/out)
    # Assume flow id contains '68X' or '960'
    
    stats = {'68X': [], '960': []}
    
    for ti in root.findall('tripinfo'):
        vid = ti.get('id')
        duration = float(ti.get('duration'))
        
        if '68X' in vid:
            stats['68X'].append(duration)
        elif '960' in vid:
            stats['960'].append(duration)
            
    # Return means
    res = {}
    for k, v in stats.items():
        res[k] = statistics.mean(v) if v else None
    return res

def main():
    results = []
    
    for name, config in SCENARIOS.items():
        # Optimization: use existing files if available?
        # s0.05 (cap1) -> summary_s005.xml exists? tripinfo_s005.xml
        # s0.08 (cap1) -> tripinfo_s008.xml
        # I can skip re-running if I trust them. But file paths differ.
        # I will re-run to be safe and consistent in one folder.
        
        tif = run_simulation(name, config)
        tt_data = parse_bus_tt(tif)
        
        row = {'Scenario': name}
        row.update(tt_data)
        results.append(row)
        
    df = pd.DataFrame(results)
    print("\n=== P7 Sensitivity Results ===")
    print(df.to_string())
    df.to_csv(OUTPUT_BASE / "p7_results.csv", index=False)

if __name__ == "__main__":
    main()
