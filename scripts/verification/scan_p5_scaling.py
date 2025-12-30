
import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import statistics

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_FILE = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"
ADDITIONAL_FILE = PROJECT_ROOT / "sumo/additional/bus_stops_cropped.add.xml"
OUTPUT_BASE = PROJECT_ROOT / "sumo/output/p5_scan"

SCALES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

def run_simulation(scale):
    scale_str = f"{scale:.2f}"
    suffix = f"s{scale_str.replace('.', '')}"
    route_file = PROJECT_ROOT / f"sumo/routes/background_corridor_{suffix}.rou.xml"
    
    if not route_file.exists():
        print(f"File not found: {route_file}")
        return None

    out_dir = OUTPUT_BASE / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = out_dir / "summary.xml"
    tripinfo_file = out_dir / "tripinfo.xml"
    
    cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", f"{FIXED_ROUTES},{route_file}",
        "-a", str(ADDITIONAL_FILE),
        "--begin", "0",
        "--end", "3600",
        "--no-step-log", "true",
        "--summary-output", str(summary_file),
        "--tripinfo-output", str(tripinfo_file),
        "--ignore-route-errors", "true",
        "--time-to-teleport", "300"
    ]
    
    print(f"[{suffix}] Running simulation...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"[{suffix}] Simulation failed!")
        return None
        
    return parse_results(scale, summary_file, tripinfo_file)

def parse_results(scale, summary_file, tripinfo_file):
    metrics = {'Scale': scale}
    
    # 1. Summary
    if summary_file.exists():
        try:
            tree = ET.parse(summary_file)
            root = tree.getroot()
            steps = root.findall('step')
            if steps:
                last_step = steps[-1]
                metrics['Inserted'] = int(last_step.get('inserted', 0))
                metrics['Loaded'] = int(last_step.get('loaded', 0))
                metrics['InsertionRate'] = metrics['Inserted'] / metrics['Loaded'] if metrics['Loaded'] > 0 else 0
        except Exception as e:
            print(f"Error parsing summary: {e}")

    # 2. Tripinfo
    if tripinfo_file.exists():
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            bg_delays = []
            bus_times = []
            
            for ti in root.findall('tripinfo'):
                vtype = ti.get('vType', '')
                if 'bg_p5' in vtype:
                    bg_delays.append(float(ti.get('departDelay')))
                elif 'bus' in vtype or 'flow' in ti.get('id', ''):
                    # Collect bus duration
                    bus_times.append(float(ti.get('duration')))
            
            if bg_delays:
                metrics['BG_MedianDelay'] = statistics.median(bg_delays)
                metrics['BG_P95Delay'] = statistics.quantiles(bg_delays, n=20)[-1] if len(bg_delays) >= 20 else max(bg_delays)
            else:
                metrics['BG_MedianDelay'] = 0
                metrics['BG_P95Delay'] = 0
                
            if bus_times:
                metrics['Bus_MeanTT'] = statistics.mean(bus_times)
            else:
                metrics['Bus_MeanTT'] = 0
                
        except Exception as e:
            print(f"Error parsing tripinfo: {e}")
            
    return metrics

def main():
    results = []
    print(f"Starting scan for scales: {SCALES}")
    
    for scale in SCALES:
        res = run_simulation(scale)
        if res:
            results.append(res)
            
    # Print results table
    if results:
        df = pd.DataFrame(results)
        print("\n=== Scan Results ===")
        print(df.to_string(index=False))
        
        # Save to csv
        df.to_csv(OUTPUT_BASE / "scan_results.csv", index=False)
        print(f"\nResults saved to {OUTPUT_BASE / 'scan_results.csv'}")

if __name__ == "__main__":
    main()
