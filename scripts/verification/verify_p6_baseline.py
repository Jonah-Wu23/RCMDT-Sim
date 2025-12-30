
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import statistics
import sys

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_FILE = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"
BG_ROUTES = PROJECT_ROOT / "sumo/routes/background_cropped.rou.xml"
ADDITIONAL_FILE = PROJECT_ROOT / "sumo/additional/bus_stops_cropped.add.xml"
OUTPUT_DIR = PROJECT_ROOT / "sumo/output/p6_baseline"

def run_simulation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / "summary.xml"
    tripinfo_file = OUTPUT_DIR / "tripinfo.xml"
    stopinfo_file = OUTPUT_DIR / "stopinfo.xml"
    
    cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", f"{FIXED_ROUTES},{BG_ROUTES}",
        "-a", str(ADDITIONAL_FILE),
        "--begin", "0",
        "--end", "3600",
        "--no-step-log", "true",
        "--summary-output", str(summary_file),
        "--tripinfo-output", str(tripinfo_file),
        "--stop-output", str(stopinfo_file),
        "--ignore-route-errors", "true",
        "--time-to-teleport", "300",
        "--device.rerouting.probability", "0" # Explicitly disable rerouting to check pure physical insertion
    ]
    
    print("Running P6 Baseline Simulation (s0.16)...")
    try:
        subprocess.run(cmd, check=True)
        print("Simulation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        return False
    return True

def analyze_results():
    summary_file = OUTPUT_DIR / "summary.xml"
    tripinfo_file = OUTPUT_DIR / "tripinfo.xml"
    
    print("\n--- P6 Baseline Metrics ---")
    
    # 1. Insertion Rate
    try:
        tree = ET.parse(summary_file)
        root = tree.getroot()
        steps = root.findall('step')
        if not steps:
            print("Error: No steps in summary.")
            return
        last = steps[-1]
        inserted = int(last.get('inserted'))
        loaded = int(last.get('loaded'))
        rate = inserted / loaded if loaded > 0 else 0
        print(f"Inserted: {inserted} / {loaded} (Rate: {rate:.2%})")
        
        if rate < 0.90:
            print("❌ FAIL: Insertion Rate < 90%")
        else:
            print("✅ PASS: Insertion Rate >= 90%")
            
    except Exception as e:
        print(f"Error reading summary: {e}")
        
    # 2. Delay Distribution
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        delays = []
        
        for ti in root.findall('tripinfo'):
            if 'bg_p5' in ti.get('vType', ''):
                delays.append(float(ti.get('departDelay')))
                
        if not delays:
            print("Warning: No background vehicles found in tripinfo.")
            return

        median_delay = statistics.median(delays)
        quantiles = statistics.quantiles(delays, n=20)
        p95_delay = quantiles[-1] # 95th percentile
        
        print(f"BG Median Delay: {median_delay:.2f}s")
        print(f"BG P95 Delay:    {p95_delay:.2f}s")
        
        if median_delay < 10 and p95_delay < 60:
             print("✅ PASS: Delays within healthy limits.")
        else:
             print("❌ FAIL: Delays too high.")

    except Exception as e:
        print(f"Error reading tripinfo: {e}")

if __name__ == "__main__":
    if run_simulation():
        analyze_results()
