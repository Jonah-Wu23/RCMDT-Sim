
import os
import sys
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
NET_FILE = PROJECT_ROOT / "sumo/net/hk_cropped.net.xml"
FIXED_ROUTES = PROJECT_ROOT / "sumo/routes/fixed_routes_cropped.rou.xml"
BG_ROUTES = PROJECT_ROOT / "sumo/routes/background_corridor.rou.xml"
ADDITIONAL_FILE = PROJECT_ROOT / "sumo/additional/bus_stops_cropped.add.xml"
OUTPUT_DIR = PROJECT_ROOT / "sumo/output/p5_verification"

def run_verification_simulation():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    summary_file = OUTPUT_DIR / "summary.xml"
    tripinfo_file = OUTPUT_DIR / "tripinfo.xml"
    
    cmd = [
        "sumo",
        "-n", str(NET_FILE),
        "-r", f"{FIXED_ROUTES},{BG_ROUTES}",
        "-a", str(ADDITIONAL_FILE),
        "--begin", "0",
        "--end", "900", # Run for 15 minutes to verify loading
        "--no-step-log", "true",
        "--summary-output", str(summary_file),
        "--tripinfo-output", str(tripinfo_file),
        "--ignore-route-errors", "true",
        "--time-to-teleport", "300"
    ]
    
    print("Running SUMO verification...")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("Simulation finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"SUMO execution failed: {e}")
        return

    # Analyze results
    if summary_file.exists():
        check_summary(summary_file)
    else:
        print("Error: summary.xml not found.")
        
    if tripinfo_file.exists():
        check_tripinfo(tripinfo_file)
    else:
        print("Error: tripinfo.xml not found.")

def check_summary(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    steps = root.findall('step')
    if not steps:
        print("Summary: No steps recorded.")
        return
        
    last_step = steps[-1]
    time = last_step.get('time')
    running = last_step.get('running')
    inserted = last_step.get('inserted')
    ended = last_step.get('ended')
    print(f"Summary at time {time}: Inserted={inserted}, Running={running}, Ended={ended}")

def check_tripinfo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    tripinfos = root.findall('tripinfo')
    
    bg_count = 0
    bus_count = 0
    
    for ti in tripinfos:
        vtype = ti.get('vType', '')
        if 'bg_p5' in vtype:
            bg_count += 1
        elif 'bus' in vtype or '68X' in ti.get('id', '') or '960' in ti.get('id', ''):
            bus_count += 1
            
    print(f"Tripinfo Analysis (Partial - only completed trips):")
    print(f"  Background Vehicles Finished: {bg_count}")
    print(f"  Bus Vehicles Finished: {bus_count}")
    print("Note: Many vehicles might still be running at t=900.")

if __name__ == "__main__":
    run_verification_simulation()
