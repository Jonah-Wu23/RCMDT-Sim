
import subprocess
import os
from pathlib import Path

# Constants
PROJECT_ROOT = Path("d:/Documents/Bus Project/Sorce code")
SCRIPT_PATH = PROJECT_ROOT / "scripts/data_processing/generate_p5_corridor_flow.py"
OUTPUT_DIR = PROJECT_ROOT / "sumo/routes"

SCALES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

def generate_variants():
    if not SCRIPT_PATH.exists():
        print(f"Error: {SCRIPT_PATH} not found.")
        return

    for scale in SCALES:
        scale_str = f"{scale:.2f}"
        suffix = f"s{scale_str.replace('.', '')}" # e.g. s005, s010
        output_file = OUTPUT_DIR / f"background_corridor_{suffix}.rou.xml"
        
        cmd = [
            "python", str(SCRIPT_PATH),
            "--bg-scale", str(scale),
            "--output", str(output_file)
        ]
        
        print(f"Generating {output_file.name} (scale={scale})...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate {suffix}: {e}")

if __name__ == "__main__":
    generate_variants()
