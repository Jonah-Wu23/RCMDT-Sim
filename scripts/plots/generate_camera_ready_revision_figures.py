#!/usr/bin/env python
"""Generate the frozen SMC camera-ready figures from validated reporting data."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_experiments.figures import FIGURE_FILES, generate_camera_ready_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate camera-ready Fig.1-Fig.5 and Table I with provenance sidecars."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "camera_ready_revision_20260716",
        help="Frozen camera-ready run directory containing real CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "plots" / "camera_ready_revision_20260716",
        help="Destination for 300-DPI PNGs and artifact-sidecar/v1 JSON files.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURE_FILES),
        help="Generate only selected artifact keys; default generates all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace artifacts with the same camera-ready filenames.",
    )
    args = parser.parse_args()

    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    generated = generate_camera_ready_artifacts(
        args.input_dir,
        args.output_dir,
        script_path=Path(__file__),
        command=command,
        selected=args.only,
        overwrite=args.overwrite,
    )
    for path in generated:
        print(path)
        print(path.with_suffix(path.suffix + ".sidecar.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
