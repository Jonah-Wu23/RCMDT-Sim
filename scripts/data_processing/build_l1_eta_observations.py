#!/usr/bin/env python
"""Append declared ETA-derived gaps to the immutable pass-derived L1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_experiments.contracts import canonical_sha256, sha256_file  # noqa: E402
from src.paper_experiments.eta_observations import (  # noqa: E402
    BASE_SOURCE_SEMANTIC,
    ETA_GAP_SOURCE_SEMANTIC,
    HYBRID_LINK_SCHEMA,
    MAX_SPEED_KMH,
    MIN_SPEED_KMH,
    MIN_TRAVEL_TIME_S,
    build_hybrid_l1_observations,
)
from src.paper_experiments.pipeline import load_protocol_manifest  # noqa: E402


SIDECAR_SCHEMA = "l1-hybrid-observation-manifest/v1"


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def _relative(path: Path) -> str:
    return _inside_root(path).relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _dataset(manifest: dict, dataset_id: str) -> dict:
    matches = [item for item in manifest["datasets"] if item.get("id") == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"Manifest must contain exactly one dataset id={dataset_id!r}")
    return matches[0]


def generate(args: argparse.Namespace) -> dict:
    manifest_path = _inside_root(Path(args.manifest))
    manifest = load_protocol_manifest(PROJECT_ROOT, _relative(manifest_path))
    base_descriptor = _dataset(manifest, "development_events")
    eta_descriptor = _dataset(manifest, "development_eta_source")
    distance_descriptor = _dataset(manifest, "route_stop_distance")
    base_path = _inside_root(PROJECT_ROOT / str(base_descriptor["path"]))
    eta_path = _inside_root(PROJECT_ROOT / str(eta_descriptor["path"]))
    distance_path = _inside_root(PROJECT_ROOT / str(distance_descriptor["path"]))
    output_path = _inside_root(Path(args.output))
    sidecar_path = _inside_root(Path(args.sidecar))

    split = manifest["splits"]["development"]
    timezone = ZoneInfo(str(manifest["timezone"]))
    window_start = pd.Timestamp(f"{split['date']} {split['window_start_hkt']}", tz=timezone)
    window_end = pd.Timestamp(f"{split['date']} {split['window_end_hkt']}", tz=timezone)
    selected_routes = [item for item in manifest["routes"] if item.get("l1_selected") is True]
    objective = manifest["l1"]["objective_definition"]
    minimum_stops = int(objective["minimum_joined_downstream_stops"])
    augmentation = manifest["l1"].get("observation_augmentation")
    if not isinstance(augmentation, dict) or augmentation.get("method") != "targeted_eta_gap_fill":
        raise ValueError("Manifest must declare l1.observation_augmentation.targeted_eta_gap_fill")
    gap_links = augmentation.get("gap_links")
    if not isinstance(gap_links, list) or not gap_links:
        raise ValueError("Manifest observation augmentation must declare gap_links")

    result = build_hybrid_l1_observations(
        pd.read_csv(base_path, dtype={"route": str}),
        pd.read_csv(eta_path, dtype={"route": str}),
        pd.read_csv(distance_path, dtype={"route": str}),
        gap_links=gap_links,
        selected_routes=selected_routes,
        window_start=window_start,
        window_end=window_end,
        min_travel_time_s=MIN_TRAVEL_TIME_S,
        min_speed_kmh=MIN_SPEED_KMH,
        max_speed_kmh=MAX_SPEED_KMH,
        minimum_downstream_stops=minimum_stops,
    )
    csv_bytes = result.events.to_csv(
        index=False, lineterminator="\n", float_format="%.6f"
    ).encode("utf-8")
    output_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    _write_immutable(output_path, csv_bytes)

    protocol_subset = {
        "schema_version": manifest["schema_version"],
        "experiment_id": manifest["experiment_id"],
        "timezone": manifest["timezone"],
        "development_split": split,
        "selected_l1_routes": selected_routes,
        "minimum_joined_downstream_stops": minimum_stops,
        "observation_augmentation": augmentation,
    }
    inputs = {
        "paper_protocol": {
            "path": _relative(manifest_path),
            "subset_sha256": canonical_sha256(protocol_subset),
            "subset": protocol_subset,
        },
        "base_pass_derived_events": {
            "path": _relative(base_path),
            "sha256": sha256_file(base_path),
        },
        "station_eta": {"path": _relative(eta_path), "sha256": sha256_file(eta_path)},
        "route_stop_distance": {
            "path": _relative(distance_path),
            "sha256": sha256_file(distance_path),
        },
    }
    scripts = {
        "core": {
            "path": "src/paper_experiments/eta_observations.py",
            "sha256": sha256_file(PROJECT_ROOT / "src/paper_experiments/eta_observations.py"),
        },
        "cli": {
            "path": "scripts/data_processing/build_l1_eta_observations.py",
            "sha256": sha256_file(Path(__file__)),
        },
    }
    rules = {
        "base_source_semantic": BASE_SOURCE_SEMANTIC,
        "supplement_source_semantic": ETA_GAP_SOURCE_SEMANTIC,
        "scope": "append only manifest-declared missing adjacent links",
        "matching": "same capture_ts, route, bound, service_type, and stop-local eta_seq",
        "travel_time": "downstream eta minus upstream eta",
        "capture_departure_arrival_window": "all timestamps inside the half-open development window",
        "scheduled_bus_policy": "exclude when either endpoint rmk_en contains Scheduled Bus",
        "travel_time_operator": ">",
        "minimum_travel_time_s": MIN_TRAVEL_TIME_S,
        "minimum_speed_kmh_inclusive": MIN_SPEED_KMH,
        "maximum_speed_kmh_inclusive": MAX_SPEED_KMH,
        "deduplication": "exact departure_eta/arrival_eta pair within each declared gap",
        "minimum_continuous_downstream_links_from_sequence_1": minimum_stops,
    }
    sidecar = {
        "schema_version": SIDECAR_SCHEMA,
        "output_schema_version": HYBRID_LINK_SCHEMA,
        "inputs": inputs,
        "scripts": scripts,
        "rules": rules,
        "derivation_hash": canonical_sha256({"inputs": inputs, "scripts": scripts, "rules": rules}),
        "diagnostics": dict(result.diagnostics),
        "output": {
            "path": _relative(output_path),
            "sha256": output_sha256,
            "bytes": len(csv_bytes),
            "rows": int(len(result.events)),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    sidecar_bytes = (
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_immutable(sidecar_path, sidecar_bytes)
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "config/paper_camera_ready_manifest.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "data/camera_ready_revision_20260716/observations/l1_hybrid_link_events.csv"
        ),
    )
    parser.add_argument(
        "--sidecar",
        default=str(
            PROJECT_ROOT
            / "data/camera_ready_revision_20260716/observations/l1_hybrid_link_events.manifest.json"
        ),
    )
    args = parser.parse_args()
    result = generate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
