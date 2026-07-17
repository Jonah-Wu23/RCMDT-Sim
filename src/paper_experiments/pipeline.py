"""Shared orchestration helpers for the camera-ready experiment pipeline.

This module owns protocol loading, immutable per-run manifests, environment
capture, and the single-run pilot.  Calibration, evaluation, and reporting
consume these helpers instead of constructing untracked SUMO commands.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    canonical_json,
    canonical_sha256,
    compute_manifest_hashes,
    hash_path,
    validate_paper_manifest,
)
from .simulation import SimulationRequest, SimulationResult, SimulatorInputs, execute_simulation
from .sumo_data import simulation_link_events


BASELINE_BUS_PARAMETERS = {
    "t_board": 2.0,
    "t_fixed": 5.0,
    "tau": 1.0,
    "sigma": 0.5,
    "minGap_bus": 2.5,
    "accel": 2.6,
    "decel": 4.5,
}
BASELINE_BACKGROUND_PARAMETERS = {
    "capacityFactor": 1.0,
    "minGap_background": 2.5,
    "impatience": 0.5,
}
PILOT_SUMO_SEED = 900000
MAIN_SIMULATION_COUNT = 825


class PipelineError(RuntimeError):
    """Raised when orchestration would violate a frozen protocol contract."""


@dataclass(frozen=True)
class ManifestBundle:
    manifest: Mapping[str, Any]
    manifest_hash: str
    provenance_hash: str
    simulation_effective_hash: str
    component_hashes: Mapping[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read JSON contract: {path}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON contract must be an object: {path}")
    return value


def load_protocol_manifest(project_root: Path, relative_path: str = "config/paper_camera_ready_manifest.json") -> dict[str, Any]:
    root = project_root.resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents:
        raise PipelineError(f"Manifest escapes project root: {path}")
    manifest = _read_json(path)
    validate_paper_manifest(manifest)
    return manifest


def _path_within(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents:
        raise PipelineError(f"Path escapes project root: {relative_path}")
    return path


def verify_input_hashes(project_root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify every declared dataset and simulator input against disk."""

    validate_paper_manifest(manifest)
    verified: dict[str, str] = {}
    for dataset in manifest["datasets"]:
        relative = str(dataset["path"])
        actual = hash_path(_path_within(project_root, relative))
        expected = str(dataset["sha256"]).lower()
        if actual.lower() != expected:
            raise PipelineError(f"Dataset hash mismatch for {relative}: expected {expected}, got {actual}")
        verified[relative] = actual

    simulator = manifest["simulator"]
    for input_id, expected_value in simulator["effective_input_hashes"].items():
        descriptor = simulator.get(input_id)
        if not isinstance(descriptor, Mapping) or not descriptor.get("path"):
            raise PipelineError(f"Simulator input {input_id!r} has no path descriptor")
        relative = str(descriptor["path"])
        actual = hash_path(_path_within(project_root, relative))
        expected = str(expected_value).lower()
        if actual.lower() != expected:
            raise PipelineError(f"Simulator input hash mismatch for {relative}: expected {expected}, got {actual}")
        if str(descriptor.get("sha256", "")).lower() != expected:
            raise PipelineError(f"Simulator descriptor/hash table disagreement for {input_id}")
        verified[relative] = actual
    return dict(sorted(verified.items()))


def _command_output(argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except OSError as exc:
        return {"argv": list(argv), "status": "failed", "error": str(exc)}
    return {
        "argv": list(argv),
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }


def software_versions(project_root: Path, *, sumo_binary: str = "sumo") -> dict[str, str]:
    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "pyesmda"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    sumo = _command_output([sumo_binary, "--version"], project_root)
    if sumo["status"] != "succeeded":
        raise PipelineError(f"SUMO version command failed: {sumo}")
    first_line = str(sumo["stdout"]).splitlines()[0] if sumo["stdout"] else "unknown"
    return {
        "python": platform.python_version(),
        "sumo": first_line,
        **packages,
    }


def capture_environment(project_root: Path, *, sumo_binary: str = "sumo") -> dict[str, Any]:
    return {
        "schema_version": "paper-environment/v1",
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "software_versions": software_versions(project_root, sumo_binary=sumo_binary),
        "sumo_version_command": _command_output([sumo_binary, "--version"], project_root),
        "git_head": _command_output(["git", "rev-parse", "HEAD"], project_root),
        "git_status_short": _command_output(["git", "status", "--short"], project_root),
    }


def _validate_parameter_keys(
    values: Mapping[str, float], expected_keys: set[str], label: str
) -> dict[str, float]:
    if set(values) != expected_keys:
        raise PipelineError(
            f"{label} keys differ: expected {sorted(expected_keys)}, got {sorted(values)}"
        )
    result = {key: float(values[key]) for key in sorted(values)}
    if not all(value == value and abs(value) != float("inf") for value in result.values()):
        raise PipelineError(f"{label} contains non-finite values")
    return result


def build_run_manifest(
    base_manifest: Mapping[str, Any],
    *,
    project_root: Path,
    run_directory: Path,
    run_id: str,
    config_id: str,
    method_id: str,
    split: str,
    seed: int,
    sumo_seed: int,
    bus_parameters: Mapping[str, float],
    background_parameters: Mapping[str, float],
    observation_semantic: str,
    l1_enabled: bool,
    l2_enabled: bool,
    software: Mapping[str, str],
    timeout_seconds: float,
    observation_contract: Mapping[str, Any] | None = None,
    parameter_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one complete immutable paper-manifest/v1 for a SUMO run."""

    if split not in {"development", "cross_day"}:
        raise PipelineError(f"Unknown split: {split}")
    if observation_semantic not in {"no_l2_input", "moving_only", "raw_d2d"}:
        raise PipelineError(f"Unknown observation semantic: {observation_semantic}")
    if not isinstance(seed, int) or not isinstance(sumo_seed, int):
        raise PipelineError("seed and sumo_seed must be integers")
    bus = _validate_parameter_keys(bus_parameters, set(BASELINE_BUS_PARAMETERS), "bus_parameters")
    background = _validate_parameter_keys(
        background_parameters, set(BASELINE_BACKGROUND_PARAMETERS), "background_parameters"
    )
    root = project_root.resolve()
    output = run_directory.resolve()
    if root not in output.parents:
        raise PipelineError(f"Run directory escapes project root: {output}")

    manifest = deepcopy(dict(base_manifest))
    background_key = (
        "development_background_routes" if split == "development" else "cross_day_background_routes"
    )
    simulator = manifest["simulator"]
    effective_descriptors = {
        "network": simulator["network"],
        "bus_routes": simulator["bus_routes"],
        "background_routes": simulator[background_key],
        "bus_stops": simulator["bus_stops"],
        "bus_stop_weights": simulator["bus_stop_weights"],
    }
    simulator["effective_input_hashes"] = {
        name: str(descriptor["sha256"]).lower()
        for name, descriptor in effective_descriptors.items()
    }
    simulator["effective_paths"] = {
        name: str(descriptor["path"]) for name, descriptor in effective_descriptors.items()
    }
    simulator["seed"] = sumo_seed
    simulator["timeout_seconds"] = float(timeout_seconds)

    manifest.update(
        {
            "config_id": config_id,
            "method_id": method_id,
            "seed": seed,
            "run_id": run_id,
            "split": split,
            "software_versions": dict(software),
            "mechanisms": {"l1_enabled": l1_enabled, "l2_enabled": l2_enabled},
        }
    )
    manifest["l1"]["bus_parameters"] = bus
    manifest["l1"]["enabled"] = l1_enabled
    manifest["l2"]["background_parameters"] = background
    manifest["l2"]["observation_semantic"] = observation_semantic
    manifest["l2"]["enabled"] = l2_enabled
    if observation_contract is not None:
        manifest["l2"]["observation_contract"] = deepcopy(dict(observation_contract))
    if parameter_sources is not None:
        manifest["parameter_sources"] = deepcopy(dict(parameter_sources))
    manifest["outputs"] = {
        **manifest["outputs"],
        "run_directory": output.relative_to(root).as_posix(),
        "required_artifacts": [
            "run-manifest.json",
            "run-manifest-hashes.json",
            "run-status.json",
            "attempt-*/stopinfo.xml",
        ],
    }
    validate_paper_manifest(manifest)
    return manifest


def bundle_run_manifest(manifest: Mapping[str, Any]) -> ManifestBundle:
    validate_paper_manifest(manifest)
    hashes = compute_manifest_hashes(manifest)
    return ManifestBundle(
        manifest=deepcopy(dict(manifest)),
        manifest_hash=canonical_sha256(manifest),
        provenance_hash=str(hashes["provenance_hash"]),
        simulation_effective_hash=str(hashes["simulation_effective_hash"]),
        component_hashes=dict(hashes["component_hashes"]),
    )


def write_json_immutable(path: Path, value: Any) -> Path:
    """Create a JSON artifact, or verify an existing byte-equivalent contract."""

    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != rendered:
            raise PipelineError(f"Refusing to overwrite non-identical artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def materialize_run_manifest(run_directory: Path, bundle: ManifestBundle) -> None:
    write_json_immutable(run_directory / "run-manifest.json", bundle.manifest)
    write_json_immutable(
        run_directory / "run-manifest-hashes.json",
        {
            "manifest_hash": bundle.manifest_hash,
            "provenance_hash": bundle.provenance_hash,
            "simulation_effective_hash": bundle.simulation_effective_hash,
            "component_hashes": dict(bundle.component_hashes),
        },
    )


def simulation_request_from_bundle(
    project_root: Path, bundle: ManifestBundle
) -> SimulationRequest:
    manifest = bundle.manifest
    paths = manifest["simulator"]["effective_paths"]
    return SimulationRequest(
        run_id=str(manifest["run_id"]),
        seed=int(manifest["simulator"]["seed"]),
        bus_parameters=manifest["l1"]["bus_parameters"],
        background_parameters=manifest["l2"]["background_parameters"],
        observation_semantic=str(manifest["l2"]["observation_semantic"]),
        l1_enabled=bool(manifest["mechanisms"]["l1_enabled"]),
        l2_enabled=bool(manifest["mechanisms"]["l2_enabled"]),
        simulator_inputs=SimulatorInputs(
            network=_path_within(project_root, str(paths["network"])),
            bus_routes=_path_within(project_root, str(paths["bus_routes"])),
            background_routes=_path_within(project_root, str(paths["background_routes"])),
            bus_stops=_path_within(project_root, str(paths["bus_stops"])),
            bus_stop_weights=_path_within(project_root, str(paths["bus_stop_weights"])),
        ),
        manifest_hash=bundle.manifest_hash,
        provenance_hash=bundle.provenance_hash,
        simulation_effective_hash=bundle.simulation_effective_hash,
        component_hashes=bundle.component_hashes,
        timeout_s=float(manifest["simulator"]["timeout_seconds"]),
        simulation_end_s=int(manifest["simulator"]["settings"]["end_s"]),
    )


def workload_estimate(pilot_runtimes_s: Sequence[float], *, workers: int = 1) -> dict[str, Any]:
    if workers < 1:
        raise PipelineError("workers must be positive")
    values = sorted(float(value) for value in pilot_runtimes_s)
    if not values or any(value <= 0 or value != value for value in values):
        raise PipelineError("pilot runtimes must be positive finite values")
    median = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    counts = {
        "l1_shared_bo_lhs": 325,
        "l2_ies": 450,
        "final_ablation": 50,
    }
    serial_seconds = median * MAIN_SIMULATION_COUNT
    return {
        "simulation_counts": {**counts, "total": MAIN_SIMULATION_COUNT},
        "pilot_median_runtime_s": median,
        "serial_runtime_hours": serial_seconds / 3600.0,
        "idealized_runtime_hours_at_workers": serial_seconds / workers / 3600.0,
        "workers": workers,
        "post_pilot_timeout_s": max(1800.0, 3.0 * median),
        "estimate_note": "Parallel estimate is an idealized lower bound; SUMO CPU contention is not modeled.",
    }


def run_pilot(
    project_root: Path,
    *,
    sumo_binary: str = "sumo",
    workers_for_estimate: int = 4,
) -> tuple[SimulationResult, dict[str, Any]]:
    base = load_protocol_manifest(project_root)
    verified = verify_input_hashes(project_root, base)
    software = software_versions(project_root, sumo_binary=sumo_binary)
    output_root = _path_within(project_root, str(base["outputs"]["run_directory"]))
    run_directory = output_root / "pilot" / "A0" / "development" / "seed-0"
    run_id = "pilot-A0-development-seed-0"
    manifest = build_run_manifest(
        base,
        project_root=project_root,
        run_directory=run_directory,
        run_id=run_id,
        config_id="pilot-A0",
        method_id="pilot-zero-shot",
        split="development",
        seed=0,
        sumo_seed=PILOT_SUMO_SEED,
        bus_parameters=BASELINE_BUS_PARAMETERS,
        background_parameters=BASELINE_BACKGROUND_PARAMETERS,
        observation_semantic="no_l2_input",
        l1_enabled=False,
        l2_enabled=False,
        software=software,
        timeout_seconds=float(base["simulator"]["timeout_seconds"]),
        parameter_sources={"bus": "manifest.ablation.baseline_bus_parameters", "background": "manifest.ablation.baseline_background_parameters"},
    )
    bundle = bundle_run_manifest(manifest)
    materialize_run_manifest(run_directory, bundle)
    result = execute_simulation(
        simulation_request_from_bundle(project_root, bundle),
        run_directory,
        sumo_binary=sumo_binary,
        max_attempts=int(base["simulator"]["max_attempts"]),
    )
    events = simulation_link_events(
        result.stopinfo_path,
        _path_within(project_root, "data/processed/kmb_route_stop_dist.csv"),
        window_start_s=0,
        window_end_s=3600,
    )
    if events.empty:
        raise PipelineError("Pilot succeeded but produced no valid event-level link samples")
    estimate = workload_estimate([result.duration_s], workers=workers_for_estimate)
    run_bytes = sum(path.stat().st_size for path in run_directory.rglob("*") if path.is_file())
    summary = {
        "schema_version": "paper-pilot/v1",
        "run_id": run_id,
        "sumo_seed": PILOT_SUMO_SEED,
        "duration_s": result.duration_s,
        "attempt": result.attempt,
        "event_count": int(len(events)),
        "stopinfo_sha256": result.output_hash,
        "run_directory_bytes": run_bytes,
        "projected_storage_gib": run_bytes * MAIN_SIMULATION_COUNT / (1024**3),
        "input_hash_count": len(verified),
        "manifest_hash": bundle.manifest_hash,
        "provenance_hash": bundle.provenance_hash,
        "simulation_effective_hash": bundle.simulation_effective_hash,
        "workload_estimate": estimate,
    }
    write_json_immutable(output_root / "pilot" / "pilot-summary.json", summary)
    write_json_immutable(
        output_root / "pilot" / "environment.json",
        capture_environment(project_root, sumo_binary=sumo_binary),
    )
    return result, summary


__all__ = [
    "BASELINE_BACKGROUND_PARAMETERS",
    "BASELINE_BUS_PARAMETERS",
    "MAIN_SIMULATION_COUNT",
    "ManifestBundle",
    "PILOT_SUMO_SEED",
    "PipelineError",
    "build_run_manifest",
    "bundle_run_manifest",
    "capture_environment",
    "load_protocol_manifest",
    "materialize_run_manifest",
    "run_pilot",
    "simulation_request_from_bundle",
    "software_versions",
    "verify_input_hashes",
    "workload_estimate",
    "write_json_immutable",
]
