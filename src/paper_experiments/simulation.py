"""SUMO execution utilities for the camera-ready experiment protocol.

Every simulation is isolated below its own run directory. Failed attempts are
kept in attempt-specific directories and are never used as successful output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping
from xml.etree import ElementTree as ET


RUN_STATUS_SCHEMA = "run-status/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMPONENT_HASH_KEYS = {
    "bus_parameters",
    "background_parameters",
    "observation_semantic",
    "simulator_inputs",
}


class SimulationInfrastructureError(RuntimeError):
    """Raised when SUMO cannot produce a valid simulation artifact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SimulatorInputs:
    network: Path
    bus_routes: Path
    background_routes: Path
    bus_stops: Path
    bus_stop_weights: Path


@dataclass(frozen=True)
class SimulationRequest:
    run_id: str
    seed: int
    bus_parameters: Mapping[str, float]
    background_parameters: Mapping[str, float]
    observation_semantic: str
    l1_enabled: bool
    l2_enabled: bool
    simulator_inputs: SimulatorInputs
    manifest_hash: str
    provenance_hash: str
    simulation_effective_hash: str
    component_hashes: Mapping[str, str]
    timeout_s: float = 3600.0
    simulation_end_s: int = 3900

    def __post_init__(self) -> None:
        """Require provenance computed by ``contracts.compute_provenance_hash``."""

        if not isinstance(self.manifest_hash, str) or not _SHA256_RE.fullmatch(
            self.manifest_hash
        ):
            raise ValueError("manifest_hash must be a 64-character SHA-256 digest")
        if not isinstance(self.provenance_hash, str) or not _SHA256_RE.fullmatch(
            self.provenance_hash
        ):
            raise ValueError("provenance_hash must be a 64-character SHA-256 digest")
        if not isinstance(self.simulation_effective_hash, str) or not _SHA256_RE.fullmatch(
            self.simulation_effective_hash
        ):
            raise ValueError(
                "simulation_effective_hash must be a 64-character SHA-256 digest"
            )
        if not isinstance(self.component_hashes, Mapping):
            raise ValueError("component_hashes must be a mapping")
        actual_keys = set(self.component_hashes)
        if actual_keys != _COMPONENT_HASH_KEYS:
            raise ValueError(
                "component_hashes must contain exactly: "
                + ", ".join(sorted(_COMPONENT_HASH_KEYS))
            )
        invalid = sorted(
            key
            for key, digest in self.component_hashes.items()
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
        )
        if invalid:
            raise ValueError(f"component_hashes contain invalid SHA-256 values: {invalid}")
        object.__setattr__(self, "bus_parameters", MappingProxyType(dict(self.bus_parameters)))
        object.__setattr__(
            self,
            "background_parameters",
            MappingProxyType(dict(self.background_parameters)),
        )
        object.__setattr__(
            self,
            "component_hashes",
            MappingProxyType(dict(self.component_hashes)),
        )


@dataclass(frozen=True)
class SimulationResult:
    run_id: str
    run_directory: Path
    stopinfo_path: Path
    attempt: int
    duration_s: float
    provenance_hash: str
    simulation_effective_hash: str
    component_hashes: Mapping[str, str]
    output_hash: str
    reused: bool = False


def component_hashes(request: SimulationRequest) -> dict[str, str]:
    """Return component hashes precomputed from the complete per-run manifest."""

    return dict(request.component_hashes)


def simulation_effective_hash(request: SimulationRequest) -> str:
    """Return the effective hash precomputed from the complete per-run manifest."""

    return request.simulation_effective_hash


def build_bus_route(
    source: Path,
    destination: Path,
    bus_parameters: Mapping[str, float],
    stop_weights_path: Path,
) -> Path:
    required = {
        "t_board",
        "t_fixed",
        "tau",
        "sigma",
        "minGap_bus",
        "accel",
        "decel",
    }
    missing = sorted(required.difference(bus_parameters))
    if missing:
        raise ValueError(f"Missing bus parameters: {missing}")

    tree = ET.parse(source)
    root = tree.getroot()
    bus_type = next((node for node in root.iter("vType") if node.get("id") == "kmb_double_decker"), None)
    if bus_type is None:
        raise ValueError(f"kmb_double_decker vType missing from {source}")

    for key in ("tau", "sigma", "accel", "decel"):
        bus_type.set(key, f"{float(bus_parameters[key]):.8g}")
    bus_type.set("minGap", f"{float(bus_parameters['minGap_bus']):.8g}")

    with stop_weights_path.open("r", encoding="utf-8") as stream:
        stop_weights = json.load(stream)
    t_fixed = float(bus_parameters["t_fixed"])
    t_board = float(bus_parameters["t_board"])
    for stop in root.iter("stop"):
        stop_id = stop.get("busStop")
        if not stop_id:
            continue
        weight = float(stop_weights.get(stop_id, {}).get("weight", 1.0))
        stop.set("duration", f"{t_fixed + t_board * 15.0 * weight:.8g}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def build_background_route(
    source: Path,
    destination: Path,
    background_parameters: Mapping[str, float],
) -> Path:
    required = {"capacityFactor", "minGap_background", "impatience"}
    missing = sorted(required.difference(background_parameters))
    if missing:
        raise ValueError(f"Missing background parameters: {missing}")

    tree = ET.parse(source)
    root = tree.getroot()
    background_types = [
        node
        for node in root.iter("vType")
        if node.get("id") in {"bg_p5", "passenger", "car", "background"}
    ]
    if not background_types:
        raise ValueError(f"Background vType missing from {source}")
    for vehicle_type in background_types:
        vehicle_type.set("minGap", f"{float(background_parameters['minGap_background']):.8g}")
        vehicle_type.set("impatience", f"{float(background_parameters['impatience']):.8g}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def build_sumocfg(
    destination: Path,
    network: Path,
    bus_routes: Path,
    background_routes: Path,
    bus_stops: Path,
    stopinfo_path: Path,
    seed: int,
    simulation_end_s: int,
) -> Path:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=str(network.resolve()))
    ET.SubElement(
        inputs,
        "route-files",
        value=f"{bus_routes.resolve()},{background_routes.resolve()}",
    )
    ET.SubElement(inputs, "additional-files", value=str(bus_stops.resolve()))

    times = ET.SubElement(root, "time")
    ET.SubElement(times, "begin", value="0")
    ET.SubElement(times, "end", value=str(int(simulation_end_s)))

    random = ET.SubElement(root, "random")
    ET.SubElement(random, "seed", value=str(int(seed)))

    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="true")
    ET.SubElement(processing, "time-to-teleport", value="300")

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-step-log", value="true")
    ET.SubElement(report, "no-warnings", value="true")

    output = ET.SubElement(root, "output")
    ET.SubElement(output, "stop-output", value=str(stopinfo_path.resolve()))

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def validate_stopinfo(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SimulationInfrastructureError(f"Missing or empty stopinfo output: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise SimulationInfrastructureError(f"Malformed stopinfo output: {path}") from exc
    if not root.findall("stopinfo"):
        raise SimulationInfrastructureError(f"No stopinfo records in output: {path}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_reusable_result(
    run_directory: Path,
    request: SimulationRequest,
    effective_hash: str,
) -> SimulationResult | None:
    status_path = run_directory / "run-status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "succeeded"
        or status.get("run_id") != request.run_id
        or status.get("manifest_hash") != request.manifest_hash
        or status.get("provenance_hash") != request.provenance_hash
        or status.get("simulation_effective_hash") != effective_hash
        or status.get("component_hashes") != dict(request.component_hashes)
    ):
        return None
    stopinfo = run_directory / status["stopinfo_relative_path"]
    validate_stopinfo(stopinfo)
    output_hash = sha256_file(stopinfo)
    if output_hash != status.get("produced_artifact_hashes", {}).get("stopinfo.xml"):
        return None
    return SimulationResult(
        run_id=status["run_id"],
        run_directory=run_directory,
        stopinfo_path=stopinfo,
        attempt=int(status["attempt"]),
        duration_s=float(status["duration_s"]),
        provenance_hash=request.provenance_hash,
        simulation_effective_hash=effective_hash,
        component_hashes=dict(request.component_hashes),
        output_hash=output_hash,
        reused=True,
    )


def execute_simulation(
    request: SimulationRequest,
    run_directory: Path,
    *,
    sumo_binary: str = "sumo",
    max_attempts: int = 3,
    allow_reuse: bool = True,
    post_output_validator: Callable[[Path], Any] | None = None,
) -> SimulationResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    effective_hash = simulation_effective_hash(request)
    components = component_hashes(request)
    if allow_reuse:
        reused = _load_reusable_result(run_directory, request, effective_hash)
        if reused is not None:
            return reused

    run_directory.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempt_dir = run_directory / f"attempt-{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        attempt_started = time.perf_counter()
        status: dict[str, Any] = {
            "schema_version": RUN_STATUS_SCHEMA,
            "run_id": request.run_id,
            "status": "running",
            "attempt": attempt,
            "started_at": started_at,
            "ended_at": None,
            "exit_code": None,
            "error_summary": None,
            "manifest_hash": request.manifest_hash,
            "provenance_hash": request.provenance_hash,
            "simulation_effective_hash": effective_hash,
            "component_hashes": components,
            "produced_artifact_hashes": {},
        }
        _write_json(attempt_dir / "run-status.json", status)

        try:
            bus_route = build_bus_route(
                request.simulator_inputs.bus_routes,
                attempt_dir / "bus_routes.rou.xml",
                request.bus_parameters,
                request.simulator_inputs.bus_stop_weights,
            )
            background_route = build_background_route(
                request.simulator_inputs.background_routes,
                attempt_dir / "background_routes.rou.xml",
                request.background_parameters,
            )
            stopinfo = attempt_dir / "stopinfo.xml"
            sumocfg = build_sumocfg(
                attempt_dir / "simulation.sumocfg",
                request.simulator_inputs.network,
                bus_route,
                background_route,
                request.simulator_inputs.bus_stops,
                stopinfo,
                request.seed,
                request.simulation_end_s,
            )
            command = [
                sumo_binary,
                "-c",
                str(sumocfg),
                "--scale",
                f"{float(request.background_parameters['capacityFactor']):.8g}",
            ]
            _write_json(attempt_dir / "command.json", {"argv": command})
            completed = subprocess.run(
                command,
                cwd=attempt_dir,
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
            )
            (attempt_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
            (attempt_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode != 0:
                raise SimulationInfrastructureError(f"SUMO exited with code {completed.returncode}")
            validate_stopinfo(stopinfo)
            if post_output_validator is not None:
                try:
                    post_output_validator(stopinfo)
                except Exception as exc:
                    raise SimulationInfrastructureError(
                        f"Post-output validation failed: {exc}"
                    ) from exc

            duration_s = time.perf_counter() - attempt_started
            output_hash = sha256_file(stopinfo)
            ended_at = _utc_now()
            status.update(
                {
                    "status": "succeeded",
                    "ended_at": ended_at,
                    "exit_code": completed.returncode,
                    "duration_s": duration_s,
                    "produced_artifact_hashes": {"stopinfo.xml": output_hash},
                }
            )
            _write_json(attempt_dir / "run-status.json", status)
            parent_status = dict(status)
            parent_status["stopinfo_relative_path"] = str(stopinfo.relative_to(run_directory)).replace("\\", "/")
            _write_json(run_directory / "run-status.json", parent_status)
            return SimulationResult(
                run_id=request.run_id,
                run_directory=run_directory,
                stopinfo_path=stopinfo,
                attempt=attempt,
                duration_s=duration_s,
                provenance_hash=request.provenance_hash,
                simulation_effective_hash=effective_hash,
                component_hashes=components,
                output_hash=output_hash,
            )
        except (OSError, subprocess.SubprocessError, SimulationInfrastructureError, ValueError) as exc:
            duration_s = time.perf_counter() - attempt_started
            failures.append(f"attempt {attempt}: {exc}")
            status.update(
                {
                    "status": "failed",
                    "ended_at": _utc_now(),
                    "exit_code": getattr(locals().get("completed", None), "returncode", None),
                    "duration_s": duration_s,
                    "error_summary": str(exc),
                }
            )
            _write_json(attempt_dir / "run-status.json", status)

    parent_status = {
        "schema_version": RUN_STATUS_SCHEMA,
        "run_id": request.run_id,
        "status": "failed",
        "attempt": max_attempts,
        "started_at": None,
        "ended_at": _utc_now(),
        "exit_code": None,
        "error_summary": "; ".join(failures),
        "manifest_hash": request.manifest_hash,
        "provenance_hash": request.provenance_hash,
        "simulation_effective_hash": effective_hash,
        "component_hashes": components,
        "produced_artifact_hashes": {},
    }
    _write_json(run_directory / "run-status.json", parent_status)
    raise SimulationInfrastructureError(
        f"Simulation {request.run_id} failed after {max_attempts} attempts: {'; '.join(failures)}"
    )
