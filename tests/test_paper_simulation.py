from __future__ import annotations

import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

import pytest

from src.paper_experiments import simulation as simulation_module
from src.paper_experiments.simulation import (
    SimulationInfrastructureError,
    SimulationRequest,
    SimulatorInputs,
    build_background_route,
    build_bus_route,
    execute_simulation,
    simulation_effective_hash,
)


def _simulator_inputs(tmp_path: Path) -> SimulatorInputs:
    network = tmp_path / "network.net.xml"
    network.write_text("<net/>", encoding="utf-8")
    bus_routes = tmp_path / "bus.rou.xml"
    bus_routes.write_text(
        "<routes><vType id='kmb_double_decker'/><vehicle id='v'><stop busStop='s1'/></vehicle></routes>",
        encoding="utf-8",
    )
    background_routes = tmp_path / "background.rou.xml"
    background_routes.write_text("<routes><vType id='bg_p5'/></routes>", encoding="utf-8")
    bus_stops = tmp_path / "stops.add.xml"
    bus_stops.write_text("<additional/>", encoding="utf-8")
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"s1": {"weight": 1.0}}), encoding="utf-8")
    return SimulatorInputs(
        network=network,
        bus_routes=bus_routes,
        background_routes=background_routes,
        bus_stops=bus_stops,
        bus_stop_weights=weights,
    )


def _request(
    tmp_path: Path,
    *,
    run_id: str = "paper-run",
    provenance_hash: str = "a" * 64,
    simulation_effective_hash: str = "c" * 64,
    component_hashes: dict[str, str] | None = None,
    inputs: SimulatorInputs | None = None,
) -> SimulationRequest:
    manifest_component_hashes = component_hashes or {
        "bus_parameters": "1" * 64,
        "background_parameters": "2" * 64,
        "observation_semantic": "3" * 64,
        "simulator_inputs": "4" * 64,
    }
    return SimulationRequest(
        run_id=run_id,
        seed=7,
        bus_parameters={
            "t_board": 2.0,
            "t_fixed": 5.0,
            "tau": 1.0,
            "sigma": 0.5,
            "minGap_bus": 2.5,
            "accel": 2.6,
            "decel": 4.5,
        },
        background_parameters={
            "capacityFactor": 1.0,
            "minGap_background": 2.5,
            "impatience": 0.5,
        },
        observation_semantic="moving_only",
        l1_enabled=True,
        l2_enabled=True,
        simulator_inputs=inputs or _simulator_inputs(tmp_path),
        manifest_hash="b" * 64,
        provenance_hash=provenance_hash,
        simulation_effective_hash=simulation_effective_hash,
        component_hashes=manifest_component_hashes,
    )


def _successful_sumo(command: list[str], *, cwd: Path, **_: object) -> subprocess.CompletedProcess[str]:
    (Path(cwd) / "stopinfo.xml").write_text(
        "<stopinfos><stopinfo id='bus.0' busStop='s1' started='1' ended='2'/></stopinfos>",
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def test_build_bus_route_uses_bus_scoped_min_gap_and_dwell(tmp_path: Path) -> None:
    source = tmp_path / "bus.xml"
    source.write_text(
        "<routes><vType id='kmb_double_decker'/><vehicle id='v'><stop busStop='s1'/></vehicle></routes>",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"s1": {"weight": 2.0}}), encoding="utf-8")
    output = tmp_path / "out.xml"
    params = {
        "t_board": 2.0,
        "t_fixed": 5.0,
        "tau": 1.1,
        "sigma": 0.4,
        "minGap_bus": 2.25,
        "accel": 2.6,
        "decel": 4.5,
    }

    build_bus_route(source, output, params, weights)

    root = ET.parse(output).getroot()
    vehicle_type = root.find("vType")
    assert vehicle_type is not None
    assert vehicle_type.get("minGap") == "2.25"
    assert vehicle_type.get("tau") == "1.1"
    assert root.find(".//stop").get("duration") == "65"


def test_build_background_route_changes_only_background_scope(tmp_path: Path) -> None:
    source = tmp_path / "background.xml"
    source.write_text("<routes><vType id='bg_p5'/><vType id='other'/></routes>", encoding="utf-8")
    output = tmp_path / "out.xml"
    build_background_route(
        source,
        output,
        {"capacityFactor": 1.2, "minGap_background": 1.75, "impatience": 0.8},
    )
    root = ET.parse(output).getroot()
    assert root.find("vType[@id='bg_p5']").get("minGap") == "1.75"
    assert root.find("vType[@id='bg_p5']").get("impatience") == "0.8"
    assert root.find("vType[@id='other']").get("minGap") is None


def test_effective_hash_excludes_run_id_and_output_path(tmp_path: Path) -> None:
    inputs = _simulator_inputs(tmp_path)
    common = dict(
        seed=7,
        bus_parameters={"minGap_bus": 2.5},
        background_parameters={"capacityFactor": 1.0, "minGap_background": 2.5, "impatience": 0.5},
        observation_semantic="moving_only",
        l1_enabled=True,
        l2_enabled=True,
        simulator_inputs=inputs,
        manifest_hash="b" * 64,
        simulation_effective_hash="c" * 64,
        component_hashes={
            "bus_parameters": "1" * 64,
            "background_parameters": "2" * 64,
            "observation_semantic": "3" * 64,
            "simulator_inputs": "4" * 64,
        },
    )
    request_a = SimulationRequest(run_id="a", provenance_hash="a" * 64, **common)
    request_b = SimulationRequest(run_id="b", provenance_hash="c" * 64, **common)
    assert simulation_effective_hash(request_a) == simulation_effective_hash(request_b)


@pytest.mark.parametrize(
    "digest",
    ["", "a" * 63, "a" * 65, "z" * 64, None],
)
def test_request_rejects_invalid_provenance_hash(tmp_path: Path, digest: str | None) -> None:
    with pytest.raises(ValueError, match="provenance_hash"):
        _request(tmp_path, provenance_hash=digest)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", ["", "a" * 63, "z" * 64, None])
def test_request_rejects_invalid_simulation_effective_hash(
    tmp_path: Path,
    digest: str | None,
) -> None:
    with pytest.raises(ValueError, match="simulation_effective_hash"):
        _request(tmp_path, simulation_effective_hash=digest)  # type: ignore[arg-type]


def test_request_requires_exact_manifest_component_hashes(tmp_path: Path) -> None:
    missing_key = {
        "bus_parameters": "1" * 64,
        "background_parameters": "2" * 64,
        "observation_semantic": "3" * 64,
    }
    with pytest.raises(ValueError, match="contain exactly"):
        _request(tmp_path, component_hashes=missing_key)

    invalid_digest = {
        "bus_parameters": "1" * 64,
        "background_parameters": "invalid",
        "observation_semantic": "3" * 64,
        "simulator_inputs": "4" * 64,
    }
    with pytest.raises(ValueError, match="invalid SHA-256"):
        _request(tmp_path, component_hashes=invalid_digest)


def test_success_status_persists_precomputed_provenance_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.subprocess, "run", _successful_sumo)
    request = _request(tmp_path, provenance_hash="d" * 64)
    run_directory = tmp_path / "run-output"

    result = execute_simulation(request, run_directory, sumo_binary="sumo-test")

    parent_status = json.loads((run_directory / "run-status.json").read_text(encoding="utf-8"))
    attempt_status = json.loads(
        (run_directory / "attempt-01" / "run-status.json").read_text(encoding="utf-8")
    )
    assert result.provenance_hash == request.provenance_hash
    assert result.simulation_effective_hash == request.simulation_effective_hash
    assert result.component_hashes == request.component_hashes
    assert parent_status["provenance_hash"] == request.provenance_hash
    assert attempt_status["provenance_hash"] == request.provenance_hash
    assert parent_status["simulation_effective_hash"] == result.simulation_effective_hash
    assert parent_status["simulation_effective_hash"] == request.simulation_effective_hash
    assert parent_status["component_hashes"] == request.component_hashes
    assert parent_status["produced_artifact_hashes"]["stopinfo.xml"] == result.output_hash
    assert len({result.provenance_hash, result.simulation_effective_hash, result.output_hash}) == 3


def test_reuse_keeps_the_same_precomputed_provenance_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.subprocess, "run", _successful_sumo)
    request = _request(tmp_path, provenance_hash="e" * 64)
    run_directory = tmp_path / "reuse-output"
    original = execute_simulation(request, run_directory, sumo_binary="sumo-test")

    def unexpected_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("SUMO must not run when the request is safely reusable")

    monkeypatch.setattr(simulation_module.subprocess, "run", unexpected_run)
    reused = execute_simulation(request, run_directory, sumo_binary="sumo-test")
    status = json.loads((run_directory / "run-status.json").read_text(encoding="utf-8"))

    assert reused.reused is True
    assert reused.provenance_hash == request.provenance_hash == original.provenance_hash
    assert reused.simulation_effective_hash == original.simulation_effective_hash
    assert reused.output_hash == original.output_hash
    assert status["provenance_hash"] == request.provenance_hash
    assert status["simulation_effective_hash"] == request.simulation_effective_hash
    assert status["component_hashes"] == request.component_hashes


def test_failed_status_persists_precomputed_provenance_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_sumo(
        command: list[str], *, cwd: Path, **_: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="failed")

    monkeypatch.setattr(simulation_module.subprocess, "run", failed_sumo)
    request = _request(tmp_path, provenance_hash="f" * 64)
    run_directory = tmp_path / "failed-output"

    with pytest.raises(SimulationInfrastructureError, match="failed after 1 attempts"):
        execute_simulation(
            request,
            run_directory,
            sumo_binary="sumo-test",
            max_attempts=1,
        )

    parent_status = json.loads((run_directory / "run-status.json").read_text(encoding="utf-8"))
    attempt_status = json.loads(
        (run_directory / "attempt-01" / "run-status.json").read_text(encoding="utf-8")
    )
    assert parent_status["status"] == "failed"
    assert parent_status["provenance_hash"] == request.provenance_hash
    assert parent_status["simulation_effective_hash"] == request.simulation_effective_hash
    assert parent_status["component_hashes"] == request.component_hashes
    assert attempt_status["provenance_hash"] == request.provenance_hash
    assert attempt_status["simulation_effective_hash"] == request.simulation_effective_hash
    assert attempt_status["component_hashes"] == request.component_hashes


def test_post_output_validator_failure_retries_before_marking_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.subprocess, "run", _successful_sumo)
    request = _request(tmp_path)
    run_directory = tmp_path / "validator-retry"
    calls: list[Path] = []

    def validator(stopinfo_path: Path) -> None:
        calls.append(stopinfo_path)
        attempt_status = json.loads(
            (stopinfo_path.parent / "run-status.json").read_text(encoding="utf-8")
        )
        assert attempt_status["status"] == "running"
        if len(calls) < 3:
            raise RuntimeError(f"invalid candidate output {len(calls)}")

    result = execute_simulation(
        request,
        run_directory,
        sumo_binary="sumo-test",
        max_attempts=3,
        post_output_validator=validator,
    )

    assert result.attempt == 3
    assert [path.parent.name for path in calls] == ["attempt-01", "attempt-02", "attempt-03"]
    assert [
        json.loads(
            (run_directory / f"attempt-{attempt:02d}" / "run-status.json").read_text(
                encoding="utf-8"
            )
        )["status"]
        for attempt in range(1, 4)
    ] == ["failed", "failed", "succeeded"]


def test_post_output_validator_exhaustion_keeps_all_attempts_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.subprocess, "run", _successful_sumo)
    request = _request(tmp_path)
    run_directory = tmp_path / "validator-failed"

    def validator(_: Path) -> None:
        raise ValueError("candidate is unevaluable")

    with pytest.raises(
        SimulationInfrastructureError,
        match="failed after 3 attempts",
    ):
        execute_simulation(
            request,
            run_directory,
            sumo_binary="sumo-test",
            max_attempts=3,
            post_output_validator=validator,
        )

    parent_status = json.loads((run_directory / "run-status.json").read_text(encoding="utf-8"))
    assert parent_status["status"] == "failed"
    assert [
        json.loads(
            (run_directory / f"attempt-{attempt:02d}" / "run-status.json").read_text(
                encoding="utf-8"
            )
        )["status"]
        for attempt in range(1, 4)
    ] == ["failed", "failed", "failed"]
