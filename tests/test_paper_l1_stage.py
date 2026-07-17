from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from src.calibration.objective import L1UnevaluableError
from src.paper_experiments import l1_stage as l1_stage_module
from src.paper_experiments.l1_stage import (
    AGGREGATE_EVALUATION_COLUMNS,
    EVALUATION_COLUMNS,
    L1PreflightError,
    L1StageError,
    PARAMETER_NAMES,
    preflight_l1_observation_chains,
    run_l1_stage,
)
from src.paper_experiments.simulation import (
    RUN_STATUS_SCHEMA,
    SimulationInfrastructureError,
    SimulationRequest,
    SimulationResult,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_real_links(path: Path, *, complete: bool) -> None:
    rows: list[dict[str, Any]] = []
    for route in ("68X", "960"):
        link_count = 3 if complete else 1
        rows.extend(
            {
                "route": route,
                "bound": "inbound",
                "from_seq": sequence,
                "to_seq": sequence + 1,
                "travel_time_s": 10.0 + sequence,
            }
            for sequence in range(1, link_count + 1)
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _manifest(tmp_path: Path, *, complete_chain: bool = True) -> dict[str, Any]:
    manifest = json.loads(
        (PROJECT_ROOT / "config" / "paper_camera_ready_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    real_links = tmp_path / "development_l1_events.csv"
    _write_real_links(real_links, complete=complete_chain)
    route_stops = tmp_path / "route_stops.csv"
    route_stops.write_text("route,bound,stop_id,seq\n", encoding="utf-8")
    manifest["datasets"].append(
        {
            "id": "development_l1_events",
            "path": real_links.name,
            "sha256": sha256_file(real_links),
            "hash_kind": "file",
            "observation_date": "2025-12-19",
            "timezone": "Asia/Hong_Kong",
            "time_window": {"start": "17:00:00", "end": "18:00:00"},
        }
    )
    for dataset in manifest["datasets"]:
        if dataset.get("id") == "route_stop_distance":
            dataset["path"] = route_stops.name
            dataset["sha256"] = sha256_file(route_stops)
    manifest["outputs"]["run_directory"] = "camera-ready-output"
    return manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


class FakeSimulationExecutor:
    def __init__(self) -> None:
        self.physical_runs: list[str] = []
        self.attempts: list[tuple[str, int]] = []

    def __call__(
        self,
        request: SimulationRequest,
        run_directory: Path,
        *,
        max_attempts: int,
        post_output_validator: Callable[[Path], Any] | None,
        **_: Any,
    ) -> SimulationResult:
        self.physical_runs.append(request.run_id)
        failures: list[str] = []
        for attempt in range(1, max_attempts + 1):
            self.attempts.append((request.run_id, attempt))
            attempt_directory = run_directory / f"attempt-{attempt:02d}"
            attempt_directory.mkdir(parents=True, exist_ok=True)
            (attempt_directory / "stdout.log").write_text("fake stdout", encoding="utf-8")
            (attempt_directory / "stderr.log").write_text("", encoding="utf-8")
            score = sum(float(request.bus_parameters[name]) ** 2 for name in PARAMETER_NAMES)
            stopinfo = attempt_directory / "stopinfo.xml"
            stopinfo.write_text(
                "<stopinfos><stopinfo id='bus.0' busStop='s1' "
                f"started='1' ended='2' score='{score:.17g}'/></stopinfos>",
                encoding="utf-8",
            )
            status = {
                "schema_version": RUN_STATUS_SCHEMA,
                "run_id": request.run_id,
                "status": "running",
                "attempt": attempt,
                "manifest_hash": request.manifest_hash,
                "provenance_hash": request.provenance_hash,
                "simulation_effective_hash": request.simulation_effective_hash,
                "component_hashes": dict(request.component_hashes),
                "produced_artifact_hashes": {},
            }
            _write_json(attempt_directory / "run-status.json", status)
            try:
                if post_output_validator is not None:
                    post_output_validator(stopinfo)
            except Exception as exc:
                failures.append(str(exc))
                status.update({"status": "failed", "error_summary": str(exc)})
                _write_json(attempt_directory / "run-status.json", status)
                continue
            output_hash = sha256_file(stopinfo)
            status.update(
                {
                    "status": "succeeded",
                    "duration_s": 0.01,
                    "produced_artifact_hashes": {"stopinfo.xml": output_hash},
                }
            )
            _write_json(attempt_directory / "run-status.json", status)
            parent_status = {
                **status,
                "stopinfo_relative_path": stopinfo.relative_to(run_directory).as_posix(),
            }
            _write_json(run_directory / "run-status.json", parent_status)
            return SimulationResult(
                run_id=request.run_id,
                run_directory=run_directory,
                stopinfo_path=stopinfo,
                attempt=attempt,
                duration_s=0.01,
                provenance_hash=request.provenance_hash,
                simulation_effective_hash=request.simulation_effective_hash,
                component_hashes=dict(request.component_hashes),
                output_hash=output_hash,
            )
        parent_status = {
            "schema_version": RUN_STATUS_SCHEMA,
            "run_id": request.run_id,
            "status": "failed",
            "attempt": max_attempts,
            "manifest_hash": request.manifest_hash,
            "provenance_hash": request.provenance_hash,
            "simulation_effective_hash": request.simulation_effective_hash,
            "component_hashes": dict(request.component_hashes),
            "produced_artifact_hashes": {},
            "error_summary": "; ".join(failures),
        }
        _write_json(run_directory / "run-status.json", parent_status)
        raise SimulationInfrastructureError("fake candidate failed after deterministic retries")


def _objective(*args: Any, feasible: bool = True, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    stopinfo = Path(args[0])
    score = float(ET.parse(stopinfo).getroot().find("stopinfo").get("score"))
    return {
        "status": "succeeded",
        "score": score if feasible else 2000.0 + score,
        "feasible": feasible,
        "jl1_68x": score,
        "rmse_68x": score / 4.0,
        "mae_68x": score / 5.0,
        "std_abs_68x": score / 10.0,
        "q90_abs_68x": score / 3.0,
        "rmse_960": 100.0 if feasible else 360.0,
        "constraint_violation_s": 0.0 if feasible else 10.0,
        "penalty": 0.0 if feasible else 2000.0 + score,
        "n_errors_68x": 4,
        "n_errors_960": 4,
    }


def _small_run(
    tmp_path: Path,
    executor: FakeSimulationExecutor,
    *,
    objective_function: Callable[..., dict[str, Any]] = _objective,
) -> dict[str, Any]:
    return run_l1_stage(
        tmp_path,
        _manifest(tmp_path),
        {"python": "test", "sumo": "test"},
        "sumo-test",
        1,
        30.0,
        optimization_seeds=[0],
        initial_evaluations=3,
        subsequent_evaluations=2,
        bo_candidate_pool_size=16,
        minimum_valid_seeds=1,
        simulation_executor=executor,
        objective_function=objective_function,
    )


def test_preflight_uses_reconstructed_l1_dataset_and_stops_before_simulation(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, complete_chain=False)
    executor = FakeSimulationExecutor()

    with pytest.raises(
        L1PreflightError,
        match=r"route=68X, bound=inbound, matched_downstream_stops=1",
    ):
        run_l1_stage(
            tmp_path,
            manifest,
            {"python": "test", "sumo": "test"},
            "sumo-test",
            1,
            30.0,
            optimization_seeds=[0],
            initial_evaluations=3,
            subsequent_evaluations=0,
            minimum_valid_seeds=1,
            simulation_executor=executor,
            objective_function=_objective,
        )

    assert executor.physical_runs == []
    assert not (tmp_path / "camera-ready-output").exists()


def test_small_stage_shares_initial_runs_matches_seed_schedule_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = FakeSimulationExecutor()

    first = _small_run(tmp_path, executor)

    assert first["valid_common_seeds"] == [0]
    assert len(executor.physical_runs) == 7
    aggregate = tmp_path / "camera-ready-output" / "l1" / "bo_lhs_evaluations.csv"
    with aggregate.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == AGGREGATE_EVALUATION_COLUMNS
        rows = list(reader)
    assert len(rows) == 10
    assert {row["method"] for row in rows} == {"BO", "LHS"}
    for method in ("BO", "LHS"):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["evaluation_index"]),
        )
        assert [int(row["evaluation_index"]) for row in method_rows] == [1, 2, 3, 4, 5]
        assert [int(row["evaluation_index"]) + 100000 for row in method_rows] == [
            100001,
            100002,
            100003,
            100004,
            100005,
        ]
    bo_initial = [row for row in rows if row["method"] == "BO"][:3]
    lhs_initial = [row for row in rows if row["method"] == "LHS"][:3]
    assert [row["candidate_hash"] for row in bo_initial] == [
        row["candidate_hash"] for row in lhs_initial
    ]
    assert [row["objective"] for row in bo_initial] == [row["objective"] for row in lhs_initial]

    detailed = pd.read_csv(
        tmp_path / "camera-ready-output" / "l1" / "seed-0" / "evaluations.csv"
    )
    assert tuple(detailed.columns) == EVALUATION_COLUMNS
    assert detailed.groupby("method").size().to_dict() == {"BO": 5, "LHS": 5}
    assert (detailed["sumo_seed"] == 100000 + detailed["evaluation_index"]).all()

    aggregate_before = aggregate.read_bytes()
    selected_path = (
        tmp_path / "camera-ready-output" / "l1" / "seed-0" / "selected.json"
    )
    selected_before = selected_path.read_bytes()

    def candidate_generation_must_not_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("complete checkpoint regenerated a search candidate")

    monkeypatch.setattr(
        l1_stage_module, "shared_initial_lhs", candidate_generation_must_not_run
    )
    monkeypatch.setattr(l1_stage_module, "continued_lhs", candidate_generation_must_not_run)
    monkeypatch.setattr(
        l1_stage_module, "select_bo_candidate", candidate_generation_must_not_run
    )
    resumed_executor = FakeSimulationExecutor()
    second = _small_run(tmp_path, resumed_executor)
    assert second["valid_common_seeds"] == [0]
    assert len(executor.physical_runs) == 7
    assert resumed_executor.physical_runs == []
    assert aggregate.read_bytes() == aggregate_before
    assert selected_path.read_bytes() == selected_before


def test_complete_checkpoint_recovery_rejects_tampered_manifest_hash(
    tmp_path: Path,
) -> None:
    _small_run(tmp_path, FakeSimulationExecutor())
    hash_contract_path = next(
        (
            tmp_path
            / "camera-ready-output"
            / "l1"
            / "seed-0"
            / "runs"
        ).rglob("run-manifest-hashes.json")
    )
    hash_contract = json.loads(hash_contract_path.read_text(encoding="utf-8"))
    hash_contract["manifest_hash"] = "0" * 64
    _write_json(hash_contract_path, hash_contract)
    selected_path = (
        tmp_path / "camera-ready-output" / "l1" / "seed-0" / "selected.json"
    )
    selected_before = selected_path.read_bytes()
    resumed_executor = FakeSimulationExecutor()

    with pytest.raises(
        L1StageError,
        match=r"Completed checkpoint validation failed.*run-manifest hashes differ",
    ):
        _small_run(tmp_path, resumed_executor)

    assert resumed_executor.physical_runs == []
    assert selected_path.read_bytes() == selected_before


def test_seed_without_feasible_shared_candidate_is_not_replaced(tmp_path: Path) -> None:
    executor = FakeSimulationExecutor()

    def infeasible_objective(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _objective(*args, feasible=False, **kwargs)

    with pytest.raises(L1StageError, match="0 valid common seeds"):
        run_l1_stage(
            tmp_path,
            _manifest(tmp_path),
            {"python": "test", "sumo": "test"},
            "sumo-test",
            1,
            30.0,
            optimization_seeds=[0],
            initial_evaluations=3,
            subsequent_evaluations=2,
            bo_candidate_pool_size=16,
            minimum_valid_seeds=1,
            simulation_executor=executor,
            objective_function=infeasible_objective,
        )

    assert len(executor.physical_runs) == 3
    selected = json.loads(
        (
            tmp_path
            / "camera-ready-output"
            / "l1"
            / "seed-0"
            / "selected.json"
        ).read_text(encoding="utf-8")
    )
    assert selected["status"] == "failed"
    assert selected["successful_evaluations"] == {"BO": 3, "LHS": 3}


def test_preflight_function_reports_all_complete_selected_routes(tmp_path: Path) -> None:
    result = preflight_l1_observation_chains(tmp_path, _manifest(tmp_path))
    assert result["status"] == "succeeded"
    assert {(row["route"], row["matched_downstream_stops"]) for row in result["routes"]} == {
        ("68X", 3),
        ("960", 3),
    }


def test_target_reach_ignores_constraint_penalties() -> None:
    def row(index: int, objective: float, feasible: bool) -> dict[str, Any]:
        return {
            "evaluation_index": index,
            "objective": objective,
            "feasible": feasible,
            "candidate_hash": f"candidate-{index}",
            **{name: float(index) for name in PARAMETER_NAMES},
            "run_id": f"run-{index}",
            "run_directory": f"runs/{index}",
            "manifest_hash": "d" * 64,
            "provenance_hash": "a" * 64,
            "simulation_effective_hash": "b" * 64,
            "component_hashes_json": "{}",
            "output_hash": "c" * 64,
            "stdout_log_path": f"runs/{index}/stdout.log",
            "stderr_log_path": f"runs/{index}/stderr.log",
            "jl1_68x": objective,
            "rmse_68x": objective,
            "mae_68x": objective,
            "std_abs_68x": 0.0,
            "q90_abs_68x": objective,
            "rmse_960": 300.0 if feasible else 400.0,
        }

    summary, selected = l1_stage_module._method_summary(
        0,
        "BO",
        [row(1, 80.0, False), row(2, 120.0, True), row(3, 90.0, True)],
        100.0,
    )

    assert summary["evaluations_to_target"] == 3
    assert summary["final_best_objective"] == 90.0
    assert summary["selected_objective"] == 90.0
    assert selected["evaluation_index"] == 3
