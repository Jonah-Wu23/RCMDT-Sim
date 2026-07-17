from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from config.calibration.l2_protocol_config import L2_CONFIG
from src.paper_experiments.ablation_stage import (
    AblationStageError,
    BLOCKED_RUN_DISPOSITION_SCHEMA,
    FINAL_SUMO_SEED_FORMULA,
    L2_MEMBER_SUMO_SEED_FORMULA,
    _execute,
    run_ablation_stage,
    validate_selected_l1_sources,
)
from src.paper_experiments.contracts import canonical_sha256
from src.paper_experiments.evaluation_stage import _validate_run_sources
from src.paper_experiments.pipeline import (
    BASELINE_BACKGROUND_PARAMETERS,
    BASELINE_BUS_PARAMETERS,
    build_run_manifest,
    bundle_run_manifest,
    materialize_run_manifest,
)
from src.paper_experiments.simulation import SimulationResult, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_manifest(tmp_path: Path) -> dict[str, Any]:
    manifest = json.loads(
        (PROJECT_ROOT / "config" / "paper_camera_ready_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["outputs"]["run_directory"] = "outputs/camera-ready"
    return manifest


def _write_fixture_data(tmp_path: Path) -> None:
    event_path = tmp_path / "data" / "processed" / "link_speeds.csv"
    index_path = tmp_path / "data" / "calibration" / "l2_observation_vector_corridor_M11.csv"
    route_stop_path = tmp_path / "data" / "processed" / "kmb_route_stop_dist.csv"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    event_rows = []
    index_rows = []
    for index in range(1, 12):
        flagged = index <= 6
        event_rows.append(
            {
                "route": "68X",
                "bound": "inbound",
                "from_seq": index,
                "to_seq": index + 1,
                "departure_ts": f"2025-12-19T17:{index:02d}:00+08:00",
                "travel_time_s": 400.0 if flagged else 40.0,
                "dist_m": 400.0,
                "speed_kmh": 3.6 if flagged else 36.0,
            }
        )
        index_rows.append(
            {
                "observation_id": index,
                "route": "68X",
                "bound": "inbound",
                "from_seq": index,
                "to_seq": index + 1,
            }
        )
    pd.DataFrame(event_rows).to_csv(event_path, index=False)
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    pd.DataFrame(
        [
            {
                "route": "68X",
                "bound": "inbound",
                "stop_id": f"S{seq}",
                "seq": seq,
                "cum_dist_m": float((seq - 1) * 400),
            }
            for seq in range(1, 13)
        ]
    ).to_csv(route_stop_path, index=False)


def _write_stopinfo(path: Path) -> None:
    root = ET.Element("stopinfos")
    for seq in range(1, 13):
        started = float((seq - 1) * 20)
        ET.SubElement(
            root,
            "stopinfo",
            id="flow_68X_inbound.0",
            busStop=f"S{seq}",
            arrival=str(started),
            started=str(started),
            ended=str(started + 1.0),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class FakeExecute:
    def __init__(self, fail_when: Callable[[str], bool] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.validator_calls: list[str] = []
        self.fail_when = fail_when or (lambda _: False)
        self._lock = Lock()

    def __call__(
        self,
        request: Any,
        run_directory: Path,
        *,
        sumo_binary: str,
        max_attempts: int,
        allow_reuse: bool = True,
        post_output_validator: Callable[[Path], Any] | None = None,
    ) -> SimulationResult:
        del sumo_binary, max_attempts
        if post_output_validator is None:
            raise AssertionError("ablation execution must provide a post-output validator")
        with self._lock:
            self.calls.append(
                {
                    "run_id": request.run_id,
                    "sumo_seed": request.seed,
                    "bus_parameters": dict(request.bus_parameters),
                    "background_parameters": dict(request.background_parameters),
                    "semantic": request.observation_semantic,
                    "l1_enabled": request.l1_enabled,
                    "l2_enabled": request.l2_enabled,
                }
            )
        if self.fail_when(request.run_id):
            raise RuntimeError(f"injected failure: {request.run_id}")

        status_path = run_directory / "run-status.json"
        if allow_reuse and status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "succeeded":
                stopinfo = run_directory / status["stopinfo_relative_path"]
                return SimulationResult(
                    run_id=request.run_id,
                    run_directory=run_directory,
                    stopinfo_path=stopinfo,
                    attempt=int(status["attempt"]),
                    duration_s=float(status["duration_s"]),
                    provenance_hash=request.provenance_hash,
                    simulation_effective_hash=request.simulation_effective_hash,
                    component_hashes=dict(request.component_hashes),
                    output_hash=sha256_file(stopinfo),
                    reused=True,
                )

        stopinfo = run_directory / "attempt-01" / "stopinfo.xml"
        _write_stopinfo(stopinfo)
        post_output_validator(stopinfo)
        with self._lock:
            self.validator_calls.append(request.run_id)
        output_hash = sha256_file(stopinfo)
        status = {
            "schema_version": "run-status/v1",
            "run_id": request.run_id,
            "status": "succeeded",
            "attempt": 1,
            "duration_s": 0.01,
            "manifest_hash": request.manifest_hash,
            "provenance_hash": request.provenance_hash,
            "simulation_effective_hash": request.simulation_effective_hash,
            "component_hashes": dict(request.component_hashes),
            "produced_artifact_hashes": {"stopinfo.xml": output_hash},
            "stopinfo_relative_path": "attempt-01/stopinfo.xml",
        }
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return SimulationResult(
            run_id=request.run_id,
            run_directory=run_directory,
            stopinfo_path=stopinfo,
            attempt=1,
            duration_s=0.01,
            provenance_hash=request.provenance_hash,
            simulation_effective_hash=request.simulation_effective_hash,
            component_hashes=dict(request.component_hashes),
            output_hash=output_hash,
        )


def _selected_l1() -> dict[int, dict[str, float]]:
    return {
        seed: {**BASELINE_BUS_PARAMETERS, "t_board": 1.25 + seed * 0.01}
        for seed in range(5)
    }


def _l1_candidate_hash(parameters: dict[str, float]) -> str:
    return canonical_sha256(
        {
            "schema_version": "l1-candidate/v1",
            "bus_parameters": {
                name: float(parameters[name]) for name in BASELINE_BUS_PARAMETERS
            },
        }
    )


def _write_l1_sources(
    tmp_path: Path,
    manifest: dict[str, Any],
    selected: dict[int, dict[str, float]],
) -> None:
    output = tmp_path / manifest["outputs"]["run_directory"]
    budget = int(manifest["l1"]["budget"]["successful_evaluations_per_method"])
    for seed, parameters in selected.items():
        seed_directory = output / "l1" / f"seed-{seed}"
        run_directory = seed_directory / "source-run"
        run_id = f"l1-source-seed-{seed}"
        run_manifest = build_run_manifest(
            manifest,
            project_root=tmp_path,
            run_directory=run_directory,
            run_id=run_id,
            config_id="L1-BO",
            method_id="l1_bo",
            split="development",
            seed=seed,
            sumo_seed=100000 + 1000 * seed + 1,
            bus_parameters=parameters,
            background_parameters=BASELINE_BACKGROUND_PARAMETERS,
            observation_semantic="no_l2_input",
            l1_enabled=True,
            l2_enabled=False,
            software={"python": "test", "sumo": "fake"},
            timeout_seconds=10,
        )
        bundle = bundle_run_manifest(run_manifest)
        materialize_run_manifest(run_directory, bundle)
        stopinfo = run_directory / "attempt-01" / "stopinfo.xml"
        _write_stopinfo(stopinfo)
        output_hash = sha256_file(stopinfo)
        status = {
            "schema_version": "run-status/v1",
            "run_id": run_id,
            "status": "succeeded",
            "attempt": 1,
            "duration_s": 0.01,
            "manifest_hash": bundle.manifest_hash,
            "provenance_hash": bundle.provenance_hash,
            "simulation_effective_hash": bundle.simulation_effective_hash,
            "component_hashes": dict(bundle.component_hashes),
            "produced_artifact_hashes": {"stopinfo.xml": output_hash},
            "stopinfo_relative_path": "attempt-01/stopinfo.xml",
        }
        (run_directory / "run-status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
        candidate_hash = _l1_candidate_hash(parameters)
        relative_run = run_directory.relative_to(tmp_path).as_posix()
        relative_stopinfo = stopinfo.relative_to(tmp_path).as_posix()
        relative_status = (run_directory / "run-status.json").relative_to(tmp_path).as_posix()
        selection = {
            "method": "BO",
            "evaluation_index": 1,
            "candidate_hash": candidate_hash,
            "objective": 1.0,
            "feasible": True,
            "parameters": parameters,
            "run_id": run_id,
            "run_directory": relative_run,
            "manifest_hash": bundle.manifest_hash,
            "provenance_hash": bundle.provenance_hash,
            "simulation_effective_hash": bundle.simulation_effective_hash,
            "component_hashes": dict(bundle.component_hashes),
            "output_hash": output_hash,
        }
        seed_directory.mkdir(parents=True, exist_ok=True)
        (seed_directory / "selected.json").write_text(
            json.dumps(
                {
                    "schema_version": "l1-selected/v1",
                    "optimization_seed": seed,
                    "status": "succeeded",
                    "successful_evaluations_per_method": budget,
                    "methods": {"BO": selection},
                    "selected_for_l2": selection,
                }
            ),
            encoding="utf-8",
        )
        rows: list[dict[str, Any]] = []
        for evaluation_index in range(1, budget + 1):
            rows.append(
                {
                    "schema_version": "bo-lhs-evaluations/v1",
                    "optimization_seed": seed,
                    "method": "BO",
                    "evaluation_index": evaluation_index,
                    "candidate_hash": candidate_hash,
                    "objective": float(evaluation_index),
                    "feasible": True,
                    **parameters,
                    "parameters_json": json.dumps(
                        parameters, sort_keys=True, separators=(",", ":")
                    ),
                    "run_id": run_id,
                    "run_directory": relative_run,
                    "stopinfo_path": relative_stopinfo,
                    "run_status_path": relative_status,
                    "manifest_hash": bundle.manifest_hash,
                    "provenance_hash": bundle.provenance_hash,
                    "simulation_effective_hash": bundle.simulation_effective_hash,
                    "component_hashes_json": json.dumps(
                        dict(bundle.component_hashes),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "output_hash": output_hash,
                    "status": "succeeded",
                }
            )
        pd.DataFrame(rows).to_csv(seed_directory / "evaluations.csv", index=False)


def _run(tmp_path: Path, fake: FakeExecute) -> dict[str, Any]:
    _write_fixture_data(tmp_path)
    manifest = _base_manifest(tmp_path)
    _write_l1_sources(tmp_path, manifest, _selected_l1())
    return run_ablation_stage(
        tmp_path,
        selected_l1_by_seed=_selected_l1(),
        workers=3,
        timeout=10,
        base_manifest=manifest,
        software={"python": "test", "sumo": "fake"},
        execute_fn=fake,
        ensemble_size=3,
        iterations=1,
        verify_inputs=False,
    )


def test_stage_builds_observations_l2_results_and_fifty_valid_runs(tmp_path: Path) -> None:
    fake = FakeExecute()
    summary = _run(tmp_path, fake)

    assert summary["status"] == "succeeded"
    assert summary["common_successful_seeds"] == [0, 1, 2, 3, 4]
    assert summary["final_run_count"] == summary["successful_final_run_count"] == 50
    assert summary["l2_member_sumo_seed_formula"] == L2_MEMBER_SUMO_SEED_FORMULA
    assert summary["final_sumo_seed_formula"] == FINAL_SUMO_SEED_FORMULA

    output = tmp_path / "outputs" / "camera-ready"
    raw = pd.read_csv(output / "l2" / "observations" / "raw_d2d.csv")
    moving = pd.read_csv(output / "l2" / "observations" / "moving_only.csv")
    contract = json.loads(
        (output / "l2" / "observations" / "observation-contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert (len(raw), len(moving)) == (11, 5)
    for semantic in ("raw_d2d", "moving_only"):
        descriptor = contract["semantics"][semantic]
        assert all(
            len(descriptor[name]) == 64
            for name in ("schema_hash", "key_hash", "content_hash")
        )

    l2_calls = [call for call in fake.calls if call["run_id"].startswith("l2-")]
    final_calls = [call for call in fake.calls if call["run_id"].startswith("final-")]
    assert len(l2_calls) == 3 * 5 * 3
    assert len(final_calls) == 50
    assert len(fake.validator_calls) == len(l2_calls) + len(final_calls)
    member_pattern = re.compile(
        r"l2-(A[234])-development-seed-(\d+)-iteration-(\d+)-member-(\d+)"
    )
    for call in l2_calls:
        match = member_pattern.fullmatch(call["run_id"])
        assert match is not None
        _, seed, iteration, member = match.groups()
        assert call["sumo_seed"] == 200000 + 10000 * int(seed) + 100 * int(iteration) + int(member)

    for call in final_calls:
        _, config_id, split, _, seed = call["run_id"].split("-")
        split_index = 0 if split == "development" else 1
        assert call["sumo_seed"] == 300000 + 1000 * int(seed) + split_index
        assert call["semantic"] == {
            "A0": "no_l2_input",
            "A1": "no_l2_input",
            "A2": "moving_only",
            "A3": "raw_d2d",
            "A4": "moving_only",
        }[config_id]

    runs = pd.read_csv(output / "ablation" / "ablation_runs.csv")
    assert len(runs) == 50
    assert set(runs["status"]) == {"succeeded"}
    assert runs["simulation_effective_hash"].str.len().eq(64).all()
    assert runs["simulator_inputs_hash"].str.len().eq(64).all()
    for config_id in ("A2", "A3", "A4"):
        for seed in range(5):
            l2_dir = output / "l2" / config_id / f"seed-{seed}"
            for name in (
                "final_parameters.json",
                "iterations.csv",
                "ensemble_parameters.csv",
                "ensemble_simulations.csv",
                "l2-status.json",
            ):
                assert (l2_dir / name).is_file()

    call_count = len(fake.calls)
    recovered = _run(tmp_path, fake)
    new_calls = fake.calls[call_count:]
    assert all(not call["run_id"].startswith("l2-") for call in new_calls)
    assert len(new_calls) == 50
    assert all(item["reused"] for item in recovered["l2_runs"])


def test_one_failed_l2_config_seed_yields_four_seed_partial_release(
    tmp_path: Path,
) -> None:
    failed_l2_run_id = "l2-A3-development-seed-4-iteration-01-member-00"
    fake = FakeExecute(lambda run_id: run_id == failed_l2_run_id)

    summary = _run(tmp_path, fake)

    assert summary["status"] == "partial"
    assert summary["common_successful_seeds"] == [0, 1, 2, 3]
    assert summary["final_run_count"] == 50
    assert summary["successful_final_run_count"] == 40

    final_calls = [call for call in fake.calls if call["run_id"].startswith("final-")]
    assert len(final_calls) == 40
    assert all(not call["run_id"].endswith("seed-4") for call in final_calls)

    output = tmp_path / "outputs" / "camera-ready"
    runs_path = output / "ablation" / "ablation_runs.csv"
    runs = pd.read_csv(runs_path)
    assert len(runs) == 50
    assert (runs["status"] == "succeeded").sum() == 40
    blocked = runs.loc[runs["status"] == "blocked"]
    assert len(blocked) == 10
    assert set(blocked["seed"]) == {4}
    assert set(blocked["config_id"]) == {"A0", "A1", "A2", "A3", "A4"}
    assert set(blocked["split"]) == {"development", "cross_day"}

    for row in blocked.to_dict("records"):
        run_directory = (
            output
            / "ablation"
            / "final"
            / str(row["config_id"])
            / str(row["split"])
            / "seed-4"
        )
        disposition_path = run_directory / "blocked-disposition.json"
        disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
        split_index = 0 if row["split"] == "development" else 1
        assert disposition == {
            "schema_version": BLOCKED_RUN_DISPOSITION_SCHEMA,
            "planned_run_id": row["run_id"],
            "config_id": row["config_id"],
            "seed": 4,
            "split": row["split"],
            "sumo_seed": 304000 + split_index,
            "status": "blocked",
            "reason": "Missing successful L2 results for seed=4: A3",
            "created_artifacts": {},
        }
        assert not (run_directory / "run-status.json").exists()

    manifest = _base_manifest(tmp_path)
    _, common_seeds, run_manifests = _validate_run_sources(
        tmp_path, runs_path, manifest
    )
    assert common_seeds == (0, 1, 2, 3)
    assert len(run_manifests) == 40


def test_stage_records_failures_and_blocks_below_three_common_seeds(tmp_path: Path) -> None:
    def fail_when(run_id: str) -> bool:
        match = re.fullmatch(r"final-A4-(?:development|cross_day)-seed-([0-2])", run_id)
        return match is not None

    fake = FakeExecute(fail_when)
    _write_fixture_data(tmp_path)
    manifest = _base_manifest(tmp_path)
    _write_l1_sources(tmp_path, manifest, _selected_l1())
    with pytest.raises(AblationStageError, match="Fewer than three seeds"):
        run_ablation_stage(
            tmp_path,
            selected_l1_by_seed=_selected_l1(),
            workers=3,
            timeout=10,
            base_manifest=manifest,
            software={"python": "test", "sumo": "fake"},
            execute_fn=fake,
            ensemble_size=3,
            iterations=1,
            verify_inputs=False,
        )

    output = tmp_path / "outputs" / "camera-ready"
    runs = pd.read_csv(output / "ablation" / "ablation_runs.csv")
    summary = json.loads(
        (output / "ablation" / "stage-summary.json").read_text(encoding="utf-8")
    )
    assert len(runs) == 50
    assert (runs["status"] == "failed").sum() == 6
    assert summary["status"] == "blocked"
    assert summary["common_successful_seeds"] == [3, 4]
    failed_status = json.loads(
        (
            output
            / "ablation"
            / "final"
            / "A4"
            / "development"
            / "seed-0"
            / "run-status.json"
        ).read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("candidate_hash", "candidate_hash does not match"),
        ("output_hash", "output_hash differs from its BO evaluation row"),
    ],
)
def test_selected_l1_gate_rejects_tampered_selection_fields(
    tmp_path: Path, field: str, message: str
) -> None:
    manifest = _base_manifest(tmp_path)
    selected = _selected_l1()
    _write_l1_sources(tmp_path, manifest, selected)
    path = (
        tmp_path
        / manifest["outputs"]["run_directory"]
        / "l1"
        / "seed-0"
        / "selected.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_for_l2"][field] = "0" * 64
    payload["methods"]["BO"][field] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AblationStageError, match=message):
        validate_selected_l1_sources(tmp_path, manifest, selected)


def test_selected_l1_gate_rejects_bo_row_and_out_of_range_parameters(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    selected = _selected_l1()
    _write_l1_sources(tmp_path, manifest, selected)
    seed_directory = (
        tmp_path / manifest["outputs"]["run_directory"] / "l1" / "seed-0"
    )
    evaluations_path = seed_directory / "evaluations.csv"
    evaluations = pd.read_csv(evaluations_path)
    evaluations.loc[evaluations["evaluation_index"] == 1, "candidate_hash"] = "f" * 64
    evaluations.to_csv(evaluations_path, index=False)
    with pytest.raises(AblationStageError, match="BO evaluation row differs"):
        validate_selected_l1_sources(tmp_path, manifest, selected)

    _write_l1_sources(tmp_path, manifest, selected)
    payload_path = seed_directory / "selected.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["selected_for_l2"]["parameters"]["t_board"] = 99.0
    payload["methods"]["BO"]["parameters"]["t_board"] = 99.0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    selected[0]["t_board"] = 99.0
    with pytest.raises(AblationStageError, match="outside the frozen range"):
        validate_selected_l1_sources(tmp_path, manifest, selected)


def test_legacy_l2_configs_match_the_camera_ready_manifest() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "config" / "paper_camera_ready_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    priors = json.loads(
        (PROJECT_ROOT / "config" / "calibration" / "l2_priors.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_priors = manifest["l2"]["priors"]
    prior_rows = {row["name"]: row for row in priors["parameters"]}
    for name, expected in manifest_priors.items():
        row = prior_rows[name]
        assert (row["mu"], row["sigma"]) == (expected["mean"], expected["std"])
        assert [row["min"], row["max"]] == expected["bounds"]

    assert priors["canonical_manifest"] == "config/paper_camera_ready_manifest.json"
    assert priors["l1_frozen_source"]["seeds"] == manifest["ablation"]["seeds"]
    assert priors["observation_contract"]["raw_d2d_dimension"] == 11
    assert priors["observation_contract"]["moving_only_dimension"] == 5
    assert priors["ensemble_config"]["ensemble_size"] == manifest["l2"]["ensemble_size"]
    assert priors["ensemble_config"]["max_iterations"] == manifest["l2"]["iterations"]
    assert priors["ensemble_config"]["initial_damping"] == manifest["l2"]["damping"]
    assert priors["ensemble_config"]["adaptive_damping"] is False
    assert priors["ensemble_config"]["localization"] is None

    assert L2_CONFIG.observation.raw_dimension == 11
    assert L2_CONFIG.observation.moving_dimension == 5
    assert L2_CONFIG.ies.optimization_seeds == tuple(manifest["ablation"]["seeds"])
    assert L2_CONFIG.ies.adaptive_damping is False
    assert L2_CONFIG.ies.localization is None
    state = {component["name"]: component for component in L2_CONFIG.state.components}
    for name, expected in manifest_priors.items():
        assert state[name]["prior_mean"] == expected["mean"]
        assert state[name]["prior_std"] == expected["std"]
        assert state[name]["bounds"] == expected["bounds"]


def test_execute_revalidates_bad_reuse_then_forces_normal_attempts(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.xml"
    valid = tmp_path / "valid.xml"
    invalid.write_text("invalid", encoding="utf-8")
    valid.write_text("valid", encoding="utf-8")
    calls: list[bool] = []
    validations: list[str] = []

    def validator(path: Path) -> None:
        validations.append(path.name)
        if path.read_text(encoding="utf-8") != "valid":
            raise ValueError("invalid reused output")

    def executor(
        request: Any,
        run_directory: Path,
        *,
        sumo_binary: str,
        max_attempts: int,
        allow_reuse: bool = True,
        post_output_validator: Callable[[Path], Any],
    ) -> SimulationResult:
        del request, sumo_binary, max_attempts
        calls.append(allow_reuse)
        stopinfo = invalid if allow_reuse else valid
        if not allow_reuse:
            post_output_validator(stopinfo)
        return SimulationResult(
            run_id="reuse-check",
            run_directory=run_directory,
            stopinfo_path=stopinfo,
            attempt=1,
            duration_s=0.01,
            provenance_hash="a" * 64,
            simulation_effective_hash="b" * 64,
            component_hashes={},
            output_hash="c" * 64,
            reused=allow_reuse,
        )

    result = _execute(
        executor,
        object(),
        tmp_path / "run",
        sumo_binary="fake",
        max_attempts=3,
        post_output_validator=validator,
    )

    assert calls == [True, False]
    assert validations == ["invalid.xml", "valid.xml"]
    assert result.reused is False


@pytest.mark.parametrize("workers", [0, 9])
def test_stage_rejects_worker_counts_outside_authorized_limit(tmp_path: Path, workers: int) -> None:
    with pytest.raises(AblationStageError, match="1 through 8"):
        run_ablation_stage(
            tmp_path,
            selected_l1_by_seed=_selected_l1(),
            workers=workers,
            base_manifest=_base_manifest(tmp_path),
            software={"python": "test", "sumo": "fake"},
            execute_fn=FakeExecute(),
            verify_inputs=False,
        )
