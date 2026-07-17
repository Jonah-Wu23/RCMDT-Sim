"""Deterministic L1 BO-versus-LHS calibration stage.

The stage owns the equal-budget search, per-seed checkpoints, immutable run
manifests, and camera-ready CSV artifacts.  SUMO remains isolated in
``simulation.execute_simulation``; an attempt is successful only after the L1
objective validates its stop output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.calibration.objective import (
    L1ObjectiveError,
    L1UnevaluableError,
    build_observed_cumulative_times,
    calculate_l1_candidate_score,
    load_real_link_speeds,
)

from .contracts import canonical_sha256, validate_paper_manifest
from .pipeline import (
    BASELINE_BACKGROUND_PARAMETERS,
    ManifestBundle,
    PipelineError,
    build_run_manifest,
    bundle_run_manifest,
    materialize_run_manifest,
    simulation_request_from_bundle,
    write_json_immutable,
)
from .search import (
    continued_lhs,
    predeclared_target,
    select_bo_candidate,
    shared_initial_lhs,
)
from .simulation import (
    RUN_STATUS_SCHEMA,
    SimulationInfrastructureError,
    SimulationRequest,
    SimulationResult,
    execute_simulation,
    sha256_file,
    validate_stopinfo,
)


L1_EVALUATION_SCHEMA = "bo-lhs-evaluations/v1"
L1_SUMMARY_SCHEMA = "bo-lhs-summary/v1"
L1_SELECTED_SCHEMA = "l1-selected/v1"
PARAMETER_NAMES = (
    "t_board",
    "t_fixed",
    "tau",
    "sigma",
    "minGap_bus",
    "accel",
    "decel",
)
METRIC_FIELDS = (
    "objective",
    "jl1_68x",
    "rmse_68x",
    "mae_68x",
    "std_abs_68x",
    "q90_abs_68x",
    "rmse_960",
    "constraint_violation_s",
    "penalty",
)
EVALUATION_COLUMNS = (
    "schema_version",
    "optimization_seed",
    "method",
    "phase",
    "evaluation_index",
    "sumo_seed",
    "candidate_hash",
    *METRIC_FIELDS,
    "feasible",
    "n_errors_68x",
    "n_errors_960",
    *PARAMETER_NAMES,
    "parameters_json",
    "run_id",
    "run_directory",
    "stopinfo_path",
    "run_status_path",
    "attempt",
    "duration_s",
    "reused",
    "manifest_hash",
    "provenance_hash",
    "simulation_effective_hash",
    "component_hashes_json",
    "bus_parameters_hash",
    "background_parameters_hash",
    "observation_semantic_hash",
    "simulator_inputs_hash",
    "output_hash",
    "stdout_log_path",
    "stderr_log_path",
    "status",
)
AGGREGATE_EVALUATION_COLUMNS = (
    "schema_version",
    "optimization_seed",
    "method",
    "evaluation_index",
    "candidate_hash",
    "objective",
    "feasible",
)
SUMMARY_COLUMNS = (
    "schema_version",
    "optimization_seed",
    "method",
    "status",
    "successful_evaluations",
    "target_objective",
    "evaluations_to_target",
    "final_best_objective",
    "selected_candidate_hash",
    "selected_objective",
    "selected_feasible",
    *PARAMETER_NAMES,
    "selected_run_id",
    "selected_provenance_hash",
    "selected_simulation_effective_hash",
    "selected_output_hash",
)

SimulationExecutor = Callable[..., SimulationResult]
ObjectiveFunction = Callable[..., Mapping[str, Any]]
PreflightFunction = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


class L1StageError(RuntimeError):
    """Raised when the L1 protocol cannot produce valid camera-ready evidence."""


class L1PreflightError(L1StageError):
    """Raised before scheduling when the observed cumulative chain is incomplete."""


class _L1SeedFailure(L1StageError):
    """Internal marker for a structurally failed optimization seed."""


def _path_within(project_root: Path, relative_path: str | Path) -> Path:
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise L1StageError(f"Path escapes project root: {relative_path}")
    return candidate


def _dataset_by_id(manifest: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["datasets"] if item.get("id") == dataset_id]
    if len(matches) != 1:
        raise L1StageError(f"Manifest requires exactly one dataset id={dataset_id!r}")
    return matches[0]


def _development_l1_dataset(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Use the manifest-declared reconstructed L1 events when present."""

    ids = {str(item.get("id")): item for item in manifest["datasets"]}
    for dataset_id in ("development_l1_events", "development_events"):
        if dataset_id in ids:
            return ids[dataset_id]
    raise L1StageError(
        "Manifest requires development_l1_events or development_events for L1"
    )


def _l1_route_bounds(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    selected = [
        (str(item["route"]), str(item["direction"]))
        for item in manifest["routes"]
        if bool(item.get("l1_selected", False))
    ]
    if not selected:
        raise L1StageError("Manifest has no routes with l1_selected=true")
    return selected


def preflight_l1_observation_chains(
    project_root: Path,
    manifest: Mapping[str, Any],
    *,
    min_downstream_stops: int = 3,
) -> Mapping[str, Any]:
    """Verify the seq=1 observed chain before any simulation is scheduled."""

    validate_paper_manifest(manifest)
    dataset = _development_l1_dataset(manifest)
    real_path = _path_within(project_root, str(dataset["path"]))
    try:
        real_links = load_real_link_speeds(str(real_path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise L1PreflightError(f"Cannot load L1 development events: {real_path}: {exc}") from exc

    checks: list[dict[str, Any]] = []
    for route, bound in _l1_route_bounds(manifest):
        try:
            observed = build_observed_cumulative_times(real_links, route, bound, origin_seq=1)
            matched_count = int(len(observed))
        except L1UnevaluableError:
            matched_count = 0
        except L1ObjectiveError as exc:
            raise L1PreflightError(
                f"L1 observation-chain preflight failed for route={route}, bound={bound}, "
                f"matched_downstream_stops=unknown: {exc}"
            ) from exc
        check = {
            "route": route,
            "bound": bound,
            "matched_downstream_stops": matched_count,
        }
        checks.append(check)
        if matched_count < int(min_downstream_stops):
            raise L1PreflightError(
                f"L1 observation-chain preflight failed for route={route}, bound={bound}, "
                f"matched_downstream_stops={matched_count}; required={min_downstream_stops}"
            )
    return {
        "status": "succeeded",
        "real_links_path": real_path.relative_to(project_root.resolve()).as_posix(),
        "minimum_downstream_stops": int(min_downstream_stops),
        "routes": checks,
    }


def _positive_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise L1StageError(f"{label} must be an integer")
    integer = int(value)
    minimum = 0 if allow_zero else 1
    if integer < minimum:
        raise L1StageError(f"{label} must be >= {minimum}")
    return integer


def _effective_protocol(
    base_manifest: Mapping[str, Any],
    *,
    optimization_seeds: Sequence[int] | None,
    initial_evaluations: int | None,
    subsequent_evaluations: int | None,
    bo_candidate_pool_size: int | None,
    minimum_valid_seeds: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_paper_manifest(base_manifest)
    manifest = deepcopy(dict(base_manifest))
    l1 = manifest["l1"]
    bounds_mapping = l1.get("parameter_bounds")
    if not isinstance(bounds_mapping, Mapping) or set(bounds_mapping) != set(PARAMETER_NAMES):
        raise L1StageError(f"L1 parameter bounds must contain exactly {list(PARAMETER_NAMES)}")
    bounds: list[tuple[float, float]] = []
    for name in PARAMETER_NAMES:
        pair = bounds_mapping[name]
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise L1StageError(f"l1.parameter_bounds.{name} must contain [lower, upper]")
        lower, upper = float(pair[0]), float(pair[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise L1StageError(f"l1.parameter_bounds.{name} must be finite and increasing")
        bounds.append((lower, upper))

    initial_contract = l1.get("initial_design")
    budget_contract = l1.get("budget")
    if not isinstance(initial_contract, Mapping) or not isinstance(budget_contract, Mapping):
        raise L1StageError("L1 initial_design and budget must be objects")
    declared_initial = initial_contract.get("shared_evaluations", initial_contract.get("n"))
    declared_subsequent = budget_contract.get("subsequent_evaluations")
    declared_pool = budget_contract.get("bo_candidate_pool", 16384)
    declared_seeds = l1.get("optimization_seeds")
    if declared_seeds is None and isinstance(l1.get("seed_schedule"), Mapping):
        declared_seeds = l1["seed_schedule"].get("optimization_seeds")
    if not isinstance(declared_seeds, Sequence) or isinstance(declared_seeds, (str, bytes)):
        raise L1StageError("l1.optimization_seeds must be an integer array")

    initial = _positive_integer(
        declared_initial if initial_evaluations is None else initial_evaluations,
        "initial_evaluations",
    )
    subsequent = _positive_integer(
        declared_subsequent if subsequent_evaluations is None else subsequent_evaluations,
        "subsequent_evaluations",
        allow_zero=True,
    )
    pool_size = _positive_integer(
        declared_pool if bo_candidate_pool_size is None else bo_candidate_pool_size,
        "bo_candidate_pool_size",
    )
    if pool_size & (pool_size - 1):
        raise L1StageError("bo_candidate_pool_size must be a power of two")
    if subsequent and initial < 3:
        raise L1StageError("BO requires at least three shared initial evaluations")

    seed_values = declared_seeds if optimization_seeds is None else optimization_seeds
    seeds = tuple(_positive_integer(seed, "optimization_seed", allow_zero=True) for seed in seed_values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise L1StageError("optimization_seeds must be non-empty and unique")
    default_minimum = int(manifest.get("evaluation", {}).get("common_seed_minimum", 3))
    common_minimum = _positive_integer(
        default_minimum if minimum_valid_seeds is None else minimum_valid_seeds,
        "minimum_valid_seeds",
    )

    max_attempts = int(
        manifest.get("execution", {}).get(
            "max_attempts_per_run", manifest["simulator"].get("max_attempts", 3)
        )
    )
    if not 1 <= max_attempts <= 3:
        raise L1StageError("L1 max_attempts_per_run must be between one and three")

    manifest["l1"]["initial_design"] = dict(initial_contract)
    manifest["l1"]["initial_design"]["shared_evaluations"] = initial
    if "n" in manifest["l1"]["initial_design"]:
        manifest["l1"]["initial_design"]["n"] = initial
    manifest["l1"]["budget"] = dict(budget_contract)
    manifest["l1"]["budget"].update(
        {
            "successful_evaluations_per_method": initial + subsequent,
            "subsequent_evaluations": subsequent,
            "bo_candidate_pool": pool_size,
        }
    )
    manifest["l1"]["optimization_seeds"] = list(seeds)
    if isinstance(manifest["l1"].get("seed_schedule"), Mapping):
        manifest["l1"]["seed_schedule"] = dict(manifest["l1"]["seed_schedule"])
        manifest["l1"]["seed_schedule"]["optimization_seeds"] = list(seeds)
    manifest.setdefault("evaluation", {})["common_seed_minimum"] = common_minimum
    validate_paper_manifest(manifest)
    settings = {
        "bounds": bounds,
        "seeds": seeds,
        "initial": initial,
        "subsequent": subsequent,
        "total": initial + subsequent,
        "candidate_pool": pool_size,
        "minimum_valid_seeds": common_minimum,
        "max_attempts": max_attempts,
    }
    return manifest, settings


_INTEGER_COLUMNS = {
    "optimization_seed",
    "evaluation_index",
    "sumo_seed",
    "n_errors_68x",
    "n_errors_960",
    "attempt",
}
_FLOAT_COLUMNS = set(METRIC_FIELDS) | set(PARAMETER_NAMES) | {"duration_s"}
_BOOLEAN_COLUMNS = {"feasible", "reused"}


def _parse_bool(value: str, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise L1StageError(f"Invalid boolean in checkpoint {label}: {value!r}")


def _read_evaluations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise L1StageError(f"Unexpected L1 checkpoint columns: {path}")
            raw_rows = list(reader)
    except OSError as exc:
        raise L1StageError(f"Cannot read L1 checkpoint: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str, int]] = set()
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for column in _INTEGER_COLUMNS:
            row[column] = int(row[column])
        for column in _FLOAT_COLUMNS:
            row[column] = float(row[column])
            if not math.isfinite(row[column]):
                raise L1StageError(f"Checkpoint has non-finite {column}: {path}")
        for column in _BOOLEAN_COLUMNS:
            row[column] = _parse_bool(str(row[column]), column)
        key = (row["optimization_seed"], str(row["method"]), row["evaluation_index"])
        if key in keys:
            raise L1StageError(f"Duplicate L1 checkpoint key {key}: {path}")
        keys.add(key)
        if row["schema_version"] != L1_EVALUATION_SCHEMA or row["status"] != "succeeded":
            raise L1StageError(f"Checkpoint contains a non-success row: {path}")
        rows.append(row)
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sort_evaluations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row["optimization_seed"]),
            str(row["method"]),
            int(row["evaluation_index"]),
        ),
    )


def _relative(project_root: Path, path: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise L1StageError(f"Artifact escapes project root: {path}")
    return resolved.relative_to(root).as_posix()


def _objective_inputs(project_root: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path, str, float]:
    real_links = _path_within(project_root, str(_development_l1_dataset(manifest)["path"]))
    route_stop_dist = _path_within(
        project_root, str(_dataset_by_id(manifest, "route_stop_distance")["path"])
    )
    selected_bounds = {bound for _, bound in _l1_route_bounds(manifest)}
    if len(selected_bounds) != 1:
        raise L1StageError("All L1-selected routes must use one common direction")
    objective_definition = manifest["l1"]["objective_definition"]
    if not isinstance(objective_definition, Mapping):
        raise L1StageError("l1.objective_definition must be an object")
    limit = float(objective_definition.get("constraint_rmse_max_s", 350.0))
    if limit != 350.0:
        raise L1StageError("L1 RMSE_960 feasibility limit must remain 350 seconds")
    return real_links, route_stop_dist, next(iter(selected_bounds)), limit


def _validate_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "status",
        "score",
        "feasible",
        "jl1_68x",
        "rmse_68x",
        "mae_68x",
        "std_abs_68x",
        "q90_abs_68x",
        "rmse_960",
        "constraint_violation_s",
        "penalty",
        "n_errors_68x",
        "n_errors_960",
    }
    missing = sorted(required - set(value))
    if missing:
        raise L1StageError(f"L1 objective result is missing fields: {missing}")
    if value["status"] != "succeeded" or not isinstance(value["feasible"], (bool, np.bool_)):
        raise L1StageError("L1 objective result has invalid status or feasibility")
    result = dict(value)
    result["objective"] = float(value["score"])
    for key in METRIC_FIELDS:
        numeric = float(result[key])
        if not math.isfinite(numeric):
            raise L1StageError(f"L1 objective result has non-finite {key}")
        result[key] = numeric
    for key in ("n_errors_68x", "n_errors_960"):
        result[key] = int(result[key])
        if result[key] < 3:
            raise L1StageError(f"L1 objective result has fewer than three {key}")
    result["feasible"] = bool(result["feasible"])
    return result


def _call_objective(
    objective_function: ObjectiveFunction,
    stopinfo_path: Path,
    real_links_path: Path,
    route_stop_dist_path: Path,
    bound: str,
    rmse_960_limit_s: float,
) -> dict[str, Any]:
    value = objective_function(
        str(stopinfo_path),
        str(real_links_path),
        str(route_stop_dist_path),
        bound=bound,
        rmse_960_limit_s=rmse_960_limit_s,
    )
    if not isinstance(value, Mapping):
        raise L1StageError("L1 objective function must return a mapping")
    return _validate_metrics(value)


def _candidate_parameters(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(PARAMETER_NAMES),) or not np.isfinite(vector).all():
        raise L1StageError("L1 candidate has invalid shape or non-finite values")
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, vector, strict=True)}


def _candidate_hash(parameters: Mapping[str, float]) -> str:
    return canonical_sha256(
        {
            "schema_version": "l1-candidate/v1",
            "bus_parameters": {name: float(parameters[name]) for name in PARAMETER_NAMES},
        }
    )


def _candidate_contract(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    software: Mapping[str, str],
    timeout_s: float,
    seed_directory: Path,
    optimization_seed: int,
    evaluation_index: int,
    parameters: Mapping[str, float],
    physical_method: str,
) -> tuple[str, int, str, Path, ManifestBundle]:
    """Rebuild the immutable contract for one physical L1 candidate."""

    candidate_hash = _candidate_hash(parameters)
    sumo_seed = 100000 + 1000 * optimization_seed + evaluation_index
    run_id = (
        f"l1-seed-{optimization_seed}-{physical_method}-"
        f"eval-{evaluation_index:03d}-{candidate_hash[:16]}"
    )
    run_directory = (
        seed_directory
        / "runs"
        / physical_method.lower()
        / f"eval-{evaluation_index:03d}-{candidate_hash}"
    )
    run_manifest = build_run_manifest(
        manifest,
        project_root=project_root,
        run_directory=run_directory,
        run_id=run_id,
        config_id=f"L1-{physical_method.upper()}",
        method_id={
            "shared": "shared-lhs",
            "bo": "bayesian-optimization",
            "lhs": "continued-lhs",
        }[physical_method],
        split="development",
        seed=optimization_seed,
        sumo_seed=sumo_seed,
        bus_parameters=parameters,
        background_parameters=manifest.get("ablation", {}).get(
            "baseline_background_parameters", BASELINE_BACKGROUND_PARAMETERS
        ),
        observation_semantic="no_l2_input",
        l1_enabled=True,
        l2_enabled=False,
        software=software,
        timeout_seconds=timeout_s,
        parameter_sources={
            "bus": {
                "stage": "l1",
                "physical_method": physical_method,
                "optimization_seed": optimization_seed,
                "evaluation_index": evaluation_index,
                "candidate_hash": candidate_hash,
            },
            "background": "manifest.ablation.baseline_background_parameters",
        },
    )
    return candidate_hash, sumo_seed, run_id, run_directory, bundle_run_manifest(run_manifest)


def _existing_records(
    records: Sequence[Mapping[str, Any]], methods: Sequence[str], evaluation_index: int
) -> list[dict[str, Any]]:
    found = [
        dict(row)
        for row in records
        if str(row["method"]) in methods and int(row["evaluation_index"]) == evaluation_index
    ]
    if found and {str(row["method"]) for row in found} != set(methods):
        raise _L1SeedFailure(
            f"Partial checkpoint at evaluation_index={evaluation_index}: expected {list(methods)}"
        )
    return sorted(found, key=lambda row: str(row["method"]))


def _verify_checkpoint_record(
    project_root: Path,
    row: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_index: int,
    expected_sumo_seed: int,
    expected_candidate_hash: str,
    expected_run_id: str,
    manifest_hash: str,
    provenance_hash: str,
    simulation_effective_hash: str,
    component_hashes: Mapping[str, str],
    expected_run_directory: Path,
    expected_manifest: Mapping[str, Any],
) -> Path:
    expected = {
        "optimization_seed": expected_seed,
        "evaluation_index": expected_index,
        "sumo_seed": expected_sumo_seed,
        "candidate_hash": expected_candidate_hash,
        "run_id": expected_run_id,
        "manifest_hash": manifest_hash,
        "provenance_hash": provenance_hash,
        "simulation_effective_hash": simulation_effective_hash,
    }
    mismatches = [key for key, value in expected.items() if row.get(key) != value]
    if mismatches:
        raise _L1SeedFailure(f"Checkpoint content differs for fields: {mismatches}")
    try:
        recorded_components = json.loads(str(row["component_hashes_json"]))
    except json.JSONDecodeError as exc:
        raise _L1SeedFailure("Checkpoint component_hashes_json is malformed") from exc
    if recorded_components != dict(component_hashes):
        raise _L1SeedFailure("Checkpoint component hashes differ from the run manifest")

    run_directory = _path_within(project_root, str(row["run_directory"]))
    if run_directory != expected_run_directory.resolve():
        raise _L1SeedFailure("Checkpoint run directory differs from the candidate contract")
    status_path = _path_within(project_root, str(row["run_status_path"]))
    if status_path != run_directory / "run-status.json":
        raise _L1SeedFailure("Checkpoint run-status path differs from the candidate contract")
    stopinfo_path = _path_within(project_root, str(row["stopinfo_path"]))
    manifest_path = run_directory / "run-manifest.json"
    manifest_hashes_path = run_directory / "run-manifest-hashes.json"
    try:
        recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_manifest_hashes = json.loads(manifest_hashes_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _L1SeedFailure(f"Cannot verify checkpoint run artifacts: {run_directory}") from exc
    expected_manifest_hashes = {
        "manifest_hash": manifest_hash,
        "provenance_hash": provenance_hash,
        "simulation_effective_hash": simulation_effective_hash,
        "component_hashes": dict(component_hashes),
    }
    if recorded_manifest != dict(expected_manifest):
        raise _L1SeedFailure(f"Checkpoint run manifest differs: {manifest_path}")
    if canonical_sha256(recorded_manifest) != manifest_hash:
        raise _L1SeedFailure(f"Checkpoint run manifest hash differs: {manifest_path}")
    if recorded_manifest_hashes != expected_manifest_hashes:
        raise _L1SeedFailure(f"Checkpoint run-manifest hashes differ: {manifest_hashes_path}")
    status_expected = {
        "schema_version": RUN_STATUS_SCHEMA,
        "status": "succeeded",
        "run_id": expected_run_id,
        "manifest_hash": manifest_hash,
        "provenance_hash": provenance_hash,
        "simulation_effective_hash": simulation_effective_hash,
        "component_hashes": dict(component_hashes),
    }
    if any(status.get(key) != value for key, value in status_expected.items()):
        raise _L1SeedFailure(f"Checkpoint run-status contract differs: {status_path}")
    if int(status.get("attempt", -1)) != int(row["attempt"]):
        raise _L1SeedFailure(f"Checkpoint attempt differs: {status_path}")
    if not math.isclose(
        float(status.get("duration_s", math.nan)),
        float(row["duration_s"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _L1SeedFailure(f"Checkpoint duration differs: {status_path}")
    relative_stopinfo = status.get("stopinfo_relative_path")
    if (
        not isinstance(relative_stopinfo, str)
        or (run_directory / relative_stopinfo).resolve() != stopinfo_path
    ):
        raise _L1SeedFailure(f"Checkpoint stopinfo path differs: {status_path}")
    attempt_directory = run_directory / f"attempt-{int(row['attempt']):02d}"
    expected_logs = {
        "stdout_log_path": attempt_directory / "stdout.log",
        "stderr_log_path": attempt_directory / "stderr.log",
    }
    for field, expected_path in expected_logs.items():
        if _path_within(project_root, str(row[field])) != expected_path.resolve():
            raise _L1SeedFailure(f"Checkpoint {field} differs from the selected attempt")
        if not expected_path.is_file():
            raise _L1SeedFailure(f"Checkpoint log is missing: {expected_path}")
    component_columns = {
        "bus_parameters_hash": "bus_parameters",
        "background_parameters_hash": "background_parameters",
        "observation_semantic_hash": "observation_semantic",
        "simulator_inputs_hash": "simulator_inputs",
    }
    for column, component in component_columns.items():
        if row[column] != component_hashes[component]:
            raise _L1SeedFailure(f"Checkpoint component column differs: {column}")
    validate_stopinfo(stopinfo_path)
    actual_output_hash = sha256_file(stopinfo_path)
    if (
        actual_output_hash != row["output_hash"]
        or actual_output_hash
        != status.get("produced_artifact_hashes", {}).get("stopinfo.xml")
    ):
        raise _L1SeedFailure(f"Checkpoint output hash differs: {stopinfo_path}")
    return stopinfo_path


def _compare_metrics(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    for field in METRIC_FIELDS:
        if not math.isclose(float(row[field]), float(metrics[field]), rel_tol=0.0, abs_tol=1e-12):
            raise _L1SeedFailure(f"Checkpoint metric differs after validation: {field}")
    if bool(row["feasible"]) != bool(metrics["feasible"]):
        raise _L1SeedFailure("Checkpoint feasibility differs after validation")
    for field in ("n_errors_68x", "n_errors_960"):
        if int(row[field]) != int(metrics[field]):
            raise _L1SeedFailure(f"Checkpoint count differs after validation: {field}")


def _expected_checkpoint_keys(total: int) -> set[tuple[str, int]]:
    return {
        (method, evaluation_index)
        for method in ("BO", "LHS")
        for evaluation_index in range(1, total + 1)
    }


def _checkpoint_is_complete(records: Sequence[Mapping[str, Any]], total: int) -> bool:
    actual = {(str(row["method"]), int(row["evaluation_index"])) for row in records}
    return actual == _expected_checkpoint_keys(total)


def _checkpoint_parameters(
    row: Mapping[str, Any],
    bounds: Sequence[tuple[float, float]],
    *,
    expected_seed: int,
    expected_method: str,
    expected_index: int,
    expected_phase: str,
) -> dict[str, float]:
    expected_fields = {
        "optimization_seed": expected_seed,
        "method": expected_method,
        "phase": expected_phase,
        "evaluation_index": expected_index,
        "sumo_seed": 100000 + 1000 * expected_seed + expected_index,
    }
    mismatches = [key for key, value in expected_fields.items() if row.get(key) != value]
    if mismatches:
        raise _L1SeedFailure(f"Checkpoint protocol fields differ: {mismatches}")

    parameters = {name: float(row[name]) for name in PARAMETER_NAMES}
    for name, value, (lower, upper) in zip(
        PARAMETER_NAMES, parameters.values(), bounds, strict=True
    ):
        if not lower <= value <= upper:
            raise _L1SeedFailure(
                f"Checkpoint parameter is outside the frozen bound: {name}={value}"
            )
    try:
        encoded = json.loads(str(row["parameters_json"]))
    except json.JSONDecodeError as exc:
        raise _L1SeedFailure("Checkpoint parameters_json is malformed") from exc
    if not isinstance(encoded, Mapping) or set(encoded) != set(PARAMETER_NAMES):
        raise _L1SeedFailure("Checkpoint parameters_json keys differ")
    try:
        encoded_parameters = {name: float(encoded[name]) for name in PARAMETER_NAMES}
    except (TypeError, ValueError) as exc:
        raise _L1SeedFailure("Checkpoint parameters_json contains non-numeric values") from exc
    if encoded_parameters != parameters:
        raise _L1SeedFailure("Checkpoint parameter columns differ from parameters_json")
    if _candidate_hash(parameters) != row["candidate_hash"]:
        raise _L1SeedFailure("Checkpoint candidate hash differs from its parameters")
    return parameters


def _validate_complete_checkpoint(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    software: Mapping[str, str],
    timeout_s: float,
    objective_function: ObjectiveFunction,
    real_links_path: Path,
    route_stop_dist_path: Path,
    bound: str,
    rmse_960_limit_s: float,
    seed_directory: Path,
    records: Sequence[Mapping[str, Any]],
    optimization_seed: int,
    settings: Mapping[str, Any],
) -> float:
    """Validate a complete checkpoint without regenerating search candidates."""

    initial = int(settings["initial"])
    total = int(settings["total"])
    bounds = settings["bounds"]
    if not _checkpoint_is_complete(records, total):
        raise _L1SeedFailure("Complete-checkpoint recovery received an incomplete checkpoint")

    for evaluation_index in range(1, total + 1):
        if evaluation_index <= initial:
            candidate_rows = _existing_records(
                records, ("BO", "LHS"), evaluation_index
            )
            parameters_by_method = {
                str(row["method"]): _checkpoint_parameters(
                    row,
                    bounds,
                    expected_seed=optimization_seed,
                    expected_method=str(row["method"]),
                    expected_index=evaluation_index,
                    expected_phase="shared_initial",
                )
                for row in candidate_rows
            }
            bo_row = next(row for row in candidate_rows if row["method"] == "BO")
            lhs_row = next(row for row in candidate_rows if row["method"] == "LHS")
            if {key: value for key, value in bo_row.items() if key != "method"} != {
                key: value for key, value in lhs_row.items() if key != "method"
            }:
                raise _L1SeedFailure(
                    f"Shared BO/LHS checkpoint rows differ at evaluation_index={evaluation_index}"
                )
            if parameters_by_method["BO"] != parameters_by_method["LHS"]:
                raise _L1SeedFailure(
                    f"Shared BO/LHS parameters differ at evaluation_index={evaluation_index}"
                )
            physical_method = "shared"
            parameters = parameters_by_method["BO"]
        else:
            candidate_rows = []
            for method, physical_method in (("BO", "bo"), ("LHS", "lhs")):
                row = _existing_records(records, (method,), evaluation_index)[0]
                parameters = _checkpoint_parameters(
                    row,
                    bounds,
                    expected_seed=optimization_seed,
                    expected_method=method,
                    expected_index=evaluation_index,
                    expected_phase="subsequent",
                )
                candidate_rows.append((row, physical_method, parameters))

        physical_candidates = (
            [(candidate_rows, physical_method, parameters)]
            if evaluation_index <= initial
            else [([row], method, values) for row, method, values in candidate_rows]
        )
        for reported_rows, physical_method, parameters in physical_candidates:
            candidate_hash, sumo_seed, run_id, run_directory, bundle = _candidate_contract(
                project_root=project_root,
                manifest=manifest,
                software=software,
                timeout_s=timeout_s,
                seed_directory=seed_directory,
                optimization_seed=optimization_seed,
                evaluation_index=evaluation_index,
                parameters=parameters,
                physical_method=physical_method,
            )
            stopinfo_path: Path | None = None
            for row in reported_rows:
                verified = _verify_checkpoint_record(
                    project_root,
                    row,
                    expected_seed=optimization_seed,
                    expected_index=evaluation_index,
                    expected_sumo_seed=sumo_seed,
                    expected_candidate_hash=candidate_hash,
                    expected_run_id=run_id,
                    manifest_hash=bundle.manifest_hash,
                    provenance_hash=bundle.provenance_hash,
                    simulation_effective_hash=bundle.simulation_effective_hash,
                    component_hashes=bundle.component_hashes,
                    expected_run_directory=run_directory,
                    expected_manifest=bundle.manifest,
                )
                stopinfo_path = verified if stopinfo_path is None else stopinfo_path
                if verified != stopinfo_path:
                    raise _L1SeedFailure(
                        "Shared checkpoint rows refer to different outputs"
                    )
            assert stopinfo_path is not None
            metrics = _call_objective(
                objective_function,
                stopinfo_path,
                real_links_path,
                route_stop_dist_path,
                bound,
                rmse_960_limit_s,
            )
            for row in reported_rows:
                _compare_metrics(row, metrics)

    initial_bo = sorted(
        (
            row
            for row in records
            if row["method"] == "BO" and int(row["evaluation_index"]) <= initial
        ),
        key=lambda row: int(row["evaluation_index"]),
    )
    return predeclared_target(
        [float(row["objective"]) for row in initial_bo],
        [bool(row["feasible"]) for row in initial_bo],
    )


def _evaluation_row(
    project_root: Path,
    *,
    optimization_seed: int,
    method: str,
    phase: str,
    evaluation_index: int,
    sumo_seed: int,
    candidate_hash: str,
    parameters: Mapping[str, float],
    metrics: Mapping[str, Any],
    result: SimulationResult,
    manifest_hash: str,
) -> dict[str, Any]:
    attempt_directory = result.run_directory / f"attempt-{result.attempt:02d}"
    components = dict(result.component_hashes)
    row: dict[str, Any] = {
        "schema_version": L1_EVALUATION_SCHEMA,
        "optimization_seed": optimization_seed,
        "method": method,
        "phase": phase,
        "evaluation_index": evaluation_index,
        "sumo_seed": sumo_seed,
        "candidate_hash": candidate_hash,
        **{field: metrics[field] for field in METRIC_FIELDS},
        "feasible": bool(metrics["feasible"]),
        "n_errors_68x": int(metrics["n_errors_68x"]),
        "n_errors_960": int(metrics["n_errors_960"]),
        **{name: float(parameters[name]) for name in PARAMETER_NAMES},
        "parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        "run_id": result.run_id,
        "run_directory": _relative(project_root, result.run_directory),
        "stopinfo_path": _relative(project_root, result.stopinfo_path),
        "run_status_path": _relative(project_root, result.run_directory / "run-status.json"),
        "attempt": int(result.attempt),
        "duration_s": float(result.duration_s),
        "reused": bool(result.reused),
        "manifest_hash": manifest_hash,
        "provenance_hash": result.provenance_hash,
        "simulation_effective_hash": result.simulation_effective_hash,
        "component_hashes_json": json.dumps(components, sort_keys=True, separators=(",", ":")),
        "bus_parameters_hash": components["bus_parameters"],
        "background_parameters_hash": components["background_parameters"],
        "observation_semantic_hash": components["observation_semantic"],
        "simulator_inputs_hash": components["simulator_inputs"],
        "output_hash": result.output_hash,
        "stdout_log_path": _relative(project_root, attempt_directory / "stdout.log"),
        "stderr_log_path": _relative(project_root, attempt_directory / "stderr.log"),
        "status": "succeeded",
    }
    return row


def _evaluate_candidate(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    software: Mapping[str, str],
    timeout_s: float,
    sumo_binary: str,
    executor: SimulationExecutor,
    objective_function: ObjectiveFunction,
    real_links_path: Path,
    route_stop_dist_path: Path,
    bound: str,
    rmse_960_limit_s: float,
    max_attempts: int,
    output_root: Path,
    seed_directory: Path,
    records: Sequence[Mapping[str, Any]],
    optimization_seed: int,
    evaluation_index: int,
    parameters: Mapping[str, float],
    physical_method: str,
    reported_methods: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    del output_root
    candidate_hash, sumo_seed, run_id, run_directory, bundle = _candidate_contract(
        project_root=project_root,
        manifest=manifest,
        software=software,
        timeout_s=timeout_s,
        seed_directory=seed_directory,
        optimization_seed=optimization_seed,
        evaluation_index=evaluation_index,
        parameters=parameters,
        physical_method=physical_method,
    )
    materialize_run_manifest(run_directory, bundle)

    existing = _existing_records(records, reported_methods, evaluation_index)
    if existing:
        stopinfo_path: Path | None = None
        for row in existing:
            row_stopinfo = _verify_checkpoint_record(
                project_root,
                row,
                expected_seed=optimization_seed,
                expected_index=evaluation_index,
                expected_sumo_seed=sumo_seed,
                expected_candidate_hash=candidate_hash,
                expected_run_id=run_id,
                manifest_hash=bundle.manifest_hash,
                provenance_hash=bundle.provenance_hash,
                simulation_effective_hash=bundle.simulation_effective_hash,
                component_hashes=bundle.component_hashes,
                expected_run_directory=run_directory,
                expected_manifest=bundle.manifest,
            )
            stopinfo_path = row_stopinfo if stopinfo_path is None else stopinfo_path
            if row_stopinfo != stopinfo_path:
                raise _L1SeedFailure("Shared checkpoint rows refer to different outputs")
        assert stopinfo_path is not None
        metrics = _call_objective(
            objective_function,
            stopinfo_path,
            real_links_path,
            route_stop_dist_path,
            bound,
            rmse_960_limit_s,
        )
        for row in existing:
            _compare_metrics(row, metrics)
        return existing, False

    status_path = run_directory / "run-status.json"
    if status_path.exists():
        try:
            existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _L1SeedFailure(f"Existing run status is unreadable: {status_path}") from exc
        if existing_status.get("status") != "succeeded":
            raise _L1SeedFailure(
                f"Existing run is non-reusable and will not be overwritten: {status_path}"
            )

    captured: dict[str, Any] = {}

    def validator(stopinfo_path: Path) -> None:
        captured["metrics"] = _call_objective(
            objective_function,
            stopinfo_path,
            real_links_path,
            route_stop_dist_path,
            bound,
            rmse_960_limit_s,
        )

    request: SimulationRequest = simulation_request_from_bundle(project_root, bundle)
    try:
        result = executor(
            request,
            run_directory,
            sumo_binary=sumo_binary,
            max_attempts=max_attempts,
            allow_reuse=True,
            post_output_validator=validator,
        )
    except (SimulationInfrastructureError, L1ObjectiveError, L1StageError, OSError, ValueError) as exc:
        raise _L1SeedFailure(
            f"Candidate evaluation_index={evaluation_index} failed after deterministic retries: {exc}"
        ) from exc
    if (
        result.run_id != run_id
        or result.provenance_hash != bundle.provenance_hash
        or result.simulation_effective_hash != bundle.simulation_effective_hash
        or dict(result.component_hashes) != dict(bundle.component_hashes)
    ):
        raise _L1SeedFailure(f"Executor returned hashes from another run: {run_id}")
    metrics = captured.get("metrics")
    if metrics is None:
        metrics = _call_objective(
            objective_function,
            result.stopinfo_path,
            real_links_path,
            route_stop_dist_path,
            bound,
            rmse_960_limit_s,
        )
    phase = "shared_initial" if physical_method == "shared" else "subsequent"
    rows = [
        _evaluation_row(
            project_root,
            optimization_seed=optimization_seed,
            method=method,
            phase=phase,
            evaluation_index=evaluation_index,
            sumo_seed=sumo_seed,
            candidate_hash=candidate_hash,
            parameters=parameters,
            metrics=metrics,
            result=result,
            manifest_hash=bundle.manifest_hash,
        )
        for method in reported_methods
    ]
    return rows, True


def _method_summary(
    optimization_seed: int,
    method: str,
    rows: Sequence[Mapping[str, Any]],
    target: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["evaluation_index"]))
    reached = next(
        (
            int(row["evaluation_index"])
            for row in ordered
            if bool(row["feasible"]) and float(row["objective"]) <= float(target)
        ),
        None,
    )
    feasible = [row for row in ordered if bool(row["feasible"])]
    if not feasible:
        raise _L1SeedFailure(f"seed={optimization_seed} method={method} has no feasible candidate")
    selected = min(feasible, key=lambda row: (float(row["objective"]), int(row["evaluation_index"])))
    summary = {
        "schema_version": L1_SUMMARY_SCHEMA,
        "optimization_seed": optimization_seed,
        "method": method,
        "status": "succeeded",
        "successful_evaluations": len(ordered),
        "target_objective": float(target),
        "evaluations_to_target": "" if reached is None else reached,
        "final_best_objective": min(float(row["objective"]) for row in feasible),
        "selected_candidate_hash": selected["candidate_hash"],
        "selected_objective": float(selected["objective"]),
        "selected_feasible": True,
        **{name: float(selected[name]) for name in PARAMETER_NAMES},
        "selected_run_id": selected["run_id"],
        "selected_provenance_hash": selected["provenance_hash"],
        "selected_simulation_effective_hash": selected["simulation_effective_hash"],
        "selected_output_hash": selected["output_hash"],
    }
    selected_payload = {
        "method": method,
        "evaluation_index": int(selected["evaluation_index"]),
        "candidate_hash": selected["candidate_hash"],
        "objective": float(selected["objective"]),
        "jl1_68x": float(selected["jl1_68x"]),
        "rmse_68x": float(selected["rmse_68x"]),
        "mae_68x": float(selected["mae_68x"]),
        "std_abs_68x": float(selected["std_abs_68x"]),
        "q90_abs_68x": float(selected["q90_abs_68x"]),
        "rmse_960": float(selected["rmse_960"]),
        "feasible": True,
        "parameters": {name: float(selected[name]) for name in PARAMETER_NAMES},
        "run_id": selected["run_id"],
        "run_directory": selected["run_directory"],
        "manifest_hash": selected["manifest_hash"],
        "provenance_hash": selected["provenance_hash"],
        "simulation_effective_hash": selected["simulation_effective_hash"],
        "component_hashes": json.loads(str(selected["component_hashes_json"])),
        "output_hash": selected["output_hash"],
        "stdout_log_path": selected["stdout_log_path"],
        "stderr_log_path": selected["stderr_log_path"],
    }
    return summary, selected_payload


def _finalize_seed(
    *,
    optimization_seed: int,
    records: Sequence[Mapping[str, Any]],
    target: float,
    settings: Mapping[str, Any],
    selected_path: Path,
) -> dict[str, Any]:
    total = int(settings["total"])
    actual_keys = {(str(row["method"]), int(row["evaluation_index"])) for row in records}
    if actual_keys != _expected_checkpoint_keys(total):
        raise _L1SeedFailure("Seed checkpoint differs from the equal-budget protocol")

    summaries: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for method in ("BO", "LHS"):
        summary, selection = _method_summary(
            optimization_seed,
            method,
            [row for row in records if row["method"] == method],
            target,
        )
        summaries.append(summary)
        selections[method] = selection
    selected_payload = {
        "schema_version": L1_SELECTED_SCHEMA,
        "optimization_seed": optimization_seed,
        "status": "succeeded",
        "target_objective": target,
        "shared_initial_evaluations": int(settings["initial"]),
        "subsequent_evaluations_per_method": int(settings["subsequent"]),
        "successful_evaluations_per_method": total,
        "physical_candidate_runs": int(settings["initial"])
        + 2 * int(settings["subsequent"]),
        "methods": selections,
        "selected_for_l2": selections["BO"],
    }
    write_json_immutable(selected_path, selected_payload)
    return {
        "optimization_seed": optimization_seed,
        "status": "succeeded",
        "records": _sort_evaluations(records),
        "summaries": summaries,
        "selected": selected_payload,
    }


def _run_seed(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    software: Mapping[str, str],
    sumo_binary: str,
    timeout_s: float,
    executor: SimulationExecutor,
    objective_function: ObjectiveFunction,
    settings: Mapping[str, Any],
    output_root: Path,
    optimization_seed: int,
) -> dict[str, Any]:
    seed_directory = output_root / "l1" / f"seed-{optimization_seed}"
    checkpoint_path = seed_directory / "evaluations.csv"
    selected_path = seed_directory / "selected.json"
    records = _read_evaluations(checkpoint_path)
    if any(int(row["optimization_seed"]) != optimization_seed for row in records):
        raise L1StageError(f"Checkpoint contains another optimization seed: {checkpoint_path}")

    real_links_path, route_stop_dist_path, bound, limit = _objective_inputs(project_root, manifest)
    bounds = settings["bounds"]
    initial = int(settings["initial"])
    subsequent = int(settings["subsequent"])
    complete_checkpoint = _checkpoint_is_complete(records, int(settings["total"]))
    try:
        if complete_checkpoint:
            target = _validate_complete_checkpoint(
                project_root=project_root,
                manifest=manifest,
                software=software,
                timeout_s=timeout_s,
                objective_function=objective_function,
                real_links_path=real_links_path,
                route_stop_dist_path=route_stop_dist_path,
                bound=bound,
                rmse_960_limit_s=limit,
                seed_directory=seed_directory,
                records=records,
                optimization_seed=optimization_seed,
                settings=settings,
            )
            return _finalize_seed(
                optimization_seed=optimization_seed,
                records=records,
                target=target,
                settings=settings,
                selected_path=selected_path,
            )

        initial_points = shared_initial_lhs(
            bounds,
            optimization_seed=optimization_seed,
            evaluations=initial,
        )
        for offset, point in enumerate(initial_points, start=1):
            rows, created = _evaluate_candidate(
                project_root=project_root,
                manifest=manifest,
                software=software,
                timeout_s=timeout_s,
                sumo_binary=sumo_binary,
                executor=executor,
                objective_function=objective_function,
                real_links_path=real_links_path,
                route_stop_dist_path=route_stop_dist_path,
                bound=bound,
                rmse_960_limit_s=limit,
                max_attempts=int(settings["max_attempts"]),
                output_root=output_root,
                seed_directory=seed_directory,
                records=records,
                optimization_seed=optimization_seed,
                evaluation_index=offset,
                parameters=_candidate_parameters(point),
                physical_method="shared",
                reported_methods=("BO", "LHS"),
            )
            if created:
                records.extend(rows)
                _write_csv(checkpoint_path, EVALUATION_COLUMNS, _sort_evaluations(records))

        initial_bo = sorted(
            (
                row
                for row in records
                if row["method"] == "BO" and int(row["evaluation_index"]) <= initial
            ),
            key=lambda row: int(row["evaluation_index"]),
        )
        if len(initial_bo) != initial:
            raise _L1SeedFailure("Shared initial checkpoint is incomplete")
        target = predeclared_target(
            [float(row["objective"]) for row in initial_bo],
            [bool(row["feasible"]) for row in initial_bo],
        )

        lhs_points = (
            continued_lhs(
                bounds,
                optimization_seed=optimization_seed,
                evaluations=subsequent,
            )
            if subsequent
            else np.empty((0, len(PARAMETER_NAMES)))
        )
        for subsequent_offset in range(subsequent):
            evaluation_index = initial + subsequent_offset + 1
            bo_rows = sorted(
                (
                    row
                    for row in records
                    if row["method"] == "BO"
                    and int(row["evaluation_index"]) < evaluation_index
                ),
                key=lambda row: int(row["evaluation_index"]),
            )
            if len(bo_rows) != evaluation_index - 1:
                raise _L1SeedFailure(
                    f"BO checkpoint is not sequential before evaluation_index={evaluation_index}"
                )
            evaluated_parameters = np.asarray(
                [[float(row[name]) for name in PARAMETER_NAMES] for row in bo_rows],
                dtype=float,
            )
            evaluated_scores = np.asarray(
                [float(row["objective"]) for row in bo_rows], dtype=float
            )
            bo_point = select_bo_candidate(
                evaluated_parameters,
                evaluated_scores,
                bounds,
                optimization_seed=optimization_seed,
                evaluation_index=evaluation_index,
                candidate_pool_size=int(settings["candidate_pool"]),
            )
            bo_new, bo_created = _evaluate_candidate(
                project_root=project_root,
                manifest=manifest,
                software=software,
                timeout_s=timeout_s,
                sumo_binary=sumo_binary,
                executor=executor,
                objective_function=objective_function,
                real_links_path=real_links_path,
                route_stop_dist_path=route_stop_dist_path,
                bound=bound,
                rmse_960_limit_s=limit,
                max_attempts=int(settings["max_attempts"]),
                output_root=output_root,
                seed_directory=seed_directory,
                records=records,
                optimization_seed=optimization_seed,
                evaluation_index=evaluation_index,
                parameters=_candidate_parameters(bo_point),
                physical_method="bo",
                reported_methods=("BO",),
            )
            if bo_created:
                records.extend(bo_new)
                _write_csv(checkpoint_path, EVALUATION_COLUMNS, _sort_evaluations(records))

            lhs_new, lhs_created = _evaluate_candidate(
                project_root=project_root,
                manifest=manifest,
                software=software,
                timeout_s=timeout_s,
                sumo_binary=sumo_binary,
                executor=executor,
                objective_function=objective_function,
                real_links_path=real_links_path,
                route_stop_dist_path=route_stop_dist_path,
                bound=bound,
                rmse_960_limit_s=limit,
                max_attempts=int(settings["max_attempts"]),
                output_root=output_root,
                seed_directory=seed_directory,
                records=records,
                optimization_seed=optimization_seed,
                evaluation_index=evaluation_index,
                parameters=_candidate_parameters(lhs_points[subsequent_offset]),
                physical_method="lhs",
                reported_methods=("LHS",),
            )
            if lhs_created:
                records.extend(lhs_new)
                _write_csv(checkpoint_path, EVALUATION_COLUMNS, _sort_evaluations(records))

        return _finalize_seed(
            optimization_seed=optimization_seed,
            records=records,
            target=target,
            settings=settings,
            selected_path=selected_path,
        )
    except (ValueError, L1ObjectiveError, SimulationInfrastructureError, _L1SeedFailure) as exc:
        if complete_checkpoint:
            raise _L1SeedFailure(
                f"Completed checkpoint validation failed; selected artifact was not changed: {exc}"
            ) from exc
        failure = {
            "schema_version": L1_SELECTED_SCHEMA,
            "optimization_seed": optimization_seed,
            "status": "failed",
            "error": str(exc),
            "successful_evaluations": {
                method: sum(1 for row in records if row["method"] == method)
                for method in ("BO", "LHS")
            },
        }
        write_json_immutable(selected_path, failure)
        return {
            "optimization_seed": optimization_seed,
            "status": "failed",
            "records": _sort_evaluations(records),
            "summaries": [],
            "selected": failure,
        }


def run_l1_stage(
    project_root: Path,
    base_manifest: Mapping[str, Any],
    software: Mapping[str, str],
    sumo_binary: str,
    workers: int,
    timeout_s: float,
    *,
    optimization_seeds: Sequence[int] | None = None,
    initial_evaluations: int | None = None,
    subsequent_evaluations: int | None = None,
    bo_candidate_pool_size: int | None = None,
    minimum_valid_seeds: int | None = None,
    simulation_executor: SimulationExecutor = execute_simulation,
    objective_function: ObjectiveFunction = calculate_l1_candidate_score,
    preflight_function: PreflightFunction = preflight_l1_observation_chains,
) -> dict[str, Any]:
    """Run the complete deterministic L1 search and write camera-ready artifacts.

    Test callers may inject smaller budgets and a fake executor.  Production
    callers omit those arguments and therefore use the manifest's 15+25
    budget, seeds 0..4, and 16,384-point BO candidate pool.
    """

    root = Path(project_root).resolve()
    worker_count = _positive_integer(workers, "workers")
    authorized_workers = min(
        8, int(base_manifest.get("execution", {}).get("authorized_max_cpu_workers", 8))
    )
    if worker_count > authorized_workers:
        raise L1StageError(
            f"workers={worker_count} exceeds the authorized maximum {authorized_workers}"
        )
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0:
        raise L1StageError("timeout_s must be a positive finite number")
    if not isinstance(software, Mapping) or not software:
        raise L1StageError("software versions are required")

    manifest, settings = _effective_protocol(
        base_manifest,
        optimization_seeds=optimization_seeds,
        initial_evaluations=initial_evaluations,
        subsequent_evaluations=subsequent_evaluations,
        bo_candidate_pool_size=bo_candidate_pool_size,
        minimum_valid_seeds=minimum_valid_seeds,
    )
    preflight = dict(preflight_function(root, manifest))
    if preflight.get("status") != "succeeded":
        raise L1PreflightError("L1 observation-chain preflight did not succeed")

    output_root = _path_within(root, str(manifest["outputs"]["run_directory"]))
    seeds = tuple(int(seed) for seed in settings["seeds"])
    results: dict[int, dict[str, Any]] = {}
    fatal_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(worker_count, len(seeds))) as pool:
        futures = {
            pool.submit(
                _run_seed,
                project_root=root,
                manifest=manifest,
                software=software,
                sumo_binary=sumo_binary,
                timeout_s=timeout,
                executor=simulation_executor,
                objective_function=objective_function,
                settings=settings,
                output_root=output_root,
                optimization_seed=seed,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                results[seed] = future.result()
            except Exception as exc:
                fatal_errors.append(f"seed={seed}: {exc}")

    if fatal_errors:
        raise L1StageError("L1 worker failures: " + "; ".join(sorted(fatal_errors)))
    valid_seeds = sorted(seed for seed, result in results.items() if result["status"] == "succeeded")
    failed_seeds = sorted(seed for seed, result in results.items() if result["status"] != "succeeded")
    aggregate_rows = _sort_evaluations(
        [row for seed in valid_seeds for row in results[seed]["records"]]
    )
    summary_rows = sorted(
        [row for seed in valid_seeds for row in results[seed]["summaries"]],
        key=lambda row: (int(row["optimization_seed"]), str(row["method"])),
    )
    aggregate_evaluations = output_root / "l1" / "bo_lhs_evaluations.csv"
    aggregate_summary = output_root / "l1" / "bo_lhs_summary.csv"
    figure_rows = [
        {column: row[column] for column in AGGREGATE_EVALUATION_COLUMNS}
        for row in aggregate_rows
    ]
    _write_csv(
        aggregate_evaluations,
        AGGREGATE_EVALUATION_COLUMNS,
        figure_rows,
    )
    _write_csv(aggregate_summary, SUMMARY_COLUMNS, summary_rows)

    minimum = int(settings["minimum_valid_seeds"])
    if len(valid_seeds) < minimum:
        raise L1StageError(
            f"L1 BO-LHS evidence has {len(valid_seeds)} valid common seeds; required={minimum}; "
            f"failed_seeds={failed_seeds}"
        )
    return {
        "schema_version": "l1-stage-result/v1",
        "status": "succeeded",
        "preflight": preflight,
        "optimization_seeds": list(seeds),
        "valid_common_seeds": valid_seeds,
        "failed_seeds": failed_seeds,
        "minimum_valid_common_seeds": minimum,
        "successful_evaluations_per_method": int(settings["total"]),
        "physical_candidate_runs_per_valid_seed": int(settings["initial"])
        + 2 * int(settings["subsequent"]),
        "bo_lhs_evaluations": _relative(root, aggregate_evaluations),
        "bo_lhs_summary": _relative(root, aggregate_summary),
    }


__all__ = [
    "AGGREGATE_EVALUATION_COLUMNS",
    "EVALUATION_COLUMNS",
    "L1_EVALUATION_SCHEMA",
    "L1_SELECTED_SCHEMA",
    "L1_SUMMARY_SCHEMA",
    "L1PreflightError",
    "L1StageError",
    "PARAMETER_NAMES",
    "SUMMARY_COLUMNS",
    "preflight_l1_observation_chains",
    "run_l1_stage",
]
