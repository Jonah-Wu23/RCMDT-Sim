"""Recoverable L2 IES and final A0-A4 simulation orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    canonical_sha256,
    compute_component_hashes,
    compute_manifest_hashes,
    sha256_file,
    validate_mechanism_matrix,
)
from .ies import IESResult, run_ies
from .pipeline import (
    BASELINE_BACKGROUND_PARAMETERS,
    BASELINE_BUS_PARAMETERS,
    ManifestBundle,
    PipelineError,
    build_run_manifest,
    bundle_run_manifest,
    load_protocol_manifest,
    materialize_run_manifest,
    simulation_request_from_bundle,
    software_versions,
    verify_input_hashes,
    write_json_immutable,
)
from .simulation import SimulationResult, execute_simulation
from .sumo_data import (
    LINK_KEY_COLUMNS,
    build_l2_observation_pair,
    extract_simulation_vector,
    real_event_window,
    simulation_link_events,
)


OBSERVATION_SCHEMA = "l2-observations/v1"
L2_STATUS_SCHEMA = "l2-run-status/v1"
BLOCKED_RUN_DISPOSITION_SCHEMA = "blocked-run-disposition/v1"
STAGE_SCHEMA = "paper-ablation-stage/v1"
FINAL_SUMO_SEED_FORMULA = "300000 + 1000*seed + split_index"
L2_MEMBER_SUMO_SEED_FORMULA = "200000 + 10000*seed + 100*iteration + member"
L1_SELECTED_SCHEMA = "l1-selected/v1"
L1_EVALUATION_SCHEMA = "bo-lhs-evaluations/v1"
_EXPECTED_SEEDS = (0, 1, 2, 3, 4)
_SPLIT_INDEX = {"development": 0, "cross_day": 1}
_CONFIGS = {
    "A0": (False, False, "no_l2_input"),
    "A1": (True, False, "no_l2_input"),
    "A2": (False, True, "moving_only"),
    "A3": (True, True, "raw_d2d"),
    "A4": (True, True, "moving_only"),
}
_L2_CONFIGS = ("A2", "A3", "A4")


class AblationStageError(PipelineError):
    """Raised when the L2/final ablation evidence cannot satisfy the protocol."""


@dataclass(frozen=True)
class _L2Outcome:
    config_id: str
    seed: int
    status: str
    run_directory: Path
    final_parameters: Mapping[str, float] | None
    error_summary: str | None
    reused: bool


@dataclass(frozen=True)
class _PreparedFinalRun:
    config_id: str
    seed: int
    split: str
    run_directory: Path
    bundle: ManifestBundle


def _json_status(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv_immutable(path: Path, frame: pd.DataFrame) -> Path:
    rendered = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise AblationStageError(f"Refusing to overwrite non-identical artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def _write_csv_snapshot(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return path


def _path_within(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    path = (root / relative_path).resolve()
    if path == root or root not in path.parents:
        raise AblationStageError(f"Path escapes project root: {relative_path}")
    return path


def _dataset(manifest: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["datasets"] if item.get("id") == dataset_id]
    if len(matches) != 1:
        raise AblationStageError(f"Expected exactly one dataset {dataset_id!r}")
    return matches[0]


def _normal_parameters(
    values: Mapping[str, Any], expected: Mapping[str, float], label: str
) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(expected):
        raise AblationStageError(
            f"{label} keys must be exactly {sorted(expected)}"
        )
    result = {key: float(values[key]) for key in expected}
    if not all(math.isfinite(value) for value in result.values()):
        raise AblationStageError(f"{label} contains non-finite values")
    return result


def _selected_by_seed(
    selected_l1_by_seed: Mapping[int | str, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    selected: dict[int, dict[str, float]] = {}
    for seed in _EXPECTED_SEEDS:
        value = selected_l1_by_seed.get(seed)
        if value is None:
            value = selected_l1_by_seed.get(str(seed))
        if value is None:
            raise AblationStageError(f"Missing frozen L1 parameters for seed={seed}")
        selected[seed] = _normal_parameters(
            value, BASELINE_BUS_PARAMETERS, f"selected_l1_by_seed[{seed}]"
        )
        if selected[seed] == dict(BASELINE_BUS_PARAMETERS):
            raise AblationStageError(
                f"selected_l1_by_seed[{seed}] equals the disabled-layer baseline"
            )
    return selected


def _required_digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise AblationStageError(f"{label} must be a SHA-256 digest")
    return text.lower()


def _required_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AblationStageError(f"{label} must be boolean")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationStageError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise AblationStageError(f"{label} must contain a JSON object: {path}")
    return dict(value)


def _candidate_hash(parameters: Mapping[str, float]) -> str:
    return canonical_sha256(
        {
            "schema_version": "l1-candidate/v1",
            "bus_parameters": {
                name: float(parameters[name]) for name in BASELINE_BUS_PARAMETERS
            },
        }
    )


def _bounded_selected_parameters(
    values: Mapping[str, Any], manifest: Mapping[str, Any], label: str
) -> dict[str, float]:
    parameters = _normal_parameters(values, BASELINE_BUS_PARAMETERS, label)
    bounds = manifest["l1"].get("parameter_bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != set(BASELINE_BUS_PARAMETERS):
        raise AblationStageError("l1.parameter_bounds must cover every bus parameter")
    for name, value in parameters.items():
        interval = bounds[name]
        if (
            not isinstance(interval, Sequence)
            or isinstance(interval, (str, bytes))
            or len(interval) != 2
        ):
            raise AblationStageError(f"l1.parameter_bounds[{name}] must contain two values")
        lower, upper = (float(interval[0]), float(interval[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise AblationStageError(f"l1.parameter_bounds[{name}] is invalid")
        if not lower <= value <= upper:
            raise AblationStageError(
                f"{label}.{name}={value} is outside the frozen range [{lower}, {upper}]"
            )
    return parameters


def validate_selected_l1_sources(
    project_root: Path,
    manifest: Mapping[str, Any],
    selected_l1_by_seed: Mapping[int | str, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Reject L1 parameters that are not traceable to the selected BO output."""

    root = project_root.resolve()
    supplied = _selected_by_seed(selected_l1_by_seed)
    output_root = _path_within(root, str(manifest["outputs"]["run_directory"]))
    expected_budget = int(manifest["l1"]["budget"]["successful_evaluations_per_method"])
    required_columns = {
        "schema_version",
        "optimization_seed",
        "method",
        "evaluation_index",
        "candidate_hash",
        "objective",
        "feasible",
        *BASELINE_BUS_PARAMETERS,
        "parameters_json",
        "run_id",
        "run_directory",
        "stopinfo_path",
        "run_status_path",
        "manifest_hash",
        "provenance_hash",
        "simulation_effective_hash",
        "component_hashes_json",
        "output_hash",
        "status",
    }
    validated: dict[int, dict[str, float]] = {}

    for seed in _EXPECTED_SEEDS:
        seed_directory = output_root / "l1" / f"seed-{seed}"
        selected_path = seed_directory / "selected.json"
        evaluations_path = seed_directory / "evaluations.csv"
        payload = _read_json_object(selected_path, f"L1 selected record for seed={seed}")
        if payload.get("schema_version") != L1_SELECTED_SCHEMA:
            raise AblationStageError(f"seed={seed} selected.json has the wrong schema")
        if (
            int(payload.get("optimization_seed", -1)) != seed
            or payload.get("status") != "succeeded"
        ):
            raise AblationStageError(f"seed={seed} selected.json is not a successful matching seed")
        if int(payload.get("successful_evaluations_per_method", -1)) != expected_budget:
            raise AblationStageError(f"seed={seed} selected.json has the wrong BO budget")
        selection = payload.get("selected_for_l2")
        methods = payload.get("methods")
        if not isinstance(selection, Mapping) or not isinstance(methods, Mapping):
            raise AblationStageError(f"seed={seed} selected.json lacks BO selection records")
        bo_copy = methods.get("BO")
        if not isinstance(bo_copy, Mapping) or canonical_sha256(
            bo_copy
        ) != canonical_sha256(selection):
            raise AblationStageError(f"seed={seed} selected_for_l2 differs from methods.BO")
        if selection.get("method") != "BO" or not _required_bool(
            selection.get("feasible"), f"seed={seed} selected BO feasibility"
        ):
            raise AblationStageError(f"seed={seed} selected_for_l2 is not a feasible BO result")
        selected_parameters = _bounded_selected_parameters(
            selection.get("parameters", {}), manifest, f"seed={seed} selected parameters"
        )
        if selected_parameters != supplied[seed]:
            raise AblationStageError(
                f"seed={seed} supplied L1 parameters differ from selected.json"
            )
        candidate_hash = _required_digest(
            selection.get("candidate_hash"), f"seed={seed} selected candidate_hash"
        )
        if candidate_hash != _candidate_hash(selected_parameters):
            raise AblationStageError(
                f"seed={seed} selected candidate_hash does not match its parameters"
            )
        try:
            evaluation_index = int(selection["evaluation_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AblationStageError(f"seed={seed} selected evaluation_index is invalid") from exc
        if not 1 <= evaluation_index <= expected_budget:
            raise AblationStageError(
                f"seed={seed} selected evaluation_index is outside the BO budget"
            )

        try:
            evaluations = pd.read_csv(
                evaluations_path, dtype=str, keep_default_na=False
            )
        except (OSError, pd.errors.ParserError) as exc:
            raise AblationStageError(f"Cannot read L1 evaluations: {evaluations_path}") from exc
        missing_columns = sorted(required_columns.difference(evaluations.columns))
        if missing_columns:
            raise AblationStageError(
                f"seed={seed} evaluations.csv is missing columns: {missing_columns}"
            )
        bo_rows = evaluations.loc[
            (evaluations["optimization_seed"] == str(seed))
            & (evaluations["method"] == "BO")
            & (evaluations["status"] == "succeeded")
        ].copy()
        try:
            bo_rows["_evaluation_index"] = bo_rows["evaluation_index"].map(int)
            bo_rows["_objective"] = bo_rows["objective"].map(float)
        except (TypeError, ValueError) as exc:
            raise AblationStageError(f"seed={seed} BO evaluations contain invalid numbers") from exc
        if (
            len(bo_rows) != expected_budget
            or set(bo_rows["_evaluation_index"]) != set(range(1, expected_budget + 1))
            or not np.isfinite(bo_rows["_objective"].to_numpy(dtype=float)).all()
        ):
            raise AblationStageError(f"seed={seed} BO evaluations do not satisfy the full budget")
        matches = bo_rows.loc[bo_rows["_evaluation_index"] == evaluation_index]
        if len(matches) != 1:
            raise AblationStageError(f"seed={seed} selected BO evaluation row is not unique")
        row = matches.iloc[0]
        if row["schema_version"] != L1_EVALUATION_SCHEMA or not _required_bool(
            row["feasible"], f"seed={seed} BO row feasibility"
        ):
            raise AblationStageError(
                f"seed={seed} selected BO row is not a successful feasible row"
            )
        feasible_rows = bo_rows.loc[
            bo_rows["feasible"].map(
                lambda value: _required_bool(value, f"seed={seed} BO row feasibility")
            )
        ].sort_values(["_objective", "_evaluation_index"], kind="stable")
        if (
            feasible_rows.empty
            or int(feasible_rows.iloc[0]["_evaluation_index"]) != evaluation_index
        ):
            raise AblationStageError(
                f"seed={seed} selected BO row is not the best feasible candidate"
            )

        row_parameters = _bounded_selected_parameters(
            {name: row[name] for name in BASELINE_BUS_PARAMETERS},
            manifest,
            f"seed={seed} BO evaluation parameters",
        )
        try:
            row_parameters_json = json.loads(row["parameters_json"])
        except json.JSONDecodeError as exc:
            raise AblationStageError(f"seed={seed} BO parameters_json is malformed") from exc
        if (
            row_parameters != selected_parameters
            or _normal_parameters(
                row_parameters_json,
                BASELINE_BUS_PARAMETERS,
                f"seed={seed} BO parameters_json",
            )
            != selected_parameters
            or _required_digest(row["candidate_hash"], f"seed={seed} BO candidate_hash")
            != candidate_hash
        ):
            raise AblationStageError(f"seed={seed} BO evaluation row differs from selected.json")
        if not math.isclose(
            float(row["objective"]), float(selection.get("objective")), rel_tol=0.0, abs_tol=0.0
        ):
            raise AblationStageError(f"seed={seed} selected objective differs from its BO row")

        selection_fields = (
            "run_id",
            "run_directory",
            "manifest_hash",
            "provenance_hash",
            "simulation_effective_hash",
            "output_hash",
        )
        for field in selection_fields:
            if str(selection.get(field)) != str(row[field]):
                raise AblationStageError(
                    f"seed={seed} selected {field} differs from its BO evaluation row"
                )
        manifest_hash = _required_digest(row["manifest_hash"], f"seed={seed} manifest_hash")
        provenance_hash = _required_digest(
            row["provenance_hash"], f"seed={seed} provenance_hash"
        )
        effective_hash = _required_digest(
            row["simulation_effective_hash"], f"seed={seed} simulation_effective_hash"
        )
        output_hash = _required_digest(row["output_hash"], f"seed={seed} output_hash")
        try:
            row_components = json.loads(row["component_hashes_json"])
        except json.JSONDecodeError as exc:
            raise AblationStageError(f"seed={seed} BO component hashes are malformed") from exc
        if not isinstance(row_components, Mapping) or dict(row_components) != dict(
            selection.get("component_hashes", {})
        ):
            raise AblationStageError(f"seed={seed} BO component hashes differ from selected.json")

        run_directory = _path_within(root, str(row["run_directory"]))
        if run_directory != _path_within(root, str(selection["run_directory"])):
            raise AblationStageError(f"seed={seed} selected run directory differs from its BO row")
        run_manifest = _read_json_object(
            run_directory / "run-manifest.json", f"L1 run manifest for seed={seed}"
        )
        if canonical_sha256(run_manifest) != manifest_hash:
            raise AblationStageError(f"seed={seed} L1 run manifest hash mismatch")
        recomputed = compute_manifest_hashes(run_manifest)
        if (
            recomputed["provenance_hash"] != provenance_hash
            or recomputed["simulation_effective_hash"] != effective_hash
            or dict(recomputed["component_hashes"]) != dict(row_components)
        ):
            raise AblationStageError(f"seed={seed} L1 run manifest provenance mismatch")
        if (
            int(run_manifest.get("seed", -1)) != seed
            or str(run_manifest.get("run_id")) != str(row["run_id"])
            or _normal_parameters(
                run_manifest.get("l1", {}).get("bus_parameters", {}),
                BASELINE_BUS_PARAMETERS,
                f"seed={seed} run-manifest bus parameters",
            )
            != selected_parameters
        ):
            raise AblationStageError(f"seed={seed} L1 run manifest selects another candidate")

        status_path = _path_within(root, str(row["run_status_path"]))
        if status_path != run_directory / "run-status.json":
            raise AblationStageError(f"seed={seed} BO run-status path is not run-local")
        status = _read_json_object(status_path, f"L1 run status for seed={seed}")
        if (
            status.get("status") != "succeeded"
            or str(status.get("run_id")) != str(row["run_id"])
            or str(status.get("manifest_hash", "")).lower() != manifest_hash
            or str(status.get("provenance_hash", "")).lower() != provenance_hash
            or str(status.get("simulation_effective_hash", "")).lower() != effective_hash
            or dict(status.get("component_hashes", {})) != dict(row_components)
        ):
            raise AblationStageError(f"seed={seed} L1 run status provenance mismatch")
        relative_stopinfo = status.get("stopinfo_relative_path")
        if not isinstance(relative_stopinfo, str) or not relative_stopinfo:
            raise AblationStageError(f"seed={seed} L1 run status lacks stopinfo_relative_path")
        stopinfo_path = _path_within(root, str(row["stopinfo_path"]))
        expected_stopinfo = (run_directory / relative_stopinfo).resolve()
        if stopinfo_path != expected_stopinfo or not stopinfo_path.is_file():
            raise AblationStageError(f"seed={seed} BO stopinfo path is missing or inconsistent")
        if (
            sha256_file(stopinfo_path) != output_hash
            or status.get("produced_artifact_hashes", {}).get("stopinfo.xml") != output_hash
        ):
            raise AblationStageError(f"seed={seed} selected output_hash differs from stopinfo.xml")
        validated[seed] = selected_parameters

    return validated


def _key_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame[["observation_id", *LINK_KEY_COLUMNS]].itertuples(index=False):
        records.append(
            {
                "observation_id": int(row.observation_id),
                "route": str(row.route),
                "bound": str(row.bound),
                "from_seq": int(row.from_seq),
                "to_seq": int(row.to_seq),
            }
        )
    return records


def _observation_descriptor(
    project_root: Path, path: Path, frame: pd.DataFrame
) -> dict[str, Any]:
    schema = [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]
    return {
        "path": path.relative_to(project_root).as_posix(),
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "schema_hash": canonical_sha256(schema),
        "key_hash": canonical_sha256(_key_records(frame)),
        "content_hash": sha256_file(path),
    }


def _build_observations(
    project_root: Path,
    manifest: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    development = _dataset(manifest, "development_events")
    time_window = development["time_window"]
    real_events = real_event_window(
        _path_within(project_root, str(development["path"])),
        observation_date=str(development["observation_date"]),
        window_start_hkt=str(time_window["start"]),
        window_end_hkt=str(time_window["end"]),
    )
    index_descriptor = manifest["simulator"].get("m11_observation_index")
    if not isinstance(index_descriptor, Mapping) or not index_descriptor.get("path"):
        raise AblationStageError("simulator.m11_observation_index is required")
    raw, moving = build_l2_observation_pair(
        real_events,
        _path_within(project_root, str(index_descriptor["path"])),
    )
    if len(raw) != 11 or len(moving) != 5:
        raise AblationStageError(
            f"Frozen L2 dimensions must be raw_d2d=11 and moving_only=5; "
            f"got {len(raw)} and {len(moving)}"
        )
    frames = {"raw_d2d": raw, "moving_only": moving}
    observation_dir = output_root / "l2" / "observations"
    descriptors: dict[str, Any] = {}
    for semantic, frame in frames.items():
        path = _write_csv_immutable(observation_dir / f"{semantic}.csv", frame)
        descriptors[semantic] = _observation_descriptor(project_root, path, frame)
    contract: dict[str, Any] = {
        "schema_version": OBSERVATION_SCHEMA,
        "source_dataset": {
            "id": development["id"],
            "path": development["path"],
            "sha256": development["sha256"],
            "observation_date": development["observation_date"],
            "timezone": development["timezone"],
            "time_window": deepcopy(time_window),
        },
        "observation_index": {
            "path": index_descriptor["path"],
            "sha256": index_descriptor["sha256"],
        },
        "semantics": descriptors,
    }
    contract["contract_hash"] = canonical_sha256(contract)
    write_json_immutable(observation_dir / "observation-contract.json", contract)
    return frames, contract


def _l2_protocol(manifest: Mapping[str, Any]) -> dict[str, Any]:
    l2 = manifest["l2"]
    components = list(l2["state_components"])
    if components != list(BASELINE_BACKGROUND_PARAMETERS):
        raise AblationStageError(
            "L2 state_components must follow capacityFactor, minGap_background, impatience"
        )
    priors = l2["priors"]
    return {
        "components": components,
        "prior_mean": [float(priors[name]["mean"]) for name in components],
        "prior_std": [float(priors[name]["std"]) for name in components],
        "bounds": [list(map(float, priors[name]["bounds"])) for name in components],
        "ensemble_size": int(l2["ensemble_size"]),
        "iterations": int(l2["iterations"]),
        "damping": float(l2["damping"]),
        "variance_floor": float(l2.get("observation_variance_floor", 1.0)),
        "priors": deepcopy(priors),
        "ensemble_seed_schedule": l2["ensemble_seed_schedule"],
    }


def _l2_contract_hash(
    config_id: str,
    seed: int,
    bus_parameters: Mapping[str, float],
    semantic: str,
    observation_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "config_id": config_id,
            "seed": seed,
            "bus_parameters": dict(bus_parameters),
            "semantic": semantic,
            "observation": observation_contract["semantics"][semantic],
            "components": protocol["components"],
            "priors": protocol["priors"],
            "ensemble_size": protocol["ensemble_size"],
            "iterations": protocol["iterations"],
            "damping": protocol["damping"],
            "variance_floor": protocol["variance_floor"],
            "ensemble_seed_schedule": protocol["ensemble_seed_schedule"],
            "sumo_seed_formula": L2_MEMBER_SUMO_SEED_FORMULA,
            "member_index_base": 0,
        }
    )


def _load_reusable_l2(
    run_directory: Path,
    contract_hash: str,
    components: Sequence[str],
) -> dict[str, float] | None:
    status_path = run_directory / "l2-status.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if status.get("status") != "succeeded" or status.get("contract_hash") != contract_hash:
        return None
    artifacts = status.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        return None
    for relative, expected in artifacts.items():
        path = run_directory / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise AblationStageError(f"Existing L2 artifact hash mismatch: {path}")
    final_path = run_directory / "final_parameters.json"
    try:
        payload = json.loads(final_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationStageError(f"Cannot reuse L2 parameters: {final_path}") from exc
    return _normal_parameters(payload.get("parameters", {}), dict.fromkeys(components, 0.0), "L2 final parameters")


def _member_seed(seed: int, iteration: int, member: int) -> int:
    return 200000 + 10000 * seed + 100 * iteration + member


def _execute(
    execute_fn: Callable[..., SimulationResult],
    request: Any,
    run_directory: Path,
    *,
    sumo_binary: str,
    max_attempts: int,
    post_output_validator: Callable[[Path], Any],
) -> SimulationResult:
    result = execute_fn(
        request,
        run_directory,
        sumo_binary=sumo_binary,
        max_attempts=max_attempts,
        post_output_validator=post_output_validator,
    )
    if result.reused:
        try:
            post_output_validator(result.stopinfo_path)
        except Exception:
            result = execute_fn(
                request,
                run_directory,
                sumo_binary=sumo_binary,
                max_attempts=max_attempts,
                allow_reuse=False,
                post_output_validator=post_output_validator,
            )
    return result


def _run_l2_task(
    *,
    project_root: Path,
    output_root: Path,
    base_manifest: Mapping[str, Any],
    software: Mapping[str, str],
    selected_l1: Mapping[int, Mapping[str, float]],
    observation_frames: Mapping[str, pd.DataFrame],
    observation_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    config_id: str,
    seed: int,
    timeout: float,
    sumo_binary: str,
    execute_fn: Callable[..., SimulationResult],
) -> _L2Outcome:
    run_directory = output_root / "l2" / config_id / f"seed-{seed}"
    l1_enabled, _, semantic = _CONFIGS[config_id]
    bus_parameters = (
        selected_l1[seed] if l1_enabled else dict(BASELINE_BUS_PARAMETERS)
    )
    contract_hash = _l2_contract_hash(
        config_id,
        seed,
        bus_parameters,
        semantic,
        observation_contract,
        protocol,
    )
    try:
        reusable = _load_reusable_l2(
            run_directory, contract_hash, protocol["components"]
        )
        if reusable is not None:
            return _L2Outcome(
                config_id, seed, "succeeded", run_directory, reusable, None, True
            )
        _json_status(
            run_directory / "l2-status.json",
            {
                "schema_version": L2_STATUS_SCHEMA,
                "config_id": config_id,
                "seed": seed,
                "status": "running",
                "contract_hash": contract_hash,
                "error_summary": None,
                "artifact_hashes": {},
            },
        )
        frame = observation_frames[semantic]
        observation = frame["mean_speed_kmh"].to_numpy(dtype=float)
        observation_std = frame["std_speed_kmh"].to_numpy(dtype=float)
        member_metadata: list[dict[str, Any]] = []
        route_stop = _path_within(
            project_root, str(_dataset(base_manifest, "route_stop_distance")["path"])
        )

        def simulate(member: np.ndarray, iteration: int, member_index: int) -> np.ndarray:
            member_parameters = {
                name: float(value)
                for name, value in zip(protocol["components"], member, strict=True)
            }
            sumo_seed = _member_seed(seed, iteration, member_index)
            member_directory = (
                run_directory
                / "members"
                / f"iteration-{iteration:02d}"
                / f"member-{member_index:02d}"
            )
            run_id = (
                f"l2-{config_id}-development-seed-{seed}-"
                f"iteration-{iteration:02d}-member-{member_index:02d}"
            )
            manifest = build_run_manifest(
                base_manifest,
                project_root=project_root,
                run_directory=member_directory,
                run_id=run_id,
                config_id=config_id,
                method_id=f"{config_id.lower()}-ies-member",
                split="development",
                seed=seed,
                sumo_seed=sumo_seed,
                bus_parameters=bus_parameters,
                background_parameters=member_parameters,
                observation_semantic=semantic,
                l1_enabled=l1_enabled,
                l2_enabled=True,
                software=software,
                timeout_seconds=timeout,
                observation_contract=observation_contract["semantics"][semantic],
                parameter_sources={
                    "bus": "frozen_l1" if l1_enabled else "design_baseline",
                    "background": "l2_ies_ensemble_member",
                },
            )
            manifest["l2"]["ensemble_member"] = {
                "iteration": iteration,
                "member": member_index,
                "member_index_base": 0,
            }
            manifest["simulator"]["seed_schedule"] = {
                "formula": L2_MEMBER_SUMO_SEED_FORMULA,
                "optimization_seed": seed,
                "iteration": iteration,
                "member": member_index,
                "value": sumo_seed,
            }
            bundle = bundle_run_manifest(manifest)
            materialize_run_manifest(member_directory, bundle)

            def validate_member_output(stopinfo_path: Path) -> None:
                member_events = simulation_link_events(
                    stopinfo_path,
                    route_stop,
                    window_start_s=0,
                    window_end_s=3600,
                )
                if member_events.empty:
                    raise AblationStageError(
                        f"L2 member produced no link events: {run_id}"
                    )
                extract_simulation_vector(member_events, frame)

            result = _execute(
                execute_fn,
                simulation_request_from_bundle(project_root, bundle),
                member_directory,
                sumo_binary=sumo_binary,
                max_attempts=int(base_manifest["simulator"]["max_attempts"]),
                post_output_validator=validate_member_output,
            )
            events = simulation_link_events(
                result.stopinfo_path,
                route_stop,
                window_start_s=0,
                window_end_s=3600,
            )
            if events.empty:
                raise AblationStageError(f"L2 member produced no link events: {run_id}")
            vector = extract_simulation_vector(events, frame)
            member_metadata.append(
                {
                    "iteration": iteration,
                    "member": member_index,
                    "sumo_seed": sumo_seed,
                    "run_id": run_id,
                    "reused": bool(result.reused),
                    "manifest_hash": bundle.manifest_hash,
                    "simulation_effective_hash": bundle.simulation_effective_hash,
                    "stopinfo_sha256": result.output_hash,
                }
            )
            return vector

        result: IESResult = run_ies(
            prior_mean=protocol["prior_mean"],
            prior_std=protocol["prior_std"],
            bounds=protocol["bounds"],
            observation=observation,
            observation_std=observation_std,
            seed=seed,
            simulate=simulate,
            ensemble_size=int(protocol["ensemble_size"]),
            iterations=int(protocol["iterations"]),
            damping=float(protocol["damping"]),
            variance_floor=float(protocol["variance_floor"]),
        )
        final_parameters = {
            name: float(value)
            for name, value in zip(
                protocol["components"], result.final_mean, strict=True
            )
        }
        final_payload = {
            "schema_version": "l2-final-parameters/v1",
            "config_id": config_id,
            "seed": seed,
            "contract_hash": contract_hash,
            "parameters": final_parameters,
        }
        write_json_immutable(run_directory / "final_parameters.json", final_payload)

        iteration_rows: list[dict[str, Any]] = []
        for record in result.iterations:
            row: dict[str, Any] = {
                "config_id": config_id,
                "seed": seed,
                "iteration": record.iteration,
                "ensemble_seed": record.ensemble_seed,
                "rmse": record.rmse,
                "clipped_components": record.clipped_components,
            }
            for name, before, after in zip(
                protocol["components"],
                record.mean_before,
                record.mean_after,
                strict=True,
            ):
                row[f"{name}_before"] = before
                row[f"{name}_after"] = after
            iteration_rows.append(row)
        _write_csv_immutable(
            run_directory / "iterations.csv", pd.DataFrame(iteration_rows)
        )

        metadata = {
            (int(item["iteration"]), int(item["member"])): item
            for item in member_metadata
        }
        ensemble_rows: list[dict[str, Any]] = []
        simulation_rows: list[dict[str, Any]] = []
        for iteration_index, (ensemble, simulations) in enumerate(
            zip(result.ensembles, result.simulations, strict=True), start=1
        ):
            for member_index, (parameters, simulated) in enumerate(
                zip(ensemble, simulations, strict=True)
            ):
                member_info = metadata[(iteration_index, member_index)]
                parameter_row = {
                    "config_id": config_id,
                    "seed": seed,
                    **member_info,
                }
                for name, value in zip(
                    protocol["components"], parameters, strict=True
                ):
                    parameter_row[name] = float(value)
                ensemble_rows.append(parameter_row)
                for observation_row, value in zip(
                    frame.itertuples(index=False), simulated, strict=True
                ):
                    simulation_rows.append(
                        {
                            "config_id": config_id,
                            "seed": seed,
                            "iteration": iteration_index,
                            "member": member_index,
                            "observation_id": int(observation_row.observation_id),
                            "route": str(observation_row.route),
                            "bound": str(observation_row.bound),
                            "from_seq": int(observation_row.from_seq),
                            "to_seq": int(observation_row.to_seq),
                            "simulated_speed_kmh": float(value),
                        }
                    )
        _write_csv_immutable(
            run_directory / "ensemble_parameters.csv", pd.DataFrame(ensemble_rows)
        )
        _write_csv_immutable(
            run_directory / "ensemble_simulations.csv", pd.DataFrame(simulation_rows)
        )
        artifact_names = (
            "final_parameters.json",
            "iterations.csv",
            "ensemble_parameters.csv",
            "ensemble_simulations.csv",
        )
        artifact_hashes = {
            name: sha256_file(run_directory / name) for name in artifact_names
        }
        _json_status(
            run_directory / "l2-status.json",
            {
                "schema_version": L2_STATUS_SCHEMA,
                "config_id": config_id,
                "seed": seed,
                "status": "succeeded",
                "contract_hash": contract_hash,
                "error_summary": None,
                "artifact_hashes": artifact_hashes,
            },
        )
        return _L2Outcome(
            config_id,
            seed,
            "succeeded",
            run_directory,
            final_parameters,
            None,
            False,
        )
    except Exception as exc:
        _json_status(
            run_directory / "l2-status.json",
            {
                "schema_version": L2_STATUS_SCHEMA,
                "config_id": config_id,
                "seed": seed,
                "status": "failed",
                "contract_hash": contract_hash,
                "error_summary": str(exc),
                "artifact_hashes": {},
            },
        )
        return _L2Outcome(
            config_id, seed, "failed", run_directory, None, str(exc), False
        )


def _final_seed(seed: int, split: str) -> int:
    return 300000 + 1000 * seed + _SPLIT_INDEX[split]


def _prepare_final_run(
    *,
    project_root: Path,
    output_root: Path,
    base_manifest: Mapping[str, Any],
    software: Mapping[str, str],
    selected_l1: Mapping[int, Mapping[str, float]],
    l2_parameters: Mapping[tuple[str, int], Mapping[str, float]],
    observation_contract: Mapping[str, Any],
    config_id: str,
    seed: int,
    split: str,
    timeout: float,
) -> _PreparedFinalRun:
    l1_enabled, l2_enabled, semantic = _CONFIGS[config_id]
    bus_parameters = (
        selected_l1[seed] if l1_enabled else dict(BASELINE_BUS_PARAMETERS)
    )
    background_parameters = (
        l2_parameters[(config_id, seed)]
        if l2_enabled
        else dict(BASELINE_BACKGROUND_PARAMETERS)
    )
    run_directory = output_root / "ablation" / "final" / config_id / split / f"seed-{seed}"
    sumo_seed = _final_seed(seed, split)
    run_id = f"final-{config_id}-{split}-seed-{seed}"
    manifest = build_run_manifest(
        base_manifest,
        project_root=project_root,
        run_directory=run_directory,
        run_id=run_id,
        config_id=config_id,
        method_id=config_id.lower(),
        split=split,
        seed=seed,
        sumo_seed=sumo_seed,
        bus_parameters=bus_parameters,
        background_parameters=background_parameters,
        observation_semantic=semantic,
        l1_enabled=l1_enabled,
        l2_enabled=l2_enabled,
        software=software,
        timeout_seconds=timeout,
        observation_contract=(
            observation_contract["semantics"][semantic] if l2_enabled else None
        ),
        parameter_sources={
            "bus": "frozen_l1" if l1_enabled else "design_baseline",
            "background": (
                f"l2/{config_id}/seed-{seed}/final_parameters.json"
                if l2_enabled
                else "design_baseline"
            ),
        },
    )
    manifest["simulator"]["seed_schedule"] = {
        "formula": FINAL_SUMO_SEED_FORMULA,
        "optimization_seed": seed,
        "split": split,
        "split_index": _SPLIT_INDEX[split],
        "value": sumo_seed,
    }
    bundle = bundle_run_manifest(manifest)
    materialize_run_manifest(run_directory, bundle)
    return _PreparedFinalRun(config_id, seed, split, run_directory, bundle)


def _base_final_row(prepared: _PreparedFinalRun) -> dict[str, Any]:
    bundle = prepared.bundle
    components = dict(bundle.component_hashes)
    manifest = bundle.manifest
    return {
        "run_id": manifest["run_id"],
        "config_id": prepared.config_id,
        "seed": prepared.seed,
        "split": prepared.split,
        "sumo_seed": int(manifest["simulator"]["seed"]),
        "l1_enabled": bool(manifest["mechanisms"]["l1_enabled"]),
        "l2_enabled": bool(manifest["mechanisms"]["l2_enabled"]),
        "observation_semantic": manifest["l2"]["observation_semantic"],
        "status": "pending",
        "error_summary": None,
        "reused": False,
        "attempt": None,
        "duration_s": None,
        "event_count": None,
        "stopinfo_path": None,
        "stopinfo_sha256": None,
        "manifest_path": f"{manifest['outputs']['run_directory']}/run-manifest.json",
        "manifest_hash": bundle.manifest_hash,
        "provenance_hash": bundle.provenance_hash,
        "simulation_effective_hash": bundle.simulation_effective_hash,
        "bus_parameters_hash": components["bus_parameters"],
        "background_parameters_hash": components["background_parameters"],
        "observation_semantic_hash": components["observation_semantic"],
        "simulator_inputs_hash": components["simulator_inputs"],
    }


def _execute_final_run(
    prepared: _PreparedFinalRun,
    *,
    project_root: Path,
    route_stop_path: Path,
    sumo_binary: str,
    max_attempts: int,
    execute_fn: Callable[..., SimulationResult],
) -> dict[str, Any]:
    row = _base_final_row(prepared)
    try:
        def validate_final_output(stopinfo_path: Path) -> None:
            validated_events = simulation_link_events(
                stopinfo_path,
                route_stop_path,
                window_start_s=0,
                window_end_s=3600,
            )
            if validated_events.empty:
                raise AblationStageError(
                    f"Final simulation produced no valid link events: {row['run_id']}"
                )

        result = _execute(
            execute_fn,
            simulation_request_from_bundle(project_root, prepared.bundle),
            prepared.run_directory,
            sumo_binary=sumo_binary,
            max_attempts=max_attempts,
            post_output_validator=validate_final_output,
        )
        events = simulation_link_events(
            result.stopinfo_path,
            route_stop_path,
            window_start_s=0,
            window_end_s=3600,
        )
        if events.empty:
            raise AblationStageError(
                f"Final simulation produced no valid link events: {row['run_id']}"
            )
        row.update(
            {
                "status": "succeeded",
                "reused": bool(result.reused),
                "attempt": int(result.attempt),
                "duration_s": float(result.duration_s),
                "event_count": int(len(events)),
                "stopinfo_path": result.stopinfo_path.relative_to(project_root).as_posix(),
                "stopinfo_sha256": result.output_hash,
            }
        )
    except Exception as exc:
        row.update({"status": "failed", "error_summary": str(exc)})
        status_path = prepared.run_directory / "run-status.json"
        if not status_path.exists():
            _json_status(
                status_path,
                {
                    "schema_version": "run-status/v1",
                    "run_id": row["run_id"],
                    "status": "failed",
                    "attempt": 0,
                    "started_at": None,
                    "ended_at": None,
                    "exit_code": None,
                    "error_summary": str(exc),
                    "manifest_hash": row["manifest_hash"],
                    "provenance_hash": row["provenance_hash"],
                    "simulation_effective_hash": row["simulation_effective_hash"],
                    "component_hashes": compute_component_hashes(prepared.bundle.manifest),
                    "produced_artifact_hashes": {},
                },
            )
    return row


def _blocked_row(
    output_root: Path, config_id: str, seed: int, split: str, error: str
) -> dict[str, Any]:
    run_directory = output_root / "ablation" / "final" / config_id / split / f"seed-{seed}"
    _json_status(
        run_directory / "blocked-disposition.json",
        {
            "schema_version": BLOCKED_RUN_DISPOSITION_SCHEMA,
            "planned_run_id": f"final-{config_id}-{split}-seed-{seed}",
            "config_id": config_id,
            "seed": seed,
            "split": split,
            "sumo_seed": _final_seed(seed, split),
            "status": "blocked",
            "reason": error,
            "created_artifacts": {},
        },
    )
    return {
        "run_id": f"final-{config_id}-{split}-seed-{seed}",
        "config_id": config_id,
        "seed": seed,
        "split": split,
        "sumo_seed": _final_seed(seed, split),
        "l1_enabled": _CONFIGS[config_id][0],
        "l2_enabled": _CONFIGS[config_id][1],
        "observation_semantic": _CONFIGS[config_id][2],
        "status": "blocked",
        "error_summary": error,
        "reused": False,
        "attempt": None,
        "duration_s": None,
        "event_count": None,
        "stopinfo_path": None,
        "stopinfo_sha256": None,
        "manifest_path": None,
        "manifest_hash": None,
        "provenance_hash": None,
        "simulation_effective_hash": None,
        "bus_parameters_hash": None,
        "background_parameters_hash": None,
        "observation_semantic_hash": None,
        "simulator_inputs_hash": None,
    }


def run_ablation_stage(
    project_root: Path,
    *,
    selected_l1_by_seed: Mapping[int | str, Mapping[str, float]],
    workers: int,
    timeout: float | None = None,
    sumo_binary: str = "sumo",
    base_manifest: Mapping[str, Any] | None = None,
    software: Mapping[str, str] | None = None,
    execute_fn: Callable[..., SimulationResult] = execute_simulation,
    ensemble_size: int | None = None,
    iterations: int | None = None,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    """Run recoverable L2 calibration followed by the 50 final simulations."""

    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise AblationStageError("workers must be an integer from 1 through 8")
    root = project_root.resolve()
    manifest = deepcopy(
        dict(base_manifest) if base_manifest is not None else load_protocol_manifest(root)
    )
    seeds = tuple(int(seed) for seed in manifest.get("ablation", {}).get("seeds", ()))
    if seeds != _EXPECTED_SEEDS:
        raise AblationStageError("Ablation seeds must be exactly 0,1,2,3,4")
    if ensemble_size is not None:
        manifest["l2"]["ensemble_size"] = int(ensemble_size)
    if iterations is not None:
        manifest["l2"]["iterations"] = int(iterations)
    protocol = _l2_protocol(manifest)
    if protocol["ensemble_size"] < 3 or protocol["iterations"] < 1:
        raise AblationStageError("IES requires at least three members and one iteration")
    if not math.isclose(protocol["damping"], 0.3, rel_tol=0.0, abs_tol=1e-12):
        raise AblationStageError("The frozen L2 damping must be 0.3")
    timeout_s = float(
        manifest["simulator"]["timeout_seconds"] if timeout is None else timeout
    )
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise AblationStageError("timeout must be a positive finite number")
    selected = validate_selected_l1_sources(root, manifest, selected_l1_by_seed)
    if verify_inputs:
        verify_input_hashes(root, manifest)
    versions = dict(software) if software is not None else software_versions(
        root, sumo_binary=sumo_binary
    )
    output_root = _path_within(root, str(manifest["outputs"]["run_directory"]))
    observation_frames, observation_contract = _build_observations(
        root, manifest, output_root
    )

    l2_outcomes: list[_L2Outcome] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_l2_task,
                project_root=root,
                output_root=output_root,
                base_manifest=manifest,
                software=versions,
                selected_l1=selected,
                observation_frames=observation_frames,
                observation_contract=observation_contract,
                protocol=protocol,
                config_id=config_id,
                seed=seed,
                timeout=timeout_s,
                sumo_binary=sumo_binary,
                execute_fn=execute_fn,
            )
            for seed in _EXPECTED_SEEDS
            for config_id in _L2_CONFIGS
        ]
        for future in as_completed(futures):
            l2_outcomes.append(future.result())
    l2_outcomes.sort(key=lambda item: (item.seed, item.config_id))
    l2_parameters = {
        (item.config_id, item.seed): dict(item.final_parameters)
        for item in l2_outcomes
        if item.status == "succeeded" and item.final_parameters is not None
    }

    prepared: list[_PreparedFinalRun] = []
    final_rows: list[dict[str, Any]] = []
    for seed in _EXPECTED_SEEDS:
        missing = [config for config in _L2_CONFIGS if (config, seed) not in l2_parameters]
        if missing:
            error = f"Missing successful L2 results for seed={seed}: {', '.join(missing)}"
            for split in _SPLIT_INDEX:
                for config_id in _CONFIGS:
                    final_rows.append(_blocked_row(output_root, config_id, seed, split, error))
            continue
        for split in _SPLIT_INDEX:
            group = [
                _prepare_final_run(
                    project_root=root,
                    output_root=output_root,
                    base_manifest=manifest,
                    software=versions,
                    selected_l1=selected,
                    l2_parameters=l2_parameters,
                    observation_contract=observation_contract,
                    config_id=config_id,
                    seed=seed,
                    split=split,
                    timeout=timeout_s,
                )
                for config_id in _CONFIGS
            ]
            expected_sumo_seed = _final_seed(seed, split)
            if any(
                int(item.bundle.manifest["simulator"]["seed"]) != expected_sumo_seed
                for item in group
            ):
                raise AblationStageError(
                    f"Final SUMO seed formula mismatch for seed={seed}, split={split}"
                )
            validate_mechanism_matrix([item.bundle.manifest for item in group])
            prepared.extend(group)

    route_stop_path = _path_within(
        root, str(_dataset(manifest, "route_stop_distance")["path"])
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _execute_final_run,
                item,
                project_root=root,
                route_stop_path=route_stop_path,
                sumo_binary=sumo_binary,
                max_attempts=int(manifest["simulator"]["max_attempts"]),
                execute_fn=execute_fn,
            )
            for item in prepared
        ]
        for future in as_completed(futures):
            final_rows.append(future.result())
    final_rows.sort(key=lambda row: (int(row["seed"]), str(row["split"]), str(row["config_id"])))
    if len(final_rows) != 50:
        raise AblationStageError(f"Expected 50 final run rows, got {len(final_rows)}")
    runs_path = _write_csv_snapshot(
        output_root / "ablation" / "ablation_runs.csv", pd.DataFrame(final_rows)
    )
    common_successful_seeds = [
        seed
        for seed in _EXPECTED_SEEDS
        if all(
            any(
                row["seed"] == seed
                and row["split"] == split
                and row["config_id"] == config_id
                and row["status"] == "succeeded"
                for row in final_rows
            )
            for split in _SPLIT_INDEX
            for config_id in _CONFIGS
        )
    ]
    stage_status = (
        "succeeded"
        if len(common_successful_seeds) == len(_EXPECTED_SEEDS)
        else "partial"
        if len(common_successful_seeds) >= 3
        else "blocked"
    )
    summary = {
        "schema_version": STAGE_SCHEMA,
        "status": stage_status,
        "workers": workers,
        "l2_ensemble_size": protocol["ensemble_size"],
        "l2_iterations": protocol["iterations"],
        "l2_damping": protocol["damping"],
        "l2_member_sumo_seed_formula": L2_MEMBER_SUMO_SEED_FORMULA,
        "l2_member_index_base": 0,
        "final_sumo_seed_formula": FINAL_SUMO_SEED_FORMULA,
        "split_indices": dict(_SPLIT_INDEX),
        "observation_contract_path": (
            output_root / "l2" / "observations" / "observation-contract.json"
        ).relative_to(root).as_posix(),
        "observation_contract_hash": observation_contract["contract_hash"],
        "ablation_runs_path": runs_path.relative_to(root).as_posix(),
        "l2_runs": [
            {
                "config_id": item.config_id,
                "seed": item.seed,
                "status": item.status,
                "reused": item.reused,
                "run_directory": item.run_directory.relative_to(root).as_posix(),
                "error_summary": item.error_summary,
            }
            for item in l2_outcomes
        ],
        "final_run_count": len(final_rows),
        "successful_final_run_count": sum(
            row["status"] == "succeeded" for row in final_rows
        ),
        "common_successful_seeds": common_successful_seeds,
        "final_runs": final_rows,
    }
    _json_status(output_root / "ablation" / "stage-summary.json", summary)
    if len(common_successful_seeds) < 3:
        raise AblationStageError(
            "Fewer than three seeds succeeded for every A0-A4 configuration on both splits"
        )
    return summary


__all__ = [
    "AblationStageError",
    "BLOCKED_RUN_DISPOSITION_SCHEMA",
    "FINAL_SUMO_SEED_FORMULA",
    "L2_MEMBER_SUMO_SEED_FORMULA",
    "OBSERVATION_SCHEMA",
    "STAGE_SCHEMA",
    "run_ablation_stage",
    "validate_selected_l1_sources",
]
