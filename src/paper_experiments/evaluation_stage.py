"""Immutable evaluation and reporting stage for final camera-ready runs.

The stage consumes the final A0--A4 run index and never invokes SUMO.  It
validates every source manifest, status file, and stopinfo hash before deriving
the common Rule-C-clean event population used by metrics and reporting.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .audit import (
    LINK_KEY_COLUMNS as AUDIT_LINK_KEY_COLUMNS,
    aggregate_link_hour,
    apply_isolation_forest,
    apply_mad,
    apply_quantile,
    fit_isolation_forest,
    fit_mad,
    fit_quantile,
    normalize_eligible_events,
    retention_summary,
    rule_c_flags,
)
from .contracts import (
    canonical_sha256,
    compute_component_hashes,
    compute_manifest_hashes,
    hash_path,
    sha256_file,
    validate_mechanism_matrix,
    validate_paper_manifest,
)
from .figures import (
    AUDIT_METRICS_COLUMNS,
    FIG2_CONTAMINATION_COLUMNS,
    FIG2_TRAJECTORY_COLUMNS,
    FIG3_SENSITIVITY_COLUMNS,
    FIG4_CDF_COLUMNS,
    validate_camera_ready_inputs,
)
from .irn import (
    build_link_to_irn_mapping,
    compute_irn_contradiction,
    load_irn_window_speeds,
    select_irn_window_files,
)
from .metrics import (
    METRIC_COLUMNS,
    full_window_ks,
    make_metric_row,
    metrics_to_long_form,
    worst_window_ks,
)
from .pipeline import PipelineError, load_protocol_manifest
from .sumo_data import (
    LINK_KEY_COLUMNS,
    load_stopinfo,
    real_event_window,
    simulation_link_events,
    strict_rule_c_event_mask,
)


EVALUATION_STAGE_SCHEMA = "paper-evaluation-stage/v1"
TABLE_I_SCHEMA = "table-i/v1"
_CONFIGS = ("A0", "A1", "A2", "A3", "A4")
_SPLITS = ("development", "cross_day")
_METHOD_LABELS = {
    "rule_c": "Fixed Rule C",
    "mad": "MAD",
    "isolation_forest": "Isolation Forest",
    "quantile_fallback": "Quantile Fallback",
}
_ADAPTIVE_METHOD_IDS = frozenset({"isolation_forest", "quantile_fallback"})
_CONFIGURATION_LABELS = {
    "A0": "Zero-shot",
    "A1": "BO-only",
    "A2": "IES-only",
    "A3": "Raw-RCMDT",
    "A4": "Full-RCMDT",
}
_RUN_COLUMNS = (
    "run_id",
    "config_id",
    "seed",
    "split",
    "sumo_seed",
    "l1_enabled",
    "l2_enabled",
    "observation_semantic",
    "status",
    "error_summary",
    "reused",
    "attempt",
    "duration_s",
    "event_count",
    "stopinfo_path",
    "stopinfo_sha256",
    "manifest_path",
    "manifest_hash",
    "provenance_hash",
    "simulation_effective_hash",
    "bus_parameters_hash",
    "background_parameters_hash",
    "observation_semantic_hash",
    "simulator_inputs_hash",
)
_TABLE_COLUMNS = (
    "schema_version",
    "config_id",
    "configuration",
    "n_seeds",
    "ks_speed_development_mean",
    "ks_speed_development_std",
    "worst_15min_ks_development_mean",
    "worst_15min_ks_development_std",
    "ks_speed_cross_day_mean",
    "ks_speed_cross_day_std",
    "worst_15min_ks_cross_day_mean",
    "worst_15min_ks_cross_day_std",
    "n_real_development",
    "n_real_cross_day",
    "n_sim_development_mean",
    "n_sim_development_std",
    "n_sim_cross_day_mean",
    "n_sim_cross_day_std",
    "status",
)


class EvaluationStageError(PipelineError):
    """Raised when immutable final-run evidence cannot support reporting."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationStageError(f"Cannot read JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationStageError(f"JSON source must be an object: {path}")
    return value


def _path_within(project_root: Path, value: str | Path) -> Path:
    root = project_root.resolve()
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if path == root or root not in path.parents:
        raise EvaluationStageError(f"Path escapes project root: {value}")
    return path


def _write_csv_immutable(path: Path, frame: pd.DataFrame) -> Path:
    rendered = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise EvaluationStageError(f"Refusing to overwrite non-identical artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise EvaluationStageError(f"Refusing to overwrite non-identical artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def _dataset(manifest: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["datasets"] if item.get("id") == dataset_id]
    if len(matches) != 1:
        raise EvaluationStageError(f"Expected exactly one dataset {dataset_id!r}")
    return matches[0]


def _verify_declared_path(project_root: Path, descriptor: Mapping[str, Any], label: str) -> Path:
    if not descriptor.get("path") or not descriptor.get("sha256"):
        raise EvaluationStageError(f"{label} requires path and sha256")
    path = _path_within(project_root, str(descriptor["path"]))
    actual = hash_path(path)
    expected = str(descriptor["sha256"]).lower()
    if actual.lower() != expected:
        raise EvaluationStageError(
            f"Source hash mismatch for {label}: expected {expected}, got {actual}"
        )
    return path


def _verify_evaluation_sources(
    project_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    result = {
        dataset_id: _verify_declared_path(
            project_root, _dataset(manifest, dataset_id), f"dataset.{dataset_id}"
        )
        for dataset_id in (
            "development_events",
            "cross_day_events",
            "route_stop_distance",
            "development_irn",
            "cross_day_irn",
        )
    }
    simulator = manifest["simulator"]
    for key in ("observation_index", "link_to_irn_mapping"):
        descriptor = simulator.get(key)
        if not isinstance(descriptor, Mapping):
            raise EvaluationStageError(f"manifest.simulator.{key} must be an object")
        result[key] = _verify_declared_path(
            project_root, descriptor, f"simulator.{key}"
        )
    return result


def _validate_evaluation_protocol(manifest: Mapping[str, Any]) -> None:
    expected_evaluation = {
        "full_window_min_samples_per_source": 20,
        "subwindow_duration_s": 900,
        "subwindow_step_s": 60,
        "subwindow_min_samples_per_source": 5,
        "common_seed_target": 5,
        "common_seed_minimum": 3,
    }
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise EvaluationStageError("manifest.evaluation must be an object")
    for field, expected in expected_evaluation.items():
        if evaluation.get(field) != expected:
            raise EvaluationStageError(
                f"manifest.evaluation.{field} must be {expected}"
            )
    expected_rule_c = {
        "travel_time_gt_s": 325.0,
        "speed_lt_kmh": 5.0,
        "distance_lte_m": 1500.0,
    }
    conditions = manifest.get("audit", {}).get("conditions")
    if not isinstance(conditions, Mapping):
        raise EvaluationStageError("manifest.audit.conditions must be an object")
    for field, expected in expected_rule_c.items():
        value = conditions.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationStageError(f"manifest.audit.conditions.{field} must be numeric")
        if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
            raise EvaluationStageError(
                f"manifest.audit.conditions.{field} must be {expected}"
            )


def _normalise_integer_column(frame: pd.DataFrame, column: str) -> None:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise EvaluationStageError(f"ablation_runs.{column} must contain integers")
    frame[column] = numeric.astype(int)


def _validate_run_sources(
    project_root: Path,
    runs_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[int, ...], dict[tuple[str, int, str], dict[str, Any]]]:
    if not runs_path.is_file():
        raise EvaluationStageError(f"Missing immutable ablation run index: {runs_path}")
    try:
        runs = pd.read_csv(runs_path)
    except Exception as exc:
        raise EvaluationStageError(f"Cannot read ablation run index: {runs_path}") from exc
    if tuple(runs.columns) != _RUN_COLUMNS:
        raise EvaluationStageError(
            "ablation_runs.csv columns differ from the final-run contract"
        )
    if runs.empty:
        raise EvaluationStageError("ablation_runs.csv is empty")
    for column in ("seed", "sumo_seed"):
        _normalise_integer_column(runs, column)
    if runs.duplicated(["config_id", "seed", "split"]).any():
        raise EvaluationStageError("ablation_runs.csv contains duplicate config/seed/split rows")
    if set(runs["config_id"].astype(str)) != set(_CONFIGS):
        raise EvaluationStageError("ablation_runs.csv must contain exactly A0-A4")
    if set(runs["split"].astype(str)) != set(_SPLITS):
        raise EvaluationStageError("ablation_runs.csv must contain development and cross_day")
    declared_seeds = tuple(int(value) for value in manifest["ablation"]["seeds"])
    expected_keys = {
        (config, seed, split)
        for config in _CONFIGS
        for seed in declared_seeds
        for split in _SPLITS
    }
    actual_keys = set(
        zip(runs["config_id"], runs["seed"], runs["split"], strict=True)
    )
    if actual_keys != expected_keys:
        raise EvaluationStageError("ablation_runs.csv does not cover the declared Cartesian design")
    common_seeds = tuple(
        seed
        for seed in declared_seeds
        if all(
            bool(
                (
                    (runs["seed"] == seed)
                    & (runs["config_id"] == config)
                    & (runs["split"] == split)
                    & (runs["status"] == "succeeded")
                ).any()
            )
            for config in _CONFIGS
            for split in _SPLITS
        )
    )
    if not 3 <= len(common_seeds) <= 5:
        raise EvaluationStageError(
            "Final evaluation requires three to five seeds successful for every A0-A4/split run"
        )

    manifests: dict[tuple[str, int, str], dict[str, Any]] = {}
    successful = runs.loc[runs["seed"].isin(common_seeds)].copy()
    for row in successful.to_dict("records"):
        key = (str(row["config_id"]), int(row["seed"]), str(row["split"]))
        manifest_path = _path_within(project_root, str(row["manifest_path"]))
        run_manifest = _read_json(manifest_path)
        validate_paper_manifest(run_manifest)
        if canonical_sha256(run_manifest) != str(row["manifest_hash"]):
            raise EvaluationStageError(f"Run manifest hash mismatch for {key}")
        if (
            str(run_manifest.get("config_id")) != key[0]
            or int(run_manifest.get("seed")) != key[1]
            or str(run_manifest.get("split")) != key[2]
            or str(run_manifest.get("run_id")) != str(row["run_id"])
        ):
            raise EvaluationStageError(f"Run manifest identity mismatch for {key}")
        computed = compute_manifest_hashes(run_manifest)
        components = dict(computed["component_hashes"])
        expected_hashes = {
            "provenance_hash": computed["provenance_hash"],
            "simulation_effective_hash": computed["simulation_effective_hash"],
            "bus_parameters_hash": components["bus_parameters"],
            "background_parameters_hash": components["background_parameters"],
            "observation_semantic_hash": components["observation_semantic"],
            "simulator_inputs_hash": components["simulator_inputs"],
        }
        for column, expected in expected_hashes.items():
            if str(row[column]) != str(expected):
                raise EvaluationStageError(f"{column} mismatch for {key}")
        hashes_path = manifest_path.parent / "run-manifest-hashes.json"
        hashes = _read_json(hashes_path)
        if (
            hashes.get("manifest_hash") != row["manifest_hash"]
            or hashes.get("provenance_hash") != row["provenance_hash"]
            or hashes.get("simulation_effective_hash") != row["simulation_effective_hash"]
            or hashes.get("component_hashes") != compute_component_hashes(run_manifest)
        ):
            raise EvaluationStageError(f"run-manifest-hashes.json mismatch for {key}")
        stopinfo_path = _path_within(project_root, str(row["stopinfo_path"]))
        stopinfo_hash = sha256_file(stopinfo_path)
        if stopinfo_hash != str(row["stopinfo_sha256"]):
            raise EvaluationStageError(f"stopinfo hash mismatch for {key}")
        status = _read_json(manifest_path.parent / "run-status.json")
        status_stopinfo = manifest_path.parent / str(status.get("stopinfo_relative_path", ""))
        if (
            status.get("status") != "succeeded"
            or status.get("manifest_hash") != row["manifest_hash"]
            or status.get("provenance_hash") != row["provenance_hash"]
            or status.get("simulation_effective_hash") != row["simulation_effective_hash"]
            or status.get("produced_artifact_hashes", {}).get("stopinfo.xml") != stopinfo_hash
            or status_stopinfo.resolve() != stopinfo_path
        ):
            raise EvaluationStageError(f"run-status.json mismatch for {key}")
        manifests[key] = run_manifest
    for seed in common_seeds:
        for split in _SPLITS:
            validate_mechanism_matrix(
                [manifests[(config, seed, split)] for config in _CONFIGS]
            )
    return runs, common_seeds, manifests


def _route_pairs(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs = {(str(item["route"]), str(item["direction"])) for item in manifest["routes"]}
    if not pairs:
        raise EvaluationStageError("manifest.routes contains no evaluation route")
    return pairs


def _load_real_events(
    path: Path,
    split: str,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    contract = manifest["splits"][split]
    raw = real_event_window(
        path,
        observation_date=str(contract["date"]),
        window_start_hkt=str(contract["window_start_hkt"]),
        window_end_hkt=str(contract["window_end_hkt"]),
    )
    pairs = _route_pairs(manifest)
    eligible = normalize_eligible_events(
        raw,
        routes={route for route, _ in pairs},
        directions={bound for _, bound in pairs},
        timezone=str(manifest.get("timezone", "Asia/Hong_Kong")),
    )
    pair_mask = pd.Series(
        list(zip(eligible["route"].astype(str), eligible["bound"].astype(str), strict=True)),
        index=eligible.index,
    ).isin(pairs)
    eligible = eligible.loc[pair_mask].reset_index(drop=True)
    start = pd.Timestamp(
        f"{contract['date']} {contract['window_start_hkt']}",
        tz=str(manifest.get("timezone", "Asia/Hong_Kong")),
    )
    eligible["event_time_sec"] = (
        eligible["departure_ts"] - start
    ).dt.total_seconds()
    eligible["dist_m"] = eligible["distance_m"].astype(float)
    if eligible.empty:
        raise EvaluationStageError(f"No eligible original D2D events for split={split}")
    return eligible


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str, int, int]]:
    return {
        (str(row.route), str(row.bound), int(row.from_seq), int(row.to_seq))
        for row in frame.loc[:, LINK_KEY_COLUMNS].itertuples(index=False)
    }


def _filter_keys(
    frame: pd.DataFrame, keys: set[tuple[str, str, int, int]]
) -> pd.DataFrame:
    values = pd.Series(
        [
            (str(row.route), str(row.bound), int(row.from_seq), int(row.to_seq))
            for row in frame.loc[:, LINK_KEY_COLUMNS].itertuples(index=False)
        ],
        index=frame.index,
    )
    return frame.loc[values.isin(keys)].reset_index(drop=True)


def _retained_rule_c_keys(records: pd.DataFrame) -> set[tuple[str, str, int, int]]:
    flags = rule_c_flags(records)
    clean = records.loc[~flags, LINK_KEY_COLUMNS]
    return _key_set(clean)


def _load_simulations(
    project_root: Path,
    runs: pd.DataFrame,
    common_seeds: Sequence[int],
    route_stop_path: Path,
) -> tuple[
    dict[tuple[str, int, str], pd.DataFrame],
    dict[tuple[str, int, str], pd.DataFrame],
]:
    events: dict[tuple[str, int, str], pd.DataFrame] = {}
    stops: dict[tuple[str, int, str], pd.DataFrame] = {}
    selected = runs.loc[runs["seed"].isin(common_seeds)]
    for row in selected.to_dict("records"):
        key = (str(row["config_id"]), int(row["seed"]), str(row["split"]))
        stopinfo_path = _path_within(project_root, str(row["stopinfo_path"]))
        event_frame = simulation_link_events(
            stopinfo_path, route_stop_path, window_start_s=0, window_end_s=3600
        )
        if int(row["event_count"]) != len(event_frame):
            raise EvaluationStageError(f"Simulation event_count mismatch for {key}")
        if event_frame.empty:
            raise EvaluationStageError(f"Simulation has no link events for {key}")
        events[key] = event_frame
        stops[key] = load_stopinfo(stopinfo_path, route_stop_path)
    return events, stops


def _common_clean_populations(
    real_events: Mapping[str, pd.DataFrame],
    simulations: Mapping[tuple[str, int, str], pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, set[tuple[str, str, int, int]]]]:
    clean_events: dict[str, pd.DataFrame] = {}
    common_keys: dict[str, set[tuple[str, str, int, int]]] = {}
    for split in _SPLITS:
        records = aggregate_link_hour(real_events[split])
        retained = _retained_rule_c_keys(records)
        simulation_sets = [
            _key_set(frame) for key, frame in simulations.items() if key[2] == split
        ]
        if not simulation_sets:
            raise EvaluationStageError(f"No successful simulations for split={split}")
        common = set(retained)
        for keys in simulation_sets:
            common.intersection_update(keys)
        if not common:
            raise EvaluationStageError(f"No Rule-C-clean common link keys for split={split}")
        filtered = _filter_keys(real_events[split], common)
        if len(filtered) < 20:
            raise EvaluationStageError(
                f"split={split} has fewer than 20 real events on common clean keys"
            )
        clean_events[split] = filtered
        common_keys[split] = common
    return clean_events, common_keys


def _require_succeeded(result: Any, context: str) -> None:
    if not result.succeeded:
        raise EvaluationStageError(f"{context} failed sample contract: {result.error_summary}")


def _paper_metrics(
    manifest: Mapping[str, Any],
    runs: pd.DataFrame,
    common_seeds: Sequence[int],
    run_manifests: Mapping[tuple[str, int, str], Mapping[str, Any]],
    real_clean: Mapping[str, pd.DataFrame],
    common_keys: Mapping[str, set[tuple[str, str, int, int]]],
    simulations: Mapping[tuple[str, int, str], pd.DataFrame],
) -> pd.DataFrame:
    run_index = {
        (str(row["config_id"]), int(row["seed"]), str(row["split"])): row
        for row in runs.loc[runs["seed"].isin(common_seeds)].to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for config in _CONFIGS:
        for seed in common_seeds:
            for split in _SPLITS:
                key = (config, int(seed), split)
                real = real_clean[split]
                sim = _filter_keys(simulations[key], common_keys[split])
                if len(sim) < 20 or _key_set(sim) != common_keys[split]:
                    raise EvaluationStageError(
                        f"{key} lacks 20 samples or one common clean link key"
                    )
                source_row = run_index[key]
                run_manifest = run_manifests[key]
                for domain, column in (
                    ("speed", "speed_kmh"),
                    ("travel_time", "travel_time_s"),
                ):
                    full = full_window_ks(
                        real[column],
                        sim[column],
                        window_start=0.0,
                        window_end=3600.0,
                    )
                    _require_succeeded(full, f"{key}/{domain}/full-window")
                    worst = worst_window_ks(
                        real[column],
                        real["event_time_sec"],
                        sim[column],
                        sim["event_time_sec"],
                        window_start=0.0,
                        window_end=3600.0,
                    )
                    _require_succeeded(worst, f"{key}/{domain}/worst-window")
                    for metric_name, result in (
                        (f"ks_{domain}", full),
                        (f"worst_15min_ks_{domain}", worst),
                    ):
                        rows.append(
                            make_metric_row(
                                result,
                                experiment_id=str(manifest["experiment_id"]),
                                config_id=config,
                                method_id=str(run_manifest["method_id"]),
                                seed=int(seed),
                                split=split,
                                metric_name=metric_name,
                                domain=domain,
                                unit="dimensionless",
                                n_link_keys=len(common_keys[split]),
                                manifest_hash=str(source_row["manifest_hash"]),
                                simulation_output_hash=str(source_row["stopinfo_sha256"]),
                                evaluator_version=EVALUATION_STAGE_SCHEMA,
                            )
                        )
    return metrics_to_long_form(rows)


def _sample_std(values: pd.Series) -> float:
    result = float(values.astype(float).std(ddof=1))
    if not math.isfinite(result):
        raise EvaluationStageError("Sample standard deviation requires at least two finite seeds")
    return result


def _table_i(metrics: pd.DataFrame, common_seeds: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in _CONFIGS:
        item: dict[str, Any] = {
            "schema_version": TABLE_I_SCHEMA,
            "config_id": config,
            "configuration": _CONFIGURATION_LABELS[config],
            "n_seeds": len(common_seeds),
            "status": "succeeded",
        }
        for split in _SPLITS:
            full = metrics.loc[
                (metrics["config_id"] == config)
                & (metrics["split"] == split)
                & (metrics["metric_name"] == "ks_speed")
            ]
            worst = metrics.loc[
                (metrics["config_id"] == config)
                & (metrics["split"] == split)
                & (metrics["metric_name"] == "worst_15min_ks_speed")
            ]
            if len(full) != len(common_seeds) or len(worst) != len(common_seeds):
                raise EvaluationStageError(f"Incomplete Table I metrics for {config}/{split}")
            item[f"ks_speed_{split}_mean"] = float(full["value"].mean())
            item[f"ks_speed_{split}_std"] = _sample_std(full["value"])
            item[f"worst_15min_ks_{split}_mean"] = float(worst["value"].mean())
            item[f"worst_15min_ks_{split}_std"] = _sample_std(worst["value"])
            item[f"n_real_{split}"] = int(full["n_real"].iloc[0])
            item[f"n_sim_{split}_mean"] = float(full["n_sim"].mean())
            item[f"n_sim_{split}_std"] = _sample_std(full["n_sim"])
        rows.append(item)
    return pd.DataFrame(rows, columns=list(_TABLE_COLUMNS))


def _fig2_contamination(events: pd.DataFrame) -> pd.DataFrame:
    records = aggregate_link_hour(events)
    record_flags = rule_c_flags(records)
    flags_by_key = {
        (str(row.route), str(row.bound), int(row.from_seq), int(row.to_seq)): bool(flagged)
        for row, flagged in zip(
            records.loc[:, LINK_KEY_COLUMNS].itertuples(index=False),
            record_flags.to_numpy(dtype=bool),
            strict=True,
        )
    }
    event_keys = [
        (str(row.route), str(row.bound), int(row.from_seq), int(row.to_seq))
        for row in events.loc[:, LINK_KEY_COLUMNS].itertuples(index=False)
    ]
    frame = pd.DataFrame(
        {
            "schema_version": "fig2-contamination/v1",
            "event_id": [f"development-{index:06d}" for index in range(len(events))],
            "route": events["route"].astype(str).to_numpy(),
            "bound": events["bound"].astype(str).to_numpy(),
            "from_seq": events["from_seq"].astype(int).to_numpy(),
            "to_seq": events["to_seq"].astype(int).to_numpy(),
            "travel_time_s": events["travel_time_s"].astype(float).to_numpy(),
            "speed_kmh": events["speed_kmh"].astype(float).to_numpy(),
            "distance_m": events["dist_m"].astype(float).to_numpy(),
            "rule_c_flagged": [flags_by_key[key] for key in event_keys],
        }
    )
    return frame.loc[:, list(FIG2_CONTAMINATION_COLUMNS)]


def _full_link_events(stops: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in stops.groupby("vehicle_id", sort=True):
        ordered = group.sort_values(["started", "seq"]).reset_index(drop=True)
        for current, following in zip(
            ordered.iloc[:-1].itertuples(index=False),
            ordered.iloc[1:].itertuples(index=False),
            strict=True,
        ):
            if current.route != following.route or current.bound != following.bound:
                continue
            if int(following.seq) != int(current.seq) + 1:
                continue
            moving = float(following.started) - float(current.ended)
            full = float(following.arrival_time) - float(current.arrival_time)
            distance = float(following.cum_dist_m) - float(current.cum_dist_m)
            if moving > 0 and full > 0 and distance > 0:
                rows.append(
                    {
                        "route": str(current.route),
                        "bound": str(current.bound),
                        "from_seq": int(current.seq),
                        "to_seq": int(following.seq),
                        "moving_time_s": moving,
                        "full_time_s": full,
                        "dist_m": distance,
                    }
                )
    return pd.DataFrame(rows)


def _longest_chain(frame: pd.DataFrame) -> pd.DataFrame:
    best: pd.DataFrame | None = None
    for _, group in frame.groupby(["route", "bound"], sort=True):
        ordered = group.sort_values(["from_seq", "to_seq"]).reset_index(drop=True)
        start = 0
        for index in range(1, len(ordered) + 1):
            boundary = index == len(ordered) or int(ordered.iloc[index - 1]["to_seq"]) != int(
                ordered.iloc[index]["from_seq"]
            )
            if boundary:
                candidate = ordered.iloc[start:index].copy()
                if best is None or len(candidate) > len(best):
                    best = candidate
                start = index
    if best is None or best.empty:
        raise EvaluationStageError("No aligned link chain is available for Fig2 trajectory")
    return best


def _fig2_trajectory(
    real_clean: pd.DataFrame,
    a0_stops: pd.DataFrame,
    common_keys: set[tuple[str, str, int, int]],
    reference_seed: int,
) -> pd.DataFrame:
    observed = (
        _filter_keys(real_clean, common_keys)
        .groupby(LINK_KEY_COLUMNS, as_index=False)
        .agg(observed_time_s=("travel_time_s", "mean"), dist_m=("dist_m", "median"))
    )
    simulated = _full_link_events(a0_stops)
    if simulated.empty:
        raise EvaluationStageError("A0 reference has no full trajectory link events")
    simulated = (
        _filter_keys(simulated, common_keys)
        .groupby(LINK_KEY_COLUMNS, as_index=False)
        .agg(
            simulated_moving_s=("moving_time_s", "mean"),
            simulated_full_s=("full_time_s", "mean"),
            simulated_dist_m=("dist_m", "median"),
        )
    )
    aligned = observed.merge(simulated, on=LINK_KEY_COLUMNS, how="inner", validate="one_to_one")
    aligned = _longest_chain(aligned)
    distances = [0.0, *np.cumsum(aligned["dist_m"].to_numpy(dtype=float)).tolist()]
    route = str(aligned.iloc[0]["route"])
    bound = str(aligned.iloc[0]["bound"])
    trajectory_id = f"A0-seed-{reference_seed}"
    rows: list[dict[str, Any]] = []
    series = (
        ("observed", "traffic_only", "observed_time_s"),
        ("simulated", "full", "simulated_full_s"),
        ("simulated", "traffic_only", "simulated_moving_s"),
    )
    for source, basis, column in series:
        times = [0.0, *np.cumsum(aligned[column].to_numpy(dtype=float)).tolist()]
        for point_index, (time_value, distance_value) in enumerate(zip(times, distances, strict=True)):
            rows.append(
                {
                    "schema_version": "fig2-trajectory/v1",
                    "split": "development",
                    "route": route,
                    "bound": bound,
                    "trajectory_id": trajectory_id,
                    "source": source,
                    "time_basis": basis,
                    "point_index": point_index,
                    "cumulative_time_s": float(time_value),
                    "cumulative_distance_m": float(distance_value),
                }
            )
    return pd.DataFrame(rows, columns=list(FIG2_TRAJECTORY_COLUMNS))


def _audit_decisions(
    records_by_split: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, Any]]:
    development = records_by_split["development"]
    mad = fit_mad(development)
    result: dict[str, dict[str, pd.Series]] = {}
    for split in _SPLITS:
        records = records_by_split[split]
        result[split] = {
            "rule_c": rule_c_flags(records),
            "mad": apply_mad(records, mad)["flagged"].astype(bool),
        }

    isolation_operation = "fit"
    try:
        isolation = fit_isolation_forest(development)
        isolation_decisions: dict[str, pd.Series] = {}
        for split in _SPLITS:
            isolation_operation = f"apply:{split}"
            isolation_decisions[split] = apply_isolation_forest(
                records_by_split[split], isolation
            )["flagged"].astype(bool)
    except Exception as exc:
        exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        isolation_failure = {
            "operation": isolation_operation,
            "exception_type": exception_type,
            "message": str(exc),
            "error_summary": f"{exception_type}: {exc}",
        }
    else:
        for split in _SPLITS:
            result[split]["isolation_forest"] = isolation_decisions[split]
        return result, {
            "mad": mad,
            "adaptive_method_id": "isolation_forest",
            "isolation_forest": isolation,
        }

    quantile = fit_quantile(development)
    for split in _SPLITS:
        result[split]["quantile_fallback"] = apply_quantile(
            records_by_split[split], quantile
        ).astype(bool)
    return result, {
        "mad": mad,
        "adaptive_method_id": "quantile_fallback",
        "quantile_fallback": quantile,
        "isolation_forest_failure": isolation_failure,
    }


def _audit_method_ids(
    decisions: Mapping[str, Mapping[str, pd.Series]],
) -> tuple[str, str, str]:
    expected_splits = set(_SPLITS)
    if set(decisions) != expected_splits:
        raise EvaluationStageError("Audit decisions must cover both frozen splits")
    development_methods = set(decisions["development"])
    adaptive_methods = development_methods.intersection(_ADAPTIVE_METHOD_IDS)
    expected_methods = {"rule_c", "mad", *adaptive_methods}
    if len(adaptive_methods) != 1 or development_methods != expected_methods:
        raise EvaluationStageError(
            "Audit decisions require Rule C, MAD, and exactly one adaptive method"
        )
    for split in _SPLITS:
        if set(decisions[split]) != development_methods:
            raise EvaluationStageError(
                "Audit decisions must use one frozen method set on both splits"
            )
    adaptive_method = next(iter(adaptive_methods))
    return "rule_c", "mad", adaptive_method


def _audit_key_payload(key: Sequence[Any]) -> dict[str, Any]:
    if len(key) != len(AUDIT_LINK_KEY_COLUMNS):
        raise EvaluationStageError(f"Unexpected audit link key length: {len(key)}")
    window_start = pd.Timestamp(key[4])
    return {
        "route": str(key[0]),
        "bound": str(key[1]),
        "from_seq": int(key[2]),
        "to_seq": int(key[3]),
        "window_start": window_start.isoformat(),
    }


def _audit_support_universes(
    records_by_split: Mapping[str, pd.DataFrame],
    a0_simulations: Mapping[str, pd.DataFrame],
) -> dict[str, set[tuple[str, str, int, int]]]:
    """Freeze one A0-supported raw key universe before method decisions apply."""

    universes: dict[str, set[tuple[str, str, int, int]]] = {}
    for split in _SPLITS:
        raw_keys = _key_set(records_by_split[split])
        if len(raw_keys) != len(records_by_split[split]):
            raise EvaluationStageError(
                f"Audit split={split} must contain one link-hour record per raw link key"
            )
        supported = raw_keys.intersection(_key_set(a0_simulations[split]))
        if not supported:
            raise EvaluationStageError(
                f"Audit split={split} has no raw link key supported by the fixed A0 reference"
            )
        universes[split] = supported
    return universes


def _audit_universe_payload(
    records_by_split: Mapping[str, pd.DataFrame],
    support_universes: Mapping[str, set[tuple[str, str, int, int]]],
) -> dict[str, Any]:
    """Serialize exact raw, supported, and unsupported link-hour universes."""

    result: dict[str, Any] = {}
    for split in _SPLITS:
        records = records_by_split[split]
        raw_keys = _key_set(records)
        supported = set(support_universes[split])
        if not supported.issubset(raw_keys):
            raise EvaluationStageError(
                f"Audit split={split} A0-supported universe contains a non-raw key"
            )
        unsupported = raw_keys.difference(supported)

        def payloads(keys: set[tuple[str, str, int, int]]) -> list[dict[str, Any]]:
            selected = _filter_keys(records, keys)
            ordered = selected.sort_values(
                [*LINK_KEY_COLUMNS, "window_start"], kind="stable"
            )
            return [
                _audit_key_payload(tuple(row))
                for row in ordered.loc[:, AUDIT_LINK_KEY_COLUMNS].itertuples(
                    index=False, name=None
                )
            ]

        raw_payload = payloads(raw_keys)
        supported_payload = payloads(supported)
        unsupported_payload = payloads(unsupported)
        result[split] = {
            "n_eligible_raw_link_keys": len(raw_keys),
            "n_a0_supported_raw_link_keys": len(supported),
            "n_a0_unsupported_raw_link_keys": len(unsupported),
            "raw_eligible_link_keys": raw_payload,
            "a0_supported_raw_link_keys": supported_payload,
            "a0_unsupported_raw_link_keys": unsupported_payload,
        }
    return result


def _audit_manifest(
    records_by_split: Mapping[str, pd.DataFrame],
    decisions: Mapping[str, Mapping[str, pd.Series]],
    fitted_models: Mapping[str, Any],
    irn_evidence: Mapping[str, Any],
    support_universes: Mapping[str, set[tuple[str, str, int, int]]],
    reference_seed: int,
) -> dict[str, Any]:
    mad = fitted_models["mad"]
    method_ids = _audit_method_ids(decisions)
    adaptive_method_id = method_ids[-1]
    rule_c_parameters = {
        "travel_time_operator": ">",
        "travel_time_s": 325.0,
        "speed_operator": "<",
        "speed_kmh": 5.0,
        "distance_operator": "<=",
        "distance_m": 1500.0,
    }
    methods: dict[str, dict[str, Any]] = {
        "rule_c": {
            "method_label": _METHOD_LABELS["rule_c"],
            "fit_split": "predeclared_physical_rule",
            "fitted_statistics": rule_c_parameters,
            "model_serialization": {
                "format": "canonical-json",
                "sha256": canonical_sha256(rule_c_parameters),
            },
            "package": {
                "name": "paper_experiments",
                "version": EVALUATION_STAGE_SCHEMA,
            },
        },
        "mad": {
            "method_label": _METHOD_LABELS["mad"],
            "fit_split": "development",
            "fitted_statistics": mad.frozen_parameters,
            "model_serialization": {
                "format": "canonical-json",
                "sha256": mad.model_hash,
            },
            "package": {"name": "numpy", "version": mad.package_version},
        },
    }
    if adaptive_method_id == "isolation_forest":
        isolation = fitted_models["isolation_forest"]
        methods["isolation_forest"] = {
            "method_label": _METHOD_LABELS["isolation_forest"],
            "fit_split": "development",
            "fitted_statistics": isolation.frozen_parameters,
            "model_serialization": {
                "format": "canonical-json/isolation-forest-model-state-v1",
                "sha256": isolation.model_hash,
            },
            "model_state_sha256": isolation.model_hash,
            "package": {
                "name": "scikit-learn",
                "version": isolation.package_version,
            },
        }
    else:
        quantile = fitted_models["quantile_fallback"]
        isolation_failure = fitted_models.get("isolation_forest_failure")
        if not isinstance(isolation_failure, Mapping):
            raise EvaluationStageError(
                "Quantile fallback requires the original Isolation Forest failure"
            )
        methods["quantile_fallback"] = {
            "method_label": _METHOD_LABELS["quantile_fallback"],
            "fit_split": "development",
            "fitted_statistics": quantile.frozen_parameters,
            "model_serialization": {
                "format": "canonical-json/quantile-fallback-model-state-v1",
                "sha256": quantile.model_hash,
            },
            "model_state_sha256": quantile.model_hash,
            "package": {
                "name": "numpy",
                "version": quantile.package_version,
            },
            "fallback_from_method_id": "isolation_forest",
            "isolation_forest_failure": dict(isolation_failure),
        }
    for method in method_ids:
        method_payload = methods[method]
        split_payload: dict[str, Any] = {}
        for split in _SPLITS:
            retention = retention_summary(
                records_by_split[split],
                decisions[split][method],
                key_columns=AUDIT_LINK_KEY_COLUMNS,
            )
            split_payload[split] = {
                "n_eligible_raw_link_keys": retention.n_eligible_raw_link_keys,
                "n_flagged_link_keys": retention.n_flagged_link_keys,
                "n_clean_link_keys": retention.n_clean_link_keys,
                "retention_rate": retention.retention_rate,
                "flagged_keys": [_audit_key_payload(key) for key in retention.flagged_keys],
                "retained_keys": [_audit_key_payload(key) for key in retention.retained_keys],
            }
        method_payload["decisions"] = split_payload
    return {
        "schema_version": "audit-manifest/v2",
        "fit_split": "development",
        "cross_day_model_refit": False,
        "development_fit_record_count": int(len(records_by_split["development"])),
        "comparison_support": {
            "definition": (
                "raw eligible link keys intersected with the fixed A0 reference "
                "simulation keys before any audit-method retention decision"
            ),
            "a0_reference_config_id": "A0",
            "a0_reference_seed": int(reference_seed),
            "splits": _audit_universe_payload(records_by_split, support_universes),
        },
        "irn_evidence": dict(irn_evidence),
        "methods": methods,
    }


def _retained_four_keys(
    records: pd.DataFrame, decisions: pd.Series
) -> tuple[Any, set[tuple[str, str, int, int]]]:
    retention = retention_summary(records, decisions, key_columns=AUDIT_LINK_KEY_COLUMNS)
    keys = {
        (str(item[0]), str(item[1]), int(item[2]), int(item[3]))
        for item in retention.retained_keys
    }
    return retention, keys


def _irn_inputs(
    manifest: Mapping[str, Any], source_paths: Mapping[str, Path]
) -> tuple[
    Mapping[tuple[str, str, int, int], Sequence[int]],
    dict[str, Mapping[int, float]],
    dict[str, Any],
]:
    mapping, mapping_hash = build_link_to_irn_mapping(
        source_paths["observation_index"], source_paths["link_to_irn_mapping"]
    )
    speeds: dict[str, Mapping[int, float]] = {}
    split_evidence: dict[str, Any] = {}
    timezone = str(manifest.get("timezone", "Asia/Hong_Kong"))
    for split in _SPLITS:
        contract = manifest["splits"][split]
        selected_files = select_irn_window_files(
            source_paths[f"{split}_irn"],
            observation_date=str(contract["date"]),
            window_start=str(contract["window_start_hkt"]),
            window_end=str(contract["window_end_hkt"]),
            timezone=timezone,
        )
        speeds[split] = load_irn_window_speeds(
            source_paths[f"{split}_irn"],
            observation_date=str(contract["date"]),
            window_start=str(contract["window_start_hkt"]),
            window_end=str(contract["window_end_hkt"]),
            timezone=timezone,
        )
        all_files = tuple(sorted(source_paths[f"{split}_irn"].glob("irnAvgSpeed-all-*.xml")))
        selected_descriptors = [
            {"name": path.name, "sha256": sha256_file(path)} for path in selected_files
        ]
        split_evidence[split] = {
            "irn_tree_sha256": hash_path(source_paths[f"{split}_irn"]),
            "selected_file_count": len(selected_files),
            "excluded_file_count": len(all_files) - len(selected_files),
            "selected_files": selected_descriptors,
            "selected_files_sha256": canonical_sha256(selected_descriptors),
            "segment_median_count": len(speeds[split]),
        }
    unique_segments = {segment for values in mapping.values() for segment in values}
    evidence = {
        "link_to_irn_mapping_sha256": mapping_hash,
        "link_count": len(mapping),
        "unique_segment_count": len(unique_segments),
        "observation_index_sha256": sha256_file(source_paths["observation_index"]),
        "link_edge_mapping_sha256": sha256_file(source_paths["link_to_irn_mapping"]),
        "splits": split_evidence,
    }
    return mapping, speeds, evidence


def _audit_metrics(
    records_by_split: Mapping[str, pd.DataFrame],
    real_events: Mapping[str, pd.DataFrame],
    a0_simulations: Mapping[str, pd.DataFrame],
    decisions: Mapping[str, Mapping[str, pd.Series]],
    support_universes: Mapping[str, set[tuple[str, str, int, int]]],
    irn_mapping: Mapping[tuple[str, str, int, int], Sequence[int]],
    irn_speeds: Mapping[str, Mapping[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in _audit_method_ids(decisions):
        for split in _SPLITS:
            records = records_by_split[split]
            flags = decisions[split][method]
            retention, retained_keys = _retained_four_keys(records, flags)
            supported = set(support_universes[split])
            evaluation_keys = retained_keys.intersection(supported)
            if not evaluation_keys:
                raise EvaluationStageError(
                    f"audit/{method}/{split} retains no A0-supported link key"
                )
            real = _filter_keys(real_events[split], evaluation_keys)
            sim = _filter_keys(a0_simulations[split], evaluation_keys)
            if _key_set(sim) != evaluation_keys:
                raise EvaluationStageError(
                    f"A0 reference misses a frozen supported {method}/{split} link key"
                )
            full = full_window_ks(real["speed_kmh"], sim["speed_kmh"])
            _require_succeeded(full, f"audit/{method}/{split}/full-window")
            worst = worst_window_ks(
                real["speed_kmh"],
                real["event_time_sec"],
                sim["speed_kmh"],
                sim["event_time_sec"],
                window_start=0.0,
                window_end=3600.0,
            )
            _require_succeeded(worst, f"audit/{method}/{split}/worst-window")
            contradiction = compute_irn_contradiction(
                records.loc[flags, [*LINK_KEY_COLUMNS, "speed_median"]],
                irn_mapping,
                irn_speeds[split],
            )
            if (
                contradiction["denominator"] == 0
                and contradiction["numerator"] != 0
            ):
                raise EvaluationStageError(
                    f"audit/{method}/{split} has a nonzero IRN numerator with zero denominator"
                )
            rows.append(
                {
                    "schema_version": "audit-metrics/v2",
                    "method_id": method,
                    "method_label": _METHOD_LABELS[method],
                    "split": split,
                    "n_eligible_raw_link_keys": retention.n_eligible_raw_link_keys,
                    "n_a0_supported_raw_link_keys": len(supported),
                    "n_a0_unsupported_raw_link_keys": (
                        retention.n_eligible_raw_link_keys - len(supported)
                    ),
                    "n_retained_link_keys": retention.n_clean_link_keys,
                    "n_evaluation_link_keys": len(evaluation_keys),
                    "retention_rate": float(retention.retention_rate),
                    "ks_speed": float(full.value),
                    "worst_15min_ks": float(worst.value),
                    "n_real": int(full.n_real),
                    "n_sim": int(full.n_sim),
                    "irn_numerator": int(contradiction["numerator"]),
                    "irn_denominator": int(contradiction["denominator"]),
                    "unmatched_flagged": int(contradiction["unmatched_flagged"]),
                    "status": "succeeded",
                }
            )
    return pd.DataFrame(rows, columns=list(AUDIT_METRICS_COLUMNS))


def _fig3_sensitivity(
    development_records: pd.DataFrame,
    development_events: pd.DataFrame,
    a0_simulation: pd.DataFrame,
    supported_keys: set[tuple[str, str, int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for travel_time in (275.0, 325.0, 375.0):
        for speed in (4.0, 5.0, 6.0):
            flags = rule_c_flags(
                development_records,
                travel_time_gt_s=travel_time,
                speed_lt_kmh=speed,
                distance_lte_m=1500.0,
            )
            retention, retained_keys = _retained_four_keys(development_records, flags)
            evaluation_keys = retained_keys.intersection(supported_keys)
            if not evaluation_keys:
                raise EvaluationStageError(
                    f"Sensitivity T={travel_time}, v={speed} retains no A0-supported key"
                )
            real = _filter_keys(development_events, evaluation_keys)
            sim = _filter_keys(a0_simulation, evaluation_keys)
            if _key_set(sim) != evaluation_keys:
                raise EvaluationStageError(
                    f"A0 reference misses a frozen sensitivity key for T={travel_time}, v={speed}"
                )
            metric = full_window_ks(real["speed_kmh"], sim["speed_kmh"])
            _require_succeeded(metric, f"sensitivity/T={travel_time}/v={speed}")
            rows.append(
                {
                    "schema_version": "fig3-sensitivity/v2",
                    "method_id": "rule_c",
                    "split": "development",
                    "travel_time_gt_s": travel_time,
                    "speed_lt_kmh": speed,
                    "distance_lte_m": 1500.0,
                    "n_eligible_raw_link_keys": retention.n_eligible_raw_link_keys,
                    "n_a0_supported_raw_link_keys": len(supported_keys),
                    "n_a0_unsupported_raw_link_keys": (
                        retention.n_eligible_raw_link_keys - len(supported_keys)
                    ),
                    "n_clean_link_keys": retention.n_clean_link_keys,
                    "n_evaluation_link_keys": len(evaluation_keys),
                    "retention_rate": retention.retention_rate,
                    "ks_speed": float(metric.value),
                    "status": "succeeded",
                }
            )
    return pd.DataFrame(rows, columns=list(FIG3_SENSITIVITY_COLUMNS))


def _link_key_text(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: f"{row['route']}|{row['bound']}|{int(row['from_seq'])}|{int(row['to_seq'])}",
        axis=1,
    )


def _fig4_cdf(
    real_clean: Mapping[str, pd.DataFrame],
    simulations: Mapping[tuple[str, int, str], pd.DataFrame],
    common_keys: Mapping[str, set[tuple[str, str, int, int]]],
    common_seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in _SPLITS:
        real = real_clean[split].copy()
        real["link_key"] = _link_key_text(real)
        for index, row in enumerate(real.itertuples(index=False), start=1):
            rows.append(
                {
                    "schema_version": "fig4-cdf-samples/v1",
                    "split": split,
                    "config_id": "A4",
                    "seed": -1,
                    "source": "real_clean",
                    "domain": "speed",
                    "event_id": f"{split}-real-{index:06d}",
                    "link_key": row.link_key,
                    "value": float(row.speed_kmh),
                    "unit": "km/h",
                    "status": "succeeded",
                }
            )
        for seed in common_seeds:
            sim = _filter_keys(simulations[("A4", int(seed), split)], common_keys[split]).copy()
            sim["link_key"] = _link_key_text(sim)
            for index, row in enumerate(sim.itertuples(index=False), start=1):
                rows.append(
                    {
                        "schema_version": "fig4-cdf-samples/v1",
                        "split": split,
                        "config_id": "A4",
                        "seed": int(seed),
                        "source": "simulation",
                        "domain": "speed",
                        "event_id": f"{split}-sim-{seed}-{index:06d}",
                        "link_key": row.link_key,
                        "value": float(row.speed_kmh),
                        "unit": "km/h",
                        "status": "succeeded",
                    }
                )
    return pd.DataFrame(rows, columns=list(FIG4_CDF_COLUMNS))


def run_evaluation_stage(
    project_root: Path,
    *,
    base_manifest: Mapping[str, Any] | None = None,
    ablation_runs_path: Path | None = None,
) -> dict[str, Any]:
    """Validate final runs and materialize all immutable evaluation artifacts."""

    root = project_root.resolve()
    manifest = deepcopy(
        dict(base_manifest) if base_manifest is not None else load_protocol_manifest(root)
    )
    validate_paper_manifest(manifest)
    _validate_evaluation_protocol(manifest)
    output_root = _path_within(root, str(manifest["outputs"]["run_directory"]))
    source_paths = _verify_evaluation_sources(root, manifest)
    runs_path = (
        ablation_runs_path.resolve()
        if ablation_runs_path is not None
        else output_root / "ablation" / "ablation_runs.csv"
    )
    if root not in runs_path.resolve().parents:
        raise EvaluationStageError("ablation_runs_path escapes project root")
    runs, common_seeds, run_manifests = _validate_run_sources(
        root, runs_path, manifest
    )
    simulations, stops = _load_simulations(
        root, runs, common_seeds, source_paths["route_stop_distance"]
    )
    real_events = {
        split: _load_real_events(source_paths[f"{split}_events"], split, manifest)
        for split in _SPLITS
    }
    real_clean, common_keys = _common_clean_populations(real_events, simulations)
    metrics = _paper_metrics(
        manifest,
        runs,
        common_seeds,
        run_manifests,
        real_clean,
        common_keys,
        simulations,
    )
    table = _table_i(metrics, common_seeds)
    reference_seed = int(min(common_seeds))
    a0_simulations = {
        split: simulations[("A0", reference_seed, split)] for split in _SPLITS
    }
    records_by_split = {
        split: aggregate_link_hour(real_events[split]) for split in _SPLITS
    }
    decisions, fitted_models = _audit_decisions(records_by_split)
    audit_support_universes = _audit_support_universes(
        records_by_split, a0_simulations
    )
    irn_mapping, irn_speeds, irn_evidence = _irn_inputs(manifest, source_paths)
    audit_manifest = _audit_manifest(
        records_by_split,
        decisions,
        fitted_models,
        irn_evidence,
        audit_support_universes,
        reference_seed,
    )
    audit = _audit_metrics(
        records_by_split,
        real_events,
        a0_simulations,
        decisions,
        audit_support_universes,
        irn_mapping,
        irn_speeds,
    )
    sensitivity = _fig3_sensitivity(
        records_by_split["development"],
        real_events["development"],
        a0_simulations["development"],
        audit_support_universes["development"],
    )
    contamination = _fig2_contamination(real_events["development"])
    trajectory = _fig2_trajectory(
        real_clean["development"],
        stops[("A0", reference_seed, "development")],
        common_keys["development"],
        reference_seed,
    )
    cdf = _fig4_cdf(real_clean, simulations, common_keys, common_seeds)

    paths = {
        "metrics": _write_csv_immutable(output_root / "metrics" / "paper_metrics.csv", metrics),
        "table_i": _write_csv_immutable(output_root / "tables" / "table_i.csv", table),
        "fig2_contamination": _write_csv_immutable(
            output_root / "reporting" / "fig2_contamination.csv", contamination
        ),
        "fig2_trajectory": _write_csv_immutable(
            output_root / "reporting" / "fig2_trajectory.csv", trajectory
        ),
        "fig3_sensitivity": _write_csv_immutable(
            output_root / "reporting" / "fig3_sensitivity.csv", sensitivity
        ),
        "fig4_cdf": _write_csv_immutable(
            output_root / "reporting" / "fig4_cdf_samples.csv", cdf
        ),
        "audit_metrics": _write_csv_immutable(
            output_root / "audit" / "audit_metrics.csv", audit
        ),
        "audit_manifest": _write_json_immutable(
            output_root / "audit" / "audit_manifest.json", audit_manifest
        ),
    }
    validate_camera_ready_inputs(
        output_root, selected=("fig2", "fig3", "fig4", "table_i")
    )
    return {
        "schema_version": EVALUATION_STAGE_SCHEMA,
        "status": "succeeded",
        "common_seeds": list(common_seeds),
        "a0_reference_seed": reference_seed,
        "ablation_runs_sha256": sha256_file(runs_path),
        "common_link_key_counts": {
            split: len(common_keys[split]) for split in _SPLITS
        },
        "metric_row_count": len(metrics),
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


__all__ = [
    "EVALUATION_STAGE_SCHEMA",
    "EvaluationStageError",
    "run_evaluation_stage",
]
