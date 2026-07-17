from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.paper_experiments.evaluation_stage as evaluation_stage
from src.paper_experiments.contracts import (
    canonical_sha256,
    compute_manifest_hashes,
    hash_path,
    sha256_file,
)
from src.paper_experiments.evaluation_stage import (
    EVALUATION_STAGE_SCHEMA,
    EvaluationStageError,
    run_evaluation_stage,
)
from src.paper_experiments.figures import validate_camera_ready_inputs
from src.paper_experiments.metrics import METRIC_COLUMNS


BASELINE_BUS = {
    "t_board": 2.0,
    "t_fixed": 5.0,
    "tau": 1.0,
    "sigma": 0.5,
    "minGap_bus": 2.5,
    "accel": 2.6,
    "decel": 4.5,
}
CALIBRATED_BUS = {**BASELINE_BUS, "t_board": 1.5}
BASELINE_BACKGROUND = {
    "capacityFactor": 1.0,
    "minGap_background": 2.5,
    "impatience": 0.5,
}
RUN_COLUMNS = (
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
CONFIGS = {
    "A0": (False, False, "no_l2_input"),
    "A1": (True, False, "no_l2_input"),
    "A2": (False, True, "moving_only"),
    "A3": (True, True, "raw_d2d"),
    "A4": (True, True, "moving_only"),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_real_events(path: Path, date: str, hour_utc: int) -> None:
    rows = []
    for link in range(1, 11):
        for repetition in range(3):
            travel_time = 500.0 if link == 10 else 10.0 + link
            distance = 100.0
            rows.append(
                {
                    "route": "68X",
                    "bound": "inbound",
                    "from_seq": link,
                    "to_seq": link + 1,
                    "departure_ts": (
                        f"{date}T{hour_utc:02d}:{repetition:02d}:{link:02d}Z"
                    ),
                    "travel_time_s": travel_time,
                    "dist_m": distance,
                    "speed_kmh": distance * 3.6 / travel_time,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_irn(directory: Path, date_compact: str, date_iso: str, time_text: str) -> None:
    segments = "".join(
        f"<segment><segment_id>{1000 + link}</segment_id><speed>20</speed><valid>Y</valid></segment>"
        for link in range(1, 11)
    )
    directory.mkdir(parents=True, exist_ok=True)
    time_compact = time_text.replace(":", "")
    (directory / f"irnAvgSpeed-all-{date_compact}-{time_compact}.xml").write_text(
        f"<root><date>{date_iso}</date><time>{time_text}</time>{segments}</root>",
        encoding="utf-8",
    )


def _write_stopinfo(path: Path, *, offset: float, omit_seq: int | None = None) -> None:
    nodes = []
    for vehicle in range(3):
        current = float(vehicle * 200)
        for seq in range(1, 12):
            if seq == omit_seq:
                continue
            started = current
            ended = started + 1.0
            nodes.append(
                f"<stopinfo id='flow_68X_inbound.{vehicle}' busStop='s{seq}' "
                f"arrival='{started}' started='{started}' ended='{ended}'/>"
            )
            current = ended + 12.0 + offset + seq * 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<stopinfos>{''.join(nodes)}</stopinfos>", encoding="utf-8")


def _base_manifest(root: Path) -> dict:
    data = root / "data"
    development = data / "development_events.csv"
    cross_day = data / "cross_day_events.csv"
    _write_real_events(development, "2025-12-19", 9)
    _write_real_events(cross_day, "2025-12-30", 7)

    route_stop = data / "route_stops.csv"
    pd.DataFrame(
        [
            {
                "route": "68X",
                "bound": "inbound",
                "stop_id": f"s{seq}",
                "seq": seq,
                "cum_dist_m": float((seq - 1) * 100),
            }
            for seq in range(1, 12)
        ]
    ).to_csv(route_stop, index=False)

    development_irn = data / "development_irn"
    cross_day_irn = data / "cross_day_irn"
    _write_irn(development_irn, "20251219", "2025-12-19", "17:30:00")
    _write_irn(cross_day_irn, "20251230", "2025-12-30", "15:30:00")

    observation_index = data / "observation_index.csv"
    pd.DataFrame(
        [
            {
                "observation_id": link,
                "route": "68X",
                "bound": "inbound",
                "from_seq": link,
                "to_seq": link + 1,
            }
            for link in range(1, 11)
        ]
    ).to_csv(observation_index, index=False)
    edge_mapping = data / "edge_mapping.csv"
    pd.DataFrame(
        [
            {"observation_id": link, "edge_ids": json.dumps([1000 + link])}
            for link in range(1, 11)
        ]
    ).to_csv(edge_mapping, index=False)

    def dataset(identifier: str, path: Path, date: str, start: str, end: str) -> dict:
        return {
            "id": identifier,
            "path": path.relative_to(root).as_posix(),
            "sha256": hash_path(path),
            "observation_date": date,
            "timezone": "Asia/Hong_Kong",
            "time_window": {"start": start, "end": end},
        }

    return {
        "schema_version": "paper-manifest/v1",
        "experiment_id": "fixture-evaluation",
        "config_id": "paper_protocol",
        "method_id": "multi_stage",
        "seed": 0,
        "timezone": "Asia/Hong_Kong",
        "datasets": [
            dataset("development_events", development, "2025-12-19", "17:00:00", "18:00:00"),
            dataset("cross_day_events", cross_day, "2025-12-30", "15:00:00", "16:00:00"),
            dataset("route_stop_distance", route_stop, "2025-12-19", "17:00:00", "18:00:00"),
            dataset("development_irn", development_irn, "2025-12-19", "17:00:00", "18:00:00"),
            dataset("cross_day_irn", cross_day_irn, "2025-12-30", "15:00:00", "16:00:00"),
        ],
        "routes": [
            {
                "route": "68X",
                "direction": "inbound",
                "link_key_selection": "positive time/distance and distance <= 1500 m",
            }
        ],
        "splits": {
            "development": {
                "date": "2025-12-19",
                "window_start_hkt": "17:00:00",
                "window_end_hkt": "18:00:00",
            },
            "cross_day": {
                "date": "2025-12-30",
                "window_start_hkt": "15:00:00",
                "window_end_hkt": "16:00:00",
            },
        },
        "l1": {
            "parameter_bounds": {
                "t_board": [0.5, 5.0],
                "t_fixed": [2.0, 15.0],
                "tau": [0.1, 2.0],
                "sigma": [0.1, 0.8],
                "minGap_bus": [0.1, 5.0],
                "accel": [0.5, 3.0],
                "decel": [1.0, 5.0],
            },
            "objective_definition": "JL1_68X",
            "initial_design": {"method": "lhs", "n": 15},
            "budget": 40,
            "seed_schedule": [0, 1, 2],
            "bus_parameters": deepcopy(BASELINE_BUS),
        },
        "l2": {
            "state_components": ["capacityFactor", "minGap_background", "impatience"],
            "priors": {"capacityFactor": [0.8, 1.2]},
            "bounds": {"capacityFactor": [0.5, 2.0]},
            "ensemble_size": 10,
            "iterations": 3,
            "damping": 0.3,
            "observation_semantic": "moving_only",
            "background_parameters": deepcopy(BASELINE_BACKGROUND),
            "ensemble_seed_schedule": [101, 102],
        },
        "audit": {
            "method": "rule_c",
            "fitted_on_split": "development",
            "frozen_parameters": {"version": 1},
            "conditions": {
                "travel_time_gt_s": 325.0,
                "speed_lt_kmh": 5.0,
                "distance_lte_m": 1500.0,
            },
        },
        "ablation": {"seeds": [0, 1, 2]},
        "evaluation": {
            "full_window_min_samples_per_source": 20,
            "subwindow_duration_s": 900,
            "subwindow_step_s": 60,
            "subwindow_min_samples_per_source": 5,
            "common_seed_target": 5,
            "common_seed_minimum": 3,
        },
        "simulator": {
            "sumo_version": "test",
            "effective_input_hashes": {"network": "a" * 64},
            "settings": {"begin_s": 0, "end_s": 3900},
            "seed": 0,
            "timeout_seconds": 60,
            "observation_index": {
                "path": observation_index.relative_to(root).as_posix(),
                "sha256": sha256_file(observation_index),
            },
            "link_to_irn_mapping": {
                "path": edge_mapping.relative_to(root).as_posix(),
                "sha256": sha256_file(edge_mapping),
            },
        },
        "outputs": {
            "run_directory": "output",
            "required_artifacts": ["metrics/paper_metrics.csv"],
        },
    }


def _run_manifest(base: dict, config: str, seed: int, split: str, run_dir: Path, root: Path) -> dict:
    l1_enabled, l2_enabled, semantic = CONFIGS[config]
    result = deepcopy(base)
    result.update(
        {
            "config_id": config,
            "method_id": config.lower(),
            "seed": seed,
            "split": split,
            "run_id": f"final-{config}-{split}-seed-{seed}",
            "software_versions": {"python": "test", "sumo": "test"},
            "mechanisms": {"l1_enabled": l1_enabled, "l2_enabled": l2_enabled},
        }
    )
    result["l1"]["bus_parameters"] = deepcopy(CALIBRATED_BUS if l1_enabled else BASELINE_BUS)
    result["l1"]["enabled"] = l1_enabled
    result["l2"]["background_parameters"] = deepcopy(
        BASELINE_BACKGROUND
        if config in {"A0", "A1"}
        else {**BASELINE_BACKGROUND, "capacityFactor": 1.1 + 0.1 * int(config[-1])}
    )
    result["l2"]["observation_semantic"] = semantic
    result["l2"]["enabled"] = l2_enabled
    result["simulator"]["seed"] = 300000 + 1000 * seed + (1 if split == "cross_day" else 0)
    result["outputs"]["run_directory"] = run_dir.relative_to(root).as_posix()
    return result


def _build_fixture(root: Path) -> tuple[dict, Path]:
    manifest = _base_manifest(root)
    output = root / "output"
    rows = []
    offsets = {config: index * 0.35 for index, config in enumerate(CONFIGS)}
    for seed in (0, 1, 2):
        for split in ("development", "cross_day"):
            for config in CONFIGS:
                run_dir = output / "ablation" / "final" / config / split / f"seed-{seed}"
                stopinfo = run_dir / "attempt-01" / "stopinfo.xml"
                _write_stopinfo(
                    stopinfo,
                    offset=offsets[config] + seed * 0.05 + (0.1 if split == "cross_day" else 0.0),
                    omit_seq=10 if config == "A0" and seed == 0 else None,
                )
                run_manifest = _run_manifest(manifest, config, seed, split, run_dir, root)
                hashes = compute_manifest_hashes(run_manifest)
                manifest_hash = canonical_sha256(run_manifest)
                _write_json(run_dir / "run-manifest.json", run_manifest)
                _write_json(
                    run_dir / "run-manifest-hashes.json",
                    {
                        "manifest_hash": manifest_hash,
                        "provenance_hash": hashes["provenance_hash"],
                        "simulation_effective_hash": hashes["simulation_effective_hash"],
                        "component_hashes": hashes["component_hashes"],
                    },
                )
                output_hash = sha256_file(stopinfo)
                status = {
                    "schema_version": "run-status/v1",
                    "run_id": run_manifest["run_id"],
                    "status": "succeeded",
                    "attempt": 1,
                    "manifest_hash": manifest_hash,
                    "provenance_hash": hashes["provenance_hash"],
                    "simulation_effective_hash": hashes["simulation_effective_hash"],
                    "component_hashes": hashes["component_hashes"],
                    "produced_artifact_hashes": {"stopinfo.xml": output_hash},
                    "stopinfo_relative_path": "attempt-01/stopinfo.xml",
                }
                _write_json(run_dir / "run-status.json", status)
                components = hashes["component_hashes"]
                rows.append(
                    {
                        "run_id": run_manifest["run_id"],
                        "config_id": config,
                        "seed": seed,
                        "split": split,
                        "sumo_seed": run_manifest["simulator"]["seed"],
                        "l1_enabled": CONFIGS[config][0],
                        "l2_enabled": CONFIGS[config][1],
                        "observation_semantic": CONFIGS[config][2],
                        "status": "succeeded",
                        "error_summary": None,
                        "reused": False,
                        "attempt": 1,
                        "duration_s": 1.0,
                        "event_count": 24 if config == "A0" and seed == 0 else 30,
                        "stopinfo_path": stopinfo.relative_to(root).as_posix(),
                        "stopinfo_sha256": output_hash,
                        "manifest_path": (run_dir / "run-manifest.json").relative_to(root).as_posix(),
                        "manifest_hash": manifest_hash,
                        "provenance_hash": hashes["provenance_hash"],
                        "simulation_effective_hash": hashes["simulation_effective_hash"],
                        "bus_parameters_hash": components["bus_parameters"],
                        "background_parameters_hash": components["background_parameters"],
                        "observation_semantic_hash": components["observation_semantic"],
                        "simulator_inputs_hash": components["simulator_inputs"],
                    }
                )
    runs_path = output / "ablation" / "ablation_runs.csv"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RUN_COLUMNS)).to_csv(runs_path, index=False)
    return manifest, runs_path


def test_evaluation_stage_writes_strict_reproducible_artifacts(tmp_path: Path) -> None:
    manifest, runs_path = _build_fixture(tmp_path)

    result = run_evaluation_stage(
        tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
    )

    assert result["schema_version"] == EVALUATION_STAGE_SCHEMA
    assert result["common_seeds"] == [0, 1, 2]
    assert result["a0_reference_seed"] == 0
    assert result["metric_row_count"] == 120
    metrics = pd.read_csv(tmp_path / "output" / "metrics" / "paper_metrics.csv")
    assert tuple(metrics.columns) == METRIC_COLUMNS
    assert set(metrics["status"]) == {"succeeded"}
    assert set(metrics["config_id"]) == set(CONFIGS)
    table = pd.read_csv(tmp_path / "output" / "tables" / "table_i.csv")
    assert table["config_id"].tolist() == list(CONFIGS)
    assert set(table["n_seeds"]) == {3}
    audit = pd.read_csv(tmp_path / "output" / "audit" / "audit_metrics.csv")
    assert set(audit["method_id"]) == {"rule_c", "mad", "isolation_forest"}
    assert set(audit["split"]) == {"development", "cross_day"}
    audit_manifest_path = tmp_path / "output" / "audit" / "audit_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    assert audit_manifest["schema_version"] == "audit-manifest/v2"
    assert audit_manifest["fit_split"] == "development"
    assert audit_manifest["cross_day_model_refit"] is False
    assert audit_manifest["development_fit_record_count"] == 10
    comparison_support = audit_manifest["comparison_support"]
    assert comparison_support["a0_reference_config_id"] == "A0"
    assert comparison_support["a0_reference_seed"] == 0
    for split in ("development", "cross_day"):
        universe = comparison_support["splits"][split]
        assert universe["n_eligible_raw_link_keys"] == 10
        assert universe["n_a0_supported_raw_link_keys"] == 8
        assert universe["n_a0_unsupported_raw_link_keys"] == 2
        assert len(universe["raw_eligible_link_keys"]) == 10
        assert len(universe["a0_supported_raw_link_keys"]) == 8
        assert len(universe["a0_unsupported_raw_link_keys"]) == 2
    assert set(audit_manifest["methods"]) == {"rule_c", "mad", "isolation_forest"}
    irn_evidence = audit_manifest["irn_evidence"]
    assert len(irn_evidence["link_to_irn_mapping_sha256"]) == 64
    assert irn_evidence["link_count"] > 0
    assert irn_evidence["unique_segment_count"] > 0
    assert set(irn_evidence["splits"]) == {"development", "cross_day"}
    for split in irn_evidence["splits"].values():
        assert split["selected_file_count"] == len(split["selected_files"])
        assert split["selected_file_count"] > 0
        assert split["excluded_file_count"] >= 0
        assert len(split["selected_files_sha256"]) == 64
        assert split["segment_median_count"] > 0
    for method in audit_manifest["methods"].values():
        assert len(method["model_serialization"]["sha256"]) == 64
        assert method["package"]["name"]
        assert method["package"]["version"]
        assert set(method["decisions"]) == {"development", "cross_day"}
        for split in method["decisions"].values():
            assert split["n_flagged_link_keys"] == len(split["flagged_keys"])
            assert split["n_clean_link_keys"] == len(split["retained_keys"])
            assert split["n_eligible_raw_link_keys"] == (
                split["n_flagged_link_keys"] + split["n_clean_link_keys"]
            )
            assert all("window_start" in key for key in split["flagged_keys"])
            assert all("window_start" in key for key in split["retained_keys"])
    for _split, group in audit.groupby("split"):
        assert set(group["n_eligible_raw_link_keys"]) == {10}
        assert set(group["n_a0_supported_raw_link_keys"]) == {8}
        assert set(group["n_a0_unsupported_raw_link_keys"]) == {2}
        assert (group["n_evaluation_link_keys"] <= 8).all()
    assert "model_state_sha256" in audit_manifest["methods"]["isolation_forest"]
    assert result["artifacts"]["audit_manifest"]["sha256"] == sha256_file(
        audit_manifest_path
    )
    validate_camera_ready_inputs(
        tmp_path / "output", selected=("fig2", "fig3", "fig4", "table_i")
    )

    repeated = run_evaluation_stage(
        tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
    )
    assert repeated["artifacts"] == result["artifacts"]


def test_evaluation_stage_preserves_zero_irn_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, runs_path = _build_fixture(tmp_path)

    def unmatched_only(flagged_records, *_args, **_kwargs):
        return {
            "numerator": 0,
            "denominator": 0,
            "unmatched_flagged": len(flagged_records),
        }

    monkeypatch.setattr(evaluation_stage, "compute_irn_contradiction", unmatched_only)
    run_evaluation_stage(tmp_path, base_manifest=manifest, ablation_runs_path=runs_path)

    audit = pd.read_csv(tmp_path / "output" / "audit" / "audit_metrics.csv")
    assert set(audit["irn_numerator"]) == {0}
    assert set(audit["irn_denominator"]) == {0}
    validate_camera_ready_inputs(tmp_path / "output", selected=("fig3",))


@pytest.mark.parametrize(
    ("failure_operation", "expected_operation"),
    (("fit", "fit"), ("apply_cross_day", "apply:cross_day")),
)
def test_evaluation_stage_uses_frozen_quantile_fallback_for_isolation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_operation: str,
    expected_operation: str,
) -> None:
    manifest, runs_path = _build_fixture(tmp_path)
    failure_message = f"forced Isolation Forest {failure_operation} failure"

    if failure_operation == "fit":
        def fail_fit(_records: pd.DataFrame) -> object:
            raise RuntimeError(failure_message)

        monkeypatch.setattr(evaluation_stage, "fit_isolation_forest", fail_fit)
    else:
        original_apply = evaluation_stage.apply_isolation_forest

        def fail_cross_day_apply(records: pd.DataFrame, model: object) -> pd.DataFrame:
            date = pd.Timestamp(records.iloc[0]["window_start"]).date().isoformat()
            if date == "2025-12-30":
                raise RuntimeError(failure_message)
            return original_apply(records, model)

        monkeypatch.setattr(
            evaluation_stage, "apply_isolation_forest", fail_cross_day_apply
        )

    result = run_evaluation_stage(
        tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
    )

    audit = pd.read_csv(tmp_path / "output" / "audit" / "audit_metrics.csv")
    assert set(audit["method_id"]) == {"rule_c", "mad", "quantile_fallback"}
    assert set(audit["split"]) == {"development", "cross_day"}
    audit_manifest_path = tmp_path / "output" / "audit" / "audit_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    assert set(audit_manifest["methods"]) == {
        "rule_c",
        "mad",
        "quantile_fallback",
    }
    fallback = audit_manifest["methods"]["quantile_fallback"]
    expected_tt = np.quantile([*range(11, 20), 500.0], 0.95, method="linear")
    expected_speed = np.quantile(
        [*(360.0 / value for value in range(11, 20)), 360.0 / 500.0],
        0.05,
        method="linear",
    )
    assert fallback["fitted_statistics"]["tt_q95"] == pytest.approx(expected_tt)
    assert fallback["fitted_statistics"]["speed_q05"] == pytest.approx(expected_speed)
    assert fallback["fitted_statistics"]["interpolation"] == "linear"
    assert fallback["fitted_statistics"]["package_version"] == np.__version__
    assert fallback["package"] == {"name": "numpy", "version": np.__version__}
    assert fallback["fallback_from_method_id"] == "isolation_forest"
    assert fallback["model_state_sha256"] == canonical_sha256(
        fallback["fitted_statistics"]
    )
    assert fallback["model_serialization"]["sha256"] == fallback["model_state_sha256"]
    failure = fallback["isolation_forest_failure"]
    assert failure["operation"] == expected_operation
    assert failure["exception_type"] == "builtins.RuntimeError"
    assert failure["message"] == failure_message
    assert failure["error_summary"] == f"builtins.RuntimeError: {failure_message}"
    assert set(fallback["decisions"]) == {"development", "cross_day"}
    for decision in fallback["decisions"].values():
        assert decision["n_flagged_link_keys"] == len(decision["flagged_keys"])
        assert decision["n_clean_link_keys"] == len(decision["retained_keys"])
        assert decision["n_eligible_raw_link_keys"] == (
            decision["n_flagged_link_keys"] + decision["n_clean_link_keys"]
        )
        for key in [*decision["flagged_keys"], *decision["retained_keys"]]:
            assert set(key) == {
                "route",
                "bound",
                "from_seq",
                "to_seq",
                "window_start",
            }
    assert result["artifacts"]["audit_manifest"]["sha256"] == sha256_file(
        audit_manifest_path
    )

    repeated = run_evaluation_stage(
        tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
    )
    assert repeated["artifacts"] == result["artifacts"]


@pytest.mark.parametrize("failed_callable", ("rule_c_flags", "fit_mad", "apply_mad"))
def test_audit_does_not_treat_rule_c_or_mad_errors_as_isolation_failures(
    monkeypatch: pytest.MonkeyPatch,
    failed_callable: str,
) -> None:
    development = pd.DataFrame(
        {
            "route": ["68X", "68X"],
            "bound": ["inbound", "inbound"],
            "from_seq": [1, 2],
            "to_seq": [2, 3],
            "window_start": pd.to_datetime(
                ["2025-12-19T17:00:00+08:00", "2025-12-19T17:00:00+08:00"]
            ),
            "tt_median": [30.0, 500.0],
            "speed_median": [12.0, 0.72],
            "dist_m": [100.0, 100.0],
        }
    )
    cross_day = development.copy()
    cross_day["window_start"] = pd.to_datetime(
        ["2025-12-30T15:00:00+08:00", "2025-12-30T15:00:00+08:00"]
    )
    quantile_called = False

    def fail_non_isolation(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"forced {failed_callable} failure")

    def forbidden_quantile(_records: pd.DataFrame) -> object:
        nonlocal quantile_called
        quantile_called = True
        raise AssertionError("quantile fallback must not handle Rule C or MAD errors")

    monkeypatch.setattr(evaluation_stage, failed_callable, fail_non_isolation)
    monkeypatch.setattr(evaluation_stage, "fit_quantile", forbidden_quantile)

    with pytest.raises(RuntimeError, match=f"forced {failed_callable} failure"):
        evaluation_stage._audit_decisions(
            {"development": development, "cross_day": cross_day}
        )
    assert quantile_called is False


def test_evaluation_stage_rejects_stopinfo_hash_drift(tmp_path: Path) -> None:
    manifest, runs_path = _build_fixture(tmp_path)
    stopinfo = (
        tmp_path
        / "output"
        / "ablation"
        / "final"
        / "A0"
        / "development"
        / "seed-0"
        / "attempt-01"
        / "stopinfo.xml"
    )
    stopinfo.write_text("<stopinfos/>", encoding="utf-8")

    with pytest.raises(EvaluationStageError, match="stopinfo hash mismatch"):
        run_evaluation_stage(
            tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
        )
    assert not (tmp_path / "output" / "metrics" / "paper_metrics.csv").exists()


def test_evaluation_stage_blocks_fewer_than_three_common_seeds(tmp_path: Path) -> None:
    manifest, runs_path = _build_fixture(tmp_path)
    runs = pd.read_csv(runs_path)
    mask = (
        (runs["config_id"] == "A4")
        & (runs["split"] == "cross_day")
        & (runs["seed"].isin([1, 2]))
    )
    runs.loc[mask, "status"] = "failed"
    runs.to_csv(runs_path, index=False)

    with pytest.raises(EvaluationStageError, match="three to five seeds"):
        run_evaluation_stage(
            tmp_path, base_manifest=manifest, ablation_runs_path=runs_path
        )
