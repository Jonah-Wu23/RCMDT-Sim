from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.paper_experiments.contracts import sha256_file
from src.paper_experiments.figures import (
    ARTIFACT_SIDECAR_SCHEMA,
    AUDIT_METRICS_COLUMNS,
    FIG2_CONTAMINATION_COLUMNS,
    FIG2_TRAJECTORY_COLUMNS,
    FIG3_SENSITIVITY_COLUMNS,
    FIG4_CDF_COLUMNS,
    FIGURE_FILES,
    FigureContractError,
    _STYLE,
    _feasible_cumulative_best,
    _render_fig1,
    _render_fig2,
    _render_fig3,
    generate_camera_ready_artifacts,
    validate_camera_ready_inputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "paper-manifest/v1",
        "l1": {
            "parameter_scope": "theta_bus",
            "parameter_bounds": {
                "t_board": [0.5, 5.0],
                "t_fixed": [2.0, 15.0],
                "tau": [0.1, 2.0],
                "sigma": [0.1, 0.8],
                "minGap_bus": [0.1, 5.0],
                "accel": [0.5, 3.0],
                "decel": [1.0, 5.0],
            },
        },
        "l2": {
            "state_name": "x_corr",
            "state_components": ["capacityFactor", "minGap_background", "impatience"],
        },
        "audit": {
            "conditions": {
                "travel_time_gt_s": 325.0,
                "speed_lt_kmh": 5.0,
                "distance_lte_m": 1500.0,
            }
        },
        "splits": {
            "development": {"date": "2025-12-19"},
            "cross_day": {"date": "2025-12-30"},
        },
    }


def test_reporting_input_column_contracts_are_frozen() -> None:
    assert FIG2_CONTAMINATION_COLUMNS == (
        "schema_version",
        "event_id",
        "route",
        "bound",
        "from_seq",
        "to_seq",
        "travel_time_s",
        "speed_kmh",
        "distance_m",
        "rule_c_flagged",
    )
    assert FIG2_TRAJECTORY_COLUMNS == (
        "schema_version",
        "split",
        "route",
        "bound",
        "trajectory_id",
        "source",
        "time_basis",
        "point_index",
        "cumulative_time_s",
        "cumulative_distance_m",
    )
    assert FIG3_SENSITIVITY_COLUMNS == (
        "schema_version",
        "method_id",
        "split",
        "travel_time_gt_s",
        "speed_lt_kmh",
        "distance_lte_m",
        "n_eligible_raw_link_keys",
        "n_a0_supported_raw_link_keys",
        "n_a0_unsupported_raw_link_keys",
        "n_clean_link_keys",
        "n_evaluation_link_keys",
        "retention_rate",
        "ks_speed",
        "status",
    )
    assert AUDIT_METRICS_COLUMNS == (
        "schema_version",
        "method_id",
        "method_label",
        "split",
        "n_eligible_raw_link_keys",
        "n_a0_supported_raw_link_keys",
        "n_a0_unsupported_raw_link_keys",
        "n_retained_link_keys",
        "n_evaluation_link_keys",
        "retention_rate",
        "ks_speed",
        "worst_15min_ks",
        "n_real",
        "n_sim",
        "irn_numerator",
        "irn_denominator",
        "unmatched_flagged",
        "status",
    )
    assert FIG4_CDF_COLUMNS == (
        "schema_version",
        "split",
        "config_id",
        "seed",
        "source",
        "domain",
        "event_id",
        "link_key",
        "value",
        "unit",
        "status",
    )


def _build_inputs(root: Path) -> None:
    manifest_path = root / "manifests" / "effective_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    contamination = []
    for index in range(30):
        contamination.append(
            {
                "schema_version": "fig2-contamination/v1",
                "event_id": f"clean-{index}",
                "route": "68X",
                "bound": "inbound",
                "from_seq": 1,
                "to_seq": 2,
                "travel_time_s": 100.0 + index,
                "speed_kmh": 12.0 + index / 10,
                "distance_m": 900.0,
                "rule_c_flagged": False,
            }
        )
    for index in range(10):
        contamination.append(
            {
                "schema_version": "fig2-contamination/v1",
                "event_id": f"flagged-{index}",
                "route": "68X",
                "bound": "inbound",
                "from_seq": 2,
                "to_seq": 3,
                "travel_time_s": 400.0 + index,
                "speed_kmh": 4.0,
                "distance_m": 1000.0,
                "rule_c_flagged": True,
            }
        )
    _write_csv(root / "reporting" / "fig2_contamination.csv", contamination)

    trajectory = []
    for source, time_basis, offset in (
        ("observed", "traffic_only", 0.0),
        ("simulated", "full", 35.0),
        ("simulated", "traffic_only", 8.0),
    ):
        for index, distance in enumerate((0.0, 500.0, 1000.0, 1500.0)):
            trajectory.append(
                {
                    "schema_version": "fig2-trajectory/v1",
                    "split": "development",
                    "route": "68X",
                    "bound": "inbound",
                    "trajectory_id": "68X-inbound-mean",
                    "source": source,
                    "time_basis": time_basis,
                    "point_index": index,
                    "cumulative_time_s": index * 90.0 + offset,
                    "cumulative_distance_m": distance,
                }
            )
    _write_csv(root / "reporting" / "fig2_trajectory.csv", trajectory)

    sensitivity = []
    eligible = 100
    for tt in (275.0, 325.0, 375.0):
        for speed in (4.0, 5.0, 6.0):
            clean = int(82 - (speed - 4) * 5 + (tt - 275) / 50)
            sensitivity.append(
                {
                    "schema_version": "fig3-sensitivity/v2",
                    "method_id": "rule_c",
                    "split": "development",
                    "travel_time_gt_s": tt,
                    "speed_lt_kmh": speed,
                    "distance_lte_m": 1500.0,
                    "n_eligible_raw_link_keys": eligible,
                    "n_a0_supported_raw_link_keys": 80,
                    "n_a0_unsupported_raw_link_keys": 20,
                    "n_clean_link_keys": clean,
                    "n_evaluation_link_keys": min(clean, 80),
                    "retention_rate": clean / eligible,
                    "ks_speed": 0.20 + 0.01 * (speed - 4) + 0.005 * ((tt - 275) / 50),
                    "status": "succeeded",
                }
            )
    _write_csv(root / "reporting" / "fig3_sensitivity.csv", sensitivity)

    audit_rows = []
    for method_id, label, offset in (
        ("rule_c", "Rule C", 0.00),
        ("mad", "MAD", 0.03),
        ("isolation_forest", "Isolation Forest", 0.05),
    ):
        for split, split_offset in (("development", 0.0), ("cross_day", 0.04)):
            audit_rows.append(
                {
                    "schema_version": "audit-metrics/v2",
                    "method_id": method_id,
                    "method_label": label,
                    "split": split,
                    "n_eligible_raw_link_keys": 100,
                    "n_a0_supported_raw_link_keys": 80,
                    "n_a0_unsupported_raw_link_keys": 20,
                    "n_retained_link_keys": int(round((0.82 - offset) * 100)),
                    "n_evaluation_link_keys": min(
                        int(round((0.82 - offset) * 100)), 80
                    ),
                    "retention_rate": 0.82 - offset,
                    "ks_speed": 0.22 + offset + split_offset,
                    "worst_15min_ks": 0.35 + offset + split_offset,
                    "n_real": 30,
                    "n_sim": 32,
                    "irn_numerator": 4,
                    "irn_denominator": 10,
                    "unmatched_flagged": 1,
                    "status": "succeeded",
                }
            )
    _write_csv(root / "audit" / "audit_metrics.csv", audit_rows)

    cdf_rows = []
    for split, split_offset in (("development", 0.0), ("cross_day", 1.0)):
        for index in range(20):
            cdf_rows.append(
                {
                    "schema_version": "fig4-cdf-samples/v1",
                    "split": split,
                    "config_id": "A4",
                    "seed": -1,
                    "source": "real_clean",
                    "domain": "speed",
                    "event_id": f"{split}-real-{index}",
                    "link_key": f"68X|inbound|{index % 4 + 1}|{index % 4 + 2}",
                    "value": 8.0 + split_offset + index * 0.4,
                    "unit": "km/h",
                    "status": "succeeded",
                }
            )
        for seed in (0, 1, 2):
            for index in range(20):
                cdf_rows.append(
                    {
                        "schema_version": "fig4-cdf-samples/v1",
                        "split": split,
                        "config_id": "A4",
                        "seed": seed,
                        "source": "simulation",
                        "domain": "speed",
                        "event_id": f"{split}-sim-{seed}-{index}",
                        "link_key": f"68X|inbound|{index % 4 + 1}|{index % 4 + 2}",
                        "value": 8.8 + split_offset + seed * 0.1 + index * 0.4,
                        "unit": "km/h",
                        "status": "succeeded",
                    }
                )
    _write_csv(root / "reporting" / "fig4_cdf_samples.csv", cdf_rows)

    evaluation_rows = []
    for seed in (0, 1, 2):
        initial_scores = [200.0 - index - seed for index in range(15)]
        for method in ("BO", "LHS"):
            for evaluation_index in range(1, 41):
                if evaluation_index <= 15:
                    objective = initial_scores[evaluation_index - 1]
                    candidate_key = f"seed-{seed}-initial-{evaluation_index}"
                elif method == "BO":
                    objective = 185.0 - (evaluation_index - 15) * 1.2 - seed
                    candidate_key = f"seed-{seed}-bo-{evaluation_index}"
                else:
                    objective = 187.0 - (evaluation_index - 15) * 0.25 - seed
                    candidate_key = f"seed-{seed}-lhs-{evaluation_index}"
                evaluation_rows.append(
                    {
                        "schema_version": "bo-lhs-evaluations/v1",
                        "optimization_seed": seed,
                        "method": method,
                        "evaluation_index": evaluation_index,
                        "candidate_hash": hashlib.sha256(candidate_key.encode()).hexdigest(),
                        "objective": objective,
                        "feasible": True,
                    }
                )
    _write_csv(root / "l1" / "bo_lhs_evaluations.csv", evaluation_rows)

    table_rows = []
    names = {
        "A0": "Zero-shot",
        "A1": "BO-only",
        "A2": "IES-only",
        "A3": "Raw-RCMDT",
        "A4": "Full-RCMDT",
    }
    for index, config_id in enumerate(("A0", "A1", "A2", "A3", "A4")):
        value = 0.35 - index * 0.025
        table_rows.append(
            {
                "schema_version": "table-i/v1",
                "config_id": config_id,
                "configuration": names[config_id],
                "n_seeds": 3,
                "ks_speed_development_mean": value,
                "ks_speed_development_std": 0.01,
                "worst_15min_ks_development_mean": value + 0.10,
                "worst_15min_ks_development_std": 0.02,
                "ks_speed_cross_day_mean": value + 0.04,
                "ks_speed_cross_day_std": 0.015,
                "worst_15min_ks_cross_day_mean": value + 0.14,
                "worst_15min_ks_cross_day_std": 0.025,
                "n_real_development": 30,
                "n_real_cross_day": 28,
                "n_sim_development_mean": 31.0 + index,
                "n_sim_development_std": 1.0,
                "n_sim_cross_day_mean": 29.0 + index,
                "n_sim_cross_day_std": 1.2,
                "status": "succeeded",
            }
        )
    _write_csv(root / "tables" / "table_i.csv", table_rows)


def test_generates_all_artifacts_at_300_dpi_with_complete_sidecars(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    output_root = tmp_path / "plots"
    _build_inputs(input_root)
    script_path = Path(__file__)

    generated = generate_camera_ready_artifacts(
        input_root,
        output_root,
        script_path=script_path,
        command="python synthetic-camera-ready-runner",
    )

    assert [path.name for path in generated] == list(FIGURE_FILES.values())
    assert len(list(output_root.glob("*.png"))) == 6
    expected_inputs = {
        "Fig1_camera_ready_architecture.png": {
            "manifests/effective_manifest.json",
        },
        "Fig2_camera_ready_contamination.png": {
            "manifests/effective_manifest.json",
            "reporting/fig2_contamination.csv",
            "reporting/fig2_trajectory.csv",
        },
        "Fig3_camera_ready_audit.png": {
            "manifests/effective_manifest.json",
            "reporting/fig3_sensitivity.csv",
            "audit/audit_metrics.csv",
        },
        "Fig4_camera_ready_cdf.png": {
            "manifests/effective_manifest.json",
            "reporting/fig4_cdf_samples.csv",
        },
        "Fig5_camera_ready_bo_lhs.png": {
            "manifests/effective_manifest.json",
            "l1/bo_lhs_evaluations.csv",
        },
        "Table_I_camera_ready_ablation.png": {
            "manifests/effective_manifest.json",
            "tables/table_i.csv",
        },
    }
    expected_dimensions = {
        "Fig1_camera_ready_architecture.png": (2148, 825),
        "Fig2_camera_ready_contamination.png": (2148, 795),
        "Fig3_camera_ready_audit.png": (2148, 1440),
        "Fig4_camera_ready_cdf.png": (2148, 795),
        "Fig5_camera_ready_bo_lhs.png": (1050, 840),
        "Table_I_camera_ready_ablation.png": (2148, 675),
    }
    for artifact_path in generated:
        with Image.open(artifact_path) as image:
            assert image.format == "PNG"
            assert (image.width, image.height) == expected_dimensions[artifact_path.name]
            assert image.info["dpi"] == pytest.approx((300.0, 300.0), abs=0.5)
        sidecar_path = artifact_path.with_suffix(".png.sidecar.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["schema_version"] == ARTIFACT_SIDECAR_SCHEMA
        assert sidecar["artifact_sha256"] == sha256_file(artifact_path)
        assert sidecar["manifest_sha256"] == sha256_file(
            input_root / "manifests" / "effective_manifest.json"
        )
        assert len(sidecar["script_sha256"]) == 64
        assert len(sidecar["provenance_hash"]) == 64
        assert sidecar["command"] == "python synthetic-camera-ready-runner"
        assert sidecar["figure"]["dpi_x"] == pytest.approx(300.0, abs=0.5)
        assert (
            sidecar["figure"]["width_px"],
            sidecar["figure"]["height_px"],
        ) == expected_dimensions[artifact_path.name]
        assert sidecar["figure"]["font_family"] == "Times New Roman"
        source_inputs = {item["path"]: item for item in sidecar["source_inputs"]}
        assert set(source_inputs) == expected_inputs[artifact_path.name]
        for relative_path, record in source_inputs.items():
            source_path = input_root / relative_path
            assert record["sha256"] == sha256_file(source_path)
            assert record["bytes"] == source_path.stat().st_size


def test_fig3_accepts_and_renders_quantile_fallback_as_the_adaptive_method(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "camera-ready"
    output_root = tmp_path / "plots"
    _build_inputs(input_root)
    audit_path = input_root / "audit" / "audit_metrics.csv"
    audit = pd.read_csv(audit_path)
    adaptive = audit["method_id"] == "isolation_forest"
    audit.loc[adaptive, "method_id"] = "quantile_fallback"
    audit.loc[adaptive, "method_label"] = "Quantile Fallback"
    audit.to_csv(audit_path, index=False)

    generated = generate_camera_ready_artifacts(
        input_root,
        output_root,
        script_path=Path(__file__),
        command="python synthetic-camera-ready-runner",
        selected=["fig3"],
    )

    assert [path.name for path in generated] == [FIGURE_FILES["fig3"]]
    assert generated[0].is_file()


def test_fig3_rejects_two_adaptive_methods(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    audit_path = input_root / "audit" / "audit_metrics.csv"
    audit = pd.read_csv(audit_path)
    fallback = audit.loc[audit["method_id"] == "isolation_forest"].copy()
    fallback["method_id"] = "quantile_fallback"
    fallback["method_label"] = "Quantile Fallback"
    pd.concat([audit, fallback], ignore_index=True).to_csv(audit_path, index=False)

    with pytest.raises(FigureContractError, match="exactly one adaptive"):
        validate_camera_ready_inputs(input_root, selected=["fig3"])


def test_fig3_accepts_undefined_irn_rate_with_zero_denominator(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    audit_path = input_root / "audit" / "audit_metrics.csv"
    audit = pd.read_csv(audit_path)
    mask = (audit["method_id"] == "mad") & (audit["split"] == "cross_day")
    audit.loc[mask, ["irn_numerator", "irn_denominator", "unmatched_flagged"]] = [0, 0, 0]
    audit.to_csv(audit_path, index=False)

    prepared = validate_camera_ready_inputs(input_root, selected=["fig3"])
    prepared_audit = prepared["fig3"].data[1]
    undefined = prepared_audit.loc[mask.to_numpy(), "irn_contradiction_rate"]
    assert undefined.isna().all()
    generated = generate_camera_ready_artifacts(
        input_root,
        tmp_path / "plots",
        script_path=Path(__file__),
        command="python synthetic-camera-ready-runner",
        selected=["fig3"],
    )
    assert len(generated) == 1


def test_fig3_rejects_method_specific_a0_support_universe(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    audit_path = input_root / "audit" / "audit_metrics.csv"
    audit = pd.read_csv(audit_path)
    mask = (audit["method_id"] == "mad") & (audit["split"] == "development")
    audit.loc[mask, "n_a0_supported_raw_link_keys"] = 79
    audit.loc[mask, "n_a0_unsupported_raw_link_keys"] = 21
    audit.to_csv(audit_path, index=False)

    with pytest.raises(FigureContractError, match="must share one n_a0_supported"):
        validate_camera_ready_inputs(input_root, selected=["fig3"])


@pytest.mark.parametrize("failure", ["missing", "schema", "sample", "nonfinite"])
def test_preflight_failures_create_no_png(tmp_path: Path, failure: str) -> None:
    input_root = tmp_path / "camera-ready"
    output_root = tmp_path / "plots"
    _build_inputs(input_root)

    if failure == "missing":
        (input_root / "reporting" / "fig4_cdf_samples.csv").unlink()
        selected = ["fig4"]
    elif failure == "schema":
        path = input_root / "reporting" / "fig3_sensitivity.csv"
        frame = pd.read_csv(path).drop(columns=["ks_speed"])
        frame.to_csv(path, index=False)
        selected = ["fig3"]
    elif failure == "sample":
        path = input_root / "reporting" / "fig4_cdf_samples.csv"
        frame = pd.read_csv(path)
        drop_index = frame.index[
            (frame["split"] == "cross_day")
            & (frame["source"] == "simulation")
            & (frame["seed"] == 0)
        ][0]
        frame.drop(index=drop_index).to_csv(path, index=False)
        selected = ["fig4"]
    else:
        path = input_root / "tables" / "table_i.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "ks_speed_cross_day_mean"] = np.inf
        frame.to_csv(path, index=False)
        selected = ["table_i"]

    with pytest.raises(FigureContractError):
        generate_camera_ready_artifacts(
            input_root,
            output_root,
            script_path=Path(__file__),
            command="python synthetic-camera-ready-runner",
            selected=selected,
        )
    assert not output_root.exists()


def test_equal_budget_contract_rejects_changed_shared_initial_candidate(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    path = input_root / "l1" / "bo_lhs_evaluations.csv"
    frame = pd.read_csv(path)
    mask = (
        (frame["optimization_seed"] == 0)
        & (frame["method"] == "LHS")
        & (frame["evaluation_index"] == 1)
    )
    frame.loc[mask, "candidate_hash"] = hashlib.sha256(b"different").hexdigest()
    frame.to_csv(path, index=False)

    with pytest.raises(FigureContractError, match="initial candidate hashes differ"):
        generate_camera_ready_artifacts(
            input_root,
            tmp_path / "plots",
            script_path=Path(__file__),
            command="python synthetic-camera-ready-runner",
            selected=["fig5"],
        )


def test_fig2_uses_one_median_based_decision_per_link_hour_key(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    path = input_root / "reporting" / "fig2_contamination.csv"
    frame = pd.read_csv(path)

    event = frame.index[frame["event_id"] == "clean-0"][0]
    frame.loc[event, ["travel_time_s", "speed_kmh"]] = [500.0, 2.0]
    frame.to_csv(path, index=False)

    validate_camera_ready_inputs(input_root, selected=["fig2"])

    frame.loc[event, "rule_c_flagged"] = True
    frame.to_csv(path, index=False)
    with pytest.raises(FigureContractError, match="one Rule C decision"):
        validate_camera_ready_inputs(input_root, selected=["fig2"])


def test_feasible_cumulative_best_ignores_infeasible_penalties() -> None:
    values = _feasible_cumulative_best(
        [100.0, 50.0, 3200.0, 40.0, 2800.0],
        [False, False, True, False, True],
    )
    np.testing.assert_allclose(
        values,
        [np.nan, np.nan, 3200.0, 3200.0, 2800.0],
        equal_nan=True,
    )


def test_fig1_box_text_stays_inside_its_patch() -> None:
    with plt.rc_context(_STYLE):
        figure = _render_fig1(_manifest())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[0]
        boxes = [patch for patch in axis.patches if isinstance(patch, FancyBboxPatch)]
        assert len(boxes) == 7

        for box in boxes:
            center = (box.get_x() + box.get_width() / 2, box.get_y() + box.get_height() / 2)
            text = next(
                item
                for item in axis.texts
                if np.allclose(item.get_position(), center, rtol=0.0, atol=1e-12)
            )
            text_bounds = text.get_window_extent(renderer)
            box_bounds = box.get_window_extent(renderer)
            assert text_bounds.x0 >= box_bounds.x0
            assert text_bounds.x1 <= box_bounds.x1
            assert text_bounds.y0 >= box_bounds.y0
            assert text_bounds.y1 <= box_bounds.y1
    plt.close(figure)


def test_fig3_irn_annotations_do_not_overlap_the_legend(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    prepared = validate_camera_ready_inputs(input_root, selected=["fig3"])["fig3"]

    with plt.rc_context(_STYLE):
        figure = _render_fig3(prepared.data)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[3]
        legend_bounds = axis.get_legend().get_window_extent(renderer)
        annotations = [text for text in axis.texts if text.get_text().startswith("IRN ")]

        assert len(annotations) == 3
        for annotation in annotations:
            assert not annotation.get_window_extent(renderer).overlaps(legend_bounds)
    plt.close(figure)


def test_fig2_rule_c_annotation_does_not_cover_scatter_points(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    prepared = validate_camera_ready_inputs(input_root, selected=["fig2"])["fig2"]

    with plt.rc_context(_STYLE):
        figure = _render_fig2(prepared.data)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[1]
        annotation = next(text for text in axis.texts if text.get_text().startswith("Rule C on"))
        annotation_bounds = annotation.get_window_extent(renderer)

        assert annotation.get_position()[1] >= 0.15
        for collection in axis.collections:
            display_points = axis.transData.transform(collection.get_offsets())
            assert not any(annotation_bounds.contains(x, y) for x, y in display_points)
    plt.close(figure)


def test_fig3_panel_d_leaves_headroom_above_bars(tmp_path: Path) -> None:
    input_root = tmp_path / "camera-ready"
    _build_inputs(input_root)
    prepared = validate_camera_ready_inputs(input_root, selected=["fig3"])["fig3"]

    with plt.rc_context(_STYLE):
        figure = _render_fig3(prepared.data)
        axis = figure.axes[3]
        bar_tops = [patch.get_height() for patch in axis.patches]

        assert bar_tops
        assert max(bar_tops) < axis.get_ylim()[1]
    plt.close(figure)
