from __future__ import annotations

from pathlib import Path

import pytest

from src.paper_experiments.pipeline import (
    BASELINE_BACKGROUND_PARAMETERS,
    BASELINE_BUS_PARAMETERS,
    MAIN_SIMULATION_COUNT,
    PipelineError,
    build_run_manifest,
    bundle_run_manifest,
    load_protocol_manifest,
    simulation_request_from_bundle,
    verify_input_hashes,
    workload_estimate,
    write_json_immutable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_protocol_manifest_and_declared_input_hashes_match_disk() -> None:
    manifest = load_protocol_manifest(PROJECT_ROOT)
    verified = verify_input_hashes(PROJECT_ROOT, manifest)
    assert len(verified) == len(manifest["datasets"]) + len(
        manifest["simulator"]["effective_input_hashes"]
    )


def test_run_manifest_hashes_and_simulation_request_share_one_contract(tmp_path: Path) -> None:
    base = load_protocol_manifest(PROJECT_ROOT)
    run_directory = PROJECT_ROOT / "data" / "camera_ready_revision_20260716" / "unit-not-created"
    manifest = build_run_manifest(
        base,
        project_root=PROJECT_ROOT,
        run_directory=run_directory,
        run_id="A0-development-seed-0",
        config_id="A0",
        method_id="zero-shot",
        split="development",
        seed=0,
        sumo_seed=300000,
        bus_parameters=BASELINE_BUS_PARAMETERS,
        background_parameters=BASELINE_BACKGROUND_PARAMETERS,
        observation_semantic="no_l2_input",
        l1_enabled=False,
        l2_enabled=False,
        software={"python": "test", "sumo": "test"},
        timeout_seconds=1800,
    )
    bundle = bundle_run_manifest(manifest)
    request = simulation_request_from_bundle(PROJECT_ROOT, bundle)
    assert request.provenance_hash == bundle.provenance_hash
    assert request.simulation_effective_hash == bundle.simulation_effective_hash
    assert dict(request.component_hashes) == dict(bundle.component_hashes)
    assert manifest["simulator"]["effective_paths"]["background_routes"].endswith(
        "background_cropped.rou.xml"
    )


def test_manifest_rejects_unknown_semantic_and_path_escape() -> None:
    base = load_protocol_manifest(PROJECT_ROOT)
    kwargs = dict(
        base_manifest=base,
        project_root=PROJECT_ROOT,
        run_directory=PROJECT_ROOT / "data" / "camera_ready_revision_20260716" / "test",
        run_id="bad",
        config_id="A0",
        method_id="zero-shot",
        split="development",
        seed=0,
        sumo_seed=0,
        bus_parameters=BASELINE_BUS_PARAMETERS,
        background_parameters=BASELINE_BACKGROUND_PARAMETERS,
        l1_enabled=False,
        l2_enabled=False,
        software={"python": "test"},
        timeout_seconds=1800,
    )
    with pytest.raises(PipelineError, match="semantic"):
        build_run_manifest(observation_semantic="invented", **kwargs)
    with pytest.raises(PipelineError, match="escapes"):
        build_run_manifest(
            observation_semantic="no_l2_input",
            **{**kwargs, "run_directory": PROJECT_ROOT.parent / "outside"},
        )


def test_workload_estimate_uses_exact_declared_simulation_counts() -> None:
    estimate = workload_estimate([10.0, 20.0, 30.0], workers=5)
    assert estimate["simulation_counts"] == {
        "l1_shared_bo_lhs": 325,
        "l2_ies": 450,
        "final_ablation": 50,
        "total": MAIN_SIMULATION_COUNT,
    }
    assert estimate["pilot_median_runtime_s"] == 20.0
    assert estimate["post_pilot_timeout_s"] == 1800.0


def test_immutable_json_refuses_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    write_json_immutable(path, {"a": 1})
    write_json_immutable(path, {"a": 1})
    with pytest.raises(PipelineError, match="Refusing"):
        write_json_immutable(path, {"a": 2})
