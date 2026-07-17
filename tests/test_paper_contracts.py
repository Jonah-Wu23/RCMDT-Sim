from __future__ import annotations

from copy import deepcopy

import pytest

from src.paper_experiments.contracts import (
    ContractError,
    PAPER_MANIFEST_SCHEMA,
    canonical_json,
    canonical_sha256,
    compute_component_hashes,
    compute_provenance_hash,
    compute_simulation_effective_hash,
    sha256_directory,
    validate_mechanism_matrix,
    validate_paper_manifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
BASELINE_BUS = {
    "t_board": 2.0,
    "t_fixed": 5.0,
    "tau": 1.0,
    "sigma": 0.5,
    "minGap_bus": 2.5,
    "accel": 2.6,
    "decel": 4.5,
}
CALIBRATED_BUS = {**BASELINE_BUS, "t_board": 1.4}
BASELINE_BACKGROUND = {
    "capacityFactor": 1.0,
    "minGap_background": 2.5,
    "impatience": 0.5,
}


def manifest(config_id: str = "A0", *, seed: int = 0) -> dict:
    return {
        "schema_version": PAPER_MANIFEST_SCHEMA,
        "experiment_id": "e2-main-ablation",
        "config_id": config_id,
        "method_id": config_id.lower(),
        "seed": seed,
        "split": "development",
        "run_id": f"{config_id}-seed-{seed}",
        "created_at": "2026-07-16T00:00:00Z",
        "software_versions": {"python": "3.11.4", "sumo": "1.20.0"},
        "datasets": [
            {
                "path": "data/processed/link_speeds.csv",
                "sha256": SHA_A,
                "observation_date": "2025-12-19",
                "timezone": "Asia/Hong_Kong",
                "time_window": {"start": "17:00", "end": "18:00"},
            }
        ],
        "routes": [{"route": "68X", "direction": "outbound", "link_key_selection": "rule-c-clean"}],
        "l1": {
            "parameter_bounds": {"t_board": [0.5, 2.0]},
            "objective_definition": "JL1_68X",
            "initial_design": {"method": "lhs", "n": 15},
            "budget": 40,
            "seed_schedule": [seed],
            "bus_parameters": deepcopy(BASELINE_BUS),
            "enabled": False,
        },
        "l2": {
            "state_components": ["capacityFactor", "minGap_background", "impatience"],
            "priors": {"capacityFactor": [0.8, 1.2]},
            "bounds": {"capacityFactor": [0.5, 2.0]},
            "ensemble_size": 10,
            "iterations": 3,
            "damping": 0.3,
            "observation_semantic": "none",
            "background_parameters": deepcopy(BASELINE_BACKGROUND),
            "ensemble_seed_schedule": [101, 102],
            "enabled": False,
        },
        "audit": {
            "method": "rule_c",
            "fitted_on_split": "predeclared",
            "frozen_parameters": {"version": 1},
            "conditions": {
                "travel_time_gt_s": 325,
                "speed_lt_kmh": 5,
                "distance_lte_m": 1500,
            },
        },
        "simulator": {
            "sumo_version": "1.20.0",
            "effective_input_hashes": {"network": SHA_B},
            "settings": {"begin": 0, "end": 3900},
            "seed": seed,
            "timeout_seconds": 3600,
        },
        "outputs": {"run_directory": f"runs/{config_id}/{seed}", "required_artifacts": ["stopinfo.xml"]},
        "mechanisms": {"l1_enabled": False, "l2_enabled": False},
    }


def ablation_manifests() -> list[dict]:
    matrix = {
        "A0": (False, False, "none", BASELINE_BUS),
        "A1": (True, False, "none", CALIBRATED_BUS),
        "A2": (False, True, "moving_only", BASELINE_BUS),
        "A3": (True, True, "raw_d2d", CALIBRATED_BUS),
        "A4": (True, True, "moving_only", CALIBRATED_BUS),
    }
    result = []
    for config_id, (l1_enabled, l2_enabled, semantic, bus) in matrix.items():
        item = manifest(config_id)
        item["mechanisms"] = {"l1_enabled": l1_enabled, "l2_enabled": l2_enabled}
        item["l1"]["enabled"] = l1_enabled
        item["l1"]["bus_parameters"] = deepcopy(bus)
        item["l2"]["enabled"] = l2_enabled
        item["l2"]["observation_semantic"] = semantic
        # Equal final values are legitimate; the validator must not invent an inequality.
        item["l2"]["background_parameters"] = deepcopy(BASELINE_BACKGROUND)
        result.append(item)
    return result


def test_canonical_json_and_directory_hashes_are_deterministic(tmp_path):
    assert canonical_json({"z": 1, "á": [2, 3]}) == '{"z":1,"á":[2,3]}'
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})

    left = tmp_path / "left"
    right = tmp_path / "right"
    (left / "nested").mkdir(parents=True)
    right.mkdir()
    (right / "nested").mkdir()
    (left / "a.txt").write_text("alpha", encoding="utf-8")
    (left / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    (right / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    (right / "a.txt").write_text("alpha", encoding="utf-8")
    assert sha256_directory(left) == sha256_directory(right)
    (right / "a.txt").write_text("changed", encoding="utf-8")
    assert sha256_directory(left) != sha256_directory(right)


def test_manifest_required_fields_and_hash_scopes():
    base = manifest()
    assert validate_paper_manifest(base) is base
    assert set(compute_component_hashes(base)) == {
        "bus_parameters",
        "background_parameters",
        "observation_semantic",
        "simulator_inputs",
    }

    relocated = deepcopy(base)
    relocated["outputs"]["run_directory"] = "different/output/path"
    relocated["run_id"] = "different-run"
    relocated["created_at"] = "2026-07-17T00:00:00Z"
    assert compute_simulation_effective_hash(base) == compute_simulation_effective_hash(relocated)
    assert compute_provenance_hash(base) != compute_provenance_hash(relocated)

    invalid = deepcopy(base)
    del invalid["simulator"]["effective_input_hashes"]
    with pytest.raises(ContractError):
        validate_paper_manifest(invalid)


@pytest.mark.parametrize(
    ("section", "alias", "duplicate"),
    [
        ("l1", "bus_parameters", "selected_parameters"),
        ("l2", "background_parameters", "final_parameters"),
    ],
)
def test_effective_parameter_aliases_must_appear_exactly_once(section, alias, duplicate):
    missing = manifest()
    del missing[section][alias]
    with pytest.raises(ContractError, match="exactly one"):
        validate_paper_manifest(missing)

    ambiguous = manifest()
    ambiguous[section][duplicate] = deepcopy(ambiguous[section][alias])
    with pytest.raises(ContractError, match="exactly one"):
        validate_paper_manifest(ambiguous)


def test_a0_a4_matrix_enforces_mechanisms_and_freezes_not_final_inequality():
    runs = ablation_manifests()
    hashes = validate_mechanism_matrix(runs)
    assert len(hashes) == 5
    assert hashes[(0, "A2")]["background_parameters"] == hashes[(0, "A4")]["background_parameters"]

    wrong_semantic = deepcopy(runs)
    wrong_semantic[3]["l2"]["observation_semantic"] = "moving_only"
    with pytest.raises(ContractError, match="A3"):
        validate_mechanism_matrix(wrong_semantic)

    wrong_frozen_bus = deepcopy(runs)
    wrong_frozen_bus[4]["l1"]["bus_parameters"]["t_board"] = 99.0
    with pytest.raises(ContractError, match="frozen L1"):
        validate_mechanism_matrix(wrong_frozen_bus)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("routes", 0, "direction"), "inbound", "same routes"),
        (("datasets", 0, "time_window", "start"), "16:00", "dataset windows"),
        (("audit", "frozen_parameters", "version"), 2, "frozen audit"),
        (("simulator", "settings", "begin"), 1, "SUMO settings"),
        (("simulator", "effective_input_hashes", "network"), SHA_A, "effective input hashes"),
    ],
)
def test_matrix_rejects_non_common_frozen_inputs(path, value, message):
    runs = ablation_manifests()
    target = runs[4]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError, match=message):
        validate_mechanism_matrix(runs)


def test_matrix_rejects_wrong_sumo_seed_and_mixed_splits():
    formula_seed = ablation_manifests()
    for item in formula_seed:
        item["simulator"]["seed"] = 300000
    assert len(validate_mechanism_matrix(formula_seed)) == 5

    wrong_seed = ablation_manifests()
    wrong_seed[4]["simulator"]["seed"] = 999
    with pytest.raises(ContractError, match="SUMO seed"):
        validate_mechanism_matrix(wrong_seed)

    mixed_splits = ablation_manifests()
    mixed_splits[4]["split"] = "cross-day"
    with pytest.raises(ContractError, match="exactly one split"):
        validate_mechanism_matrix(mixed_splits)


def test_matrix_requires_exact_design_baselines_and_enabled_l1_difference():
    wrong_bus_baseline = ablation_manifests()
    for index in (0, 2):
        wrong_bus_baseline[index]["l1"]["bus_parameters"]["t_board"] = 1.9
    with pytest.raises(ContractError, match="exact design baseline bus"):
        validate_mechanism_matrix(wrong_bus_baseline)

    wrong_background_baseline = ablation_manifests()
    for index in (0, 1):
        wrong_background_baseline[index]["l2"]["background_parameters"]["capacityFactor"] = 0.9
    with pytest.raises(ContractError, match="exact design baseline background"):
        validate_mechanism_matrix(wrong_background_baseline)

    ineffective_l1 = ablation_manifests()
    for index in (1, 3, 4):
        ineffective_l1[index]["l1"]["bus_parameters"] = deepcopy(BASELINE_BUS)
    with pytest.raises(ContractError, match="baseline bus hash"):
        validate_mechanism_matrix(ineffective_l1)


def test_matrix_freezes_l2_priors_and_ensemble_schedule():
    wrong_priors = ablation_manifests()
    wrong_priors[4]["l2"]["priors"]["capacityFactor"] = [0.7, 1.3]
    with pytest.raises(ContractError, match="L2 priors"):
        validate_mechanism_matrix(wrong_priors)

    wrong_schedule = ablation_manifests()
    wrong_schedule[4]["l2"]["ensemble_seed_schedule"] = [101, 999]
    with pytest.raises(ContractError, match="ensemble seed schedule"):
        validate_mechanism_matrix(wrong_schedule)
