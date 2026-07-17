"""Versioned contracts and hashes for camera-ready paper experiments.

The simulation-effective hash intentionally excludes run identifiers, output
paths, and timestamps.  The provenance hash intentionally includes them via
the complete manifest and the explicit run metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PAPER_MANIFEST_SCHEMA = "paper-manifest/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MISSING = object()


class ContractError(ValueError):
    """Raised when a paper experiment artifact violates its contract."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON with sorted keys and no NaN values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file():
        raise ContractError(f"Not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash directory contents using relative POSIX paths, sizes, and digests."""

    root = Path(path)
    if not root.is_dir():
        raise ContractError(f"Not a directory: {root}")
    entries = []
    for item in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        entries.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return canonical_sha256(entries)


def hash_path(path: str | Path) -> str:
    """Hash a file or directory, rejecting missing and unsupported paths."""

    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return sha256_directory(target)
    raise ContractError(f"Path does not exist: {target}")


def _require_mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key, _MISSING)
    if not isinstance(value, Mapping):
        raise ContractError(f"{context}.{key} must be an object")
    return value


def _require_sequence(parent: Mapping[str, Any], key: str, context: str) -> Sequence[Any]:
    value = parent.get(key, _MISSING)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ContractError(f"{context}.{key} must be a non-empty array")
    return value


def _require_nonempty(parent: Mapping[str, Any], key: str, context: str) -> Any:
    value = parent.get(key, _MISSING)
    if value is _MISSING or value is None or value == "":
        raise ContractError(f"Missing required field: {context}.{key}")
    return value


def _require_one(parent: Mapping[str, Any], keys: Sequence[str], context: str) -> Any:
    for key in keys:
        value = parent.get(key, _MISSING)
        if value is not _MISSING and value is not None and value != "":
            return value
    raise ContractError(f"{context} requires one of: {', '.join(keys)}")


def _require_exactly_one(parent: Mapping[str, Any], keys: Sequence[str], context: str) -> Any:
    """Return the sole declared alias, rejecting ambiguous manifests."""

    present = [key for key in keys if key in parent]
    if len(present) != 1:
        found = ", ".join(present) if present else "none"
        raise ContractError(
            f"{context} requires exactly one of: {', '.join(keys)}; found: {found}"
        )
    return parent[present[0]]


def _require_finite_number(parent: Mapping[str, Any], key: str, context: str, *, minimum: float | None = None) -> float:
    value = _require_nonempty(parent, key, context)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{context}.{key} must be a finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ContractError(f"{context}.{key} must be >= {minimum}")
    return numeric


def validate_paper_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate required ``paper-manifest/v1`` fields and nested contracts.

    The function returns the original mapping on success and raises
    :class:`ContractError` on the first violation.
    """

    if not isinstance(manifest, Mapping):
        raise ContractError("manifest must be an object")
    if manifest.get("schema_version") != PAPER_MANIFEST_SCHEMA:
        raise ContractError(f"schema_version must be {PAPER_MANIFEST_SCHEMA!r}")
    for key in ("experiment_id", "config_id", "method_id"):
        _require_nonempty(manifest, key, "manifest")
    seed = _require_nonempty(manifest, "seed", "manifest")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("manifest.seed must be an integer")

    datasets = _require_sequence(manifest, "datasets", "manifest")
    for index, dataset in enumerate(datasets):
        context = f"manifest.datasets[{index}]"
        if not isinstance(dataset, Mapping):
            raise ContractError(f"{context} must be an object")
        for key in ("path", "sha256", "observation_date", "timezone", "time_window"):
            _require_nonempty(dataset, key, context)
        if not _SHA256_RE.fullmatch(str(dataset["sha256"])):
            raise ContractError(f"{context}.sha256 must be a 64-character hexadecimal digest")
        window = dataset["time_window"]
        if not isinstance(window, Mapping):
            raise ContractError(f"{context}.time_window must be an object")
        _require_nonempty(window, "start", f"{context}.time_window")
        _require_nonempty(window, "end", f"{context}.time_window")

    routes = _require_sequence(manifest, "routes", "manifest")
    for index, route in enumerate(routes):
        context = f"manifest.routes[{index}]"
        if not isinstance(route, Mapping):
            raise ContractError(f"{context} must be an object")
        for key in ("route", "direction", "link_key_selection"):
            _require_nonempty(route, key, context)

    l1 = _require_mapping(manifest, "l1", "manifest")
    for key in ("parameter_bounds", "objective_definition", "initial_design", "budget", "seed_schedule"):
        _require_nonempty(l1, key, "manifest.l1")
    bus_parameters = _require_exactly_one(
        l1,
        ("bus_parameters", "selected_parameters", "effective_parameters"),
        "manifest.l1",
    )
    if not isinstance(bus_parameters, Mapping) or not bus_parameters:
        raise ContractError("manifest.l1 effective bus parameters must be a non-empty object")

    l2 = _require_mapping(manifest, "l2", "manifest")
    for key in ("state_components", "priors", "bounds", "ensemble_size", "iterations", "damping", "observation_semantic"):
        _require_nonempty(l2, key, "manifest.l2")
    _require_finite_number(l2, "ensemble_size", "manifest.l2", minimum=1)
    _require_finite_number(l2, "iterations", "manifest.l2", minimum=1)
    _require_finite_number(l2, "damping", "manifest.l2", minimum=0)
    background_parameters = _require_exactly_one(
        l2,
        ("background_parameters", "final_parameters", "effective_parameters"),
        "manifest.l2",
    )
    if not isinstance(background_parameters, Mapping) or not background_parameters:
        raise ContractError("manifest.l2 effective background parameters must be a non-empty object")
    _require_one(l2, ("ensemble_seed_schedule", "seed_schedule", "ensemble_seed_hash"), "manifest.l2")

    audit = _require_mapping(manifest, "audit", "manifest")
    method = str(_require_nonempty(audit, "method", "manifest.audit")).lower().replace("-", "_")
    _require_one(audit, ("fitted_on_split", "fitted_on"), "manifest.audit")
    _require_one(audit, ("frozen_parameters", "model_hash"), "manifest.audit")
    if method in {"rule_c", "fixed_rule_c"}:
        conditions = _require_mapping(audit, "conditions", "manifest.audit")
        for key in ("travel_time_gt_s", "speed_lt_kmh", "distance_lte_m"):
            _require_finite_number(conditions, key, "manifest.audit.conditions", minimum=0)

    simulator = _require_mapping(manifest, "simulator", "manifest")
    for key in ("sumo_version", "effective_input_hashes", "settings", "seed", "timeout_seconds"):
        _require_nonempty(simulator, key, "manifest.simulator")
    if not isinstance(simulator["effective_input_hashes"], Mapping) or not simulator["effective_input_hashes"]:
        raise ContractError("manifest.simulator.effective_input_hashes must be a non-empty object")
    for name, digest in simulator["effective_input_hashes"].items():
        if not _SHA256_RE.fullmatch(str(digest)):
            raise ContractError(f"manifest.simulator.effective_input_hashes[{name!r}] is not SHA-256")
    _require_finite_number(simulator, "timeout_seconds", "manifest.simulator", minimum=1)

    outputs = _require_mapping(manifest, "outputs", "manifest")
    _require_one(outputs, ("run_directory", "run_dir"), "manifest.outputs")
    artifacts = _require_one(outputs, ("required_artifacts", "required_artifact_names"), "manifest.outputs")
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence) or not artifacts:
        raise ContractError("manifest.outputs required artifacts must be a non-empty array")
    return manifest


def _first_mapping(section: Mapping[str, Any], keys: Sequence[str], context: str) -> Mapping[str, Any]:
    value = _require_exactly_one(section, keys, context)
    if not isinstance(value, Mapping) or not value:
        raise ContractError(f"{context}.{keys[0]} must resolve to an object")
    return value


def _input_content_hashes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    dataset_hashes = {str(item["path"]): str(item["sha256"]).lower() for item in manifest["datasets"]}
    simulator_hashes = {
        str(name): str(value).lower()
        for name, value in manifest["simulator"]["effective_input_hashes"].items()
    }
    return {"datasets": dataset_hashes, "simulator_inputs": simulator_hashes}


def _effective_sections(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_paper_manifest(manifest)
    l1 = manifest["l1"]
    l2 = manifest["l2"]
    simulator = manifest["simulator"]
    return {
        "input_content_hashes": _input_content_hashes(manifest),
        "bus_parameters": dict(_first_mapping(l1, ("bus_parameters", "selected_parameters", "effective_parameters"), "manifest.l1")),
        "background_parameters": dict(_first_mapping(l2, ("background_parameters", "final_parameters", "effective_parameters"), "manifest.l2")),
        "observation_semantic": l2["observation_semantic"],
        "simulator_settings": dict(simulator["settings"]),
        "sumo_seed": simulator["seed"],
    }


def compute_component_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Compute the four component hashes required by the validator."""

    effective = _effective_sections(manifest)
    return {
        "bus_parameters": canonical_sha256(effective["bus_parameters"]),
        "background_parameters": canonical_sha256(effective["background_parameters"]),
        "observation_semantic": canonical_sha256(effective["observation_semantic"]),
        "simulator_inputs": canonical_sha256(
            {
                "input_content_hashes": effective["input_content_hashes"],
                "simulator_settings": effective["simulator_settings"],
                "sumo_seed": effective["sumo_seed"],
            }
        ),
    }


def compute_simulation_effective_hash(manifest: Mapping[str, Any]) -> str:
    """Hash only inputs that can change an effective SUMO simulation."""

    return canonical_sha256(_effective_sections(manifest))


def compute_provenance_hash(
    manifest: Mapping[str, Any],
    *,
    software_versions: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> str:
    """Hash the complete manifest together with explicit run provenance."""

    validate_paper_manifest(manifest)
    versions = software_versions if software_versions is not None else manifest.get("software_versions")
    effective_run_id = run_id if run_id is not None else manifest.get("run_id")
    if not isinstance(versions, Mapping) or not versions:
        raise ContractError("software_versions are required for provenance_hash")
    if not effective_run_id:
        raise ContractError("run_id is required for provenance_hash")
    payload = {
        "manifest": manifest,
        "input_content_hashes": _input_content_hashes(manifest),
        "software_versions": dict(versions),
        "seed": manifest["seed"],
        "run_id": effective_run_id,
        "schema_version": manifest["schema_version"],
    }
    return canonical_sha256(payload)


def compute_manifest_hashes(
    manifest: Mapping[str, Any],
    *,
    software_versions: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return provenance, simulation-effective, and component hashes."""

    return {
        "provenance_hash": compute_provenance_hash(
            manifest,
            software_versions=software_versions,
            run_id=run_id,
        ),
        "simulation_effective_hash": compute_simulation_effective_hash(manifest),
        "component_hashes": compute_component_hashes(manifest),
    }


_MECHANISM_MATRIX = {
    "A0": (False, False, {"none", "no_l2_input"}),
    "A1": (True, False, {"none", "no_l2_input"}),
    "A2": (False, True, {"moving_only"}),
    "A3": (True, True, {"raw_d2d"}),
    "A4": (True, True, {"moving_only"}),
}

_BASELINE_BUS_PARAMETERS = {
    "t_board": 2.0,
    "t_fixed": 5.0,
    "tau": 1.0,
    "sigma": 0.5,
    "minGap_bus": 2.5,
    "accel": 2.6,
    "decel": 4.5,
}
_BASELINE_BACKGROUND_PARAMETERS = {
    "capacityFactor": 1.0,
    "minGap_background": 2.5,
    "impatience": 0.5,
}


def _mechanism_value(manifest: Mapping[str, Any], layer: str) -> bool:
    mechanisms = manifest.get("mechanisms", {})
    key = f"{layer}_enabled"
    value = mechanisms.get(key, manifest.get(key, manifest[layer].get("enabled", _MISSING)))
    if not isinstance(value, bool):
        raise ContractError(f"{manifest.get('config_id')}.{key} must be boolean")
    return value


def _l2_freeze_hashes(manifest: Mapping[str, Any]) -> tuple[str, str]:
    l2 = manifest["l2"]
    prior_hash = canonical_sha256(l2["priors"])
    seed_value = _require_one(l2, ("ensemble_seed_schedule", "seed_schedule", "ensemble_seed_hash"), "manifest.l2")
    seed_hash = str(seed_value).lower() if isinstance(seed_value, str) and _SHA256_RE.fullmatch(seed_value) else canonical_sha256(seed_value)
    return prior_hash, seed_hash


def _comparison_context_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Hash the frozen, non-mechanism comparison context."""

    routes = sorted(canonical_json(dict(route)) for route in manifest["routes"])
    dataset_windows = sorted(
        canonical_json(
            {
                "path": dataset["path"],
                "observation_date": dataset["observation_date"],
                "timezone": dataset["timezone"],
                "time_window": dataset["time_window"],
            }
        )
        for dataset in manifest["datasets"]
    )
    return {
        "routes": canonical_sha256(routes),
        "dataset_windows": canonical_sha256(dataset_windows),
        "audit": canonical_sha256(manifest["audit"]),
    }


def _matches_exact_parameters(actual: Mapping[str, Any], expected: Mapping[str, float]) -> bool:
    """Compare a JSON parameter object by keys and numerical values."""

    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        value = actual[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)) or float(value) != expected_value:
            return False
    return True


def _require_baseline_parameters(
    actual: Mapping[str, Any],
    expected: Mapping[str, float],
    message: str,
) -> None:
    if not _matches_exact_parameters(actual, expected):
        raise ContractError(message)


def _require_equal(values: Sequence[str], message: str) -> None:
    if len(set(values)) != 1:
        raise ContractError(message)


def validate_mechanism_matrix(manifests: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, str]]:
    """Validate A0-A4 mechanisms and required frozen-input equalities.

    Validation is performed independently for each seed on exactly one split.
    Final background parameter hashes are deliberately not required to differ
    (or to be equal).
    """

    if not manifests:
        raise ContractError("At least one manifest is required")
    validated: list[Mapping[str, Any]] = []
    splits: set[str] = set()
    for manifest in manifests:
        validate_paper_manifest(manifest)
        split = _require_nonempty(manifest, "split", "manifest")
        if not isinstance(split, str):
            raise ContractError("manifest.split must be a non-empty string")
        splits.add(split)
        validated.append(manifest)
    if len(splits) != 1:
        raise ContractError("A mechanism-matrix validation call must contain exactly one split")

    grouped: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    hashes: dict[tuple[int, str], dict[str, str]] = {}
    freeze_hashes: dict[tuple[int, str], tuple[str, str]] = {}
    context_hashes: dict[tuple[int, str], dict[str, str]] = {}

    for manifest in validated:
        config_id = str(manifest["config_id"])
        if config_id not in _MECHANISM_MATRIX:
            raise ContractError(f"Unknown ablation config_id: {config_id}")
        seed = int(manifest["seed"])
        simulator_seed = manifest["simulator"]["seed"]
        if isinstance(simulator_seed, bool) or not isinstance(simulator_seed, int):
            raise ContractError(
                f"seed={seed} {config_id}: SUMO seed must be an integer"
            )
        if config_id in grouped[seed]:
            raise ContractError(f"Duplicate manifest for seed={seed}, config_id={config_id}")
        grouped[seed][config_id] = manifest
        hashes[(seed, config_id)] = compute_component_hashes(manifest)
        freeze_hashes[(seed, config_id)] = _l2_freeze_hashes(manifest)
        context_hashes[(seed, config_id)] = _comparison_context_hashes(manifest)

    for seed, configs in grouped.items():
        missing = sorted(set(_MECHANISM_MATRIX) - set(configs))
        if missing:
            raise ContractError(f"seed={seed} is missing configurations: {', '.join(missing)}")
        for config_id, (expected_l1, expected_l2, semantics) in _MECHANISM_MATRIX.items():
            manifest = configs[config_id]
            if _mechanism_value(manifest, "l1") is not expected_l1:
                raise ContractError(f"seed={seed} {config_id} has incorrect l1_enabled")
            if _mechanism_value(manifest, "l2") is not expected_l2:
                raise ContractError(f"seed={seed} {config_id} has incorrect l2_enabled")
            semantic = str(manifest["l2"]["observation_semantic"]).lower()
            if semantic not in semantics:
                raise ContractError(f"seed={seed} {config_id} has incorrect observation_semantic={semantic!r}")

        component = {config: hashes[(seed, config)] for config in _MECHANISM_MATRIX}
        context = {config: context_hashes[(seed, config)] for config in _MECHANISM_MATRIX}
        for context_name, description in (
            ("routes", "routes"),
            ("dataset_windows", "dataset windows"),
            ("audit", "frozen audit information"),
        ):
            _require_equal(
                [context[config][context_name] for config in _MECHANISM_MATRIX],
                f"seed={seed}: A0-A4 must share the same {description}",
            )
        _require_equal(
            [component[config]["simulator_inputs"] for config in _MECHANISM_MATRIX],
            (
                f"seed={seed}: A0-A4 must share the SUMO seed, SUMO settings, "
                "and effective input hashes"
            ),
        )

        bus_parameters = {
            config: _first_mapping(
                configs[config]["l1"],
                ("bus_parameters", "selected_parameters", "effective_parameters"),
                "manifest.l1",
            )
            for config in _MECHANISM_MATRIX
        }
        background_parameters = {
            config: _first_mapping(
                configs[config]["l2"],
                ("background_parameters", "final_parameters", "effective_parameters"),
                "manifest.l2",
            )
            for config in _MECHANISM_MATRIX
        }
        _require_equal(
            [component[config]["bus_parameters"] for config in ("A0", "A2")],
            f"seed={seed}: A0 and A2 must share baseline bus parameters",
        )
        for config in ("A0", "A2"):
            _require_baseline_parameters(
                bus_parameters[config],
                _BASELINE_BUS_PARAMETERS,
                f"seed={seed} {config} must use the exact design baseline bus parameters",
            )
        _require_equal(
            [component[config]["bus_parameters"] for config in ("A1", "A3", "A4")],
            f"seed={seed}: A1, A3, and A4 must share frozen L1 bus parameters",
        )
        if any(
            component[config]["bus_parameters"] == component["A0"]["bus_parameters"]
            or _matches_exact_parameters(bus_parameters[config], _BASELINE_BUS_PARAMETERS)
            for config in ("A1", "A3", "A4")
        ):
            raise ContractError(
                f"seed={seed}: enabled L1 configurations cannot share the baseline bus hash"
            )
        _require_equal(
            [component[config]["background_parameters"] for config in ("A0", "A1")],
            f"seed={seed}: A0 and A1 must share baseline background parameters",
        )
        for config in ("A0", "A1"):
            _require_baseline_parameters(
                background_parameters[config],
                _BASELINE_BACKGROUND_PARAMETERS,
                f"seed={seed} {config} must use the exact design baseline background parameters",
            )
        _require_equal(
            [freeze_hashes[(seed, config)][0] for config in ("A2", "A3", "A4")],
            f"seed={seed}: A2, A3, and A4 must share L2 priors",
        )
        _require_equal(
            [freeze_hashes[(seed, config)][1] for config in ("A2", "A3", "A4")],
            f"seed={seed}: A2, A3, and A4 must share the ensemble seed schedule",
        )
        if component["A3"]["observation_semantic"] == component["A4"]["observation_semantic"]:
            raise ContractError(f"seed={seed}: A3 and A4 observation semantics must differ")
    return hashes


__all__ = [
    "ContractError",
    "PAPER_MANIFEST_SCHEMA",
    "canonical_json",
    "canonical_sha256",
    "compute_component_hashes",
    "compute_manifest_hashes",
    "compute_provenance_hash",
    "compute_simulation_effective_hash",
    "hash_path",
    "sha256_directory",
    "sha256_file",
    "validate_mechanism_matrix",
    "validate_paper_manifest",
]
