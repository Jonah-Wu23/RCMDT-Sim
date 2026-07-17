"""Strict camera-ready reporting figures and artifact provenance.

The reporting layer consumes only frozen CSV/JSON artifacts under one run
directory.  It never synthesizes, imputes, clips, or silently drops production
data.  The explicit input contracts are:

``manifests/effective_manifest.json``
    ``paper-manifest/v1`` with the frozen ``theta_bus``, ``x_corr``, Rule C,
    development split, and cross-day split.
``reporting/fig2_contamination.csv``
    ``fig2-contamination/v1`` columns ``schema_version,event_id,``
    ``travel_time_s,speed_kmh,distance_m,rule_c_flagged``.  ``event_id`` is
    the key; the numeric units are seconds, km/h, and metres, respectively.
``reporting/fig2_trajectory.csv``
    ``fig2-trajectory/v1`` columns ``schema_version,split,route,bound,``
    ``trajectory_id,source,time_basis,point_index,cumulative_time_s,``
    ``cumulative_distance_m``.  The key is ``(split, route, bound,``
    ``trajectory_id, source, time_basis, point_index)``; cumulative units are
    seconds and metres.  The three exact series are
    ``observed/traffic_only``, ``simulated/full``, and
    ``simulated/traffic_only``.
``reporting/fig3_sensitivity.csv``
    ``fig3-sensitivity/v2`` columns ``schema_version,method_id,split,``
    ``travel_time_gt_s,speed_lt_kmh,distance_lte_m,``
    ``n_eligible_raw_link_keys,n_a0_supported_raw_link_keys,``
    ``n_a0_unsupported_raw_link_keys,n_clean_link_keys,``
    ``n_evaluation_link_keys,retention_rate,ks_speed,status``.  The threshold
    triple plus ``method_id`` and ``split`` is the key; threshold units are
    seconds, km/h, and metres, and both rates are dimensionless on [0, 1].
``audit/audit_metrics.csv``
    ``audit-metrics/v2`` columns ``schema_version,method_id,method_label,``
    ``split,n_eligible_raw_link_keys,n_a0_supported_raw_link_keys,``
    ``n_a0_unsupported_raw_link_keys,n_retained_link_keys,``
    ``n_evaluation_link_keys,retention_rate,ks_speed,worst_15min_ks,``
    ``n_real,n_sim,irn_numerator,irn_denominator,unmatched_flagged,status``.
    The key is ``(method_id, split)`` and rate/K-S values are dimensionless on
    [0, 1].  A zero IRN denominator is an explicitly undefined rate.
``reporting/fig4_cdf_samples.csv``
    ``fig4-cdf-samples/v1`` columns ``schema_version,split,config_id,seed,``
    ``source,domain,event_id,link_key,value,unit,status``.  The key is
    ``(split, config_id, seed, source, domain, event_id)``.  ``config_id`` is
    A4, ``domain`` is speed, ``unit`` is km/h, ``source`` is ``real_clean``
    or ``simulation``, and seed -1 denotes the single seed-independent real
    population; simulations use three to five common non-negative seeds.
``l1/bo_lhs_evaluations.csv``
    ``bo-lhs-evaluations/v1`` complete equal-budget optimization histories.
``tables/table_i.csv``
    ``table-i/v1`` validated A0-A4 aggregate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .contracts import canonical_sha256, sha256_file


ARTIFACT_SIDECAR_SCHEMA = "artifact-sidecar/v1"
FIGURE_DPI = 300
FONT_FAMILY = "Times New Roman"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
BLACK = "#111111"

FIGURE_FILES = {
    "fig1": "Fig1_camera_ready_architecture.png",
    "fig2": "Fig2_camera_ready_contamination.png",
    "fig3": "Fig3_camera_ready_audit.png",
    "fig4": "Fig4_camera_ready_cdf.png",
    "fig5": "Fig5_camera_ready_bo_lhs.png",
    "table_i": "Table_I_camera_ready_ablation.png",
}

INPUT_FILES = {
    "manifest": "manifests/effective_manifest.json",
    "fig2_contamination": "reporting/fig2_contamination.csv",
    "fig2_trajectory": "reporting/fig2_trajectory.csv",
    "fig3_sensitivity": "reporting/fig3_sensitivity.csv",
    "audit_metrics": "audit/audit_metrics.csv",
    "fig4_cdf": "reporting/fig4_cdf_samples.csv",
    "fig5_evaluations": "l1/bo_lhs_evaluations.csv",
    "table_i": "tables/table_i.csv",
}

_STYLE = {
    "font.family": "serif",
    "font.serif": [FONT_FAMILY, "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_THETA_BUS = ("t_board", "t_fixed", "tau", "sigma", "minGap_bus", "accel", "decel")
_X_CORR = ("capacityFactor", "minGap_background", "impatience")
_SENSITIVITY_T = (275.0, 325.0, 375.0)
_SENSITIVITY_V = (4.0, 5.0, 6.0)
_SPLITS = ("development", "cross_day")
_CDF_SOURCES = ("real_clean", "simulation")
_TRAJECTORY_SERIES = (
    ("observed", "traffic_only"),
    ("simulated", "full"),
    ("simulated", "traffic_only"),
)
_CONFIG_ORDER = ("A0", "A1", "A2", "A3", "A4")

FIG2_CONTAMINATION_COLUMNS = (
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
FIG2_TRAJECTORY_COLUMNS = (
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
FIG3_SENSITIVITY_COLUMNS = (
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
AUDIT_METRICS_COLUMNS = (
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
FIG4_CDF_COLUMNS = (
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


class FigureContractError(ValueError):
    """Raised before plotting when a reporting artifact violates its schema."""


@dataclass(frozen=True)
class PreparedArtifact:
    artifact_id: str
    input_paths: tuple[Path, ...]
    schema_versions: Mapping[str, str]
    data: Any
    render: Callable[[Any], Figure]


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FigureContractError(f"Required reporting input is missing: {path}")
    return path


def _relative_input(path: Path, input_root: Path) -> str:
    try:
        return path.resolve().relative_to(input_root.resolve()).as_posix()
    except ValueError as exc:
        raise FigureContractError(f"Reporting input is outside the run directory: {path}") from exc


def _read_json(path: Path) -> Mapping[str, Any]:
    _require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FigureContractError(f"Malformed JSON reporting input: {path}") from exc
    if not isinstance(value, Mapping):
        raise FigureContractError(f"JSON reporting input must be an object: {path}")
    return value


def _read_csv(
    path: Path,
    *,
    schema_version: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    _require_file(path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise FigureContractError(f"Cannot read CSV reporting input: {path}") from exc
    required = set(required_columns) | {"schema_version"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FigureContractError(f"{path} is missing required columns: {missing}")
    unexpected = sorted(set(frame.columns).difference(required))
    if unexpected:
        raise FigureContractError(f"{path} has unexpected columns: {unexpected}")
    if frame.empty:
        raise FigureContractError(f"{path} contains no rows")
    versions = set(frame["schema_version"].astype(str))
    if versions != {schema_version}:
        raise FigureContractError(
            f"{path} schema_version must be {schema_version!r}; got {sorted(versions)}"
        )
    return frame


def _finite_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source: Path,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FigureContractError(f"{source}:{column} contains missing or non-finite values")
        if minimum is not None and np.any(values < minimum):
            raise FigureContractError(f"{source}:{column} contains values below {minimum}")
        if maximum is not None and np.any(values > maximum):
            raise FigureContractError(f"{source}:{column} contains values above {maximum}")
        frame[column] = values


def _integer_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source: Path,
    minimum: int = 0,
) -> None:
    _finite_numeric(frame, columns, source=source, minimum=float(minimum))
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise FigureContractError(f"{source}:{column} must contain integers")
        frame[column] = values.astype(int)


def _nonempty_strings(frame: pd.DataFrame, columns: Sequence[str], *, source: Path) -> None:
    for column in columns:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise FigureContractError(f"{source}:{column} contains empty values")
        frame[column] = values.str.strip().astype(str)


def _boolean_series(values: pd.Series, *, source: Path, column: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = normalized.isna() | ~normalized.isin(mapping)
    if invalid.any():
        raise FigureContractError(f"{source}:{column} must contain only true/false values")
    return normalized.map(mapping).astype(bool)


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], *, source: Path) -> None:
    if frame.duplicated(list(columns), keep=False).any():
        raise FigureContractError(f"{source} has duplicate keys for columns {list(columns)}")


def _manifest_value(manifest: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = manifest
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise FigureContractError(f"effective manifest is missing {'.'.join(path)}")
        value = value[key]
    return value


def _prepare_fig1(manifest_path: Path) -> PreparedArtifact:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "paper-manifest/v1":
        raise FigureContractError("effective manifest must use paper-manifest/v1")
    if _manifest_value(manifest, ("l1", "parameter_scope")) != "theta_bus":
        raise FigureContractError("effective manifest l1.parameter_scope must be theta_bus")
    bounds = _manifest_value(manifest, ("l1", "parameter_bounds"))
    if not isinstance(bounds, Mapping) or set(bounds) != set(_THETA_BUS):
        raise FigureContractError(f"theta_bus must contain exactly {list(_THETA_BUS)}")
    if _manifest_value(manifest, ("l2", "state_name")) != "x_corr":
        raise FigureContractError("effective manifest l2.state_name must be x_corr")
    components = _manifest_value(manifest, ("l2", "state_components"))
    if list(components) != list(_X_CORR):
        raise FigureContractError(f"x_corr must contain exactly {list(_X_CORR)}")
    conditions = _manifest_value(manifest, ("audit", "conditions"))
    expected_conditions = {
        "travel_time_gt_s": 325.0,
        "speed_lt_kmh": 5.0,
        "distance_lte_m": 1500.0,
    }
    for key, expected in expected_conditions.items():
        try:
            actual = float(conditions[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise FigureContractError(f"effective manifest audit.conditions.{key} is invalid") from exc
        if not math.isfinite(actual) or actual != expected:
            raise FigureContractError(
                f"effective manifest audit.conditions.{key} must be {expected}"
            )
    for split in _SPLITS:
        _manifest_value(manifest, ("splits", split, "date"))
    return PreparedArtifact(
        artifact_id="Fig1",
        input_paths=(manifest_path,),
        schema_versions={manifest_path.name: "paper-manifest/v1"},
        data=manifest,
        render=_render_fig1,
    )


def _prepare_fig2(contamination_path: Path, trajectory_path: Path) -> PreparedArtifact:
    contamination = _read_csv(
        contamination_path,
        schema_version="fig2-contamination/v1",
        required_columns=FIG2_CONTAMINATION_COLUMNS[1:],
    )
    _nonempty_strings(
        contamination,
        ("event_id", "route", "bound"),
        source=contamination_path,
    )
    _require_unique(contamination, ("event_id",), source=contamination_path)
    _integer_columns(
        contamination,
        ("from_seq", "to_seq"),
        source=contamination_path,
        minimum=1,
    )
    _finite_numeric(
        contamination,
        ("travel_time_s", "speed_kmh", "distance_m"),
        source=contamination_path,
        minimum=0.0,
    )
    if (
        (contamination[["travel_time_s", "speed_kmh", "distance_m"]] <= 0).any().any()
        or (contamination["distance_m"] > 1500.0).any()
    ):
        raise FigureContractError(
            "Fig2 contamination rows must satisfy positive time/speed/distance and distance <= 1500 m"
        )
    contamination["rule_c_flagged"] = _boolean_series(
        contamination["rule_c_flagged"], source=contamination_path, column="rule_c_flagged"
    )
    link_key = ["route", "bound", "from_seq", "to_seq"]
    flag_counts = contamination.groupby(link_key, observed=True)["rule_c_flagged"].nunique()
    if (flag_counts != 1).any():
        raise FigureContractError(
            "Fig2 events sharing one link-hour key must have one Rule C decision"
        )
    link_hour = (
        contamination.groupby(link_key, as_index=False, observed=True)
        .agg(
            tt_median=("travel_time_s", "median"),
            speed_median=("speed_kmh", "median"),
            distance_median=("distance_m", "median"),
            rule_c_flagged=("rule_c_flagged", "first"),
        )
    )
    expected_flags = (
        (link_hour["tt_median"] > 325.0)
        & (link_hour["speed_median"] < 5.0)
        & (link_hour["distance_median"] <= 1500.0)
    )
    if not np.array_equal(link_hour["rule_c_flagged"].to_numpy(), expected_flags.to_numpy()):
        raise FigureContractError(
            "Fig2 link-hour Rule C decisions do not match strict median-based Rule C"
        )
    n_clean = int((~contamination["rule_c_flagged"]).sum())
    n_flagged = int(contamination["rule_c_flagged"].sum())
    if len(contamination) < 20 or n_clean < 20 or n_flagged < 1:
        raise FigureContractError(
            "Fig2 requires at least 20 raw events, 20 clean events, and one flagged event"
        )

    trajectory = _read_csv(
        trajectory_path,
        schema_version="fig2-trajectory/v1",
        required_columns=FIG2_TRAJECTORY_COLUMNS[1:],
    )
    _nonempty_strings(
        trajectory,
        ("split", "route", "bound", "trajectory_id", "source", "time_basis"),
        source=trajectory_path,
    )
    trajectory["split"] = trajectory["split"].str.casefold().str.replace("-", "_", regex=False)
    trajectory["source"] = trajectory["source"].str.casefold()
    trajectory["time_basis"] = trajectory["time_basis"].str.casefold()
    if set(trajectory["split"]) != {"development"}:
        raise FigureContractError("Fig2 trajectory split must be exactly development")
    actual_series = set(zip(trajectory["source"], trajectory["time_basis"], strict=True))
    if actual_series != set(_TRAJECTORY_SERIES):
        raise FigureContractError(
            "Fig2 trajectory series must be observed/traffic_only, simulated/full, "
            "and simulated/traffic_only"
        )
    _integer_columns(trajectory, ("point_index",), source=trajectory_path, minimum=0)
    _finite_numeric(
        trajectory,
        ("cumulative_time_s", "cumulative_distance_m"),
        source=trajectory_path,
        minimum=0.0,
    )
    trajectory_key = (
        "split",
        "route",
        "bound",
        "trajectory_id",
        "source",
        "time_basis",
        "point_index",
    )
    _require_unique(trajectory, trajectory_key, source=trajectory_path)
    group_key = ("split", "route", "bound", "trajectory_id", "source", "time_basis")
    for key, group in trajectory.groupby(list(group_key), sort=True):
        ordered = group.sort_values("point_index")
        if len(ordered) < 2:
            raise FigureContractError(f"Fig2 trajectory {key!r} has fewer than two points")
        indices = ordered["point_index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise FigureContractError(
                f"Fig2 trajectory {key!r} point_index must be consecutive from zero"
            )
        if np.any(np.diff(ordered["cumulative_time_s"].to_numpy()) <= 0):
            raise FigureContractError(
                f"Fig2 trajectory {key!r} cumulative_time_s must increase strictly"
            )
        if np.any(np.diff(ordered["cumulative_distance_m"].to_numpy()) < 0):
            raise FigureContractError(
                f"Fig2 trajectory {key!r} cumulative_distance_m must not decrease"
            )
    comparison_keys = set(
        zip(
            trajectory["split"],
            trajectory["route"],
            trajectory["bound"],
            trajectory["trajectory_id"],
            strict=True,
        )
    )
    for comparison_key in comparison_keys:
        split, route, bound, trajectory_id = comparison_key
        comparison = trajectory.loc[
            (trajectory["split"] == split)
            & (trajectory["route"] == route)
            & (trajectory["bound"] == bound)
            & (trajectory["trajectory_id"] == trajectory_id)
        ]
        if set(zip(comparison["source"], comparison["time_basis"], strict=True)) != set(
            _TRAJECTORY_SERIES
        ):
            raise FigureContractError(
                f"Fig2 trajectory comparison {comparison_key!r} lacks one required series"
            )
        expected_points: set[tuple[int, float]] | None = None
        for series in _TRAJECTORY_SERIES:
            series_rows = comparison.loc[
                (comparison["source"] == series[0])
                & (comparison["time_basis"] == series[1])
            ]
            points = set(
                zip(
                    series_rows["point_index"].astype(int),
                    series_rows["cumulative_distance_m"].astype(float),
                    strict=True,
                )
            )
            if expected_points is None:
                expected_points = points
            elif points != expected_points:
                raise FigureContractError(
                    f"Fig2 trajectory comparison {comparison_key!r} uses unmatched points/distances"
                )

    return PreparedArtifact(
        artifact_id="Fig2",
        input_paths=(contamination_path, trajectory_path),
        schema_versions={
            contamination_path.name: "fig2-contamination/v1",
            trajectory_path.name: "fig2-trajectory/v1",
        },
        data=(contamination, trajectory),
        render=_render_fig2,
    )


def _prepare_fig3(sensitivity_path: Path, audit_path: Path) -> PreparedArtifact:
    sensitivity = _read_csv(
        sensitivity_path,
        schema_version="fig3-sensitivity/v2",
        required_columns=FIG3_SENSITIVITY_COLUMNS[1:],
    )
    _nonempty_strings(sensitivity, ("method_id", "split", "status"), source=sensitivity_path)
    sensitivity["method_id"] = sensitivity["method_id"].str.casefold()
    sensitivity["split"] = sensitivity["split"].str.casefold().str.replace("-", "_", regex=False)
    if set(sensitivity["method_id"]) != {"rule_c"}:
        raise FigureContractError("Fig3 sensitivity method_id must be exactly rule_c")
    if set(sensitivity["split"]) != {"development"}:
        raise FigureContractError("Fig3 sensitivity split must be exactly development")
    if set(sensitivity["status"]) != {"succeeded"}:
        raise FigureContractError("Fig3 accepts only succeeded sensitivity rows")
    _finite_numeric(
        sensitivity,
        ("travel_time_gt_s", "speed_lt_kmh", "distance_lte_m"),
        source=sensitivity_path,
        minimum=0.0,
    )
    _integer_columns(
        sensitivity,
        (
            "n_eligible_raw_link_keys",
            "n_a0_supported_raw_link_keys",
            "n_clean_link_keys",
            "n_evaluation_link_keys",
        ),
        source=sensitivity_path,
        minimum=1,
    )
    _integer_columns(
        sensitivity,
        ("n_a0_unsupported_raw_link_keys",),
        source=sensitivity_path,
        minimum=0,
    )
    _finite_numeric(
        sensitivity,
        ("retention_rate", "ks_speed"),
        source=sensitivity_path,
        minimum=0.0,
        maximum=1.0,
    )
    _require_unique(
        sensitivity,
        ("method_id", "split", "travel_time_gt_s", "speed_lt_kmh", "distance_lte_m"),
        source=sensitivity_path,
    )
    actual_grid = set(
        zip(sensitivity["travel_time_gt_s"], sensitivity["speed_lt_kmh"], strict=True)
    )
    expected_grid = {(tt, speed) for tt in _SENSITIVITY_T for speed in _SENSITIVITY_V}
    if actual_grid != expected_grid or len(sensitivity) != 9:
        raise FigureContractError("Fig3 sensitivity must contain the exact predeclared 3x3 grid")
    if not np.equal(sensitivity["distance_lte_m"].to_numpy(), 1500.0).all():
        raise FigureContractError("Fig3 sensitivity distance_lte_m must be 1500")
    eligible_counts = set(sensitivity["n_eligible_raw_link_keys"])
    if len(eligible_counts) != 1:
        raise FigureContractError("Fig3 sensitivity must use one common eligible link-key universe")
    supported_counts = set(sensitivity["n_a0_supported_raw_link_keys"])
    unsupported_counts = set(sensitivity["n_a0_unsupported_raw_link_keys"])
    if len(supported_counts) != 1 or len(unsupported_counts) != 1:
        raise FigureContractError("Fig3 sensitivity must use one frozen A0-supported universe")
    if not np.equal(
        sensitivity["n_a0_supported_raw_link_keys"]
        + sensitivity["n_a0_unsupported_raw_link_keys"],
        sensitivity["n_eligible_raw_link_keys"],
    ).all():
        raise FigureContractError("Fig3 A0-supported and unsupported counts must partition raw keys")
    if (
        (sensitivity["n_clean_link_keys"] > sensitivity["n_eligible_raw_link_keys"])
        | (
            sensitivity["n_evaluation_link_keys"]
            > sensitivity["n_a0_supported_raw_link_keys"]
        )
        | (sensitivity["n_evaluation_link_keys"] > sensitivity["n_clean_link_keys"])
    ).any():
        raise FigureContractError("Fig3 sensitivity link-key counts are inconsistent")
    expected_retention = (
        sensitivity["n_clean_link_keys"] / sensitivity["n_eligible_raw_link_keys"]
    )
    if not np.allclose(
        sensitivity["retention_rate"], expected_retention, rtol=0.0, atol=1e-12
    ):
        raise FigureContractError("Fig3 retention_rate does not match clean/eligible counts")

    audit = _read_csv(
        audit_path,
        schema_version="audit-metrics/v2",
        required_columns=AUDIT_METRICS_COLUMNS[1:],
    )
    _nonempty_strings(audit, ("method_id", "method_label", "split", "status"), source=audit_path)
    audit["method_id"] = audit["method_id"].str.casefold()
    audit["split"] = audit["split"].str.casefold().str.replace("-", "_", regex=False)
    if set(audit["split"]) != set(_SPLITS):
        raise FigureContractError(f"audit_metrics splits must be exactly {list(_SPLITS)}")
    if set(audit["status"]) != {"succeeded"}:
        raise FigureContractError("Fig3 accepts only succeeded audit metric rows")
    allowed_methods = {"rule_c", "mad", "iqr", "isolation_forest", "quantile_fallback"}
    methods = set(audit["method_id"])
    if not methods.issubset(allowed_methods):
        raise FigureContractError(f"audit_metrics has unsupported methods: {sorted(methods - allowed_methods)}")
    adaptive_methods = methods.intersection({"isolation_forest", "quantile_fallback"})
    if not {"rule_c", "mad"}.issubset(methods) or len(adaptive_methods) != 1:
        raise FigureContractError(
            "Fig3 requires Rule C, MAD, and exactly one adaptive audit method"
        )
    _require_unique(audit, ("method_id", "split"), source=audit_path)
    expected_pairs = {(method, split) for method in methods for split in _SPLITS}
    actual_pairs = set(zip(audit["method_id"], audit["split"], strict=True))
    if actual_pairs != expected_pairs:
        raise FigureContractError("Every Fig3 audit method must report both fixed splits")
    _finite_numeric(
        audit,
        ("retention_rate", "ks_speed", "worst_15min_ks"),
        source=audit_path,
        minimum=0.0,
        maximum=1.0,
    )
    _integer_columns(
        audit,
        (
            "n_eligible_raw_link_keys",
            "n_a0_supported_raw_link_keys",
            "n_a0_unsupported_raw_link_keys",
            "n_retained_link_keys",
            "n_evaluation_link_keys",
            "n_real",
            "n_sim",
            "irn_numerator",
            "irn_denominator",
            "unmatched_flagged",
        ),
        source=audit_path,
        minimum=0,
    )
    if (audit[["n_real", "n_sim"]] < 20).any().any():
        raise FigureContractError("Fig3 full-window audit metrics require n_real and n_sim >= 20")
    if (
        audit[
            [
                "n_eligible_raw_link_keys",
                "n_a0_supported_raw_link_keys",
                "n_retained_link_keys",
                "n_evaluation_link_keys",
            ]
        ]
        < 1
    ).any().any():
        raise FigureContractError("Fig3 audit requires positive raw, supported, retained, and evaluation counts")
    if (audit["irn_numerator"] > audit["irn_denominator"]).any():
        raise FigureContractError("Fig3 IRN numerator cannot exceed denominator")
    if ((audit["irn_denominator"] == 0) & (audit["irn_numerator"] != 0)).any():
        raise FigureContractError("Fig3 zero IRN denominator requires a zero numerator")
    for split, group in audit.groupby("split", observed=True):
        for field in (
            "n_eligible_raw_link_keys",
            "n_a0_supported_raw_link_keys",
            "n_a0_unsupported_raw_link_keys",
        ):
            if group[field].nunique() != 1:
                raise FigureContractError(
                    f"Fig3 audit methods must share one {field} for split={split}"
                )
    if not np.equal(
        audit["n_a0_supported_raw_link_keys"] + audit["n_a0_unsupported_raw_link_keys"],
        audit["n_eligible_raw_link_keys"],
    ).all():
        raise FigureContractError("Fig3 audit support counts must partition raw eligible keys")
    if (
        (audit["n_retained_link_keys"] > audit["n_eligible_raw_link_keys"])
        | (audit["n_evaluation_link_keys"] > audit["n_a0_supported_raw_link_keys"])
        | (audit["n_evaluation_link_keys"] > audit["n_retained_link_keys"])
    ).any():
        raise FigureContractError("Fig3 audit link-key counts are inconsistent")
    expected_audit_retention = (
        audit["n_retained_link_keys"] / audit["n_eligible_raw_link_keys"]
    )
    if not np.allclose(audit["retention_rate"], expected_audit_retention, rtol=0.0, atol=1e-12):
        raise FigureContractError("Fig3 audit retention_rate must use all raw eligible keys")
    contradiction_rate = np.full(len(audit), np.nan, dtype=float)
    denominator = audit["irn_denominator"].to_numpy(dtype=float)
    np.divide(
        audit["irn_numerator"].to_numpy(dtype=float),
        denominator,
        out=contradiction_rate,
        where=denominator > 0,
    )
    audit["irn_contradiction_rate"] = contradiction_rate

    return PreparedArtifact(
        artifact_id="Fig3",
        input_paths=(sensitivity_path, audit_path),
        schema_versions={
            sensitivity_path.name: "fig3-sensitivity/v2",
            audit_path.name: "audit-metrics/v2",
        },
        data=(sensitivity, audit),
        render=_render_fig3,
    )


def _prepare_fig4(cdf_path: Path) -> PreparedArtifact:
    frame = _read_csv(
        cdf_path,
        schema_version="fig4-cdf-samples/v1",
        required_columns=FIG4_CDF_COLUMNS[1:],
    )
    _nonempty_strings(
        frame,
        ("split", "config_id", "source", "domain", "event_id", "link_key", "unit", "status"),
        source=cdf_path,
    )
    frame["split"] = frame["split"].str.casefold().str.replace("-", "_", regex=False)
    frame["source"] = frame["source"].str.casefold()
    frame["domain"] = frame["domain"].str.casefold()
    if set(frame["split"]) != set(_SPLITS):
        raise FigureContractError(f"Fig4 splits must be exactly {list(_SPLITS)}")
    if set(frame["config_id"]) != {"A4"}:
        raise FigureContractError("Fig4 config_id must be exactly A4")
    if set(frame["source"]) != set(_CDF_SOURCES):
        raise FigureContractError(f"Fig4 source must be exactly {list(_CDF_SOURCES)}")
    if set(frame["domain"]) != {"speed"} or set(frame["unit"]) != {"km/h"}:
        raise FigureContractError("Fig4 requires domain=speed and unit=km/h")
    if set(frame["status"]) != {"succeeded"}:
        raise FigureContractError("Fig4 accepts only succeeded sample rows")
    _integer_columns(frame, ("seed",), source=cdf_path, minimum=-1)
    _finite_numeric(frame, ("value",), source=cdf_path, minimum=0.0)
    if (frame["value"] <= 0).any():
        raise FigureContractError("Fig4 speed values must be positive")
    real = frame.loc[frame["source"] == "real_clean"]
    simulation = frame.loc[frame["source"] == "simulation"]
    if set(real["seed"]) != {-1} or (simulation["seed"] < 0).any():
        raise FigureContractError(
            "Fig4 real_clean rows must use seed=-1 and simulation rows non-negative seeds"
        )
    simulation_seeds = set(simulation["seed"].astype(int))
    if not 3 <= len(simulation_seeds) <= 5:
        raise FigureContractError("Fig4 requires three to five common simulation seeds")
    _require_unique(
        frame,
        ("split", "config_id", "seed", "source", "domain", "event_id"),
        source=cdf_path,
    )
    for split in _SPLITS:
        real_split = real.loc[real["split"] == split]
        if len(real_split) < 20:
            raise FigureContractError(f"Fig4 {split}/real_clean requires at least 20 samples")
        real_links = set(real_split["link_key"])
        split_simulation = simulation.loc[simulation["split"] == split]
        if set(split_simulation["seed"].astype(int)) != simulation_seeds:
            raise FigureContractError("Fig4 simulation seeds must be common across both splits")
        for seed in sorted(simulation_seeds):
            group = split_simulation.loc[split_simulation["seed"] == seed]
            if len(group) < 20:
                raise FigureContractError(
                    f"Fig4 {split}/simulation seed {seed} requires at least 20 samples"
                )
            if set(group["link_key"]) != real_links:
                raise FigureContractError(
                    f"Fig4 {split} real and simulation seed {seed} link-key sets differ"
                )
    return PreparedArtifact(
        artifact_id="Fig4",
        input_paths=(cdf_path,),
        schema_versions={cdf_path.name: "fig4-cdf-samples/v1"},
        data=frame,
        render=_render_fig4,
    )


def _prepare_fig5(evaluations_path: Path) -> PreparedArtifact:
    frame = _read_csv(
        evaluations_path,
        schema_version="bo-lhs-evaluations/v1",
        required_columns=(
            "optimization_seed",
            "method",
            "evaluation_index",
            "candidate_hash",
            "objective",
            "feasible",
        ),
    )
    _nonempty_strings(frame, ("method", "candidate_hash"), source=evaluations_path)
    frame["method"] = frame["method"].str.upper()
    if set(frame["method"]) != {"BO", "LHS"}:
        raise FigureContractError("Fig5 methods must be exactly BO and LHS")
    invalid_hash = ~frame["candidate_hash"].str.casefold().str.fullmatch(_SHA256_RE)
    if invalid_hash.any():
        raise FigureContractError("Fig5 candidate_hash must contain SHA-256 digests")
    frame["candidate_hash"] = frame["candidate_hash"].str.casefold()
    _integer_columns(
        frame, ("optimization_seed", "evaluation_index"), source=evaluations_path, minimum=0
    )
    _finite_numeric(frame, ("objective",), source=evaluations_path)
    frame["feasible"] = _boolean_series(
        frame["feasible"], source=evaluations_path, column="feasible"
    )
    _require_unique(
        frame, ("optimization_seed", "method", "evaluation_index"), source=evaluations_path
    )
    seeds_by_method = {
        method: set(frame.loc[frame["method"] == method, "optimization_seed"])
        for method in ("BO", "LHS")
    }
    if seeds_by_method["BO"] != seeds_by_method["LHS"]:
        raise FigureContractError("Fig5 BO and LHS must use the same optimization seeds")
    seeds = sorted(seeds_by_method["BO"])
    if not 3 <= len(seeds) <= 5:
        raise FigureContractError("Fig5 requires three to five common valid optimization seeds")
    targets: dict[int, float] = {}
    for seed in seeds:
        groups = {
            method: frame.loc[
                (frame["optimization_seed"] == seed) & (frame["method"] == method)
            ].sort_values("evaluation_index")
            for method in ("BO", "LHS")
        }
        expected_indices = list(range(1, 41))
        for method, group in groups.items():
            if group["evaluation_index"].tolist() != expected_indices:
                raise FigureContractError(
                    f"Fig5 seed={seed} method={method} must contain evaluations 1..40"
                )
        initial_bo = groups["BO"].iloc[:15]
        initial_lhs = groups["LHS"].iloc[:15]
        if initial_bo["candidate_hash"].tolist() != initial_lhs["candidate_hash"].tolist():
            raise FigureContractError(f"Fig5 seed={seed} initial candidate hashes differ")
        if not np.allclose(
            initial_bo["objective"], initial_lhs["objective"], rtol=0.0, atol=1e-12
        ) or not np.array_equal(
            initial_bo["feasible"].to_numpy(), initial_lhs["feasible"].to_numpy()
        ):
            raise FigureContractError(f"Fig5 seed={seed} shared initial evaluations differ")
        feasible_initial = initial_bo.loc[initial_bo["feasible"], "objective"]
        if feasible_initial.empty:
            raise FigureContractError(f"Fig5 seed={seed} shared initial design has no feasible candidate")
        targets[seed] = float(0.95 * feasible_initial.min())
    return PreparedArtifact(
        artifact_id="Fig5",
        input_paths=(evaluations_path,),
        schema_versions={evaluations_path.name: "bo-lhs-evaluations/v1"},
        data=(frame, targets),
        render=_render_fig5,
    )


def _prepare_table_i(table_path: Path) -> PreparedArtifact:
    metric_fields = (
        "ks_speed_development_mean",
        "ks_speed_development_std",
        "worst_15min_ks_development_mean",
        "worst_15min_ks_development_std",
        "ks_speed_cross_day_mean",
        "ks_speed_cross_day_std",
        "worst_15min_ks_cross_day_mean",
        "worst_15min_ks_cross_day_std",
    )
    frame = _read_csv(
        table_path,
        schema_version="table-i/v1",
        required_columns=(
            "config_id",
            "configuration",
            "n_seeds",
            *metric_fields,
            "n_real_development",
            "n_real_cross_day",
            "n_sim_development_mean",
            "n_sim_development_std",
            "n_sim_cross_day_mean",
            "n_sim_cross_day_std",
            "status",
        ),
    )
    _nonempty_strings(frame, ("config_id", "configuration", "status"), source=table_path)
    if set(frame["status"]) != {"succeeded"}:
        raise FigureContractError("Table I accepts only succeeded rows")
    _require_unique(frame, ("config_id",), source=table_path)
    if set(frame["config_id"]) != set(_CONFIG_ORDER) or len(frame) != 5:
        raise FigureContractError("Table I must contain exactly A0-A4")
    _integer_columns(
        frame,
        ("n_seeds", "n_real_development", "n_real_cross_day"),
        source=table_path,
        minimum=1,
    )
    if len(set(frame["n_seeds"])) != 1 or not 3 <= int(frame["n_seeds"].iloc[0]) <= 5:
        raise FigureContractError("Table I must use one common set of three to five seeds")
    if len(set(frame["n_real_development"])) != 1 or len(set(frame["n_real_cross_day"])) != 1:
        raise FigureContractError("Table I real event counts must be fixed across A0-A4")
    if (frame[["n_real_development", "n_real_cross_day"]] < 20).any().any():
        raise FigureContractError("Table I full-window real event counts must be at least 20")
    _finite_numeric(frame, metric_fields, source=table_path, minimum=0.0, maximum=1.0)
    _finite_numeric(
        frame,
        (
            "n_sim_development_mean",
            "n_sim_development_std",
            "n_sim_cross_day_mean",
            "n_sim_cross_day_std",
        ),
        source=table_path,
        minimum=0.0,
    )
    if (frame[["n_sim_development_mean", "n_sim_cross_day_mean"]] < 20).any().any():
        raise FigureContractError("Table I mean simulation event counts must be at least 20")
    for field in metric_fields:
        if field.endswith("_std") and (frame[field] < 0).any():
            raise FigureContractError(f"Table I {field} must be non-negative")
    ordered = frame.set_index("config_id").loc[list(_CONFIG_ORDER)].reset_index()
    return PreparedArtifact(
        artifact_id="Table_I",
        input_paths=(table_path,),
        schema_versions={table_path.name: "table-i/v1"},
        data=ordered,
        render=_render_table_i,
    )


def _box(ax: Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=BLACK,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7)


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], label: str | None = None) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=BLACK, linewidth=1.0)
    )
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, label, ha="center", fontsize=6)


def _render_fig1(manifest: Mapping[str, Any]) -> Figure:
    fig, ax = plt.subplots(figsize=(7.16, 2.75), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "Operator-aware two-level calibration and freeze protocol", ha="center", fontweight="bold")

    _box(ax, (0.02, 0.54), 0.14, 0.25, "D2D observations\nDec. 19 development", LIGHT_GRAY)
    _box(ax, (0.21, 0.54), 0.16, 0.25, "Observation audit\nT > 325 s\nv < 5 km/h\nd ≤ 1500 m", "#F0E442")
    _box(
        ax,
        (0.42, 0.54),
        0.16,
        0.25,
        "L1: BO ($\\theta_{bus}$)\n"
        "$t_{board}, t_{fixed}, \\tau, \\sigma$\n"
        "$minGap_{bus}, accel, decel$",
        SKY,
    )
    _box(
        ax,
        (0.63, 0.54),
        0.15,
        0.25,
        "L2: IES ($x_{corr}$)\ncapacityFactor\nminGap_background\nimpatience",
        "#BDE3D7",
    )
    _box(ax, (0.83, 0.54), 0.15, 0.25, "Frozen validation\nDevelopment +\nDec. 30 cross-day", "#F3C5B8")
    for left, right in ((0.16, 0.21), (0.37, 0.42), (0.58, 0.63), (0.78, 0.83)):
        _arrow(ax, (left, 0.665), (right, 0.665))

    _box(ax, (0.42, 0.12), 0.16, 0.22, "Bus scope\nstop service +\nbus car-following", "#E5F3FA")
    _box(ax, (0.63, 0.12), 0.15, 0.22, "Background scope\ncapacity, gap,\nimpatience", "#E4F4EE")
    _arrow(ax, (0.50, 0.54), (0.50, 0.34), "freeze $\\theta_{bus}$")
    _arrow(ax, (0.705, 0.54), (0.705, 0.34), "freeze $x_{corr}$")
    ax.text(
        0.02,
        0.08,
        "Calibration and threshold fitting stop before cross-day evaluation; validation never updates the optimizer.",
        fontsize=6.5,
    )
    return fig


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def _panel_label(ax: Axes, label: str) -> None:
    ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", fontweight="bold")


def _render_fig2(data: tuple[pd.DataFrame, pd.DataFrame]) -> Figure:
    contamination, trajectory = data
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65), constrained_layout=True)
    raw = contamination["speed_kmh"]
    clean = contamination.loc[~contamination["rule_c_flagged"], "speed_kmh"]
    for values, label, color, linestyle in (
        (raw, f"Raw (n={len(raw)})", GRAY, ":"),
        (clean, f"Rule-C clean (n={len(clean)})", BLUE, "-"),
    ):
        x, y = _ecdf(values)
        axes[0].plot(x, y, label=label, color=color, linestyle=linestyle)
    axes[0].set_xlabel("Effective speed (km/h)")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    _panel_label(axes[0], "(a)")

    flagged = contamination["rule_c_flagged"]
    axes[1].scatter(
        contamination.loc[~flagged, "travel_time_s"],
        contamination.loc[~flagged, "speed_kmh"],
        s=11,
        alpha=0.55,
        color=BLUE,
        label="Retained link-hour key",
    )
    axes[1].scatter(
        contamination.loc[flagged, "travel_time_s"],
        contamination.loc[flagged, "speed_kmh"],
        s=16,
        alpha=0.85,
        color=ORANGE,
        marker="x",
        label="Flagged link-hour key",
    )
    axes[1].axvline(325, color=BLACK, linestyle="--", linewidth=1)
    axes[1].axhline(5, color=GRAY, linestyle=":", linewidth=1)
    axes[1].set_xlabel("Travel time (s)")
    axes[1].set_ylabel("Effective speed (km/h)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.25)
    axes[1].text(
        0.98,
        0.18,
        "Rule C on link-hour medians\ndistance ≤ 1500 m",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    _panel_label(axes[1], "(b)")

    style = {
        ("observed", "traffic_only"): (BLUE, "-", "Observed traffic-only"),
        ("simulated", "full"): (ORANGE, "-", "Simulated full time"),
        ("simulated", "traffic_only"): (ORANGE, "--", "Simulated traffic-only"),
    }
    labelled: set[tuple[str, str]] = set()
    for key, group in trajectory.groupby(
        ["split", "route", "bound", "trajectory_id", "source", "time_basis"], sort=True
    ):
        source, time_basis = key[-2:]
        series = (source, time_basis)
        color, linestyle, label = style[series]
        ordered = group.sort_values("point_index")
        axes[2].plot(
            ordered["cumulative_time_s"],
            ordered["cumulative_distance_m"],
            color=color,
            linestyle=linestyle,
            alpha=0.8,
            label=label if series not in labelled else None,
        )
        labelled.add(series)
    axes[2].set_xlabel("Cumulative time (s)")
    axes[2].set_ylabel("Cumulative distance (m)")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle=":", alpha=0.35)
    _panel_label(axes[2], "(c)")
    return fig


def _matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    pivot = frame.pivot(index="travel_time_gt_s", columns="speed_lt_kmh", values=value)
    return pivot.loc[list(_SENSITIVITY_T), list(_SENSITIVITY_V)].to_numpy(dtype=float)


def _heatmap(ax: Axes, matrix: np.ndarray, *, title: str, fmt: str, cmap: str) -> None:
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(3), [f"{value:g}" for value in _SENSITIVITY_V])
    ax.set_yticks(range(3), [f"{value:g}" for value in _SENSITIVITY_T])
    ax.set_xlabel("Speed threshold (km/h)")
    ax.set_ylabel("Travel-time threshold (s)")
    ax.set_title(title)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, format(matrix[row, column], fmt), ha="center", va="center", fontsize=6.5)
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor=BLACK, linewidth=1.5))
    plt.colorbar(image, ax=ax, fraction=0.05, pad=0.03)


def _method_order(methods: Iterable[str]) -> list[str]:
    order = ["rule_c", "mad", "iqr", "isolation_forest", "quantile_fallback"]
    present = set(methods)
    return [method for method in order if method in present]


def _render_fig3(data: tuple[pd.DataFrame, pd.DataFrame]) -> Figure:
    sensitivity, audit = data
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), constrained_layout=True)
    _heatmap(
        axes[0, 0],
        _matrix(sensitivity, "retention_rate"),
        title="(a) Rule-C retention rate",
        fmt=".2f",
        cmap="Blues",
    )
    _heatmap(
        axes[0, 1],
        _matrix(sensitivity, "ks_speed"),
        title="(b) Development KS-speed",
        fmt=".2f",
        cmap="Oranges",
    )
    cross = audit.loc[audit["split"] == "cross_day"].set_index("method_id")
    methods = _method_order(cross.index)
    labels = [str(cross.loc[method, "method_label"]) for method in methods]
    x = np.arange(len(methods), dtype=float)
    width = 0.36
    axes[1, 0].bar(x - width / 2, cross.loc[methods, "ks_speed"], width, color=BLUE, label="Full window")
    axes[1, 0].bar(
        x + width / 2,
        cross.loc[methods, "worst_15min_ks"],
        width,
        color=ORANGE,
        label="Worst 15 min",
    )
    axes[1, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 0].set_ylabel("KS-speed")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("(c) Frozen cross-day audit comparison")
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1, 1].bar(
        x - width / 2,
        cross.loc[methods, "retention_rate"],
        width,
        color=GREEN,
        label="Retention",
    )
    contradiction_rate = cross.loc[methods, "irn_contradiction_rate"].to_numpy(dtype=float)
    rendered_rate = np.where(np.isfinite(contradiction_rate), contradiction_rate, 0.0)
    axes[1, 1].bar(
        x + width / 2,
        rendered_rate,
        width,
        color=SKY,
        label="IRN contradiction",
    )
    for index, method in enumerate(methods):
        numerator = int(cross.loc[method, "irn_numerator"])
        denominator = int(cross.loc[method, "irn_denominator"])
        unmatched = int(cross.loc[method, "unmatched_flagged"])
        rate_label = "N/A" if denominator == 0 else f"{numerator}/{denominator}"
        axes[1, 1].text(
            index,
            0.04,
            f"IRN {rate_label}\nunmatched={unmatched}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=90,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        )
    axes[1, 1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_title("(d) Retention and IRN consistency")
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.35)
    return fig


def _render_fig4(frame: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), constrained_layout=True, sharey=True)
    titles = {"development": "Dec. 19 development", "cross_day": "Dec. 30 cross-day"}
    for panel, split in enumerate(_SPLITS):
        ax = axes[panel]
        real = frame.loc[(frame["split"] == split) & (frame["source"] == "real_clean")]
        x, y = _ecdf(real["value"])
        ax.plot(x, y, color=BLUE, linestyle="-", label=f"Real Rule-C clean (n={len(real)})")
        simulation = frame.loc[
            (frame["split"] == split) & (frame["source"] == "simulation")
        ]
        seeds = sorted(set(simulation["seed"].astype(int)))
        for index, seed in enumerate(seeds):
            values = simulation.loc[simulation["seed"] == seed, "value"]
            x, y = _ecdf(values)
            ax.plot(
                x,
                y,
                color=ORANGE,
                linestyle="--",
                alpha=0.45,
                linewidth=1.0,
                label=f"A4 simulations ({len(seeds)} seeds)" if index == 0 else None,
            )
        ax.set_xlabel("Effective speed (km/h)")
        ax.set_title(f"({'ab'[panel]}) {titles[split]}")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Empirical CDF")
    return fig


def _render_fig5(data: tuple[pd.DataFrame, Mapping[int, float]]) -> Figure:
    frame, targets = data
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    evaluations = np.arange(1, 41)
    for method, color, linestyle in (("BO", BLUE, "-"), ("LHS", ORANGE, "--")):
        curves = []
        subset = frame.loc[frame["method"] == method]
        for _, group in subset.groupby("optimization_seed", sort=True):
            ordered = group.sort_values("evaluation_index")
            curves.append(
                _feasible_cumulative_best(
                    ordered["objective"].to_numpy(dtype=float),
                    ordered["feasible"].to_numpy(dtype=bool),
                )
            )
        values = np.vstack(curves)
        counts = np.isfinite(values).sum(axis=0)
        mean = np.full(values.shape[1], np.nan, dtype=float)
        valid_mean = counts > 0
        mean[valid_mean] = np.nansum(values[:, valid_mean], axis=0) / counts[valid_mean]
        std = np.full(values.shape[1], np.nan, dtype=float)
        for index in np.flatnonzero(counts > 1):
            std[index] = float(np.nanstd(values[:, index], ddof=1))
        ax.plot(evaluations, mean, color=color, linestyle=linestyle, label=f"{method} mean")
        ax.fill_between(evaluations, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.axvline(15.5, color=GRAY, linestyle=":", linewidth=1, label="15 shared LHS")
    ax.axhline(float(np.mean(list(targets.values()))), color=GREEN, linestyle="-.", linewidth=1, label="Mean target")
    ax.set_xlabel("Successful candidate evaluations")
    ax.set_ylabel("JL1 score (s; lower is better)")
    ax.set_xlim(1, 40)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    return fig


def _feasible_cumulative_best(
    objectives: Sequence[float], feasible: Sequence[bool]
) -> np.ndarray:
    """Return the running best feasible objective without promoting penalties."""

    scores = np.asarray(objectives, dtype=float)
    decisions = np.asarray(feasible, dtype=bool)
    if scores.ndim != 1 or decisions.ndim != 1 or scores.shape != decisions.shape:
        raise FigureContractError("Fig5 objective and feasible arrays must be aligned vectors")
    result = np.full(scores.shape, np.nan, dtype=float)
    best = math.inf
    for index, (score, is_feasible) in enumerate(zip(scores, decisions, strict=True)):
        if is_feasible:
            best = min(best, float(score))
        if math.isfinite(best):
            result[index] = best
    return result


def _mean_std(row: pd.Series, mean_field: str, std_field: str) -> str:
    return f"{float(row[mean_field]):.3f} ± {float(row[std_field]):.3f}"


def _render_table_i(frame: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7.16, 2.25), constrained_layout=True)
    ax.axis("off")
    columns = [
        "Configuration",
        "Dev. KS",
        "Dev. worst",
        "Cross-day KS",
        "Cross-day worst",
        "$n_{sim}$ dev.",
        "$n_{sim}$ cross-day",
    ]
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                f"{row['config_id']} {row['configuration']}",
                _mean_std(row, "ks_speed_development_mean", "ks_speed_development_std"),
                _mean_std(
                    row,
                    "worst_15min_ks_development_mean",
                    "worst_15min_ks_development_std",
                ),
                _mean_std(row, "ks_speed_cross_day_mean", "ks_speed_cross_day_std"),
                _mean_std(
                    row,
                    "worst_15min_ks_cross_day_mean",
                    "worst_15min_ks_cross_day_std",
                ),
                _mean_std(row, "n_sim_development_mean", "n_sim_development_std"),
                _mean_std(row, "n_sim_cross_day_mean", "n_sim_cross_day_std"),
            ]
        )
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.35)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(LIGHT_GRAY)
            cell.set_text_props(weight="bold")
    n_dev = int(frame["n_real_development"].iloc[0])
    n_cross = int(frame["n_real_cross_day"].iloc[0])
    seeds = int(frame["n_seeds"].iloc[0])
    ax.text(
        0.0,
        0.02,
        f"Mean ± sample SD over {seeds} common seeds. Fixed real events: development n={n_dev}; cross-day n={n_cross}.",
        transform=ax.transAxes,
        fontsize=6.2,
    )
    return fig


def _prepare_all(input_root: Path, selected: Sequence[str]) -> dict[str, PreparedArtifact]:
    manifest_path = input_root / INPUT_FILES["manifest"]
    prepared: dict[str, PreparedArtifact] = {}
    if "fig1" in selected:
        prepared["fig1"] = _prepare_fig1(manifest_path)
    if "fig2" in selected:
        prepared["fig2"] = _prepare_fig2(
            input_root / INPUT_FILES["fig2_contamination"],
            input_root / INPUT_FILES["fig2_trajectory"],
        )
    if "fig3" in selected:
        prepared["fig3"] = _prepare_fig3(
            input_root / INPUT_FILES["fig3_sensitivity"],
            input_root / INPUT_FILES["audit_metrics"],
        )
    if "fig4" in selected:
        prepared["fig4"] = _prepare_fig4(input_root / INPUT_FILES["fig4_cdf"])
    if "fig5" in selected:
        prepared["fig5"] = _prepare_fig5(input_root / INPUT_FILES["fig5_evaluations"])
    if "table_i" in selected:
        prepared["table_i"] = _prepare_table_i(input_root / INPUT_FILES["table_i"])
    return prepared


def validate_camera_ready_inputs(
    input_dir: str | Path,
    *,
    selected: Sequence[str] | None = None,
) -> dict[str, PreparedArtifact]:
    """Validate every requested input before any output directory is created."""

    root = Path(input_dir)
    if not root.is_dir():
        raise FigureContractError(f"Camera-ready input directory is missing: {root}")
    chosen = tuple(selected or FIGURE_FILES)
    unknown = sorted(set(chosen).difference(FIGURE_FILES))
    if unknown:
        raise FigureContractError(f"Unknown camera-ready artifacts: {unknown}")
    if len(set(chosen)) != len(chosen):
        raise FigureContractError("selected artifacts contain duplicates")
    return _prepare_all(root, chosen)


def _sidecar_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(artifact_path.suffix + ".sidecar.json")


def _png_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise FigureContractError(f"Generated artifact is not PNG: {path}")
        dpi = image.info.get("dpi")
        if not dpi or len(dpi) != 2:
            raise FigureContractError(f"Generated PNG has no embedded DPI: {path}")
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
        if not math.isclose(dpi_x, FIGURE_DPI, abs_tol=0.5) or not math.isclose(
            dpi_y, FIGURE_DPI, abs_tol=0.5
        ):
            raise FigureContractError(f"Generated PNG is not 300 DPI: {path} ({dpi_x}, {dpi_y})")
        return {
            "format": "PNG",
            "width_px": int(image.width),
            "height_px": int(image.height),
            "dpi_x": round(dpi_x, 3),
            "dpi_y": round(dpi_y, 3),
            "font_family": FONT_FAMILY,
            "palette": [BLUE, ORANGE, GREEN, SKY, GRAY, BLACK],
        }


def _script_records(script_path: Path) -> list[dict[str, str]]:
    paths = {Path(__file__).resolve(), script_path.resolve()}
    records = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        _require_file(path)
        records.append({"path": path.as_posix(), "sha256": sha256_file(path)})
    return records


def _write_sidecar(
    *,
    artifact: PreparedArtifact,
    artifact_path: Path,
    input_root: Path,
    manifest_path: Path,
    script_path: Path,
    command: str,
) -> Path:
    if not command.strip():
        raise FigureContractError("Generation command must not be empty")
    source_paths = {manifest_path.resolve(), *(path.resolve() for path in artifact.input_paths)}
    source_inputs = [
        {
            "path": _relative_input(path, input_root),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(source_paths, key=lambda item: item.as_posix())
    ]
    scripts = _script_records(script_path)
    manifest_relative = _relative_input(manifest_path, input_root)
    manifest_hash = sha256_file(manifest_path)
    figure_metadata = _png_metadata(artifact_path)
    provenance_payload = {
        "artifact_id": artifact.artifact_id,
        "artifact_sha256": sha256_file(artifact_path),
        "source_inputs": source_inputs,
        "input_schema_versions": dict(artifact.schema_versions),
        "manifest": {"path": manifest_relative, "sha256": manifest_hash},
        "scripts": scripts,
        "command": command,
        "figure": figure_metadata,
        "sidecar_schema": ARTIFACT_SIDECAR_SCHEMA,
    }
    sidecar = {
        "schema_version": ARTIFACT_SIDECAR_SCHEMA,
        "manuscript_artifact_id": artifact.artifact_id,
        "artifact_path": artifact_path.name,
        "artifact_sha256": provenance_payload["artifact_sha256"],
        "source_inputs": source_inputs,
        "input_schema_versions": dict(artifact.schema_versions),
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_hash,
        "script_files": scripts,
        "script_sha256": canonical_sha256(scripts),
        "command": command,
        "figure": figure_metadata,
        "provenance_hash": canonical_sha256(provenance_payload),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path = _sidecar_path(artifact_path)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def generate_camera_ready_artifacts(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    script_path: str | Path,
    command: str,
    selected: Sequence[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Generate requested artifacts and ``artifact-sidecar/v1`` files.

    All requested inputs are preflighted before the output directory is
    created.  Existing artifacts are protected unless ``overwrite=True`` is
    explicitly supplied by the caller.
    """

    input_root = Path(input_dir)
    chosen = tuple(selected or FIGURE_FILES)
    prepared = validate_camera_ready_inputs(input_root, selected=chosen)
    manifest_path = _require_file(input_root / INPUT_FILES["manifest"])
    runner_path = _require_file(Path(script_path))
    destination = Path(output_dir)
    output_paths = [destination / FIGURE_FILES[key] for key in chosen]
    conflicts = [
        path
        for path in output_paths
        for candidate in (path, _sidecar_path(path))
        if candidate.exists()
    ]
    if conflicts and not overwrite:
        raise FigureContractError(f"Refusing to overwrite existing artifacts: {conflicts}")

    destination.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    with plt.rc_context(_STYLE):
        for key, output_path in zip(chosen, output_paths, strict=True):
            artifact = prepared[key]
            figure = artifact.render(artifact.data)
            try:
                figure.savefig(output_path, format="png", dpi=FIGURE_DPI, facecolor="white")
            finally:
                plt.close(figure)
            _write_sidecar(
                artifact=artifact,
                artifact_path=output_path,
                input_root=input_root,
                manifest_path=manifest_path,
                script_path=runner_path,
                command=command,
            )
            generated.append(output_path)
    return generated


__all__ = [
    "ARTIFACT_SIDECAR_SCHEMA",
    "FIGURE_DPI",
    "FIGURE_FILES",
    "FONT_FAMILY",
    "FigureContractError",
    "INPUT_FILES",
    "generate_camera_ready_artifacts",
    "validate_camera_ready_inputs",
]
