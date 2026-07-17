"""Frozen observation-audit methods used by the camera-ready experiments.

The functions in this module are deliberately side-effect free.  Development
records are fitted once, the returned model objects are serializable, and the
same objects can then be applied to the December 30 cross-day split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from .contracts import canonical_sha256


RULE_C_TRAVEL_TIME_S = 325.0
RULE_C_SPEED_KMH = 5.0
RULE_C_DISTANCE_M = 1500.0
RULE_C_SENSITIVITY_TRAVEL_TIME_S = (275.0, 325.0, 375.0)
RULE_C_SENSITIVITY_SPEED_KMH = (4.0, 5.0, 6.0)
LINK_KEY_COLUMNS = ("route", "bound", "from_seq", "to_seq", "window_start")

_EVENT_ALIASES = {
    "route": ("route", "route_id"),
    "bound": ("bound", "direction", "dir"),
    "from_seq": ("from_seq", "from_stop_seq", "origin_seq"),
    "to_seq": ("to_seq", "to_stop_seq", "destination_seq"),
    "departure_ts": ("departure_ts", "departure_time", "timestamp", "event_time"),
    "travel_time_s": ("travel_time_s", "tt_s", "travel_time", "duration_s"),
    "distance_m": ("distance_m", "dist_m", "distance"),
    "speed_kmh": ("speed_kmh", "effective_speed_kmh", "v_eff_kmh", "v_eff"),
}


class AuditError(ValueError):
    """Raised when audit inputs cannot satisfy the frozen protocol."""


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_state_value(value: Any) -> Any:
    """Convert fitted sklearn state into deterministic JSON-compatible data."""

    if isinstance(value, np.generic):
        return _canonical_state_value(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {"type": "float", "value": "nan"}
        if np.isposinf(value):
            return {"type": "float", "value": "+inf"}
        if np.isneginf(value):
            return {"type": "float", "value": "-inf"}
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, np.ndarray):
        if value.dtype.names is not None:
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "itemsize": value.dtype.itemsize,
                "fields": {
                    name: _canonical_state_value(value[name])
                    for name in value.dtype.names
                },
            }
        return {
            "type": "numpy.ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _canonical_state_value(value.tolist()),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical_state_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_state_value(item) for item in value]
    raise TypeError(f"Unsupported fitted-state value: {_qualified_type_name(value)}")


def _canonical_tree_estimator_state(estimator: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for name, value in vars(estimator).items():
        if name == "tree_":
            raw_state = value.__getstate__()
            nodes = raw_state["nodes"]
            # sklearn 1.5 leaves missing-value routing bytes undefined for
            # these finite-only random trees, so they are not fitted state.
            node_fields = {
                field: _canonical_state_value(nodes[field])
                for field in nodes.dtype.names or ()
                if field != "missing_go_to_left"
            }
            attributes[name] = {
                "type": _qualified_type_name(value),
                "state": {
                    "max_depth": _canonical_state_value(raw_state["max_depth"]),
                    "node_count": _canonical_state_value(raw_state["node_count"]),
                    "nodes": {
                        "type": "numpy.structured-array",
                        "shape": list(nodes.shape),
                        "fields": node_fields,
                    },
                    "values": _canonical_state_value(raw_state["values"]),
                    "missing_value_routing": "not-applicable-to-finite-audit-features",
                },
            }
        else:
            attributes[name] = _canonical_state_value(value)
    return {"type": _qualified_type_name(estimator), "attributes": attributes}


def _canonical_isolation_forest_state(estimator: IsolationForest) -> dict[str, Any]:
    """Serialize every semantic fitted attribute without allocator-dependent bytes."""

    attributes: dict[str, Any] = {}
    for name, value in vars(estimator).items():
        if name == "estimator_":
            attributes[name] = _canonical_tree_estimator_state(value)
        elif name == "estimators_":
            attributes[name] = [_canonical_tree_estimator_state(tree) for tree in value]
        else:
            attributes[name] = _canonical_state_value(value)
    return {"type": _qualified_type_name(estimator), "attributes": attributes}


@dataclass(frozen=True)
class MADModel:
    """Frozen robust-location model for the two Rule-C features."""

    tt_log_median: float
    speed_log_median: float
    tt_log_scale: float | None
    speed_log_scale: float | None
    tt_threshold: float = 3.5
    speed_threshold: float = -3.5
    package_version: str = np.__version__

    @property
    def frozen_parameters(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def model_hash(self) -> str:
        return canonical_sha256(self.frozen_parameters)


@dataclass(frozen=True)
class QuantileModel:
    """Frozen empirical-quantile adaptive fallback."""

    tt_q95: float
    speed_q05: float
    interpolation: str = "linear"
    package_version: str = np.__version__

    @property
    def frozen_parameters(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def model_hash(self) -> str:
        return canonical_sha256(self.frozen_parameters)


@dataclass(frozen=True)
class IsolationForestModel:
    """Frozen Isolation Forest and its development-split standardization."""

    estimator: IsolationForest
    feature_medians: tuple[float, float, float]
    feature_scales: tuple[float, float, float]
    raw_tt_median: float
    raw_speed_median: float
    zero_scale_features: tuple[str, ...]
    package_version: str = sklearn.__version__

    @property
    def frozen_parameters(self) -> dict[str, Any]:
        return {
            "feature_order": ["log1p(tt_median)", "log1p(speed_median)", "log1p(dist_m)"],
            "feature_medians": list(self.feature_medians),
            "feature_scales": list(self.feature_scales),
            "raw_tt_median": self.raw_tt_median,
            "raw_speed_median": self.raw_speed_median,
            "zero_scale_features": list(self.zero_scale_features),
            "estimator_parameters": {
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": "auto",
                "random_state": 42,
            },
            "package_version": self.package_version,
        }

    @property
    def model_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "isolation-forest-model-state/v1",
                "frozen_parameters": self.frozen_parameters,
                "fitted_estimator_state": _canonical_isolation_forest_state(self.estimator),
            }
        )


@dataclass(frozen=True)
class RetentionResult:
    """Link-key retention counts and the exact frozen key sets."""

    n_eligible_raw_link_keys: int
    n_flagged_link_keys: int
    n_clean_link_keys: int
    retention_rate: float
    flagged_keys: tuple[tuple[Any, ...], ...]
    retained_keys: tuple[tuple[Any, ...], ...]


def _resolve_column(frame: pd.DataFrame, canonical: str, columns: Mapping[str, str] | None) -> str:
    if columns and canonical in columns:
        selected = columns[canonical]
        if selected not in frame.columns:
            raise AuditError(f"Configured {canonical!r} column does not exist: {selected!r}")
        return selected
    for candidate in _EVENT_ALIASES[canonical]:
        if candidate in frame.columns:
            return candidate
    raise AuditError(f"Missing {canonical!r} column; accepted names: {_EVENT_ALIASES[canonical]}")


def _coerce_timestamps(values: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise AuditError("departure timestamps contain missing or malformed values")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert(timezone)


def normalize_eligible_events(
    events: pd.DataFrame,
    *,
    routes: Iterable[str] | None = None,
    directions: Iterable[str] | None = None,
    window_start: str | pd.Timestamp | None = None,
    window_end: str | pd.Timestamp | None = None,
    timezone: str = "Asia/Hong_Kong",
    columns: Mapping[str, str] | None = None,
    max_distance_m: float = RULE_C_DISTANCE_M,
) -> pd.DataFrame:
    """Apply common E1 eligibility and return canonical event columns.

    Time filtering is half-open.  Eligibility requires a declared route and
    direction when those filters are supplied, finite positive travel time and
    distance, and distance no greater than 1500 m.  Effective speed is derived
    from distance and travel time when it is not present in the input.
    """

    if not isinstance(events, pd.DataFrame):
        raise AuditError("events must be a pandas DataFrame")
    if max_distance_m <= 0:
        raise AuditError("max_distance_m must be positive")

    required = ("route", "bound", "from_seq", "to_seq", "departure_ts", "travel_time_s", "distance_m")
    resolved = {name: _resolve_column(events, name, columns) for name in required}
    normalized = pd.DataFrame(index=events.index)
    for name in ("route", "bound", "from_seq", "to_seq"):
        normalized[name] = events[resolved[name]]
    normalized["departure_ts"] = _coerce_timestamps(events[resolved["departure_ts"]], timezone)
    normalized["travel_time_s"] = pd.to_numeric(events[resolved["travel_time_s"]], errors="coerce")
    normalized["distance_m"] = pd.to_numeric(events[resolved["distance_m"]], errors="coerce")

    try:
        speed_column = _resolve_column(events, "speed_kmh", columns)
    except AuditError:
        normalized["speed_kmh"] = normalized["distance_m"] / normalized["travel_time_s"] * 3.6
    else:
        normalized["speed_kmh"] = pd.to_numeric(events[speed_column], errors="coerce")

    mask = (
        np.isfinite(normalized["travel_time_s"])
        & np.isfinite(normalized["distance_m"])
        & np.isfinite(normalized["speed_kmh"])
        & (normalized["travel_time_s"] > 0)
        & (normalized["distance_m"] > 0)
        & (normalized["distance_m"] <= max_distance_m)
        & (normalized["speed_kmh"] > 0)
    )
    if routes is not None:
        allowed_routes = {str(value) for value in routes}
        mask &= normalized["route"].astype(str).isin(allowed_routes)
    if directions is not None:
        allowed_directions = {str(value) for value in directions}
        mask &= normalized["bound"].astype(str).isin(allowed_directions)

    if (window_start is None) != (window_end is None):
        raise AuditError("window_start and window_end must be supplied together")
    if window_start is not None and window_end is not None:
        start = pd.Timestamp(window_start)
        end = pd.Timestamp(window_end)
        start = start.tz_localize(timezone) if start.tzinfo is None else start.tz_convert(timezone)
        end = end.tz_localize(timezone) if end.tzinfo is None else end.tz_convert(timezone)
        if end <= start:
            raise AuditError("window_end must be later than window_start")
        mask &= (normalized["departure_ts"] >= start) & (normalized["departure_ts"] < end)

    return normalized.loc[mask].reset_index(drop=True)


def filter_eligible_events(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Backward-friendly alias for :func:`normalize_eligible_events`."""

    return normalize_eligible_events(*args, **kwargs)


def aggregate_link_hour(
    eligible_events: pd.DataFrame,
    *,
    window_start: str | pd.Timestamp | None = None,
    timezone: str = "Asia/Hong_Kong",
) -> pd.DataFrame:
    """Aggregate canonical eligible events to one median record per link-hour."""

    required = {
        "route",
        "bound",
        "from_seq",
        "to_seq",
        "departure_ts",
        "travel_time_s",
        "speed_kmh",
        "distance_m",
    }
    missing = sorted(required.difference(eligible_events.columns))
    if missing:
        raise AuditError(f"eligible_events lacks canonical columns: {missing}")
    frame = eligible_events.copy()
    frame["departure_ts"] = _coerce_timestamps(frame["departure_ts"], timezone)
    if window_start is None:
        frame["window_start"] = frame["departure_ts"].dt.floor("h")
    else:
        start = pd.Timestamp(window_start)
        start = start.tz_localize(timezone) if start.tzinfo is None else start.tz_convert(timezone)
        frame["window_start"] = start
    frame["window_end"] = frame["window_start"] + pd.Timedelta(hours=1)

    grouped = (
        frame.groupby(list(LINK_KEY_COLUMNS), dropna=False, sort=True, observed=True)
        .agg(
            window_end=("window_end", "first"),
            tt_median=("travel_time_s", "median"),
            speed_median=("speed_kmh", "median"),
            dist_m=("distance_m", "median"),
            n_events=("travel_time_s", "size"),
        )
        .reset_index()
    )
    return grouped


def _numeric_feature(records: pd.DataFrame, column: str, *, strictly_positive: bool = True) -> np.ndarray:
    if column not in records.columns:
        raise AuditError(f"records is missing {column!r}")
    values = pd.to_numeric(records[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if strictly_positive:
        valid &= values > 0
    if not valid.all():
        raise AuditError(f"{column} contains non-finite or non-positive values")
    return values


def rule_c_flags(
    records: pd.DataFrame,
    *,
    travel_time_gt_s: float = RULE_C_TRAVEL_TIME_S,
    speed_lt_kmh: float = RULE_C_SPEED_KMH,
    distance_lte_m: float = RULE_C_DISTANCE_M,
) -> pd.Series:
    """Return strict Rule-C decisions: ``T >``, ``v <``, ``distance <=``."""

    tt = _numeric_feature(records, "tt_median")
    speed = _numeric_feature(records, "speed_median")
    distance = _numeric_feature(records, "dist_m")
    values = (tt > travel_time_gt_s) & (speed < speed_lt_kmh) & (distance <= distance_lte_m)
    return pd.Series(values, index=records.index, name="flagged", dtype=bool)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float | None]:
    median = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - median)))
    if np.isfinite(mad_scale) and mad_scale > 0:
        return median, mad_scale
    q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
    iqr_scale = float((q75 - q25) / 1.349)
    if np.isfinite(iqr_scale) and iqr_scale > 0:
        return median, iqr_scale
    return median, None


def _robust_scores(values: np.ndarray, median: float, scale: float | None) -> np.ndarray:
    if scale is None:
        return np.zeros(values.shape, dtype=float)
    return (values - median) / scale


def fit_mad(records: pd.DataFrame) -> MADModel:
    """Fit the frozen MAD model jointly over all development records."""

    if records.empty:
        raise AuditError("cannot fit MAD on an empty development split")
    tt_log = np.log1p(_numeric_feature(records, "tt_median"))
    speed_log = np.log1p(_numeric_feature(records, "speed_median"))
    tt_median, tt_scale = _robust_location_scale(tt_log)
    speed_median, speed_scale = _robust_location_scale(speed_log)
    return MADModel(tt_median, speed_median, tt_scale, speed_scale)


def apply_mad(records: pd.DataFrame, model: MADModel) -> pd.DataFrame:
    """Apply a frozen MAD model and expose both robust scores and decisions."""

    tt_log = np.log1p(_numeric_feature(records, "tt_median"))
    speed_log = np.log1p(_numeric_feature(records, "speed_median"))
    tt_score = _robust_scores(tt_log, model.tt_log_median, model.tt_log_scale)
    speed_score = _robust_scores(speed_log, model.speed_log_median, model.speed_log_scale)
    return pd.DataFrame(
        {
            "tt_robust_score": tt_score,
            "speed_robust_score": speed_score,
            "flagged": (tt_score > model.tt_threshold) & (speed_score < model.speed_threshold),
        },
        index=records.index,
    )


def fit_quantile(records: pd.DataFrame) -> QuantileModel:
    """Fit the predeclared Q95/Q05 fallback using linear interpolation."""

    if records.empty:
        raise AuditError("cannot fit quantiles on an empty development split")
    tt = _numeric_feature(records, "tt_median")
    speed = _numeric_feature(records, "speed_median")
    return QuantileModel(
        tt_q95=float(np.quantile(tt, 0.95, method="linear")),
        speed_q05=float(np.quantile(speed, 0.05, method="linear")),
    )


def apply_quantile(records: pd.DataFrame, model: QuantileModel) -> pd.Series:
    """Apply inclusive quantile-fallback decisions (``>= Q95`` and ``<= Q05``)."""

    tt = _numeric_feature(records, "tt_median")
    speed = _numeric_feature(records, "speed_median")
    return pd.Series(
        (tt >= model.tt_q95) & (speed <= model.speed_q05),
        index=records.index,
        name="flagged",
        dtype=bool,
    )


def _isolation_features(records: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(_numeric_feature(records, "tt_median")),
            np.log1p(_numeric_feature(records, "speed_median")),
            np.log1p(_numeric_feature(records, "dist_m")),
        ]
    )


def fit_isolation_forest(records: pd.DataFrame) -> IsolationForestModel:
    """Fit the exact predeclared Isolation Forest on development records."""

    if len(records) < 2:
        raise AuditError("Isolation Forest requires at least two development records")
    features = _isolation_features(records)
    medians: list[float] = []
    scales: list[float] = []
    zero_scale: list[str] = []
    names = ("tt_median", "speed_median", "dist_m")
    for name, values in zip(names, features.T, strict=True):
        median, scale = _robust_location_scale(values)
        medians.append(median)
        if scale is None:
            zero_scale.append(name)
            scales.append(1.0)
        else:
            scales.append(scale)
    standardized = (features - np.asarray(medians)) / np.asarray(scales)
    estimator = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=42,
    )
    estimator.fit(standardized)
    return IsolationForestModel(
        estimator=estimator,
        feature_medians=tuple(medians),
        feature_scales=tuple(scales),
        raw_tt_median=float(np.median(_numeric_feature(records, "tt_median"))),
        raw_speed_median=float(np.median(_numeric_feature(records, "speed_median"))),
        zero_scale_features=tuple(zero_scale),
    )


def apply_isolation_forest(records: pd.DataFrame, model: IsolationForestModel) -> pd.DataFrame:
    """Apply Isolation Forest plus the required long/slow directional gate."""

    features = _isolation_features(records)
    standardized = (features - np.asarray(model.feature_medians)) / np.asarray(model.feature_scales)
    prediction = model.estimator.predict(standardized)
    anomaly = prediction == -1
    tt = _numeric_feature(records, "tt_median")
    speed = _numeric_feature(records, "speed_median")
    gate = (tt > model.raw_tt_median) & (speed < model.raw_speed_median)
    return pd.DataFrame(
        {
            "isolation_anomaly": anomaly,
            "directional_gate": gate,
            "flagged": anomaly & gate,
        },
        index=records.index,
    )


def _key_tuples(records: pd.DataFrame, key_columns: Sequence[str]) -> set[tuple[Any, ...]]:
    missing = sorted(set(key_columns).difference(records.columns))
    if missing:
        raise AuditError(f"records lacks link-key columns: {missing}")
    return set(records.loc[:, list(key_columns)].itertuples(index=False, name=None))


def retention_summary(
    records: pd.DataFrame,
    flagged: Sequence[bool] | pd.Series,
    *,
    key_columns: Sequence[str] = LINK_KEY_COLUMNS,
) -> RetentionResult:
    """Compute retention over unique eligible raw link keys."""

    decisions = pd.Series(flagged, index=records.index, dtype=bool)
    if len(decisions) != len(records):
        raise AuditError("flagged decisions must have one value per record")
    eligible_keys = _key_tuples(records, key_columns)
    flagged_keys = _key_tuples(records.loc[decisions], key_columns)
    retained_keys = eligible_keys.difference(flagged_keys)
    denominator = len(eligible_keys)
    rate = float(len(retained_keys) / denominator) if denominator else float("nan")
    sort_key = lambda item: tuple(str(value) for value in item)
    return RetentionResult(
        n_eligible_raw_link_keys=denominator,
        n_flagged_link_keys=len(flagged_keys),
        n_clean_link_keys=len(retained_keys),
        retention_rate=rate,
        flagged_keys=tuple(sorted(flagged_keys, key=sort_key)),
        retained_keys=tuple(sorted(retained_keys, key=sort_key)),
    )


def rule_c_sensitivity_grid(
    records: pd.DataFrame,
    *,
    travel_time_thresholds_s: Sequence[float] = RULE_C_SENSITIVITY_TRAVEL_TIME_S,
    speed_thresholds_kmh: Sequence[float] = RULE_C_SENSITIVITY_SPEED_KMH,
    distance_lte_m: float = RULE_C_DISTANCE_M,
    key_columns: Sequence[str] = LINK_KEY_COLUMNS,
) -> pd.DataFrame:
    """Evaluate the predeclared 3x3 development-split Rule-C neighborhood."""

    rows: list[dict[str, Any]] = []
    for tt_threshold in travel_time_thresholds_s:
        for speed_threshold in speed_thresholds_kmh:
            flags = rule_c_flags(
                records,
                travel_time_gt_s=float(tt_threshold),
                speed_lt_kmh=float(speed_threshold),
                distance_lte_m=distance_lte_m,
            )
            retention = retention_summary(records, flags, key_columns=key_columns)
            rows.append(
                {
                    "travel_time_gt_s": float(tt_threshold),
                    "speed_lt_kmh": float(speed_threshold),
                    "distance_lte_m": float(distance_lte_m),
                    "n_eligible_raw_link_keys": retention.n_eligible_raw_link_keys,
                    "n_flagged_link_keys": retention.n_flagged_link_keys,
                    "n_clean_link_keys": retention.n_clean_link_keys,
                    "retention_rate": retention.retention_rate,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "AuditError",
    "IsolationForestModel",
    "LINK_KEY_COLUMNS",
    "MADModel",
    "QuantileModel",
    "RULE_C_DISTANCE_M",
    "RULE_C_SPEED_KMH",
    "RULE_C_TRAVEL_TIME_S",
    "RetentionResult",
    "aggregate_link_hour",
    "apply_isolation_forest",
    "apply_mad",
    "apply_quantile",
    "filter_eligible_events",
    "fit_isolation_forest",
    "fit_mad",
    "fit_quantile",
    "normalize_eligible_events",
    "retention_summary",
    "rule_c_flags",
    "rule_c_sensitivity_grid",
]
