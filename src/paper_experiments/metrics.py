"""Validated event-level K-S metrics for camera-ready paper artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


PAPER_METRICS_SCHEMA = "paper-metrics/v1"
FULL_WINDOW_MIN_SAMPLES = 20
SUBWINDOW_MIN_SAMPLES = 5
WORST_WINDOW_SECONDS = 900
WORST_WINDOW_STEP_SECONDS = 60

METRIC_COLUMNS = (
    "schema_version",
    "experiment_id",
    "config_id",
    "method_id",
    "seed",
    "split",
    "metric_name",
    "domain",
    "value",
    "unit",
    "n_real",
    "n_sim",
    "n_link_keys",
    "window_start",
    "window_end",
    "manifest_hash",
    "simulation_output_hash",
    "evaluator_version",
    "status",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class MetricError(ValueError):
    """Raised when metric inputs or rows violate the paper contract."""


@dataclass(frozen=True)
class MetricResult:
    """One metric calculation before experiment metadata are attached."""

    status: str
    value: float | None
    n_real: int
    n_sim: int
    window_start: Any = None
    window_end: Any = None
    error_summary: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


def _finite_values(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def full_window_ks(
    real_values: Sequence[float] | np.ndarray | pd.Series,
    sim_values: Sequence[float] | np.ndarray | pd.Series,
    *,
    min_samples: int = FULL_WINDOW_MIN_SAMPLES,
    window_start: Any = None,
    window_end: Any = None,
) -> MetricResult:
    """Compute full-window K-S ``D`` or return an explicit failed metric.

    Non-finite samples are excluded before the minimum count is checked.  A
    missing minimum never produces the numerical value zero.
    """

    if min_samples < 1:
        raise MetricError("min_samples must be at least one")
    real = _finite_values(real_values)
    sim = _finite_values(sim_values)
    if len(real) < min_samples or len(sim) < min_samples:
        return MetricResult(
            status="failed",
            value=None,
            n_real=len(real),
            n_sim=len(sim),
            window_start=window_start,
            window_end=window_end,
            error_summary=(
                f"requires at least {min_samples} finite samples per source; "
                f"got n_real={len(real)}, n_sim={len(sim)}"
            ),
        )
    statistic = float(ks_2samp(real, sim, alternative="two-sided", method="auto").statistic)
    return MetricResult(
        status="succeeded",
        value=statistic,
        n_real=len(real),
        n_sim=len(sim),
        window_start=window_start,
        window_end=window_end,
    )


@dataclass(frozen=True)
class _TimeAxis:
    real_seconds: np.ndarray
    sim_seconds: np.ndarray
    start_seconds: float
    end_seconds: float
    start_value: Any
    is_datetime: bool

    def render(self, offset_seconds: float) -> Any:
        if self.is_datetime:
            return (pd.Timestamp(self.start_value) + pd.to_timedelta(offset_seconds, unit="s")).isoformat()
        return float(self.start_seconds + offset_seconds)


def _numeric_times(values: Sequence[Any], expected_length: int, name: str) -> np.ndarray:
    if len(values) != expected_length:
        raise MetricError(f"{name} must have one timestamp per sample")
    return pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)


def _datetime_times(values: Sequence[Any], expected_length: int, name: str) -> np.ndarray:
    if len(values) != expected_length:
        raise MetricError(f"{name} must have one timestamp per sample")
    parsed = pd.to_datetime(pd.Series(values, dtype="object"), errors="coerce", utc=True)
    result = parsed.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    result[parsed.isna().to_numpy()] = np.nan
    return result


def _build_time_axis(
    real_times: Sequence[Any],
    sim_times: Sequence[Any],
    n_real: int,
    n_sim: int,
    window_start: Any,
    window_end: Any,
) -> _TimeAxis:
    numeric_window = (
        isinstance(window_start, numbers.Real)
        and not isinstance(window_start, bool)
        and isinstance(window_end, numbers.Real)
        and not isinstance(window_end, bool)
    )
    if numeric_window:
        start_seconds = float(window_start)
        end_seconds = float(window_end)
        real_seconds = _numeric_times(real_times, n_real, "real_times")
        sim_seconds = _numeric_times(sim_times, n_sim, "sim_times")
        start_value = start_seconds
    else:
        start = pd.Timestamp(window_start)
        end = pd.Timestamp(window_end)
        if pd.isna(start) or pd.isna(end):
            raise MetricError("window_start and window_end must be valid timestamps")
        start_utc = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end_utc = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        start_seconds = start_utc.timestamp()
        end_seconds = end_utc.timestamp()
        real_seconds = _datetime_times(real_times, n_real, "real_times")
        sim_seconds = _datetime_times(sim_times, n_sim, "sim_times")
        start_value = start
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise MetricError("window_end must be later than window_start")
    return _TimeAxis(
        real_seconds=real_seconds,
        sim_seconds=sim_seconds,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        start_value=start_value,
        is_datetime=not numeric_window,
    )


def worst_window_ks(
    real_values: Sequence[float] | np.ndarray | pd.Series,
    real_times: Sequence[Any],
    sim_values: Sequence[float] | np.ndarray | pd.Series,
    sim_times: Sequence[Any],
    *,
    window_start: Any,
    window_end: Any,
    subwindow_seconds: int = WORST_WINDOW_SECONDS,
    step_seconds: int = WORST_WINDOW_STEP_SECONDS,
    min_samples: int = SUBWINDOW_MIN_SAMPLES,
) -> MetricResult:
    """Return maximum valid K-S over sliding half-open 15-minute windows."""

    if subwindow_seconds <= 0 or step_seconds <= 0:
        raise MetricError("subwindow_seconds and step_seconds must be positive")
    if min_samples < 1:
        raise MetricError("min_samples must be at least one")

    real = pd.to_numeric(pd.Series(real_values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    sim = pd.to_numeric(pd.Series(sim_values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    axis = _build_time_axis(real_times, sim_times, len(real), len(sim), window_start, window_end)
    duration = axis.end_seconds - axis.start_seconds
    if duration < subwindow_seconds:
        raise MetricError("evaluation window is shorter than one subwindow")

    n_starts = int(math.floor((duration - subwindow_seconds) / step_seconds + 1e-12)) + 1
    best: tuple[float, int, int, float] | None = None
    for index in range(n_starts):
        offset = float(index * step_seconds)
        start = axis.start_seconds + offset
        end = start + subwindow_seconds
        real_mask = (
            np.isfinite(real)
            & np.isfinite(axis.real_seconds)
            & (axis.real_seconds >= start)
            & (axis.real_seconds < end)
        )
        sim_mask = (
            np.isfinite(sim)
            & np.isfinite(axis.sim_seconds)
            & (axis.sim_seconds >= start)
            & (axis.sim_seconds < end)
        )
        n_real_window = int(real_mask.sum())
        n_sim_window = int(sim_mask.sum())
        if n_real_window < min_samples or n_sim_window < min_samples:
            continue
        statistic = float(
            ks_2samp(real[real_mask], sim[sim_mask], alternative="two-sided", method="auto").statistic
        )
        if best is None or statistic > best[0]:
            best = (statistic, n_real_window, n_sim_window, offset)

    if best is None:
        global_real = int(
            (
                np.isfinite(real)
                & np.isfinite(axis.real_seconds)
                & (axis.real_seconds >= axis.start_seconds)
                & (axis.real_seconds < axis.end_seconds)
            ).sum()
        )
        global_sim = int(
            (
                np.isfinite(sim)
                & np.isfinite(axis.sim_seconds)
                & (axis.sim_seconds >= axis.start_seconds)
                & (axis.sim_seconds < axis.end_seconds)
            ).sum()
        )
        return MetricResult(
            status="failed",
            value=None,
            n_real=global_real,
            n_sim=global_sim,
            window_start=axis.render(0.0),
            window_end=axis.render(duration),
            error_summary=f"no subwindow has at least {min_samples} finite samples per source",
        )

    statistic, n_real_window, n_sim_window, offset = best
    return MetricResult(
        status="succeeded",
        value=statistic,
        n_real=n_real_window,
        n_sim=n_sim_window,
        window_start=axis.render(offset),
        window_end=axis.render(offset + subwindow_seconds),
    )


def make_metric_row(
    result: MetricResult,
    *,
    experiment_id: str,
    config_id: str,
    method_id: str,
    seed: int,
    split: str,
    metric_name: str,
    domain: str,
    unit: str,
    n_link_keys: int,
    manifest_hash: str,
    simulation_output_hash: str,
    evaluator_version: str,
) -> dict[str, Any]:
    """Attach provenance metadata to a result and validate the resulting row."""

    row = {
        "schema_version": PAPER_METRICS_SCHEMA,
        "experiment_id": experiment_id,
        "config_id": config_id,
        "method_id": method_id,
        "seed": seed,
        "split": split,
        "metric_name": metric_name,
        "domain": domain,
        "value": result.value,
        "unit": unit,
        "n_real": result.n_real,
        "n_sim": result.n_sim,
        "n_link_keys": n_link_keys,
        "window_start": result.window_start,
        "window_end": result.window_end,
        "manifest_hash": manifest_hash,
        "simulation_output_hash": simulation_output_hash,
        "evaluator_version": evaluator_version,
        "status": result.status,
    }
    validate_metric_row(row)
    return row


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def _require_nonempty(row: Mapping[str, Any], field: str) -> Any:
    if field not in row or row[field] is None or row[field] == "":
        raise MetricError(f"missing required metric field: {field}")
    return row[field]


def _validate_count(row: Mapping[str, Any], field: str) -> None:
    value = _require_nonempty(row, field)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise MetricError(f"{field} must be a non-negative integer")


def _window_duration_seconds(window_start: Any, window_end: Any) -> float:
    """Return a validated duration for numeric or timestamp window bounds."""

    start_is_numeric = isinstance(window_start, numbers.Real) and not isinstance(window_start, bool)
    end_is_numeric = isinstance(window_end, numbers.Real) and not isinstance(window_end, bool)
    if start_is_numeric != end_is_numeric:
        raise MetricError("window_start and window_end must use the same numeric or timestamp type")

    if start_is_numeric:
        start_value = float(window_start)
        end_value = float(window_end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise MetricError("numeric metric windows must be finite")
        duration = end_value - start_value
    else:
        try:
            start = pd.Timestamp(window_start)
            end = pd.Timestamp(window_end)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MetricError("metric windows must be valid timestamps") from exc
        if pd.isna(start) or pd.isna(end):
            raise MetricError("metric windows must be valid timestamps")
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise MetricError("timestamp metric windows must use consistent timezone awareness")
        if start.tzinfo is not None:
            start = start.tz_convert("UTC")
            end = end.tz_convert("UTC")
        duration = float((end - start).total_seconds())

    if not math.isfinite(duration) or duration <= 0:
        raise MetricError("window_end must be later than window_start")
    return duration


def _validate_succeeded_ks_contract(row: Mapping[str, Any], metric_name: str) -> None:
    is_worst_window = "worst" in metric_name
    minimum = SUBWINDOW_MIN_SAMPLES if is_worst_window else FULL_WINDOW_MIN_SAMPLES
    expected_seconds = WORST_WINDOW_SECONDS if is_worst_window else 3600
    if int(row["n_real"]) < minimum or int(row["n_sim"]) < minimum:
        label = "worst-15-minute" if is_worst_window else "full-window"
        raise MetricError(
            f"a succeeded {label} K-S metric requires at least {minimum} real and simulated events"
        )
    duration = _window_duration_seconds(row["window_start"], row["window_end"])
    if not math.isclose(duration, expected_seconds, rel_tol=0.0, abs_tol=1e-6):
        label = "worst-15-minute" if is_worst_window else "full-window"
        raise MetricError(
            f"a succeeded {label} K-S metric requires a {expected_seconds}-second window"
        )


def validate_metric_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one ``paper-metrics/v1`` long-form row."""

    if not isinstance(row, Mapping):
        raise MetricError("metric row must be an object")
    if row.get("schema_version") != PAPER_METRICS_SCHEMA:
        raise MetricError(f"schema_version must be {PAPER_METRICS_SCHEMA!r}")
    for field in (
        "experiment_id",
        "config_id",
        "method_id",
        "split",
        "metric_name",
        "unit",
        "window_start",
        "window_end",
        "evaluator_version",
    ):
        _require_nonempty(row, field)
    seed = _require_nonempty(row, "seed")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise MetricError("seed must be an integer")
    if row.get("domain") not in {"speed", "travel_time"}:
        raise MetricError("domain must be 'speed' or 'travel_time'")
    for field in ("n_real", "n_sim", "n_link_keys"):
        _validate_count(row, field)
    for field in ("manifest_hash", "simulation_output_hash"):
        value = str(_require_nonempty(row, field))
        if not _SHA256_RE.fullmatch(value):
            raise MetricError(f"{field} must be a 64-character hexadecimal SHA-256")

    status = row.get("status")
    value = row.get("value")
    if status == "succeeded":
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            raise MetricError("a succeeded metric requires a finite numerical value")
        metric_name = str(row["metric_name"]).lower().replace("_", "-")
        if "ks" in metric_name:
            if not 0.0 <= float(value) <= 1.0:
                raise MetricError("K-S statistic must be between zero and one")
            _validate_succeeded_ks_contract(row, metric_name)
    elif status == "failed":
        if not _is_missing(value):
            raise MetricError("a failed metric must have a null value, never a numerical placeholder")
    else:
        raise MetricError("status must be 'succeeded' or 'failed'")
    return row


def metrics_to_long_form(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate metric rows and return them in the canonical column order."""

    validated = [dict(validate_metric_row(row)) for row in rows]
    return pd.DataFrame(validated, columns=list(METRIC_COLUMNS))


def successful_metric_rows(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Return the only rows reporting scripts are permitted to consume."""

    source = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    frame = metrics_to_long_form(source)
    return frame.loc[frame["status"] == "succeeded"].reset_index(drop=True)


__all__ = [
    "FULL_WINDOW_MIN_SAMPLES",
    "METRIC_COLUMNS",
    "MetricError",
    "MetricResult",
    "PAPER_METRICS_SCHEMA",
    "SUBWINDOW_MIN_SAMPLES",
    "WORST_WINDOW_SECONDS",
    "WORST_WINDOW_STEP_SECONDS",
    "full_window_ks",
    "make_metric_row",
    "metrics_to_long_form",
    "successful_metric_rows",
    "validate_metric_row",
    "worst_window_ks",
]
