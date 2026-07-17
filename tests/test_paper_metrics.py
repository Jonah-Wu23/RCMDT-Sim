from __future__ import annotations

import numpy as np
import pytest

from src.paper_experiments.metrics import (
    MetricError,
    MetricResult,
    PAPER_METRICS_SCHEMA,
    full_window_ks,
    make_metric_row,
    metrics_to_long_form,
    successful_metric_rows,
    validate_metric_row,
    worst_window_ks,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def metadata() -> dict:
    return {
        "experiment_id": "e2",
        "config_id": "A0",
        "method_id": "zero-shot",
        "seed": 0,
        "split": "development",
        "metric_name": "KS-speed",
        "domain": "speed",
        "unit": "D",
        "n_link_keys": 12,
        "manifest_hash": SHA_A,
        "simulation_output_hash": SHA_B,
        "evaluator_version": "1.0.0",
    }


def test_full_window_requires_twenty_finite_samples():
    failed = full_window_ks(np.arange(19), np.arange(30))
    assert failed.status == "failed"
    assert failed.value is None
    assert failed.n_real == 19

    succeeded = full_window_ks(np.arange(20), np.arange(20))
    assert succeeded.status == "succeeded"
    assert succeeded.value == 0.0
    assert succeeded.n_real == succeeded.n_sim == 20


def test_worst_window_is_900_seconds_half_open_and_steps_by_sixty():
    times = [0, 1, 2, 3, 4, 900, 901, 902, 903, 904]
    real = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sim = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    result = worst_window_ks(real, times, sim, times, window_start=0, window_end=3600)
    assert result.status == "succeeded"
    assert result.value == 1.0
    assert result.window_start == 0.0
    assert result.window_end == 900.0
    assert result.n_real == result.n_sim == 5


def test_worst_window_missing_minimum_is_failed_not_zero():
    result = worst_window_ks(
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        window_start=0,
        window_end=3600,
    )
    assert result.status == "failed"
    assert result.value is None


def test_paper_metrics_long_form_validation():
    result = full_window_ks(
        np.arange(20),
        np.arange(20),
        window_start="2025-12-19T17:00:00+08:00",
        window_end="2025-12-19T18:00:00+08:00",
    )
    row = make_metric_row(result, **metadata())
    assert row["schema_version"] == PAPER_METRICS_SCHEMA
    assert validate_metric_row(row) is row
    frame = metrics_to_long_form([row])
    assert frame.shape == (1, 19)
    assert successful_metric_rows(frame).shape == (1, 19)

    failed_row = make_metric_row(
        MetricResult(
            status="failed",
            value=None,
            n_real=1,
            n_sim=0,
            window_start=0,
            window_end=1,
        ),
        **metadata(),
    )
    assert successful_metric_rows([failed_row]).empty

    bad_failed = dict(failed_row, value=0.0)
    with pytest.raises(MetricError, match="null value"):
        validate_metric_row(bad_failed)

    bad_success = dict(row, value=None)
    with pytest.raises(MetricError, match="finite numerical"):
        validate_metric_row(bad_success)

    # The shared schema also carries objectives and errors whose units are not D.
    objective = dict(row, metric_name="JL1_68X", unit="seconds", value=347.2)
    assert validate_metric_row(objective) is objective


def test_succeeded_ks_rows_accept_exact_numeric_and_timezone_windows():
    numeric_full = make_metric_row(
        MetricResult(
            status="succeeded",
            value=0.2,
            n_real=20,
            n_sim=21,
            window_start=0,
            window_end=3600,
        ),
        **metadata(),
    )
    assert validate_metric_row(numeric_full) is numeric_full

    worst_metadata = dict(metadata(), metric_name="worst-15-min KS-speed")
    numeric_worst = make_metric_row(
        MetricResult(
            status="succeeded",
            value=0.4,
            n_real=5,
            n_sim=6,
            window_start=120,
            window_end=1020,
        ),
        **worst_metadata,
    )
    assert validate_metric_row(numeric_worst) is numeric_worst

    timezone_worst = make_metric_row(
        MetricResult(
            status="succeeded",
            value=0.4,
            n_real=5,
            n_sim=5,
            window_start="2025-12-19T17:10:00+08:00",
            window_end="2025-12-19T17:25:00+08:00",
        ),
        **worst_metadata,
    )
    assert validate_metric_row(timezone_worst) is timezone_worst


def test_succeeded_ks_rows_cannot_bypass_sample_or_window_contracts():
    full_row = make_metric_row(
        MetricResult(
            status="succeeded",
            value=0.2,
            n_real=20,
            n_sim=20,
            window_start=0,
            window_end=3600,
        ),
        **metadata(),
    )
    with pytest.raises(MetricError, match="at least 20"):
        validate_metric_row(dict(full_row, n_real=19))
    with pytest.raises(MetricError, match="3600-second"):
        validate_metric_row(dict(full_row, window_end=3599))

    worst_metadata = dict(metadata(), metric_name="worst-15-min KS-speed")
    worst_row = make_metric_row(
        MetricResult(
            status="succeeded",
            value=0.4,
            n_real=5,
            n_sim=5,
            window_start=120,
            window_end=1020,
        ),
        **worst_metadata,
    )
    with pytest.raises(MetricError, match="at least 5"):
        validate_metric_row(dict(worst_row, n_sim=4))
    with pytest.raises(MetricError, match="900-second"):
        validate_metric_row(dict(worst_row, window_end=1021))

    # A failed metric remains honest evidence of an invalid sample population.
    failed_low_sample = dict(
        worst_row,
        status="failed",
        value=None,
        n_real=0,
        n_sim=1,
        window_start=0,
        window_end=1,
    )
    assert validate_metric_row(failed_low_sample) is failed_low_sample
