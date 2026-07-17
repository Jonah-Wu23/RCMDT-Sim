from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.paper_experiments.audit import (
    LINK_KEY_COLUMNS,
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
    rule_c_sensitivity_grid,
)


def test_common_eligibility_and_link_hour_aggregation():
    events = pd.DataFrame(
        {
            "route": ["68X"] * 8,
            "direction": ["O", "O", "O", "I", "O", "O", "O", "O"],
            "from_seq": [1, 1, 2, 1, 1, 1, 1, 1],
            "to_seq": [2, 2, 3, 2, 2, 2, 2, 2],
            "departure_time": [
                "2025-12-19 17:05:00",
                "2025-12-19 17:10:00",
                "2025-12-19 17:20:00",
                "2025-12-19 17:25:00",  # wrong direction
                "2025-12-19 18:00:00",  # half-open end
                "2025-12-19 17:30:00",  # too long a link
                "2025-12-19 17:35:00",  # zero travel time
                "2025-12-19 16:59:59",  # before the split
            ],
            "travel_time_s": [100, 200, 90, 100, 100, 100, 0, 100],
            "distance_m": [1000, 1000, 900, 1000, 1000, 1501, 1000, 1000],
        }
    )
    eligible = normalize_eligible_events(
        events,
        routes=["68X"],
        directions=["O"],
        window_start="2025-12-19 17:00:00",
        window_end="2025-12-19 18:00:00",
    )
    assert len(eligible) == 3
    assert np.allclose(eligible["speed_kmh"], eligible["distance_m"] / eligible["travel_time_s"] * 3.6)

    records = aggregate_link_hour(eligible, window_start="2025-12-19 17:00:00")
    assert len(records) == 2
    link = records.loc[records["from_seq"] == 1].iloc[0]
    assert link["tt_median"] == 150
    assert link["n_events"] == 2
    assert str(link["window_start"].tz) == "Asia/Hong_Kong"


def test_rule_c_uses_strict_time_and_speed_and_inclusive_distance():
    records = pd.DataFrame(
        {
            "tt_median": [325, 326, 326, 326],
            "speed_median": [4, 5, 4, 4],
            "dist_m": [1500, 1500, 1500, 1500.01],
        }
    )
    assert rule_c_flags(records).tolist() == [False, False, True, False]


def test_mad_zero_scale_assigns_zero_scores():
    records = pd.DataFrame({"tt_median": [100.0] * 6, "speed_median": [20.0] * 6})
    model = fit_mad(records)
    assert model.tt_log_scale is None
    assert model.speed_log_scale is None
    result = apply_mad(records, model)
    assert (result[["tt_robust_score", "speed_robust_score"]] == 0).all().all()
    assert not result["flagged"].any()


def test_isolation_forest_exact_parameters_and_directional_gate():
    records = pd.DataFrame(
        {
            "tt_median": [100 + value for value in range(20)] + [900],
            "speed_median": [20 + value / 10 for value in range(20)] + [1],
            "dist_m": [1000 + value for value in range(20)] + [1200],
        }
    )
    model = fit_isolation_forest(records)
    params = model.estimator.get_params()
    assert params["n_estimators"] == 200
    assert params["max_samples"] == "auto"
    assert params["contamination"] == "auto"
    assert params["random_state"] == 42
    assert model.model_hash == model.model_hash

    decisions = apply_isolation_forest(records, model)
    assert (decisions["flagged"] == (decisions["isolation_anomaly"] & decisions["directional_gate"])).all()
    assert (records.loc[decisions["flagged"], "tt_median"] > model.raw_tt_median).all()
    assert (records.loc[decisions["flagged"], "speed_median"] < model.raw_speed_median).all()


def test_isolation_forest_hash_is_stable_across_independent_fits():
    records = pd.DataFrame(
        {
            "tt_median": [100 + value for value in range(20)] + [900],
            "speed_median": [20 + value / 10 for value in range(20)] + [1],
            "dist_m": [1000 + value for value in range(20)] + [1200],
        }
    )

    first = fit_isolation_forest(records)
    second = fit_isolation_forest(records)

    assert first.model_hash == second.model_hash


def test_isolation_forest_hash_changes_with_fitted_state():
    records = pd.DataFrame(
        {
            "tt_median": [100 + value for value in range(20)] + [900],
            "speed_median": [20 + value / 10 for value in range(20)] + [1],
            "dist_m": [1000 + value for value in range(20)] + [1200],
        }
    )
    model = fit_isolation_forest(records)
    original_hash = model.model_hash

    threshold = model.estimator.estimators_[0].tree_.threshold
    threshold[0] = np.nextafter(threshold[0], np.inf)

    assert model.model_hash != original_hash

    model = fit_isolation_forest(records)
    original_hash = model.model_hash
    model.estimator.offset_ = np.nextafter(model.estimator.offset_, np.inf)

    assert model.model_hash != original_hash

    model = fit_isolation_forest(records)
    original_hash = model.model_hash
    changed_parameters = replace(
        model,
        raw_tt_median=np.nextafter(model.raw_tt_median, np.inf),
    )

    assert changed_parameters.model_hash != original_hash


def test_quantile_retention_and_predeclared_sensitivity_grid():
    records = pd.DataFrame(
        {
            "route": ["68X"] * 20,
            "bound": ["O"] * 20,
            "from_seq": list(range(20)),
            "to_seq": list(range(1, 21)),
            "window_start": [pd.Timestamp("2025-12-19 17:00", tz="Asia/Hong_Kong")] * 20,
            "tt_median": np.arange(1, 21, dtype=float),
            "speed_median": np.arange(20, 0, -1, dtype=float),
            "dist_m": [1000.0] * 20,
        }
    )
    model = fit_quantile(records)
    assert model.tt_q95 == np.quantile(records["tt_median"], 0.95, method="linear")
    assert model.speed_q05 == np.quantile(records["speed_median"], 0.05, method="linear")
    flags = apply_quantile(records, model)
    assert flags.sum() == 1

    retention = retention_summary(records, flags)
    assert retention.n_eligible_raw_link_keys == 20
    assert retention.n_flagged_link_keys == 1
    assert retention.n_clean_link_keys == 19
    assert retention.retention_rate == 0.95
    assert len(retention.retained_keys[0]) == len(LINK_KEY_COLUMNS)

    grid = rule_c_sensitivity_grid(records)
    assert len(grid) == 9
    assert set(grid["travel_time_gt_s"]) == {275.0, 325.0, 375.0}
    assert set(grid["speed_lt_kmh"]) == {4.0, 5.0, 6.0}
