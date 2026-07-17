import numpy as np
import pandas as pd
import pytest

from src.calibration import objective


def _real_route(route: str, link_times: list[float], bound: str = "inbound") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route": route,
                "bound": bound,
                "from_seq": index,
                "to_seq": index + 1,
                "travel_time_s": travel_time,
            }
            for index, travel_time in enumerate(link_times, start=1)
        ]
    )


def _sim_route(
    route: str,
    observed_cumulative: list[float],
    errors: list[float],
    bound: str = "inbound",
    vehicle_id: str | None = None,
    start_time: float = 100.0,
) -> pd.DataFrame:
    vehicle = vehicle_id or f"{route}.0"
    rows = [
        {
            "route": route,
            "bound": bound,
            "vehicle_id": vehicle,
            "seq": 1,
            "arrival_time": start_time,
        }
    ]
    rows.extend(
        {
            "route": route,
            "bound": bound,
            "vehicle_id": vehicle,
            "seq": seq,
            "arrival_time": start_time + cumulative + error,
        }
        for seq, (cumulative, error) in enumerate(
            zip(observed_cumulative, errors), start=2
        )
    )
    return pd.DataFrame(rows)


def test_observed_cumulative_times_use_route_bound_and_mean_link_time() -> None:
    real = pd.concat(
        [
            _real_route("68X", [8.0, 20.0, 30.0, 40.0]),
            pd.DataFrame(
                [
                    {
                        "route": "68X",
                        "bound": "inbound",
                        "from_seq": 1,
                        "to_seq": 2,
                        "travel_time_s": 12.0,
                    },
                    {
                        "route": "68X",
                        "bound": "outbound",
                        "from_seq": 1,
                        "to_seq": 2,
                        "travel_time_s": 999.0,
                    },
                    {
                        "route": "960",
                        "bound": "inbound",
                        "from_seq": 1,
                        "to_seq": 2,
                        "travel_time_s": 777.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    result = objective.build_observed_cumulative_times(real, "68X", "I")

    assert result[["route", "bound"]].drop_duplicates().to_dict("records") == [
        {"route": "68X", "bound": "inbound"}
    ]
    assert result["to_seq"].tolist() == [2, 3, 4, 5]
    np.testing.assert_allclose(
        result["observed_cumulative_time_s"].to_numpy(),
        [10.0, 30.0, 60.0, 100.0],
    )


def test_error_table_zeros_each_vehicle_at_its_first_matched_stop() -> None:
    real = _real_route("68X", [10.0, 20.0, 30.0, 40.0])
    first = _sim_route(
        "68X",
        [10.0, 30.0, 60.0, 100.0],
        [2.0, 5.0, 8.0, 10.0],
        vehicle_id="bus.a",
        start_time=100.0,
    )
    second = _sim_route(
        "68X",
        [10.0, 30.0, 60.0, 100.0],
        [0.0, 4.0, 7.0, 8.0],
        vehicle_id="bus.b",
        start_time=500.0,
    )
    noise = _sim_route(
        "68X",
        [10.0, 30.0, 60.0, 100.0],
        [500.0, 500.0, 500.0, 500.0],
        bound="outbound",
        vehicle_id="noise",
    )
    sim = pd.concat([first, second, noise], ignore_index=True)

    result = objective.compute_l1_error_table(sim, real, "68X", "inbound")

    assert result["to_seq"].tolist() == [2, 3, 4, 5]
    np.testing.assert_allclose(result["error_s"].to_numpy(), [1.0, 4.5, 7.5, 9.0])


def test_first_matched_stop_by_arrival_time_is_the_vehicle_time_origin() -> None:
    sim = pd.DataFrame(
        [
            {
                "route": "68X",
                "bound": "inbound",
                "vehicle_id": "bus.partial",
                "seq": 3,
                "arrival_time": 320.0,
            },
            {
                "route": "68X",
                "bound": "inbound",
                "vehicle_id": "bus.partial",
                "seq": 4,
                "arrival_time": 345.0,
            },
            {
                "route": "68X",
                "bound": "inbound",
                "vehicle_id": "bus.partial",
                "seq": 2,
                "arrival_time": 300.0,
            },
        ]
    )

    result = objective.build_simulated_cumulative_times(
        sim,
        matched_stop_seqs=[2, 3, 4],
        route="68X",
        bound="I",
    )

    assert result["to_seq"].tolist() == [2, 3, 4]
    np.testing.assert_allclose(
        result["simulated_cumulative_time_s"].to_numpy(),
        [0.0, 20.0, 45.0],
    )


def test_decreasing_arrival_time_along_stop_sequence_raises_infrastructure_error() -> None:
    sim = pd.DataFrame(
        [
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 2, "arrival_time": 100.0},
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 3, "arrival_time": 90.0},
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 4, "arrival_time": 130.0},
        ]
    )

    with pytest.raises(objective.L1InfrastructureError, match="strictly increasing"):
        objective.build_simulated_cumulative_times(sim, [2, 3, 4], "68X", "inbound")


def test_duplicate_arrival_time_along_stop_sequence_raises_infrastructure_error() -> None:
    sim = pd.DataFrame(
        [
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 2, "arrival_time": 100.0},
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 3, "arrival_time": 100.0},
            {"route": "68X", "bound": "inbound", "vehicle_id": "v", "seq": 4, "arrival_time": 130.0},
        ]
    )

    with pytest.raises(objective.L1InfrastructureError, match="strictly increasing"):
        objective.build_simulated_cumulative_times(sim, [2, 3, 4], "68X", "inbound")


def test_error_table_requires_three_joined_downstream_stops() -> None:
    real = _real_route("68X", [10.0, 20.0, 30.0, 40.0])
    sim = _sim_route(
        "68X",
        [10.0, 30.0],
        [0.0, 0.0],
    )

    with pytest.raises(objective.L1UnevaluableError, match="2 matched downstream stops"):
        objective.compute_l1_error_table(sim, real, "68X", "I")


def test_malformed_simulation_values_raise_infrastructure_error() -> None:
    real = _real_route("68X", [10.0, 20.0, 30.0])
    sim = _sim_route(
        "68X",
        [10.0, 30.0, 60.0],
        [0.0, 0.0, 0.0],
    )
    sim["arrival_time"] = sim["arrival_time"].astype(object)
    sim.loc[1, "arrival_time"] = "malformed"

    with pytest.raises(objective.L1InfrastructureError, match="arrival_time"):
        objective.compute_l1_error_table(sim, real, "68X", "I")


def test_jl1_components_follow_the_fixed_formula() -> None:
    errors = np.array([1.0, -2.0, 3.0, -4.0])

    result = objective.calculate_jl1_from_errors(errors)

    abs_errors = np.abs(errors)
    expected_rmse = float(np.sqrt(np.mean(errors**2)))
    expected_mae = float(np.mean(abs_errors))
    expected_std = float(np.std(abs_errors))
    expected_q90 = float(np.quantile(abs_errors, 0.9))
    expected_jl1 = expected_rmse + expected_mae + 0.5 * expected_std + 0.3 * expected_q90
    assert result["status"] == "succeeded"
    assert result["rmse_term"] == pytest.approx(expected_rmse)
    assert result["mae_term"] == pytest.approx(expected_mae)
    assert result["std_term"] == pytest.approx(expected_std)
    assert result["tail_term"] == pytest.approx(expected_q90)
    assert result["jl1"] == pytest.approx(expected_jl1)


def test_candidate_score_uses_jl1_when_960_constraint_is_feasible() -> None:
    real = pd.concat(
        [
            _real_route("68X", [10.0, 20.0, 30.0, 40.0]),
            _real_route("960", [15.0, 25.0, 35.0, 45.0]),
        ],
        ignore_index=True,
    )
    sim = pd.concat(
        [
            _sim_route("68X", [10.0, 30.0, 60.0, 100.0], [1.0, 2.0, 3.0, 4.0]),
            _sim_route("960", [15.0, 40.0, 75.0, 120.0], [100.0] * 4),
        ],
        ignore_index=True,
    )

    result = objective.calculate_l1_candidate_score_from_frames(sim, real)

    expected_jl1 = objective.calculate_jl1_from_errors([1.0, 2.0, 3.0, 4.0])["jl1"]
    assert result["status"] == "succeeded"
    assert result["feasible"] is True
    assert result["rmse_960"] == pytest.approx(100.0)
    assert result["score"] == pytest.approx(expected_jl1)
    assert result["penalty"] == 0.0


def test_candidate_score_applies_constraint_violation_penalty() -> None:
    real = pd.concat(
        [
            _real_route("68X", [10.0, 20.0, 30.0, 40.0]),
            _real_route("960", [15.0, 25.0, 35.0, 45.0]),
        ],
        ignore_index=True,
    )
    sim = pd.concat(
        [
            _sim_route("68X", [10.0, 30.0, 60.0, 100.0], [1.0, 2.0, 3.0, 4.0]),
            _sim_route("960", [15.0, 40.0, 75.0, 120.0], [360.0] * 4),
        ],
        ignore_index=True,
    )

    result = objective.calculate_l1_candidate_score_from_frames(sim, real)

    assert result["feasible"] is False
    assert result["rmse_960"] == pytest.approx(360.0)
    assert result["constraint_violation_s"] == pytest.approx(10.0)
    assert result["score"] == pytest.approx(2100.0)
    assert result["penalty"] == pytest.approx(2100.0)


def test_legacy_file_wrappers_reuse_the_dataframe_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    real = _real_route("68X", [10.0, 20.0, 30.0, 40.0])
    sim = _sim_route(
        "68X",
        [10.0, 30.0, 60.0, 100.0],
        [1.0, 2.0, 3.0, 4.0],
    )
    monkeypatch.setattr(objective, "_load_l1_route_frames", lambda *args: (sim, real))

    rmse = objective.calculate_l1_rmse("sim.xml", "real.csv", "dist.csv")
    jl1 = objective.calculate_jl1_loss("sim.xml", "real.csv", "dist.csv")

    assert rmse == pytest.approx(np.sqrt(np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2)))
    assert jl1["status"] == "succeeded"
    assert jl1["n_errors"] == 4
    assert jl1["errors"] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_missing_simulation_output_is_not_returned_as_numeric_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(objective, "load_sim_data", lambda _: pd.DataFrame())

    with pytest.raises(objective.L1InfrastructureError, match="missing or malformed"):
        objective.calculate_l1_rmse("missing.xml", "real.csv", "dist.csv")
