from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.paper_experiments.sumo_data import (
    SimulationDataError,
    build_l2_observation_pair,
    load_stopinfo,
    parse_vehicle_identity,
    simulation_link_events,
    strict_rule_c_event_mask,
)


def test_vehicle_identity_and_strict_rule_boundaries() -> None:
    assert parse_vehicle_identity("flow_68X_inbound.2") == ("68X", "inbound")
    frame = pd.DataFrame(
        {
            "travel_time_s": [325.0, 326.0, 326.0, 326.0],
            "speed_kmh": [4.0, 5.0, 4.0, 4.0],
            "dist_m": [1500.0, 1500.0, 1500.0, 1500.1],
        }
    )
    assert strict_rule_c_event_mask(frame).tolist() == [False, False, True, False]


def _route_stops(tmp_path: Path) -> Path:
    path = tmp_path / "route_stops.csv"
    pd.DataFrame(
        {
            "route": ["68X", "68X"],
            "bound": ["inbound", "inbound"],
            "stop_id": ["s1", "s2"],
            "seq": [1, 2],
            "cum_dist_m": [0.0, 500.0],
        }
    ).to_csv(path, index=False)
    return path


def test_simulation_events_use_origin_departure_for_half_open_window(tmp_path: Path) -> None:
    stopinfo = tmp_path / "stopinfo.xml"
    stopinfo.write_text(
        "<stops>"
        '<stopinfo id="flow_68X_inbound.0" busStop="s1" arrival="890" started="891" ended="899"/>'
        '<stopinfo id="flow_68X_inbound.0" busStop="s2" arrival="901" started="901" ended="905"/>'
        "</stops>",
        encoding="utf-8",
    )
    events = simulation_link_events(
        stopinfo,
        _route_stops(tmp_path),
        window_start_s=0,
        window_end_s=900,
    )
    assert len(events) == 1
    assert events.loc[0, "event_time_sec"] == 899.0
    assert events.loc[0, "travel_time_s"] == 2.0


def test_malformed_stopinfo_is_rejected_instead_of_silently_dropped(tmp_path: Path) -> None:
    stopinfo = tmp_path / "stopinfo.xml"
    stopinfo.write_text(
        '<stops><stopinfo id="bad" busStop="s1" started="1" ended="2"/></stops>',
        encoding="utf-8",
    )
    with pytest.raises(SimulationDataError, match="invalid vehicle id"):
        load_stopinfo(stopinfo, _route_stops(tmp_path))


def test_l2_raw_and_moving_vectors_use_aggregated_rule_c_keys(tmp_path: Path) -> None:
    index = pd.DataFrame(
        {
            "observation_id": [1, 2, 3, 4],
            "route": ["68X"] * 4,
            "bound": ["inbound"] * 4,
            "from_seq": [1, 2, 3, 4],
            "to_seq": [2, 3, 4, 5],
        }
    )
    index_path = tmp_path / "index.csv"
    index.to_csv(index_path, index=False)
    events = pd.DataFrame(
        {
            "route": ["68X"] * 9,
            "bound": ["inbound"] * 9,
            "from_seq": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "to_seq": [2, 2, 2, 3, 3, 4, 4, 5, 5],
            "travel_time_s": [400, 410, 100, 120, 130, 200, 210, 180, 190],
            "speed_kmh": [4, 4, 20, 15, 16, 10, 11, 12, 13],
            "dist_m": [500] * 9,
        }
    )
    raw, moving = build_l2_observation_pair(events, index_path)
    assert raw["observation_id"].tolist() == [1, 2, 3, 4]
    assert moving["observation_id"].tolist() == [2, 3, 4]
    assert raw["observation_semantic"].unique().tolist() == ["raw_d2d"]
    assert moving["observation_semantic"].unique().tolist() == ["moving_only"]
