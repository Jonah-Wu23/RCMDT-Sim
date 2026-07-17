from __future__ import annotations

import pandas as pd
import pytest

from src.paper_experiments.eta_observations import (
    BASE_SOURCE_SEMANTIC,
    ETA_GAP_SOURCE_SEMANTIC,
    HYBRID_LINK_SCHEMA,
    EtaObservationError,
    build_hybrid_l1_observations,
)


WINDOW_START = "2025-12-19T09:00:00Z"
WINDOW_END = "2025-12-19T10:00:00Z"
GAP = {
    "route": "68X",
    "bound": "inbound",
    "service_type": 1,
    "from_seq": 2,
    "to_seq": 3,
}


def _base(*, include_gap: bool = False) -> pd.DataFrame:
    links = [(1, 2), (3, 4)]
    if include_gap:
        links.append((2, 3))
    return pd.DataFrame(
        [
            {
                "route": "68X",
                "bound": "inbound",
                "service_type": 1,
                "from_seq": from_seq,
                "to_seq": to_seq,
                "departure_ts": f"2025-12-19T09:0{index}:00Z",
                "arrival_ts": f"2025-12-19T09:0{index + 1}:00Z",
                "travel_time_s": 60.0,
                "dist_m": 1000.0,
                "speed_kmh": 60.0,
            }
            for index, (from_seq, to_seq) in enumerate(links, start=1)
        ]
    )


def _eta() -> pd.DataFrame:
    rows = []
    for capture, departure, arrival in (
        ("2025-12-19T09:00:00Z", "2025-12-19T09:10:00Z", "2025-12-19T09:11:00Z"),
        ("2025-12-19T09:01:00Z", "2025-12-19T09:10:00Z", "2025-12-19T09:11:00Z"),
        ("2025-12-19T09:02:00Z", "2025-12-19T09:20:00Z", "2025-12-19T09:21:10Z"),
    ):
        rows.extend(
            [
                {
                    "capture_ts": capture,
                    "route": "68X",
                    "bound": "I",
                    "service_type": 1,
                    "stop_seq": 2,
                    "eta": departure,
                    "eta_seq": 1,
                    "rmk_en": "",
                },
                {
                    "capture_ts": capture,
                    "route": "68X",
                    "bound": "I",
                    "service_type": 1,
                    "stop_seq": 3,
                    "eta": arrival,
                    "eta_seq": 1,
                    "rmk_en": "",
                },
            ]
        )
    return pd.DataFrame(rows)


def _distance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "route": ["68X", "68X", "68X", "68X"],
            "bound": ["inbound"] * 4,
            "service_type": [1] * 4,
            "seq": [1, 2, 3, 4],
            "link_dist_m": [0.0, 1000.0, 1000.0, 1000.0],
        }
    )


def _build(base: pd.DataFrame | None = None, eta: pd.DataFrame | None = None):
    return build_hybrid_l1_observations(
        _base() if base is None else base,
        _eta() if eta is None else eta,
        _distance(),
        gap_links=[GAP],
        selected_routes=[("68X", "inbound")],
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )


def test_only_declared_gap_is_appended_and_exact_eta_pairs_are_deduplicated() -> None:
    result = _build()

    assert len(result.events) == 4
    assert result.events["schema_version"].eq(HYBRID_LINK_SCHEMA).all()
    assert result.events["observation_id"].is_unique
    assert result.events["source_semantic"].value_counts().to_dict() == {
        BASE_SOURCE_SEMANTIC: 2,
        ETA_GAP_SOURCE_SEMANTIC: 2,
    }
    supplement = result.events.loc[result.events["source_semantic"] == ETA_GAP_SOURCE_SEMANTIC]
    assert set(zip(supplement["from_seq"], supplement["to_seq"])) == {(2, 3)}
    assert supplement["travel_time_s"].tolist() == [60.0, 70.0]
    assert result.diagnostics["base_event_count"] == 2
    assert result.diagnostics["eta_gap_event_count"] == 2
    assert result.diagnostics["chain_diagnostics"]["68X/inbound"][
        "continuous_downstream_links_from_sequence_1"
    ] == 3


def test_eta_departure_and_arrival_must_both_be_inside_half_open_window() -> None:
    eta = _eta()
    outside = pd.DataFrame(
        [
            {
                "capture_ts": "2025-12-19T09:30:00Z",
                "route": "68X",
                "bound": "I",
                "service_type": 1,
                "stop_seq": stop,
                "eta": value,
                "eta_seq": 2,
                "rmk_en": "",
            }
            for stop, value in ((2, "2025-12-19T09:59:30Z"), (3, WINDOW_END))
        ]
    )
    result = _build(eta=pd.concat([eta, outside], ignore_index=True))
    assert result.diagnostics["eta_gap_event_count"] == 2
    assert WINDOW_END not in set(result.events["arrival_ts"])


def test_declared_gap_must_be_absent_from_base_events() -> None:
    with pytest.raises(EtaObservationError, match="already exists"):
        _build(base=_base(include_gap=True))


def test_scheduled_bus_proxy_pairs_are_excluded() -> None:
    eta = _eta()
    eta.loc[
        (eta["capture_ts"] == "2025-12-19T09:02:00Z") & (eta["stop_seq"] == 2),
        "rmk_en",
    ] = "Scheduled Bus"
    result = _build(eta=eta)
    supplement = result.events.loc[
        result.events["source_semantic"] == ETA_GAP_SOURCE_SEMANTIC
    ]
    assert supplement["travel_time_s"].tolist() == [60.0]
    assert result.diagnostics["gap_links"][0][
        "scheduled_bus_pairs_excluded_after_exact_pair_deduplication"
    ] == 1


def test_base_events_outside_window_are_rejected_not_silently_filtered() -> None:
    base = _base()
    base.loc[0, "departure_ts"] = "2025-12-19T08:59:59Z"
    with pytest.raises(EtaObservationError, match="outside the half-open window"):
        _build(base=base)


def test_missing_gap_evidence_is_a_hard_failure() -> None:
    eta = _eta()
    eta["eta"] = "2025-12-19T10:30:00Z"
    with pytest.raises(EtaObservationError, match="no complete-window physical ETA pairs"):
        _build(eta=eta)
