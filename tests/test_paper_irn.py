from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.paper_experiments.irn import (
    build_link_to_irn_mapping,
    compute_irn_contradiction,
    load_irn_window_speeds,
    select_irn_window_files,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_irn_xml(
    path: Path,
    rows: list[tuple[int, object, str]],
    *,
    source_date: str = "2025-12-19",
    source_time: str = "17:01:00",
) -> None:
    segments = "".join(
        "<segment>"
        f"<segment_id>{segment_id}</segment_id>"
        f"<speed>{speed}</speed>"
        f"<valid>{valid}</valid>"
        "</segment>"
        for segment_id, speed, valid in rows
    )
    path.write_text(
        f"<root><date>{source_date}</date><time>{source_time}</time>{segments}</root>",
        encoding="utf-8",
    )


def test_mapping_strips_reverse_suffix_and_hash_is_order_independent(tmp_path: Path) -> None:
    observation_rows = [
        {
            "observation_id": 2,
            "route": "960",
            "bound": "O",
            "from_seq": 8,
            "to_seq": 9,
        },
        {
            "observation_id": 1,
            "route": "68x",
            "bound": "I",
            "from_seq": 1,
            "to_seq": 2,
        },
    ]
    mapping_rows = [
        {"observation_id": 1, "edge_ids": json.dumps(["105735_rev", "105735", "273264_rev"])},
        {"observation_id": 2, "edge_ids": json.dumps(["42_rev"])},
    ]
    obs_a = tmp_path / "observations_a.csv"
    map_a = tmp_path / "mapping_a.csv"
    _write_csv(
        obs_a,
        ["observation_id", "route", "bound", "from_seq", "to_seq"],
        observation_rows,
    )
    _write_csv(map_a, ["observation_id", "edge_ids"], mapping_rows)

    mapping_a, hash_a = build_link_to_irn_mapping(obs_a, map_a)

    assert mapping_a == {
        ("68X", "inbound", 1, 2): (105735, 273264),
        ("960", "outbound", 8, 9): (42,),
    }
    assert len(hash_a) == 64

    obs_b = tmp_path / "observations_b.csv"
    map_b = tmp_path / "mapping_b.csv"
    _write_csv(
        obs_b,
        ["observation_id", "route", "bound", "from_seq", "to_seq"],
        list(reversed(observation_rows)),
    )
    _write_csv(map_b, ["observation_id", "edge_ids"], list(reversed(mapping_rows)))

    mapping_b, hash_b = build_link_to_irn_mapping(obs_b, map_b)
    assert mapping_b == mapping_a
    assert hash_b == hash_a


def test_mapping_rejects_non_numeric_edge_id(tmp_path: Path) -> None:
    observations = tmp_path / "observations.csv"
    mapping = tmp_path / "mapping.csv"
    _write_csv(
        observations,
        ["observation_id", "route", "bound", "from_seq", "to_seq"],
        [{"observation_id": 1, "route": "68X", "bound": "inbound", "from_seq": 1, "to_seq": 2}],
    )
    _write_csv(
        mapping,
        ["observation_id", "edge_ids"],
        [{"observation_id": 1, "edge_ids": json.dumps(["edge-7_rev"])}],
    )

    with pytest.raises(ValueError, match="must be numeric"):
        build_link_to_irn_mapping(observations, mapping)


def test_window_parser_uses_only_valid_records_and_segment_medians(tmp_path: Path) -> None:
    window = tmp_path / "irn-window"
    window.mkdir()
    _write_irn_xml(
        window / "irnAvgSpeed-all-20251219-170200.xml",
        [(101, 8, "Y"), (102, 100, "N"), (103, 12, "Y")],
        source_time="17:02:00",
    )
    _write_irn_xml(
        window / "irnAvgSpeed-all-20251219-170100.xml",
        [(101, 4, "Y"), (102, 6, "Y"), (103, 10, "Y")],
        source_time="17:01:00",
    )
    _write_irn_xml(
        window / "irnAvgSpeed-all-20251219-170020.xml",
        [(101, 1000, "Y")],
        source_time="16:55:00",
    )
    _write_irn_xml(
        window / "irnAvgSpeed-all-20251219-180000.xml",
        [(101, 1000, "Y")],
        source_time="18:00:00",
    )

    selected = select_irn_window_files(
        window,
        observation_date="2025-12-19",
        window_start="17:00:00",
        window_end="18:00:00",
    )
    assert [path.name for path in selected] == [
        "irnAvgSpeed-all-20251219-170100.xml",
        "irnAvgSpeed-all-20251219-170200.xml",
    ]
    assert load_irn_window_speeds(
        window,
        observation_date="2025-12-19",
        window_start="17:00:00",
        window_end="18:00:00",
    ) == {101: 6.0, 102: 6.0, 103: 11.0}


def test_window_parser_rejects_malformed_valid_speed(tmp_path: Path) -> None:
    window = tmp_path / "irn-window"
    window.mkdir()
    _write_irn_xml(
        window / "irnAvgSpeed-all-20251219-170100.xml",
        [(101, "bad", "Y")],
    )

    with pytest.raises(ValueError, match="must be numeric"):
        load_irn_window_speeds(
            window,
            observation_date="2025-12-19",
            window_start="17:00:00",
            window_end="18:00:00",
        )


def test_contradiction_counts_exclude_unmatched_flagged_records() -> None:
    mapping = {
        ("68X", "inbound", 1, 2): (101, 102),
        ("68X", "inbound", 2, 3): (103,),
        ("68X", "inbound", 3, 4): (104,),
        ("68X", "inbound", 4, 5): (105,),
    }
    irn_speeds = {101: 6.0, 102: 8.0, 103: 3.0, 105: 9.0}
    flagged = [
        {"route": "68x", "bound": "I", "from_seq": 1, "to_seq": 2, "speed_median": 4.0},
        {"route": "68X", "bound": "inbound", "from_seq": 2, "to_seq": 3, "speed_median": 4.0},
        {"route": "68X", "bound": "inbound", "from_seq": 3, "to_seq": 4, "speed_median": 2.0},
        {"route": "68X", "bound": "inbound", "from_seq": 4, "to_seq": 5, "speed_median": 5.0},
    ]

    assert compute_irn_contradiction(flagged, mapping, irn_speeds) == {
        "numerator": 1,
        "denominator": 3,
        "unmatched_flagged": 1,
    }
