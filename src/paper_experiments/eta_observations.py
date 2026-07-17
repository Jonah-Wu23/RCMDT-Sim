"""Targeted, auditable ETA augmentation for the L1 observation chain.

KMB ``eta_seq`` is a stop-local prediction rank rather than a vehicle ID, so
it is not suitable for rebuilding every link.  The camera-ready amendment uses
ETA differences only for explicitly declared missing adjacent links.  All
existing pass-derived observations remain unchanged and every appended row is
labelled with its distinct measurement semantic.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HYBRID_LINK_SCHEMA = "l1-hybrid-link-events/v1"
BASE_SOURCE_SEMANTIC = "pass_derived_arrival_departure_detector"
ETA_GAP_SOURCE_SEMANTIC = "eta_derived_proxy_same_capture_same_eta_seq_gap_fill"
MIN_TRAVEL_TIME_S = 10.0
MIN_SPEED_KMH = 2.0
MAX_SPEED_KMH = 100.0
MIN_DOWNSTREAM_STOPS = 3

_LINK_COLUMNS = (
    "route",
    "bound",
    "service_type",
    "from_seq",
    "to_seq",
    "departure_ts",
    "arrival_ts",
    "travel_time_s",
    "dist_m",
    "speed_kmh",
)
_ETA_COLUMNS = (
    "capture_ts",
    "route",
    "bound",
    "service_type",
    "stop_seq",
    "eta",
    "eta_seq",
    "rmk_en",
)
_DIST_COLUMNS = ("route", "bound", "service_type", "seq", "link_dist_m")
_BOUND_ALIASES = {
    "i": "inbound",
    "inbound": "inbound",
    "o": "outbound",
    "outbound": "outbound",
}


class EtaObservationError(ValueError):
    """Raised when the targeted augmentation violates its frozen contract."""


@dataclass(frozen=True)
class EtaObservationBuild:
    events: pd.DataFrame
    diagnostics: Mapping[str, Any]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise EtaObservationError(f"{label} missing required columns: {missing}")


def _normalise_bound(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in _BOUND_ALIASES:
        raise EtaObservationError(f"Unsupported bound value: {value!r}")
    return _BOUND_ALIASES[text]


def _integer_column(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if numeric.isna().any() or not np.isfinite(values).all() or not np.isclose(
        values, np.round(values)
    ).all():
        raise EtaObservationError(f"{label}.{column} must contain finite integers")
    return numeric.astype(int)


def _utc_window(start_value: Any, end_value: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_value)
    end = pd.Timestamp(end_value)
    if pd.isna(start) or pd.isna(end):
        raise EtaObservationError("Observation window timestamps must be valid")
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise EtaObservationError("window_end must be later than window_start")
    return start, end


def _continuous_chain_length(pairs: set[tuple[int, int]], origin_seq: int = 1) -> int:
    sequence = int(origin_seq)
    count = 0
    while (sequence, sequence + 1) in pairs:
        sequence += 1
        count += 1
    return count


def _selected_pairs(routes: Sequence[Mapping[str, Any] | tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    result: set[tuple[str, str]] = set()
    for item in routes:
        if isinstance(item, Mapping):
            if not bool(item.get("l1_selected", True)):
                continue
            route = str(item.get("route", "")).strip()
            bound = _normalise_bound(item.get("direction", ""))
        else:
            route = str(item[0]).strip()
            bound = _normalise_bound(item[1])
        if not route:
            raise EtaObservationError("Selected L1 route must be non-empty")
        result.add((route, bound))
    if not result:
        raise EtaObservationError("At least one L1 route/direction must be selected")
    return tuple(sorted(result))


def _gap_contracts(gaps: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int, int, int]] = set()
    for item in gaps:
        contract = {
            "route": str(item.get("route", "")).strip(),
            "bound": _normalise_bound(item.get("bound", item.get("direction", ""))),
            "service_type": int(item.get("service_type", 1)),
            "from_seq": int(item["from_seq"]),
            "to_seq": int(item["to_seq"]),
        }
        if not contract["route"] or contract["to_seq"] != contract["from_seq"] + 1:
            raise EtaObservationError(f"Gap must be one adjacent, non-empty route link: {item}")
        identity = tuple(contract[key] for key in ("route", "bound", "service_type", "from_seq", "to_seq"))
        if identity in identities:
            raise EtaObservationError(f"Duplicate gap contract: {identity}")
        identities.add(identity)
        result.append(contract)
    if not result:
        raise EtaObservationError("At least one explicit missing-link contract is required")
    return tuple(result)


def _identifier(prefix: str, values: Sequence[Any]) -> str:
    rendered = "|".join(str(value) for value in values)
    return f"{prefix}-" + sha256(rendered.encode("utf-8")).hexdigest()


def build_hybrid_l1_observations(
    base_events: pd.DataFrame,
    station_eta: pd.DataFrame,
    route_stop_distance: pd.DataFrame,
    *,
    gap_links: Sequence[Mapping[str, Any]],
    selected_routes: Sequence[Mapping[str, Any] | tuple[str, str]],
    window_start: Any,
    window_end: Any,
    min_travel_time_s: float = MIN_TRAVEL_TIME_S,
    min_speed_kmh: float = MIN_SPEED_KMH,
    max_speed_kmh: float = MAX_SPEED_KMH,
    minimum_downstream_stops: int = MIN_DOWNSTREAM_STOPS,
) -> EtaObservationBuild:
    """Append only declared ETA-derived gaps and validate the L1 chain."""

    _require_columns(base_events, _LINK_COLUMNS, "base_events")
    _require_columns(station_eta, _ETA_COLUMNS, "station_eta")
    _require_columns(route_stop_distance, _DIST_COLUMNS, "route_stop_distance")
    if min_travel_time_s < 0 or min_speed_kmh <= 0 or max_speed_kmh <= min_speed_kmh:
        raise EtaObservationError("Physical filter bounds are invalid")
    if minimum_downstream_stops < 1:
        raise EtaObservationError("minimum_downstream_stops must be positive")
    start, end = _utc_window(window_start, window_end)
    gaps = _gap_contracts(gap_links)
    selected = _selected_pairs(selected_routes)

    base = base_events.loc[:, _LINK_COLUMNS].copy()
    base["route"] = base["route"].astype(str)
    base["bound"] = base["bound"].map(_normalise_bound)
    for column in ("service_type", "from_seq", "to_seq"):
        base[column] = _integer_column(base, column, "base_events")
    for column in ("travel_time_s", "dist_m", "speed_kmh"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    if not np.isfinite(base[["travel_time_s", "dist_m", "speed_kmh"]].to_numpy(dtype=float)).all():
        raise EtaObservationError("base_events contains non-finite measurements")
    if (base[["travel_time_s", "dist_m", "speed_kmh"]] <= 0).any().any():
        raise EtaObservationError("base_events measurements must be positive")
    for column in ("departure_ts", "arrival_ts"):
        base[column] = pd.to_datetime(base[column], errors="coerce", utc=True)
        if base[column].isna().any():
            raise EtaObservationError(f"base_events.{column} contains malformed timestamps")
    outside_base_window = (base["departure_ts"] < start) | (base["arrival_ts"] >= end)
    if outside_base_window.any():
        raise EtaObservationError(
            f"base_events has {int(outside_base_window.sum())} rows outside the half-open window"
        )

    eta = station_eta.loc[:, _ETA_COLUMNS].copy()
    eta["route"] = eta["route"].astype(str)
    eta["bound"] = eta["bound"].map(_normalise_bound)
    eta["capture_ts"] = pd.to_datetime(eta["capture_ts"], errors="coerce", utc=True)
    eta["eta"] = pd.to_datetime(eta["eta"], errors="coerce", utc=True)
    if eta[["capture_ts", "eta"]].isna().any().any():
        raise EtaObservationError("station_eta contains malformed capture_ts or eta")
    for column in ("service_type", "stop_seq", "eta_seq"):
        eta[column] = _integer_column(eta, column, "station_eta")
    eta = eta.loc[(eta["capture_ts"] >= start) & (eta["capture_ts"] < end)].copy()
    raw_key = ["capture_ts", "route", "bound", "service_type", "stop_seq", "eta_seq"]
    eta = eta.drop_duplicates([*raw_key, "eta"])
    if eta.duplicated(raw_key, keep=False).any():
        raise EtaObservationError("Conflicting ETAs share one station snapshot key")

    distances = route_stop_distance.loc[:, _DIST_COLUMNS].copy()
    distances["route"] = distances["route"].astype(str)
    distances["bound"] = distances["bound"].map(_normalise_bound)
    distances["service_type"] = _integer_column(
        distances, "service_type", "route_stop_distance"
    )
    distances["to_seq"] = _integer_column(distances, "seq", "route_stop_distance")
    distances["dist_m"] = pd.to_numeric(distances["link_dist_m"], errors="coerce")
    distance_key = ["route", "bound", "service_type", "to_seq"]
    if distances.duplicated(distance_key).any():
        raise EtaObservationError("route_stop_distance has duplicate route/bound/service/seq keys")

    supplements: list[pd.DataFrame] = []
    gap_diagnostics: list[dict[str, Any]] = []
    for gap in gaps:
        existing = base.loc[
            (base["route"] == gap["route"])
            & (base["bound"] == gap["bound"])
            & (base["service_type"] == gap["service_type"])
            & (base["from_seq"] == gap["from_seq"])
            & (base["to_seq"] == gap["to_seq"])
        ]
        if not existing.empty:
            raise EtaObservationError(f"Declared gap already exists in base_events: {gap}")
        upstream = eta.loc[
            (eta["route"] == gap["route"])
            & (eta["bound"] == gap["bound"])
            & (eta["service_type"] == gap["service_type"])
            & (eta["stop_seq"] == gap["from_seq"]),
            ["capture_ts", "eta_seq", "eta", "rmk_en"],
        ].rename(columns={"eta": "departure_ts", "rmk_en": "upstream_rmk_en"})
        downstream = eta.loc[
            (eta["route"] == gap["route"])
            & (eta["bound"] == gap["bound"])
            & (eta["service_type"] == gap["service_type"])
            & (eta["stop_seq"] == gap["to_seq"]),
            ["capture_ts", "eta_seq", "eta", "rmk_en"],
        ].rename(columns={"eta": "arrival_ts", "rmk_en": "downstream_rmk_en"})
        pairs = upstream.merge(
            downstream,
            on=["capture_ts", "eta_seq"],
            how="inner",
            validate="one_to_one",
        )
        pairs["route"] = gap["route"]
        pairs["bound"] = gap["bound"]
        pairs["service_type"] = gap["service_type"]
        pairs["from_seq"] = gap["from_seq"]
        pairs["to_seq"] = gap["to_seq"]
        pairs["travel_time_s"] = (
            pairs["arrival_ts"] - pairs["departure_ts"]
        ).dt.total_seconds()
        pairs = pairs.merge(
            distances[distance_key + ["dist_m"]],
            on=distance_key,
            how="left",
            validate="many_to_one",
        )
        pairs["speed_kmh"] = pairs["dist_m"] * 3.6 / pairs["travel_time_s"]
        complete_window = (pairs["departure_ts"] >= start) & (pairs["arrival_ts"] < end)
        finite = np.isfinite(
            pairs[["travel_time_s", "dist_m", "speed_kmh"]].to_numpy(dtype=float)
        ).all(axis=1)
        physical = (
            finite
            & complete_window
            & (pairs["travel_time_s"] > float(min_travel_time_s))
            & (pairs["dist_m"] > 0.0)
            & (pairs["speed_kmh"] >= float(min_speed_kmh))
            & (pairs["speed_kmh"] <= float(max_speed_kmh))
        )
        retained = pairs.loc[physical].sort_values(
            ["departure_ts", "arrival_ts", "capture_ts", "eta_seq"], kind="stable"
        )
        before_deduplication = int(len(retained))
        retained = retained.drop_duplicates(["departure_ts", "arrival_ts"], keep="first").copy()
        scheduled = (
            retained["upstream_rmk_en"]
            .fillna("")
            .astype(str)
            .str.contains("Scheduled Bus", case=False, regex=False)
            | retained["downstream_rmk_en"]
            .fillna("")
            .astype(str)
            .str.contains("Scheduled Bus", case=False, regex=False)
        )
        scheduled_excluded = int(scheduled.sum())
        retained = retained.loc[~scheduled].copy()
        if retained.empty:
            raise EtaObservationError(f"Gap has no complete-window physical ETA pairs: {gap}")
        retained["source_semantic"] = ETA_GAP_SOURCE_SEMANTIC
        retained["observation_id"] = retained.apply(
            lambda row: _identifier(
                "eta-gap",
                [
                    gap["route"],
                    gap["bound"],
                    gap["service_type"],
                    gap["from_seq"],
                    gap["to_seq"],
                    pd.Timestamp(row["departure_ts"]).isoformat(),
                    pd.Timestamp(row["arrival_ts"]).isoformat(),
                ],
            ),
            axis=1,
        )
        supplements.append(retained)
        gap_diagnostics.append(
            {
                **gap,
                "same_rank_candidate_pairs": int(len(pairs)),
                "scheduled_bus_pairs_excluded_after_exact_pair_deduplication": scheduled_excluded,
                "retained_complete_window_physical_pairs_before_deduplication": before_deduplication,
                "exact_eta_pairs_appended": int(len(retained)),
                "travel_time_min_s": float(retained["travel_time_s"].min()),
                "travel_time_max_s": float(retained["travel_time_s"].max()),
            }
        )

    base = base.reset_index(drop=True)
    base["capture_ts"] = pd.NaT
    base["eta_seq"] = pd.NA
    base["source_semantic"] = BASE_SOURCE_SEMANTIC
    base["upstream_rmk_en"] = ""
    base["downstream_rmk_en"] = ""
    base["observation_id"] = [
        _identifier(
            "base",
            [
                index,
                row["route"],
                row["bound"],
                row["service_type"],
                row["from_seq"],
                row["to_seq"],
                pd.Timestamp(row["departure_ts"]).isoformat(),
                pd.Timestamp(row["arrival_ts"]).isoformat(),
            ],
        )
        for index, row in base.iterrows()
    ]
    combined = pd.concat([base, *supplements], ignore_index=True, sort=False)
    if combined["observation_id"].duplicated().any():
        raise EtaObservationError("Hybrid observation IDs are not unique")
    combined = combined.sort_values(
        ["route", "bound", "service_type", "from_seq", "to_seq", "source_semantic", "departure_ts"],
        kind="stable",
    ).reset_index(drop=True)

    chain_diagnostics: dict[str, Any] = {}
    for route, bound in selected:
        subset = combined.loc[(combined["route"] == route) & (combined["bound"] == bound)]
        link_pairs = set(zip(subset["from_seq"].astype(int), subset["to_seq"].astype(int)))
        chain_length = _continuous_chain_length(link_pairs)
        if chain_length < int(minimum_downstream_stops):
            raise EtaObservationError(
                f"route={route}, bound={bound} has {chain_length} downstream links from "
                f"sequence 1 after augmentation; at least {minimum_downstream_stops} required"
            )
        chain_diagnostics[f"{route}/{bound}"] = {
            "event_count": int(len(subset)),
            "unique_link_count": int(len(link_pairs)),
            "continuous_downstream_links_from_sequence_1": int(chain_length),
        }

    combined["schema_version"] = HYBRID_LINK_SCHEMA
    for column in ("capture_ts", "departure_ts", "arrival_ts"):
        combined[column] = combined[column].map(
            lambda value: "" if pd.isna(value) else pd.Timestamp(value).isoformat()
        )
    for column in ("travel_time_s", "dist_m", "speed_kmh"):
        combined[column] = combined[column].astype(float).round(6)
    output_columns = [
        "schema_version",
        "observation_id",
        *_LINK_COLUMNS,
        "capture_ts",
        "eta_seq",
        "upstream_rmk_en",
        "downstream_rmk_en",
        "source_semantic",
    ]
    events = combined.loc[:, output_columns].copy()
    diagnostics = {
        "base_event_count": int(len(base)),
        "eta_gap_event_count": int(sum(len(frame) for frame in supplements)),
        "output_event_count": int(len(events)),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "gap_links": gap_diagnostics,
        "chain_diagnostics": chain_diagnostics,
    }
    return EtaObservationBuild(events=events, diagnostics=diagnostics)


__all__ = [
    "BASE_SOURCE_SEMANTIC",
    "ETA_GAP_SOURCE_SEMANTIC",
    "HYBRID_LINK_SCHEMA",
    "EtaObservationBuild",
    "EtaObservationError",
    "build_hybrid_l1_observations",
]
