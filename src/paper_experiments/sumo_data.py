"""Data adapters shared by camera-ready calibration and evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


LINK_KEY_COLUMNS = ["route", "bound", "from_seq", "to_seq"]


class SimulationDataError(ValueError):
    """Raised when a simulation output cannot satisfy the declared data contract."""


def parse_vehicle_identity(vehicle_id: str) -> tuple[str, str]:
    """Parse IDs emitted from routes such as ``flow_68X_inbound.0``."""

    parts = str(vehicle_id).split("_")
    if len(parts) < 3:
        raise SimulationDataError(f"Unrecognized bus vehicle id: {vehicle_id}")
    return parts[1], parts[2].split(".", 1)[0]


def load_stop_index(path: Path) -> dict[tuple[str, str, str], tuple[int, float]]:
    frame = pd.read_csv(path)
    required = {"route", "bound", "stop_id", "seq", "cum_dist_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SimulationDataError(f"Route-stop file is missing columns: {missing}")
    if "cum_dist_m_dir" in frame.columns:
        distances = frame["cum_dist_m_dir"].fillna(frame["cum_dist_m"])
    else:
        distances = frame["cum_dist_m"]
    index: dict[tuple[str, str, str], tuple[int, float]] = {}
    for row, distance in zip(frame.itertuples(index=False), distances, strict=True):
        key = (str(row.route), str(row.bound), str(row.stop_id))
        if key in index:
            raise SimulationDataError(f"Duplicate route-stop key in {path}: {key}")
        if not np.isfinite(float(distance)):
            raise SimulationDataError(f"Non-finite route-stop distance in {path}: {key}")
        index[key] = (int(row.seq), float(distance))
    return index


def load_stopinfo(path: Path, route_stop_path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SimulationDataError(f"Missing or empty stopinfo: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise SimulationDataError(f"Malformed stopinfo: {path}") from exc
    stop_index = load_stop_index(route_stop_path)
    records: list[dict[str, object]] = []
    nodes = root.findall("stopinfo")
    for record_index, node in enumerate(nodes, start=1):
        vehicle_id = node.get("id")
        stop_id = node.get("busStop")
        started = node.get("started")
        ended = node.get("ended")
        if not vehicle_id or not stop_id or started is None or ended is None:
            raise SimulationDataError(
                f"stopinfo[{record_index}] is missing id, busStop, started, or ended in {path}"
            )
        try:
            route, bound = parse_vehicle_identity(vehicle_id)
        except SimulationDataError as exc:
            raise SimulationDataError(
                f"stopinfo[{record_index}] has an invalid vehicle id in {path}: {vehicle_id}"
            ) from exc
        mapping = stop_index.get((route, bound, stop_id))
        if mapping is None:
            raise SimulationDataError(
                f"stopinfo[{record_index}] has no frozen route-stop mapping in {path}: "
                f"{(route, bound, stop_id)}"
            )
        seq, distance = mapping
        arrival_text = node.get("arrival")
        try:
            started_value = float(started)
            ended_value = float(ended)
            arrival = float(arrival_text) if arrival_text is not None else started_value
        except ValueError as exc:
            raise SimulationDataError(
                f"stopinfo[{record_index}] has a non-numeric time in {path}"
            ) from exc
        if not np.isfinite([arrival, started_value, ended_value]).all():
            raise SimulationDataError(
                f"stopinfo[{record_index}] has a non-finite time in {path}"
            )
        if arrival > started_value or started_value > ended_value:
            raise SimulationDataError(
                f"stopinfo[{record_index}] violates arrival <= started <= ended in {path}"
            )
        records.append(
            {
                "vehicle_id": vehicle_id,
                "route": route,
                "bound": bound,
                "stop_id": stop_id,
                "seq": seq,
                "cum_dist_m": distance,
                "arrival_time": arrival,
                "started": started_value,
                "ended": ended_value,
            }
        )
    if not records:
        raise SimulationDataError(f"No matched bus stop records in {path}")
    return pd.DataFrame.from_records(records).sort_values(["vehicle_id", "started", "seq"])


def simulation_link_events(
    stopinfo_path: Path,
    route_stop_path: Path,
    *,
    window_start_s: float = 0.0,
    window_end_s: float = 3600.0,
) -> pd.DataFrame:
    """Build moving-time event samples on half-open simulation windows."""

    stops = load_stopinfo(stopinfo_path, route_stop_path)
    records: list[dict[str, object]] = []
    for vehicle_id, group in stops.groupby("vehicle_id", sort=True):
        ordered = group.sort_values(["started", "seq"]).reset_index(drop=True)
        for current, following in zip(
            ordered.iloc[:-1].itertuples(index=False),
            ordered.iloc[1:].itertuples(index=False),
            strict=True,
        ):
            if current.route != following.route or current.bound != following.bound:
                continue
            if int(following.seq) != int(current.seq) + 1:
                continue
            # Real events are keyed by origin departure_ts, so the simulated
            # half-open window must use the matching origin departure event.
            event_time = float(current.ended)
            if not (window_start_s <= event_time < window_end_s):
                continue
            travel_time = float(following.started) - float(current.ended)
            distance = float(following.cum_dist_m) - float(current.cum_dist_m)
            if travel_time <= 0 or distance <= 0:
                continue
            records.append(
                {
                    "vehicle_id": vehicle_id,
                    "route": str(current.route),
                    "bound": str(current.bound),
                    "from_seq": int(current.seq),
                    "to_seq": int(following.seq),
                    "travel_time_s": travel_time,
                    "dist_m": distance,
                    "speed_kmh": distance * 3.6 / travel_time,
                    "event_time_sec": event_time - window_start_s,
                }
            )
    return pd.DataFrame.from_records(records)


def real_event_window(
    path: Path,
    *,
    observation_date: str,
    window_start_hkt: str,
    window_end_hkt: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(LINK_KEY_COLUMNS + ["departure_ts", "travel_time_s", "dist_m", "speed_kmh"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SimulationDataError(f"Real event file is missing columns: {missing}")
    departure = pd.to_datetime(frame["departure_ts"], utc=True, errors="coerce").dt.tz_convert("Asia/Hong_Kong")
    start = pd.Timestamp(f"{observation_date} {window_start_hkt}", tz="Asia/Hong_Kong")
    end = pd.Timestamp(f"{observation_date} {window_end_hkt}", tz="Asia/Hong_Kong")
    mask = departure.notna() & (departure >= start) & (departure < end)
    result = frame.loc[mask].copy()
    result["event_time_sec"] = (departure.loc[mask] - start).dt.total_seconds().to_numpy()
    result["route"] = result["route"].astype(str)
    result["bound"] = result["bound"].astype(str)
    return result


def strict_rule_c_event_mask(
    frame: pd.DataFrame,
    *,
    travel_time_s: float = 325.0,
    speed_kmh: float = 5.0,
    distance_m: float = 1500.0,
) -> pd.Series:
    return (
        (frame["travel_time_s"] > travel_time_s)
        & (frame["speed_kmh"] < speed_kmh)
        & (frame["dist_m"] <= distance_m)
    )


def build_l2_observation_pair(
    events: pd.DataFrame,
    observation_index_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw-D2D and Rule-C-clean moving-only development vectors.

    Rule C is a link-hour decision in the paper protocol.  It is therefore
    applied after aggregating each frozen M11 key, never event by event.  The
    raw vector retains every frozen eligible key; the moving-only vector drops
    keys whose aggregated median satisfies Rule C.
    """

    index = pd.read_csv(observation_index_path)
    index_columns = ["observation_id", *LINK_KEY_COLUMNS]
    missing = sorted(set(index_columns).difference(index.columns))
    if missing:
        raise SimulationDataError(f"Observation index is missing columns: {missing}")
    if index[LINK_KEY_COLUMNS].duplicated().any() or index["observation_id"].duplicated().any():
        raise SimulationDataError("Observation index contains duplicate IDs or link keys")
    indexed = events.merge(index[index_columns], on=LINK_KEY_COLUMNS, how="inner")
    if indexed.empty:
        raise SimulationDataError("No real events match the L2 observation index")
    numeric_columns = ["travel_time_s", "speed_kmh", "dist_m"]
    numeric = indexed[numeric_columns].apply(pd.to_numeric, errors="coerce")
    eligible_mask = (
        np.isfinite(numeric).all(axis=1)
        & (numeric["travel_time_s"] > 0)
        & (numeric["speed_kmh"] > 0)
        & (numeric["dist_m"] > 0)
        & (numeric["dist_m"] <= 1500.0)
    )
    eligible = indexed.loc[eligible_mask].copy()
    eligible[numeric_columns] = numeric.loc[eligible_mask]
    eligible_keys = eligible[LINK_KEY_COLUMNS].drop_duplicates()
    frozen_keys = index[LINK_KEY_COLUMNS].drop_duplicates()
    missing_keys = frozen_keys.merge(
        eligible_keys,
        on=LINK_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only", LINK_KEY_COLUMNS]
    if not missing_keys.empty:
        raise SimulationDataError(
            f"Frozen L2 keys have no eligible development events: {missing_keys.to_dict('records')}"
        )

    link_hour = (
        eligible.groupby(LINK_KEY_COLUMNS, as_index=False)
        .agg(
            tt_median=("travel_time_s", "median"),
            speed_median=("speed_kmh", "median"),
            dist_m=("dist_m", "median"),
        )
    )
    flagged = (
        (link_hour["tt_median"] > 325.0)
        & (link_hour["speed_median"] < 5.0)
        & (link_hour["dist_m"] <= 1500.0)
    )
    retained_keys = link_hour.loc[~flagged, LINK_KEY_COLUMNS]
    moving_events = eligible.merge(retained_keys, on=LINK_KEY_COLUMNS, how="inner")

    def aggregate(source: pd.DataFrame, semantic: str) -> pd.DataFrame:
        grouped = (
            source.groupby(["observation_id", *LINK_KEY_COLUMNS], as_index=False)
            .agg(
                mean_speed_kmh=("speed_kmh", "mean"),
                std_speed_kmh=("speed_kmh", "std"),
                sample_count=("speed_kmh", "size"),
                dist_m=("dist_m", "median"),
            )
            .sort_values("observation_id")
            .reset_index(drop=True)
        )
        grouped["std_speed_kmh"] = grouped["std_speed_kmh"].fillna(10.0).clip(lower=1.0)
        grouped["observation_semantic"] = semantic
        return grouped

    raw_vector = aggregate(eligible, "raw_d2d")
    moving_vector = aggregate(moving_events, "moving_only")
    if len(raw_vector) != len(index):
        raise SimulationDataError(
            f"Raw L2 vector has {len(raw_vector)} keys; frozen index requires {len(index)}"
        )
    if len(moving_vector) < 3:
        raise SimulationDataError("Fewer than three clean-observable L2 links")
    return raw_vector, moving_vector


def extract_simulation_vector(
    events: pd.DataFrame,
    observation_vector: pd.DataFrame,
) -> np.ndarray:
    if events.empty:
        raise SimulationDataError("Simulation link-event table is empty")
    means = events.groupby(LINK_KEY_COLUMNS, as_index=False)["speed_kmh"].mean()
    merged = observation_vector[LINK_KEY_COLUMNS].merge(means, on=LINK_KEY_COLUMNS, how="left")
    values = merged["speed_kmh"].to_numpy(dtype=float)
    if np.isnan(values).any():
        missing = merged.loc[merged["speed_kmh"].isna(), LINK_KEY_COLUMNS].to_dict("records")
        raise SimulationDataError(f"Simulation misses required L2 links: {missing}")
    return values
