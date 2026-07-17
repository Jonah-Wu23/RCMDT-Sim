"""IRN link mapping and audit-contradiction utilities.

The paper audit compares one flagged D2D link-hour record with the median
speed of the IRN segments mapped to that link.  Unmatched flagged records are
reported separately and never enter the contradiction-rate denominator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import numbers
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


LinkKey = tuple[str, str, int, int]

_EDGE_REVERSE_SUFFIX = re.compile(r"_rev$", re.IGNORECASE)
_INTEGER_TEXT = re.compile(r"[+-]?\d+")
_IRN_FILENAME = re.compile(r"irnAvgSpeed-all-(\d{8})-(\d{6})\.xml")
_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_BOUND_ALIASES = {
    "i": "inbound",
    "in": "inbound",
    "inbound": "inbound",
    "o": "outbound",
    "out": "outbound",
    "outbound": "outbound",
}


def _path(value: str | Path, *, kind: str) -> Path:
    path = Path(value)
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Expected file: {path}")
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(f"Expected directory: {path}")
    return path


def _required_columns(fieldnames: Sequence[str] | None, required: set[str], source: Path) -> None:
    present = set(fieldnames or ())
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
        raise ValueError(f"{field} must be an integer, got {value!r}")
    text = str(value).strip()
    if not _INTEGER_TEXT.fullmatch(text):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return int(text)


def _finite_speed(value: object, *, field: str) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(speed) or speed < 0:
        raise ValueError(f"{field} must be a finite non-negative speed, got {value!r}")
    return speed


def _normalise_route(value: object) -> str:
    route = str(value).strip().upper()
    if not route:
        raise ValueError("route must not be empty")
    return route


def _normalise_bound(value: object) -> str:
    bound = str(value).strip().casefold()
    if not bound:
        raise ValueError("bound must not be empty")
    return _BOUND_ALIASES.get(bound, bound)


def _link_key(record: Mapping[str, object], *, context: str) -> LinkKey:
    required = ("route", "bound", "from_seq", "to_seq")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")
    return (
        _normalise_route(record["route"]),
        _normalise_bound(record["bound"]),
        _integer(record["from_seq"], field=f"{context}.from_seq"),
        _integer(record["to_seq"], field=f"{context}.to_seq"),
    )


def _segment_id(edge_id: object, *, context: str) -> int:
    text = _EDGE_REVERSE_SUFFIX.sub("", str(edge_id).strip())
    if not text or not text.isdecimal():
        raise ValueError(
            f"{context} edge id must be numeric after removing '_rev', got {edge_id!r}"
        )
    return int(text)


def _edge_ids(raw_value: object, *, context: str) -> tuple[int, ...]:
    text = str(raw_value).strip()
    if not text:
        values: object = []
    else:
        try:
            values = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}.edge_ids is not valid JSON") from exc
    if not isinstance(values, list):
        raise ValueError(f"{context}.edge_ids must be a JSON list")
    return tuple(sorted({_segment_id(value, context=context) for value in values}))


def hash_link_to_irn_mapping(mapping: Mapping[LinkKey, Sequence[int]]) -> str:
    """Return a deterministic SHA-256 over a canonical link/segment mapping."""

    canonical_rows = []
    for raw_key, raw_segments in mapping.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise ValueError(f"Invalid link key: {raw_key!r}")
        key = _link_key(
            {
                "route": raw_key[0],
                "bound": raw_key[1],
                "from_seq": raw_key[2],
                "to_seq": raw_key[3],
            },
            context="mapping key",
        )
        segments = sorted({_integer(value, field="segment_id") for value in raw_segments})
        if any(value < 0 for value in segments):
            raise ValueError("segment_id must be non-negative")
        canonical_rows.append(
            {
                "route": key[0],
                "bound": key[1],
                "from_seq": key[2],
                "to_seq": key[3],
                "segment_ids": segments,
            }
        )

    canonical_rows.sort(
        key=lambda row: (row["route"], row["bound"], row["from_seq"], row["to_seq"])
    )
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_link_to_irn_mapping(
    observation_csv: str | Path,
    edge_mapping_csv: str | Path,
) -> tuple[dict[LinkKey, tuple[int, ...]], str]:
    """Join observation link keys to numeric IRN segment IDs.

    ``edge_ids`` must be a JSON list.  Directional SUMO suffixes such as
    ``105735_rev`` collapse to the same numeric IRN segment as ``105735``.
    Every observation must have exactly one edge-mapping row.
    """

    observation_path = _path(observation_csv, kind="file")
    mapping_path = _path(edge_mapping_csv, kind="file")

    observations: dict[int, LinkKey] = {}
    seen_links: set[LinkKey] = set()
    with observation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _required_columns(
            reader.fieldnames,
            {"observation_id", "route", "bound", "from_seq", "to_seq"},
            observation_path,
        )
        for row_number, row in enumerate(reader, start=2):
            context = f"{observation_path}:{row_number}"
            observation_id = _integer(row["observation_id"], field=f"{context}.observation_id")
            if observation_id in observations:
                raise ValueError(f"Duplicate observation_id {observation_id} in {observation_path}")
            key = _link_key(row, context=context)
            if key in seen_links:
                raise ValueError(f"Duplicate link key {key!r} in {observation_path}")
            observations[observation_id] = key
            seen_links.add(key)
    if not observations:
        raise ValueError(f"{observation_path} contains no observations")

    segments_by_observation: dict[int, tuple[int, ...]] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _required_columns(reader.fieldnames, {"observation_id", "edge_ids"}, mapping_path)
        for row_number, row in enumerate(reader, start=2):
            context = f"{mapping_path}:{row_number}"
            observation_id = _integer(row["observation_id"], field=f"{context}.observation_id")
            if observation_id in segments_by_observation:
                raise ValueError(f"Duplicate observation_id {observation_id} in {mapping_path}")
            if observation_id not in observations:
                raise ValueError(
                    f"{mapping_path} references unknown observation_id {observation_id}"
                )
            segments_by_observation[observation_id] = _edge_ids(
                row["edge_ids"], context=context
            )

    missing_ids = sorted(set(observations) - set(segments_by_observation))
    if missing_ids:
        raise ValueError(f"{mapping_path} has no row for observation_id values {missing_ids}")

    result = {
        observations[observation_id]: segments_by_observation[observation_id]
        for observation_id in sorted(observations)
    }
    return result, hash_link_to_irn_mapping(result)


def select_irn_window_files(
    window_dir: str | Path,
    *,
    observation_date: str,
    window_start: str,
    window_end: str,
    timezone: str = "Asia/Hong_Kong",
) -> tuple[Path, ...]:
    """Select IRN files in the declared local half-open one-hour window."""

    directory = _path(window_dir, kind="directory")
    xml_paths = sorted(directory.glob("irnAvgSpeed-all-*.xml"), key=lambda path: path.name)
    if not xml_paths:
        raise FileNotFoundError(f"No irnAvgSpeed-all-*.xml files in {directory}")
    zone = ZoneInfo(timezone)
    try:
        start = datetime.fromisoformat(f"{observation_date}T{window_start}").replace(tzinfo=zone)
        end = datetime.fromisoformat(f"{observation_date}T{window_end}").replace(tzinfo=zone)
    except ValueError as exc:
        raise ValueError("Invalid IRN observation date or time-window boundary") from exc
    if end <= start:
        raise ValueError("IRN window_end must be later than window_start")

    selected: list[Path] = []
    for xml_path in xml_paths:
        match = _IRN_FILENAME.fullmatch(xml_path.name)
        if match is None:
            raise ValueError(f"Malformed IRN timestamp filename: {xml_path.name}")
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Malformed IRN XML: {xml_path}") from exc
        source_date = root.findtext("date")
        source_time = root.findtext("time")
        if not source_date or not source_time:
            raise ValueError(f"IRN XML is missing source date/time metadata: {xml_path}")
        try:
            timestamp = datetime.fromisoformat(f"{source_date.strip()}T{source_time.strip()}").replace(
                tzinfo=zone
            )
        except ValueError as exc:
            raise ValueError(f"Invalid IRN source date/time metadata: {xml_path}") from exc
        if start <= timestamp < end:
            selected.append(xml_path)
    if not selected:
        raise FileNotFoundError(
            f"No IRN XML files in [{start.isoformat()}, {end.isoformat()}) under {directory}"
        )
    return tuple(selected)


def load_irn_window_speeds(
    window_dir: str | Path,
    *,
    observation_date: str,
    window_start: str,
    window_end: str,
    timezone: str = "Asia/Hong_Kong",
) -> dict[int, float]:
    """Parse valid IRN records and return medians for one frozen window."""

    xml_paths = select_irn_window_files(
        window_dir,
        observation_date=observation_date,
        window_start=window_start,
        window_end=window_end,
        timezone=timezone,
    )

    samples: defaultdict[int, list[float]] = defaultdict(list)
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Malformed IRN XML: {xml_path}") from exc
        for index, segment in enumerate(root.findall(".//segment"), start=1):
            valid_text = segment.findtext("valid")
            if valid_text is None or valid_text.strip().upper() != "Y":
                continue
            context = f"{xml_path}:segment[{index}]"
            segment_text = segment.findtext("segment_id")
            speed_text = segment.findtext("speed")
            if segment_text is None or speed_text is None:
                raise ValueError(f"{context} is missing segment_id or speed")
            segment_id = _integer(segment_text, field=f"{context}.segment_id")
            if segment_id < 0:
                raise ValueError(f"{context}.segment_id must be non-negative")
            samples[segment_id].append(
                _finite_speed(speed_text, field=f"{context}.speed")
            )

    return {
        segment_id: float(statistics.median(samples[segment_id]))
        for segment_id in sorted(samples)
    }


def _records(value: object) -> Iterable[Mapping[str, object]]:
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict(orient="records")  # type: ignore[call-arg]
        except TypeError:
            converted = None
        if isinstance(converted, list):
            value = converted
    if isinstance(value, Mapping):
        raise TypeError("flagged_records must be an iterable of row mappings, not one mapping")
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("flagged_records must be an iterable of row mappings") from exc
    for index, record in enumerate(iterator, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"flagged_records[{index}] is not a mapping")
        yield record


def compute_irn_contradiction(
    flagged_records: object,
    link_to_segments: Mapping[LinkKey, Sequence[int]],
    segment_median_speeds: Mapping[int, float],
    *,
    speed_threshold_kmh: float = 5.0,
) -> dict[str, int]:
    """Count contradictions among IRN-matched flagged link-hour records.

    A contradiction requires ``D2D speed_median < threshold`` and the median
    across all available IRN segment medians for that link to be at least the
    same threshold.  A flagged record with no available mapped IRN segment is
    counted only in ``unmatched_flagged``.
    """

    threshold = _finite_speed(speed_threshold_kmh, field="speed_threshold_kmh")
    normalised_mapping: dict[LinkKey, tuple[int, ...]] = {}
    for raw_key, raw_segments in link_to_segments.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 4:
            raise ValueError(f"Invalid link key: {raw_key!r}")
        key = _link_key(
            {
                "route": raw_key[0],
                "bound": raw_key[1],
                "from_seq": raw_key[2],
                "to_seq": raw_key[3],
            },
            context="mapping key",
        )
        if key in normalised_mapping:
            raise ValueError(f"Duplicate normalised link key: {key!r}")
        segments = tuple(sorted({_integer(value, field="segment_id") for value in raw_segments}))
        if any(value < 0 for value in segments):
            raise ValueError("segment_id must be non-negative")
        normalised_mapping[key] = segments

    speeds = {
        _integer(segment_id, field="segment_id"): _finite_speed(
            value, field=f"segment_median_speeds[{segment_id!r}]"
        )
        for segment_id, value in segment_median_speeds.items()
    }

    numerator = 0
    denominator = 0
    unmatched = 0
    seen_links: set[LinkKey] = set()
    for index, record in enumerate(_records(flagged_records), start=1):
        context = f"flagged_records[{index}]"
        if "speed_median" not in record:
            raise ValueError(f"{context} is missing speed_median")
        key = _link_key(record, context=context)
        if key in seen_links:
            raise ValueError(f"Duplicate flagged link-hour key: {key!r}")
        seen_links.add(key)
        d2d_speed = _finite_speed(record["speed_median"], field=f"{context}.speed_median")

        irn_values = [
            speeds[segment_id]
            for segment_id in normalised_mapping.get(key, ())
            if segment_id in speeds
        ]
        if not irn_values:
            unmatched += 1
            continue

        denominator += 1
        irn_speed = float(statistics.median(irn_values))
        if d2d_speed < threshold and irn_speed >= threshold:
            numerator += 1

    return {
        "numerator": numerator,
        "denominator": denominator,
        "unmatched_flagged": unmatched,
    }


# Short aliases for reporting code that treats these as load/parse operations.
load_link_to_irn_mapping = build_link_to_irn_mapping
parse_irn_window = load_irn_window_speeds


__all__ = [
    "LinkKey",
    "build_link_to_irn_mapping",
    "compute_irn_contradiction",
    "hash_link_to_irn_mapping",
    "load_irn_window_speeds",
    "load_link_to_irn_mapping",
    "parse_irn_window",
    "select_irn_window_files",
]
