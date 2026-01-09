#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_link_stats_peak.py
========================
Aggregate link-level stats from link_speeds.csv to link_stats.csv.

This mirrors the off-peak stats schema:
route, bound, from_seq, to_seq,
tt_mean, tt_median, tt_std, sample_count, tt_p90,
speed_mean, speed_median, speed_std, speed_p90, dist_m
"""

import argparse
import pandas as pd


def build_link_stats(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    required = {"route", "bound", "from_seq", "to_seq", "travel_time_s", "speed_kmh", "dist_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

    stats = (
        df.groupby(["route", "bound", "from_seq", "to_seq"])
        .agg(
            tt_mean=("travel_time_s", "mean"),
            tt_median=("travel_time_s", "median"),
            tt_std=("travel_time_s", "std"),
            sample_count=("travel_time_s", "count"),
            tt_p90=("travel_time_s", lambda x: x.quantile(0.9)),
            speed_mean=("speed_kmh", "mean"),
            speed_median=("speed_kmh", "median"),
            speed_std=("speed_kmh", "std"),
            speed_p90=("speed_kmh", lambda x: x.quantile(0.9)),
            dist_m=("dist_m", "median"),
        )
        .reset_index()
    )

    stats.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build link_stats.csv from link_speeds.csv")
    parser.add_argument("--input", required=True, help="Input link_speeds CSV")
    parser.add_argument("--output", required=True, help="Output link_stats CSV")
    args = parser.parse_args()

    build_link_stats(args.input, args.output)


if __name__ == "__main__":
    main()
