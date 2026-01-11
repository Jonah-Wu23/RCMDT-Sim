#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_p14_robustness.py
======================
Visualize P14 Robustness Results:
1. Ghost Jam Audit: Histogram of travel times (Raw vs Clean).
2. Robustness Validation: CDF comparison of Real (Raw/Clean) vs Sim.

Style: IEEE SMC (Times New Roman, 8pt)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# IEEE Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

def load_sim_speeds_accurate(xml_path, dist_file):
    print(f"Loading Sim Speeds with Distances from {dist_file}")
    df_dist = pd.read_csv(dist_file)
    # create dist map: (route, from_stop, to_stop) -> dist
    # Dist file cols: route, bound, service_type, seq, stop_id, dist_m, link_dist_m
    # Map: (stop_id_from, stop_id_to) -> link_dist_m? 
    # Stop IDs in Sim are 'busStop' attr. Real dist file has 'stop_id'.
    # Link dist is between seq i and i+1.
    
    # Map: (stop_id_i, stop_id_next) -> link_dist
    dist_map = {}
    for r, group in df_dist.groupby(['route', 'bound', 'service_type']):
        group = group.sort_values('seq')
        stops = group['stop_id'].tolist()
        dists = group['link_dist_m'].tolist()
        for i in range(len(stops)-1):
            s1, s2 = stops[i], stops[i+1]
            d = dists[i+1] # link_dist is usually on the 'to' node or 'from'? 
            # In clean_kmb_links, link_dist comes from the row of s2.
            if pd.notna(d) and d > 0:
                dist_map[(str(s1), str(s2))] = d
    
    tree = ET.parse(xml_path)
    speeds = []
    
    # Group by vehicle
    veh_stops = {} # veh_id -> list of (stop_id, departure, arrival)
    for stop in tree.getroot().findall('.//stopinfo'):
        vid = stop.get('id')
        sid = stop.get('busStop')
        dep = float(stop.get('ended', 0))
        arr = float(stop.get('started', 0))
        if vid not in veh_stops: veh_stops[vid] = []
        veh_stops[vid].append({'sid':sid, 'dep':dep, 'arr':arr})
    
    for vid, stops in veh_stops.items():
        stops.sort(key=lambda x: x['arr'])
        for i in range(len(stops)-1):
            s1 = stops[i]['sid']
            s2 = stops[i+1]['sid']
            dep = stops[i]['dep']
            arr = stops[i+1]['arr']
            tt = arr - dep
            
            # Lookup distance
            dist = dist_map.get((s1, s2))
            if dist and tt > 0:
                speed = (dist/1000) / (tt/3600)
                if 0.1 < speed < 120:
                    speeds.append(speed)
                    
    return np.array(speeds)

def apply_rule_c(df_raw: pd.DataFrame, t_critical: float, speed_kmh: float, max_dist_m: float) -> tuple[pd.Series, pd.Series]:
    missing = [c for c in ("tt_median", "speed_median", "dist_m") if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in real stats CSV: {missing}")
    cond_ghost = (df_raw["tt_median"] > t_critical) & (df_raw["speed_median"] < speed_kmh) & (df_raw["dist_m"] < max_dist_m)
    return cond_ghost, ~cond_ghost

def _is_fixture_mode(raw_path: str, fixture_flag: bool) -> bool:
    if fixture_flag:
        return True
    p = str(raw_path).replace("\\", "/").lower()
    return "/tests/fixtures/" in p or p.endswith("/tests/fixtures") or "/fixtures/" in p

def _add_fixture_watermark(ax: plt.Axes) -> None:
    ax.text(
        0.5,
        0.5,
        "FIXTURE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=28,
        color="black",
        alpha=0.10,
        rotation=25,
        zorder=0,
    )

def plot_ghost_audit(real_file_raw, output_path, t_critical: float, speed_kmh: float, max_dist_m: float, fixture: bool = False):
    print(f"\nPlotting Ghost Audit: {real_file_raw}")
    df = pd.read_csv(real_file_raw)
    
    _, clean_mask = apply_rule_c(df, t_critical, speed_kmh, max_dist_m)
    clean_tt = df.loc[clean_mask, 'tt_median'].dropna().values
    clean_speed = df.loc[clean_mask, 'speed_median'].dropna().values
    
    raw_tt = df['tt_median'].dropna().values
    
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    
    # 1. Travel Time Hist
    ax1 = axes[0]
    # Plot Raw (Background)
    counts, bins, _ = ax1.hist(raw_tt, bins=30, color='gray', alpha=0.3, label='Raw (All)', density=True)
    # Plot Clean (Overlay)
    ax1.hist(clean_tt, bins=bins, color='#1f77b4', alpha=0.7, label='Clean (Valid)', density=True)
    
    ax1.axvline(t_critical, color='black', linestyle='--', linewidth=1, label=f'Critical T*={t_critical:g}s')
    ax1.set_xlabel('Travel Time (s)')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Measurement Audit (Raw vs Clean)', fontweight='bold')
    ax1.legend()
    if fixture:
        _add_fixture_watermark(ax1)
    
    # 2. Speed vs Time Scatter
    ax2 = axes[1]
    # Plot Raw as background (Gray)
    ax2.scatter(df['tt_median'], df['speed_median'], alpha=0.3, s=10, c='gray', label='Ghost / Raw')
    
    # Plot Clean as foreground (Blue)
    ax2.scatter(clean_tt, clean_speed, alpha=0.6, s=10, c='#1f77b4', label='Clean (Op-L2-v1.1)')
    
    ax2.axhline(speed_kmh, color='gray', linestyle=':', label=f'{speed_kmh:g} km/h limit')
    ax2.axvline(t_critical, color='black', linestyle='--', label=f'T*={t_critical:g}s')
    
    ax2.set_xlabel('Travel Time (s)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('(b) Filter Logic (Rule C)', fontweight='bold')
    ax2.legend()
    if fixture:
        _add_fixture_watermark(ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_robustness_cdf(
    real_csv_raw,
    sim_xml,
    dist_file,
    output_path,
    t_critical: float,
    speed_kmh: float,
    max_dist_m: float,
    worst_window_ks: float | None = None,
    fixture: bool = False,
):
    print(f"\nPlotting Robustness CDF: {output_path}")
    
    # Load Data
    df_raw = pd.read_csv(real_csv_raw)
    raw_speeds = df_raw['speed_median'].dropna().values
    
    # Rule M is passed as 'real_csv_clean', but User wants Rule C (T*=325).
    # Since I don't have 'link_stats_Rule_C.csv' pre-generated, I should filter Raw here or pass Rule C file.
    # Actually, find_critical_threshold.py generated Rule C stats in memory but didn't save?
    # No, it just printed. 
    # I will implement Rule C filtering on the fly from 'df_raw' or just use Rule M if the difference is small.
    # KS=0.28 vs 0.29. The User specifically asked for "Rule C KS=0.2977". 
    # To be precise, I should apply T > 325 filter on df_raw.
    
    cond_ghost, clean_mask = apply_rule_c(df_raw, t_critical, speed_kmh, max_dist_m)
    clean_speeds = df_raw.loc[clean_mask, 'speed_median'].dropna().values

    
    sim_speeds = load_sim_speeds_accurate(sim_xml, dist_file)
    
    fig, ax = plt.subplots(figsize=(3.5, 3)) # Single column width
    
    # Plot CDFs
    def plot_cdf(data, color, ls, label, lw=1.5):
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        ax.plot(sorted_data, y, color=color, linestyle=ls, label=label, linewidth=lw)
    
    plot_cdf(raw_speeds, 'gray', ':', 'Real (Raw Operator)', lw=1)
    plot_cdf(clean_speeds, '#1f77b4', '-', 'Real (Op-L2-v1.1 / Rule C)')
    plot_cdf(sim_speeds, '#ff7f0e', '--', 'Sim (Zero-Shot)')
    
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Cumulative probability')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right')
    
    # Annotate KS
    from scipy.stats import ks_2samp
    ks_raw, _ = ks_2samp(raw_speeds, sim_speeds)
    ks_clean, _ = ks_2samp(clean_speeds, sim_speeds)
    
    # Calculate improvement percentage
    improvement_pct = ((ks_raw - ks_clean) / ks_raw) * 100
    
    lines = [
        f"KS(raw)={ks_raw:.2f}",
        f"KS(clean)={ks_clean:.4f} (Improved {improvement_pct:.0f}%)"
    ]
    if (not fixture) and (worst_window_ks is not None):
        lines.append(f"Worst 15-min window: KS={worst_window_ks:.4f}")
    text = "\n".join(lines)
    # Adjust text position to avoid boundary overflow
    ax.text(0.45, 0.35, text, transform=ax.transAxes, fontsize=7, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    if fixture:
        _add_fixture_watermark(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data2/processed/link_stats_offpeak.csv")
    parser.add_argument("--sim", default="sumo/output/offpeak_v2_offpeak_stopinfo.xml")
    parser.add_argument("--dist", default="data/processed/kmb_route_stop_dist.csv")
    parser.add_argument("--out-audit", default="plots/P14_ghost_audit.png")
    parser.add_argument("--out-cdf", default="plots/P14_robustness_cdf.png")
    parser.add_argument("--t_critical", type=float, default=325, help="Rule C: travel time threshold T* (s)")
    parser.add_argument("--speed_kmh", type=float, default=5, help="Rule C: speed threshold v* (km/h)")
    parser.add_argument("--max_dist_m", type=float, default=1500, help="Rule C: apply only when dist_m < max_dist_m")
    parser.add_argument("--worst_window_ks", type=float, default=None, help="Optional: annotate worst 15-min window KS in CDF plot")
    parser.add_argument("--fixture", action="store_true", help="Fixture mode: watermark outputs and omit paper-only annotations")
    
    args = parser.parse_args()
    fixture_mode = _is_fixture_mode(args.raw, args.fixture)
    
    if os.path.exists(args.raw):
        plot_ghost_audit(args.raw, args.out_audit, args.t_critical, args.speed_kmh, args.max_dist_m, fixture=fixture_mode)
    
    if os.path.exists(args.raw) and os.path.exists(args.sim) and os.path.exists(args.dist):
        plot_robustness_cdf(
            args.raw,
            args.sim,
            args.dist,
            args.out_cdf,
            args.t_critical,
            args.speed_kmh,
            args.max_dist_m,
            worst_window_ks=args.worst_window_ks,
            fixture=fixture_mode,
        )
