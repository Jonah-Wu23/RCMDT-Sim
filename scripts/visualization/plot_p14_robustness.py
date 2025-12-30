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

def load_sim_speeds(xml_path):
    print(f"Loading Sim: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        df_stops = []
        for stop in tree.getroot().findall('.//stopinfo'):
             df_stops.append({
                'vehicle_id': stop.get('id'),
                'arrival': float(stop.get('started', 0)),
                'departure': float(stop.get('ended', 0)),
                'stop_id': stop.get('busStop')
            })
        if not df_stops:
            return np.array([])
        
        df = pd.DataFrame(df_stops)
        results = []
        for veh_id, veh_data in df.groupby('vehicle_id'):
            veh_data = veh_data.sort_values('arrival').reset_index(drop=True)
            for i in range(len(veh_data) - 1):
                departure = veh_data.loc[i, 'departure']
                arrival = veh_data.loc[i+1, 'arrival']
                tt = arrival - departure
                if tt > 0:
                    speed_kmh = (500 / 1000) / (tt / 3600)  # Approx 500m avg dist
                    # Better: if we had exact dist. For now use P14 assumptions implicitly or just align D2D TT if possible.
                    # Wait, evaluate_robustness used exact dist if available or avg.
                    # Since we don't have link dist here easily, we rely on the fact that evaluate_robustness printed speeds.
                    # Actually, evaluate_robustness logic:
                    # speed = (link_dist / 1000) / (tt / 3600).
                    # We can visualize Travel Times distribution directly?
                    # The user asked for "Speed CDF".
                    # Real data has 'speed_median'.
                    # For Sim, we need distance.
                    # QUICK FIX: Let's assume average link distance ~500m if we can't map. 
                    # OR: Use the 'evaluate_robustness.py' output speeds if we could dump them.
                    # But better: Load 'link_stats_Rule_M.csv' which has real speeds.
                    # For Sim, we really need the distances for accurate speed.
                    # Hack: Use the distribution of Real Distances to sample? No.
                    # Correct way: match Sim stops to Link definitions.
                    # Complex.
                    # OFFSET: In evaluate_robustness, map Sim (stop A->B) to Real (stop A->B).
                    pass
        
        # ACTUALLY: Let's use the speeds calculated in 'evaluate_robustness.py'.
        # I will modify evaluate_robustness to perform the PLOT, or duplicate the logic here.
        # Duplicating logic is safer.
        pass
    except Exception as e:
        print(e)
        return np.array([])
    return np.array([]) 

# Optimized: Since I cannot easily map Sim Stop->Stop to Distance without the Distance Map file,
# I will use a simplified approach:
# Load the 'link_stats_offpeak.csv' (Real) and trust its speeds.
# For Sim, I need the distances. 
# I will LOAD 'data/processed/kmb_route_stop_dist.csv' to get distances.

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

def plot_ghost_audit(real_file_raw, output_path):
    print(f"\nPlotting Ghost Audit: {real_file_raw}")
    df = pd.read_csv(real_file_raw)
    
    # Define Ghost Jams / Clean Split
    # Using T* = 325s from user finding
    T_star = 325
    clean_mask = (df['tt_median'] <= T_star) | (df['speed_median'] >= 5.0)
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
    
    ax1.axvline(325, color='black', linestyle='--', linewidth=1, label='Critical T*=325s')
    ax1.set_xlabel('Travel Time (s)')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Measurement Audit (Raw vs Clean)', fontweight='bold')
    ax1.legend()
    
    # 2. Speed vs Time Scatter
    ax2 = axes[1]
    # Plot Raw as background (Gray)
    ax2.scatter(df['tt_median'], df['speed_median'], alpha=0.3, s=10, c='gray', label='Ghost / Raw')
    
    # Plot Clean as foreground (Blue)
    ax2.scatter(clean_tt, clean_speed, alpha=0.6, s=10, c='#1f77b4', label='Clean (Op-L2-v1.1)')
    
    ax2.axhline(5, color='gray', linestyle=':', label='5 km/h limit')
    ax2.axvline(325, color='black', linestyle='--', label='T*=325s')
    
    ax2.set_xlabel('Travel Time (s)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('(b) Filter Logic (Rule C)', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_robustness_cdf(real_csv_raw, real_csv_clean, sim_xml, dist_file, output_path):
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
    
    # Filter for Rule C (T > 325, Speed < 5, Dist < 1500)
    # Re-implement filtering
    cond_ghost = (df_raw['tt_median'] > 325) & (df_raw['speed_median'] < 5.0) & (df_raw['dist_m'] < 1500)
    clean_speeds = df_raw.loc[~cond_ghost, 'speed_median'].dropna().values

    
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
    
    text = (
        f"KS(raw)={ks_raw:.2f}\n"
        f"KS(clean)={ks_clean:.4f}\n"
        "Worst 15-min window: KS=0.3337"
    )
    # Move text to bottom right (above legend) to avoid Raw curve overlap in top-left
    # User requested shift to upper-right relative to previous (0.40, 0.20)
    ax.text(0.50, 0.35, text, transform=ax.transAxes, fontsize=7, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data2/processed/link_stats_offpeak.csv")
    parser.add_argument("--clean", default="data2/processed/link_stats_Rule_M_clean.csv")
    parser.add_argument("--sim", default="sumo/output/offpeak_v2_offpeak_stopinfo.xml")
    parser.add_argument("--dist", default="data/processed/kmb_route_stop_dist.csv")
    parser.add_argument("--out-audit", default="plots/P14_ghost_audit.png")
    parser.add_argument("--out-cdf", default="plots/P14_robustness_cdf.png")
    
    args = parser.parse_args()
    
    if os.path.exists(args.raw):
        plot_ghost_audit(args.raw, args.out_audit)
    
    if os.path.exists(args.raw) and os.path.exists(args.sim) and os.path.exists(args.dist):
        plot_robustness_cdf(args.raw, args.clean, args.sim, args.dist, args.out_cdf)
