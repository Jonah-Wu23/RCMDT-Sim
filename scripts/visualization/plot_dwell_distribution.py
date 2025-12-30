import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add scripts directory to path to import common_data
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts'))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds, get_dist_map


def _cdf(values):
    if len(values) == 0:
        return np.array([]), np.array([])
    vals = np.sort(values)
    y = np.arange(1, len(vals) + 1) / len(vals)
    return vals, y


def main():
    parser = argparse.ArgumentParser(description="Plot dwell-time distribution (kinematic definition) for Real vs Sim")
    parser.add_argument('--real_links', required=True, help="Path to link_times_offpeak.csv")
    parser.add_argument('--real_dist', required=True, help="Path to route stop distance CSV")
    parser.add_argument('--sim', required=True, help="Path to simulation stopinfo XML")
    parser.add_argument('--out', required=True, help="Output image path")
    parser.add_argument('--route', default='68X', help="Route ID")
    parser.add_argument('--bound', default=None, help="Direction (inbound/outbound), optional auto-detect")
    parser.add_argument('--corridor_m', type=float, default=5000, help="Spatial corridor length in meters")
    parser.add_argument('--v_stop_kmh', type=float, default=5, help="Speed threshold for dwell episodes (km/h)")
    parser.add_argument('--min_dwell_s', type=float, default=5, help="Minimum duration to count as dwell (s)")
    args = parser.parse_args()

    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })

    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]

    sim_raw = load_sim_data(args.sim)
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[real_links['route'] == args.route]

    if args.bound:
        target_bound = args.bound
    else:
        sim_stops_set = set(sim_raw['bus_stop_id'].unique())
        bound_counts = dist_df.groupby('bound').apply(lambda x: len(set(x['stop_id']) & sim_stops_set))
        target_bound = bound_counts.idxmax() if not bound_counts.empty else dist_df['bound'].mode().iloc[0]

    dist_df = dist_df[dist_df['bound'] == target_bound]
    real_links = real_links[real_links['bound'] == target_bound]

    dist_map = get_dist_map(dist_df, 'seq')
    start_seq = dist_df['seq'].min()
    start_dist_abs = dist_map.get(start_seq, 0)

    # --- Real dwell episodes (link-based kinematic definition) ---
    real_links['speed_kmh'] = (real_links['dist_m'] / real_links['travel_time_s']) * 3.6
    real_links['pos_m'] = real_links['from_seq'].map(dist_map) - start_dist_abs
    real_links = real_links.dropna(subset=['pos_m'])
    real_links = real_links[(real_links['pos_m'] >= 0) & (real_links['pos_m'] <= args.corridor_m)]

    dwell_real = real_links[
        (real_links['speed_kmh'] < args.v_stop_kmh) &
        (real_links['travel_time_s'] >= args.min_dwell_s)
    ]['travel_time_s'].to_numpy()

    # --- Sim dwell episodes (kinematic definition on stopinfo-derived segments) ---
    sim_df = build_sim_trajectory(sim_raw, dist_df)
    sim_df['dist_rel'] = sim_df['cum_dist_m'] - start_dist_abs

    dwell_sim = []
    for vid, group in sim_df.groupby('vehicle_id'):
        group = group.sort_values('seq')
        if group.empty:
            continue

        for i in range(len(group)):
            row = group.iloc[i]
            if row['dist_rel'] < 0 or row['dist_rel'] > args.corridor_m:
                continue

            dwell_time = row['departure_time'] - row['arrival_time']
            if dwell_time >= args.min_dwell_s:
                dwell_sim.append(dwell_time)

            if i == len(group) - 1:
                continue

            next_row = group.iloc[i + 1]
            d_u = row['dist_rel']
            d_v = next_row['dist_rel']
            if d_u < 0 or d_u > args.corridor_m:
                continue

            t_move = next_row['arrival_time'] - row['departure_time']
            if t_move <= 0:
                continue

            dist_move = max(d_v - d_u, 0)
            speed_kmh = (dist_move / t_move) * 3.6
            if speed_kmh < args.v_stop_kmh and t_move >= args.min_dwell_s:
                dwell_sim.append(t_move)

    dwell_sim = np.array(dwell_sim)

    # --- Plot CDF ---
    x_real, y_real = _cdf(dwell_real)
    x_sim, y_sim = _cdf(dwell_sim)

    plt.figure(figsize=(3.5, 3.0))
    if len(x_real):
        plt.plot(x_real, y_real, color='#1f77b4', linewidth=2.0, label='Real (holding proxy)')
    if len(x_sim):
        plt.plot(x_sim, y_sim, color='#ff7f0e', linewidth=2.0, label='Simulation (dwell)')

    for vals, color, label in [(dwell_real, '#1f77b4', 'Real'), (dwell_sim, '#ff7f0e', 'Sim')]:
        if len(vals) == 0:
            continue
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        plt.axvline(p50, color=color, linestyle=':', linewidth=1.0, alpha=0.6, zorder=2)
        plt.axvline(p90, color=color, linestyle='--', linewidth=1.0, alpha=0.4, zorder=2)
        plt.text(p50 - 5, 0.52, f"{label} p50", color=color, fontsize=7, rotation=90,
                 va='bottom', ha='right', zorder=1)
        if label == 'Real':
            plt.annotate(f"{label} p90", xy=(p90, 0.88), xytext=(-12, 0),
                         textcoords='offset points', color=color, fontsize=7, rotation=90,
                         va='bottom', ha='right', zorder=1)
        else:
            plt.text(p90, 0.88, f"{label} p90", color=color, fontsize=7, rotation=90,
                     va='bottom', ha='right', zorder=1)

    plt.xlabel("Dwell Time (s)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"{args.route} Holding Proxy vs Simulated Dwell")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved dwell distribution plot to {args.out}")


if __name__ == "__main__":
    main()
