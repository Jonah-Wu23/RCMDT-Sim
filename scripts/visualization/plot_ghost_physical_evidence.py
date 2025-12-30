import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add scripts directory to path to import common_data
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts'))
from common_data import load_route_stop_dist, load_real_link_speeds, get_dist_map


def main():
    parser = argparse.ArgumentParser(description="Plot Ghost vs Clean spatial evidence (travel time vs corridor position)")
    parser.add_argument('--real_links', required=True, help="Path to link_times_offpeak.csv")
    parser.add_argument('--real_dist', required=True, help="Path to route stop distance CSV")
    parser.add_argument('--out', required=True, help="Output image path")
    parser.add_argument('--route', default='68X', help="Route ID")
    parser.add_argument('--bound', default=None, help="Direction (inbound/outbound), optional auto-detect")
    parser.add_argument('--t_critical', type=float, default=325, help="Travel time threshold (s)")
    parser.add_argument('--speed_kmh', type=float, default=5, help="Speed threshold (km/h)")
    parser.add_argument('--corridor_m', type=float, default=5000, help="Spatial corridor length in meters")
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

    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[real_links['route'] == args.route]

    if args.bound:
        target_bound = args.bound
    else:
        if real_links.empty:
            target_bound = dist_df['bound'].mode().iloc[0]
        else:
            target_bound = real_links['bound'].mode().iloc[0]

    dist_df = dist_df[dist_df['bound'] == target_bound]
    real_links = real_links[real_links['bound'] == target_bound]

    dist_map = get_dist_map(dist_df, 'seq')
    start_seq = dist_df['seq'].min()
    start_dist_abs = dist_map.get(start_seq, 0)

    real_links['speed_kmh'] = (real_links['dist_m'] / real_links['travel_time_s']) * 3.6
    real_links['is_ghost'] = (real_links['travel_time_s'] > args.t_critical) & (real_links['speed_kmh'] < args.speed_kmh)

    real_links['pos_m'] = real_links['from_seq'].map(dist_map) - start_dist_abs
    real_links = real_links.dropna(subset=['pos_m'])
    real_links = real_links[(real_links['pos_m'] >= 0) & (real_links['pos_m'] <= args.corridor_m)]

    ghost = real_links[real_links['is_ghost']]
    clean = real_links[~real_links['is_ghost']]

    plt.figure(figsize=(3.5, 3.0))
    plt.scatter(clean['travel_time_s'], clean['speed_kmh'], s=10, c='#1f77b4', alpha=0.2, label='Clean')
    plt.scatter(ghost['travel_time_s'], ghost['speed_kmh'], s=14, c='#ff7f0e', alpha=0.6, label='Ghost (Rule C)')
    plt.axvline(args.t_critical, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    plt.axhline(args.speed_kmh, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)

    plt.xlabel("Travel Time (s)")
    plt.ylabel("Effective Speed (km/h)")
    plt.title(f"{args.route} Ghost vs Clean (Kinematic Evidence)")
    max_speed = max(real_links['speed_kmh'].max(), args.speed_kmh)
    plt.ylim(0, max_speed * 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.text(0.98, 0.72, f"Nclean={len(clean)}, Nghost={len(ghost)}",
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved ghost evidence plot to {args.out}")


if __name__ == "__main__":
    main()
