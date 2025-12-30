
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds, get_dist_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_links', required=True)
    parser.add_argument('--real_dist', required=True)
    parser.add_argument('--sim', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--route', default='68X')
    parser.add_argument('--hour', type=int, default=None, help='Filter real data by hour (0-23)')
    args = parser.parse_args()

    # Unified Color Scheme
    files_colors = {'Real World': '#1f77b4', 'Simulation': '#ff7f0e'}
    # sns.set_theme(style="whitegrid", context="talk") # Disabled for IEEE style
    
    # IEEE Style Configuration (8pt fonts with small figure size)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7

    # 1. Load Data
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]
    # 使用 get_dist_map 处理 cum_dist_m_dir 的 NaN
    dist_map = get_dist_map(dist_df, 'seq')

    sim_raw = load_sim_data(args.sim)
    
    # 1. Determine common start sequence and BOUND from Simulation
    sim_raw_route = sim_raw[sim_raw['vehicle_id'].str.contains(args.route)]
    if sim_raw_route.empty:
        print(f"No simulation data found for route {args.route}")
        return

    # Auto-detect direction (Bound)
    # Check which bound (inbound/outbound) contains most of the stops visited by sim
    sim_stops_set = set(sim_raw_route['bus_stop_id'].unique())
    bound_counts = dist_df.groupby('bound').apply(lambda x: len(set(x['stop_id']) & sim_stops_set))
    target_bound = bound_counts.idxmax() if not bound_counts.empty else 'inbound'
    print(f"Detected Bound for {args.route}: {target_bound}")
    
    # Filter dist_df strictly to this bound
    dist_df = dist_df[dist_df['bound'] == target_bound]
    dist_map = get_dist_map(dist_df, 'seq')
    stop_to_seq = dist_df.set_index('stop_id')['seq'].to_dict()

    sim_seqs = [stop_to_seq[s] for s in sim_stops_set if s in stop_to_seq]
    if not sim_seqs:
        print(f"No stops found for {args.route} {target_bound} in simulation mapping.")
        return
        
    start_seq = min(sim_seqs)
    start_dist_abs = dist_map.get(start_seq, 0)
    
    print(f"Alignment: {args.route} {target_bound} Sequence {start_seq} (Abs: {start_dist_abs:.1f}m)")

    # 2. Build Simulation Curve (Vehicle-level -> Average)
    sim_traj = build_sim_trajectory(sim_raw_route, dist_df)
    sim_traj['dist_rel'] = sim_traj['cum_dist_m'] - start_dist_abs
    
    sim_trips = []
    # Group by vehicle and ensure each vehicle starts at t=0 at start_seq
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        entry_row = group[group['seq'] == start_seq]
        if entry_row.empty:
            continue
            
        t0 = entry_row.iloc[0]['arrival_time']
        for _, row in group.iterrows():
            d_rel = row['dist_rel']
            if d_rel < 0 or d_rel > 5000:
                continue
            sim_trips.append({
                'dist': d_rel,
                'cum_time': row['arrival_time'] - t0,
                'Source': 'Simulation'
            })
            
    df_sim_plot = pd.DataFrame(sim_trips)
    if df_sim_plot.empty:
        print(f"No simulation vehicles found stopping at Seq {start_seq}.")
        return

    max_sim_dist = df_sim_plot['dist'].max()
    print(f"Simulation Max Distance: {max_sim_dist:.1f}m")

    # 3. Build Real World Curve (Cumulative Link Times)
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[(real_links['route'] == args.route) & (real_links['bound'] == target_bound)]
    # Filter to specific hour if applicable
    if 'arrival_ts' in real_links.columns:
        if real_links['arrival_ts'].dt.tz is not None:
             real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')
        
        if args.hour is not None:
            real_links = real_links[real_links['arrival_ts'].dt.hour == args.hour]

    avg_link_times = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().to_dict()
    
    # Fallback speed for missing links (approx 15 km/h -> 4.1 m/s)
    FALLBACK_SPEED_MS = 15 / 3.6
    
    real_curve = [{'dist': 0, 'cum_time': 0, 'Source': 'Real World'}]
    sorted_seqs = sorted([s for s in dist_map.keys() if s >= start_seq])
    
    curr_t = 0
    for i in range(len(sorted_seqs)-1):
        u = sorted_seqs[i]
        v = sorted_seqs[i+1]
        
        d_u_abs = dist_map[u]
        d_v_abs = dist_map[v]
        d_v_rel = d_v_abs - start_dist_abs
        
        if d_v_rel > max_sim_dist + 100: # Sync end
            break
        if d_v_rel > 5000: # Global cap
            break
            
        # Get Time
        dt = avg_link_times.get((u, v))
        if dt is None:
            # Avoid flat lines (teleportation) by estimating time from distance
            dist_link = d_v_abs - d_u_abs
            dt = dist_link / FALLBACK_SPEED_MS
            # print(f"Warning: Link {u}->{v} missing real data. Using fallback speed.")
            
        curr_t += dt
        real_curve.append({'dist': d_v_rel, 'cum_time': curr_t, 'Source': 'Real World'})
        
    df_real_plot = pd.DataFrame(real_curve)

    # 4. Final Plotting
    full_df = pd.concat([df_real_plot, df_sim_plot])
    
    plt.figure(figsize=(3.5, 2.5))  # IEEE single column width
    
    # Use markers to clearly see "Stations"
    sns.lineplot(data=full_df, x="dist", y="cum_time", hue="Source", style="Source",
                 markers=True, dashes=False, palette=files_colors, markersize=7)
    
    plt.xlabel(f"Distance from Baseline Stop (Seq {start_seq}) (m)")
    plt.ylabel("Cumulative Travel Time (s)")
    
    # Give the (0,0) origin some breathing room
    plt.xlim(left=-200) 
    plt.ylim(bottom=-100)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Aligned Trajectory Plot to {args.out}")

if __name__ == "__main__":
    main()
