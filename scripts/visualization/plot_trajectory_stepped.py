import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np

# Add scripts directory to path to import common_data
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts'))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds, get_dist_map

def main():
    parser = argparse.ArgumentParser(description="Plot Stepped Trajectory (Distance vs Time) to show Dwell Mechanics")
    parser.add_argument('--real_links', required=True, help="Path to real link stats CSV")
    parser.add_argument('--real_dist', required=True, help="Path to route stop distance CSV")
    parser.add_argument('--sim', required=True, help="Path to simulation stopinfo XML")
    parser.add_argument('--out', required=True, help="Output image path")
    parser.add_argument('--route', default='68X', help="Route ID (e.g. 68X, 960)")
    parser.add_argument('--bound', default=None, help="Direction (inbound/outbound), optional auto-detect")
    parser.add_argument('--title_suffix', default="", help="Suffix for plot title")
    args = parser.parse_args()

    # IEEE Style Configuration
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7

    # --- 1. Load Data ---
    print(f"Loading stop distances from {args.real_dist}...")
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]
    
    print(f"Loading simulation data from {args.sim}...")
    sim_raw = load_sim_data(args.sim)
    
    # Auto-detect bound if not provided
    if args.bound:
        target_bound = args.bound
    else:
        # Check which bound (inbound/outbound) contains most of the stops visited by sim
        sim_stops_set = set(sim_raw['bus_stop_id'].unique())
        bound_counts = dist_df.groupby('bound').apply(lambda x: len(set(x['stop_id']) & sim_stops_set))
        target_bound = bound_counts.idxmax() if not bound_counts.empty else 'inbound'
        print(f"Detected Bound for {args.route}: {target_bound}")
    
    # Filter Dist DF
    dist_df = dist_df[dist_df['bound'] == target_bound]
    dist_map = get_dist_map(dist_df, 'seq')
    stop_to_seq = dict(zip(dist_df['stop_id'], dist_df['seq']))
    
    # Identify Sim Bounds (Start Seq / Max Dist)
    sim_stops = sim_raw['bus_stop_id'].unique()
    sim_seqs = [stop_to_seq[s] for s in sim_stops if s in stop_to_seq]
    if not sim_seqs:
        print(f"No valid stops found for {args.route} {target_bound}")
        return
        
    start_seq = min(sim_seqs)
    start_dist_abs = dist_map.get(start_seq, 0)
    print(f"Sim Start Seq: {start_seq}, Abs Dist: {start_dist_abs}m")

    # Build Sim Trajectory DataFrame
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    sim_traj['dist_rel'] = sim_traj['cum_dist_m'] - start_dist_abs
    
    # Calculate Max Sim Dist (Relative)
    max_sim_dist = sim_traj['dist_rel'].max()
    print(f"Sim Max Relative Dist: {max_sim_dist:.1f}m")

    # --- 2. Select Representative Sim Vehicle (Median Total Time) ---
    # Calculate total duration for each vehicle *within the cropped area*
    # Group by vehicle and ensure each vehicle starts at t=0 at start_seq
    sim_trips = []
    
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        # Find entry point
        entry_rows = group[group['seq'] == start_seq]
        if entry_rows.empty: continue
        
        t0 = entry_rows.iloc[0]['arrival_time']
        
        # Build trip points
        trip_points = []
        for _, row in group.iterrows():
            d_rel = row['dist_rel']
            if d_rel < -50 or d_rel > max_sim_dist + 50: continue # Loose crop
            
            trip_points.append({
                'vehicle_id': vid,
                'time_arr': row['arrival_time'] - t0,
                'time_dep': row['departure_time'] - t0,
                'dist': d_rel
            })
        
        if len(trip_points) > 2:
            duration = trip_points[-1]['time_dep']
            sim_trips.append({'vid': vid, 'duration': duration, 'points': trip_points})

    if not sim_trips:
        print("No valid complete sim trips found.")
        return
        
    df_trips = pd.DataFrame(sim_trips)
    median_dur = df_trips['duration'].median()
    # Select closest to median
    best_idx = (df_trips['duration'] - median_dur).abs().idxmin()
    rep_trip = sim_trips[best_idx]
    
    print(f"Representative Vehicle: {rep_trip['vid']} (Duration {rep_trip['duration']:.1f}s)")
    
    # Construct Stepped Line Points for Sim
    sim_x = [] # Time
    sim_y = [] # Distance
    
    for pt in rep_trip['points']:
        sim_x.append(pt['time_arr'])
        sim_y.append(pt['dist'])
        sim_x.append(pt['time_dep'])
        sim_y.append(pt['dist'])
        
    # --- 3. Construct Real World Mean Trajectory ---
    print(f"Loading real link speeds from {args.real_links}...")
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[(real_links['route'] == args.route) & (real_links['bound'] == target_bound)]
    
    # Calculate Mean Travel Time per Link
    avg_link_times = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().to_dict()
    
    real_x = [0]
    real_y = [0] # Start at Relative 0
    current_t = 0
    
    # Find sequence of stops from dist_df starting from start_seq
    sorted_seqs = sorted([s for s in dist_df['seq'] if s >= start_seq])
    
    for i in range(len(sorted_seqs) - 1):
        u = sorted_seqs[i]
        v = sorted_seqs[i+1]
        
        d_u = dist_map.get(u, 0)
        d_v = dist_map.get(v, 0)
        
        d_v_rel = d_v - start_dist_abs
        
        # STOP CONDITION: Sync with Sim Max Dist
        if d_v_rel > max_sim_dist + 100:
            break
            
        # Link Travel Time
        dt = avg_link_times.get((u, v))
        if dt is None or np.isnan(dt):
            # Fallback estimation if link data missing (e.g. 30 km/h)
            dt = (d_v - d_u) / (30 / 3.6) 
            
        current_t += dt
        
        real_x.append(current_t)
        real_y.append(d_v_rel)
        
    # --- 4. Plotting ---
    plt.figure(figsize=(3.5, 3.0)) # IEEE Small column width
    
    # Plot Real (Dashed, "Continuous")
    plt.plot(real_x, real_y, label='Real World (Avg)', color='#1f77b4', linestyle='--', linewidth=2.0, alpha=0.9)
    
    # Plot Sim (Solid, Stepped)
    plt.plot(sim_x, sim_y, label='Simulation (Rep.)', color='#ff7f0e', linewidth=2.5, alpha=0.9) # Back to Orange, thick
    
    # Style
    plt.xlabel("Cumulative Travel Time (s)", fontsize=8)
    plt.ylabel("Cumulative Distance (m)", fontsize=8)
    plt.title(f"{args.route} Trajectory (Stepped){args.title_suffix}")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=7, loc='best')
    
    # Formatting
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Stepped Trajectory plot to {args.out}")

if __name__ == "__main__":
    main()
