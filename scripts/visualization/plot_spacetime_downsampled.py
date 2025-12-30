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
    parser = argparse.ArgumentParser(description="Plot Spacetime Diagram Comparison (High Res Sim vs Downsampled Sim vs Real)")
    parser.add_argument('--real_links', required=True)
    parser.add_argument('--real_dist', required=True)
    parser.add_argument('--sim', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--route', default='68X')
    parser.add_argument('--bound', default=None)
    parser.add_argument('--downsample_interval_s', type=int, default=30, help="Downsample interval in seconds")
    parser.add_argument('--t_critical', type=float, default=325, help="Ghost filter threshold for Real Data (Rule C)")
    args = parser.parse_args()

    # IEEE Style Configuration (matching plot_calibration_convergence.py)
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })

    # --- 1. Load Data ---
    dist_df = load_route_stop_dist(args.real_dist)
    dist_df = dist_df[dist_df['route'] == args.route]
    
    sim_raw = load_sim_data(args.sim)
    
    if args.bound:
        target_bound = args.bound
    else:
        # Detect Bound
        sim_stops_set = set(sim_raw['bus_stop_id'].unique())
        bound_counts = dist_df.groupby('bound').apply(lambda x: len(set(x['stop_id']) & sim_stops_set))
        target_bound = bound_counts.idxmax() if not bound_counts.empty else 'inbound'
    
    print(f"Route: {args.route}, Bound: {target_bound}")
    
    dist_df = dist_df[dist_df['bound'] == target_bound]
    dist_map = get_dist_map(dist_df, 'seq')
    stop_to_seq = dict(zip(dist_df['stop_id'], dist_df['seq']))
    
    # Identify Sim Bounds
    sim_stops = sim_raw['bus_stop_id'].unique()
    sim_seqs = [stop_to_seq[s] for s in sim_stops if s in stop_to_seq]
    if not sim_seqs:
        print("No valid sim seqs.")
        return
    start_seq = min(sim_seqs)
    start_dist_abs = dist_map.get(start_seq, 0)
    
    # --- 2. Build Sim Full Trajectory (High Res, Interpolated 1s) ---
    sim_df = build_sim_trajectory(sim_raw, dist_df)
    sim_df['dist_rel'] = sim_df['cum_dist_m'] - start_dist_abs
    
    high_res_points = []
    
    for vid, group in sim_df.groupby('vehicle_id'):
        group = group.sort_values('seq')
        
        # Align Time to 0 at start_seq
        entry_rows = group[group['seq'] == start_seq]
        if entry_rows.empty: continue
        t0 = entry_rows.iloc[0]['arrival_time']
        
        if len(group) < 2: continue
        
        last_time = -9999
        
        for i in range(len(group) - 1):
            curr_stop = group.iloc[i]
            next_stop = group.iloc[i+1]
            
            # Dwell phase (At curr_stop)
            # t_arr -> t_dep: Dist = const
            t_arr_rel = curr_stop['arrival_time'] - t0
            t_dep_rel = curr_stop['departure_time'] - t0
            
            for t in range(int(t_arr_rel), int(t_dep_rel)):
                high_res_points.append({
                    'vehicle_id': vid,
                    'time_rel': t,
                    'dist': curr_stop['dist_rel'],
                    'type': 'Dwell'
                })
                
            # Travel phase
            t_next_arr_rel = next_stop['arrival_time'] - t0
            d_curr = curr_stop['dist_rel']
            d_next = next_stop['dist_rel']
            
            duration = t_next_arr_rel - t_dep_rel
            dist_diff = d_next - d_curr
            
            if duration > 0:
                speed = dist_diff / duration
                for t in range(int(t_dep_rel), int(t_next_arr_rel)):
                    dt = t - t_dep_rel
                    d = d_curr + speed * dt
                    high_res_points.append({
                        'vehicle_id': vid,
                        'time_rel': t,
                        'dist': d,
                        'type': 'Move'
                    })
                    
    df_sim_high = pd.DataFrame(high_res_points)
    max_sim_dist = df_sim_high['dist'].max() if not df_sim_high.empty else 0
    print(f"Max Sim Dist: {max_sim_dist}")
    
    # --- 3. Downsample Sim Data ---
    if not df_sim_high.empty:
        df_sim_low = df_sim_high[df_sim_high['time_rel'] % args.downsample_interval_s == 0].copy()
    else:
        df_sim_low = pd.DataFrame()
        
    # --- 4. Load & Reconstruct Real Trips ---
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[(real_links['route'] == args.route) & (real_links['bound'] == target_bound)]
    
    if 'dist_m' in real_links.columns:
         real_links['length_km'] = real_links['dist_m'] / 1000.0
    real_links['speed_kmh'] = (real_links['length_km'] * 3600) / real_links['travel_time_s']
    
    # Filter Ghost
    clean_links = real_links[~((real_links['travel_time_s'] > args.t_critical) & (real_links['speed_kmh'] < 5))].copy()
    
    # Reconstruct Trips
    # We need to chain links into trips to calculate Cumulative Time relative to start_seq
    # Simple heuristic: Group by departure_ts (Trip) if available.
    # link_times_offpeak.csv: route,bound,service_type,from_seq,to_seq,departure_ts,arrival_ts...
    
    # We don't have vehicle_id. But 'departure_ts' at origin establishes a trip.
    # However, link departure_ts updates along the trip.
    # We must chain them. But without TripID, it's hard.
    # Alternative: Plot unconnected segments in Relative Space?
    # Relative Dist is known (from_seq -> dist map).
    # Relative Time? We don't know T0 for each link without knowing when that trip passed start_seq.
    
    # Solution: Assume strict schedule or use inferred trips?
    # Or just Assume Mean speed to backtrack to start? No, that defeats the purpose.
    # Best effort: Group by "Approximate Start Time".
    # Sort by from_seq.
    # If we iterate valid links, we might find chains.
    # But for "Visual Comparison", maybe "Mean Trajectory" + Variance is enough?
    # No, we want "Spacetime Characteristics".
    # Let's use the provided `departure_ts` to group.
    # If links belong to same trip, they should be chemically connected.
    # Trip ID heuristic: Round(arrival_ts at destination - predicted duration)?
    # Let's try to just plot "Segments" but we need X = Relative Time.
    # Relative Time = (departure_ts - Trip_Start_TS).
    # We need Trip_Start_TS.
    
    # Heuristic: Back-calculate Trip Start TS for each link using avg speed?
    # BETTER: Group by `service_type` (if unique?) No.
    # Let's assume the dataset is small enough (1 hour).
    # Let's try to link them: 
    #   Sort by departure_ts.
    #   If (to_seq == next_from_seq) and (next_dep approx arr), it's same trip.
    
    clean_links['ts'] = pd.to_datetime(clean_links['departure_ts'])
    if clean_links['ts'].dt.tz is None: clean_links['ts'] = clean_links['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
    
    clean_links = clean_links.sort_values(['ts', 'from_seq'])
    
    trips = []
    current_trip = []
    
    # Naive chaining
    for _, row in clean_links.iterrows():
        if not current_trip:
            current_trip.append(row)
            continue
            
        last = current_trip[-1]
        # Check connectivity
        # seq connectivity
        seq_gap = row['from_seq'] - last['to_seq']
        # time connectivity (allow 5 min dwell/gap)
        time_gap = (row['ts'] - pd.to_datetime(last['arrival_ts']).tz_convert('Asia/Hong_Kong')).total_seconds()
        
        if seq_gap == 0 and 0 <= time_gap < 300:
            current_trip.append(row)
        else:
            trips.append(current_trip)
            current_trip = [row]
    if current_trip: trips.append(current_trip)
    
    real_plot_segments = []
    
    for trip in trips:
        # Convert trip to DataFrame
        df_trip = pd.DataFrame(trip)
        
        # Check if trip covers start_seq
        # Find time at start_seq
        # We need to interpolate if start_seq is not exactly a node?
        # link is (from, to). If start_seq is one of 'from', we have time.
        # If start_seq is between from/to?
        # Usually start_seq IS a node.
        
        # Find row where from_seq <= start_seq < to_seq OR from_seq == start_seq
        # Ideally, we want T=0 at start_seq.
        
        # Simple Logic: Find row where from_seq == start_seq.
        start_row = df_trip[df_trip['from_seq'] == start_seq]
        
        if not start_row.empty:
            t0 = start_row.iloc[0]['ts']
        else:
            # Trip doesn't contain start_seq link?
            # Maybe it started later? Or data missing?
            # Try to find closest preceeding or succeeding?
            # If we only keep trips that pass through start_seq:
            continue
            
        for _, row in df_trip.iterrows():
            u, v = row['from_seq'], row['to_seq']
            d_u = dist_map.get(u)
            d_v = dist_map.get(v)
            if d_u is None or d_v is None: continue
            
            d_u_rel = d_u - start_dist_abs
            d_v_rel = d_v - start_dist_abs
            
            # Crop Spatial
            if d_u_rel > max_sim_dist + 500: continue
            
            t_dep_rel = (row['ts'] - t0).total_seconds()
            t_arr_rel = t_dep_rel + row['travel_time_s']
            
            real_plot_segments.append({
                'x0': t_dep_rel, 'x1': t_arr_rel,
                'y0': d_u_rel, 'y1': d_v_rel
            })
            
    print(f"Reconstructed {len(real_plot_segments)} Clean Real Segments aligned to start_seq {start_seq}")

    # --- 5. Plotting 3-Panel (IEEE Double-Column Width: 7.16 inches) ---
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    # This matches the style of B2_phase_comparison.png for consistency
    
    # Common limits
    all_x = [s['x1'] for s in real_plot_segments] + df_sim_high['time_rel'].tolist()
    maxx = np.percentile(all_x, 98) if all_x else 3000
    
    # Panel 1: Sim High Res
    ax0 = axes[0]
    if not df_sim_high.empty:
        for vid, group in df_sim_high.groupby('vehicle_id'):
            ax0.plot(group['time_rel'], group['dist'], color='#ff7f0e', linewidth=1.2, alpha=0.8) # Back to Orange
            
    ax0.set_title("Simulated (Raw 1s)", fontsize=8)
    ax0.set_xlabel("Relative Time (s)", fontsize=8)
    ax0.set_ylabel("Relative Distance (m)", fontsize=8)
    ax0.set_xlim(0, maxx)
    ax0.set_ylim(0, max_sim_dist + 100)
    ax0.grid(True, alpha=0.4)
    
    # Panel 2: Sim Downsampled
    ax1 = axes[1]
    if not df_sim_low.empty:
        for vid, group in df_sim_low.groupby('vehicle_id'):
            # Plot as dots
            ax1.plot(group['time_rel'], group['dist'], color='#ff7f0e', marker='o', linestyle='', markersize=3.5, alpha=0.9) # Back to Orange
            
    ax1.set_title(f"Simulated (Downsampled {args.downsample_interval_s}s)", fontsize=8)
    ax1.set_xlabel("Relative Time (s)", fontsize=8)
    ax1.set_xlim(0, maxx)
    ax1.grid(True, alpha=0.4)
    
    # Panel 3: Real Clean
    ax2 = axes[2]
    # Plot segments
    for seg in real_plot_segments:
        ax2.plot([seg['x0'], seg['x1']], [seg['y0'], seg['y1']], color='#1f77b4', linewidth=1.5, alpha=0.9) # Dark blue, thicker
        
    ax2.set_title(f"Real World (Clean)", fontsize=8)
    ax2.set_xlabel("Relative Time (s)", fontsize=8)
    ax2.set_xlim(0, maxx)
    ax2.grid(True, alpha=0.4)
    
    
    # Apply tight_layout (matching B2_phase_comparison style)
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Comparison Spacetime to {args.out}")

if __name__ == "__main__":
    main()
