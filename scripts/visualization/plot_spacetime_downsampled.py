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

def _build_trips_from_origin(links_df, start_seq=1, max_gap_s=900, seq_drop=3,
                             stuck_k=5, stuck_speed_kmh=1.0, stuck_dist_m=10.0):
    trips = []
    current = []
    stuck_count = 0
    last_dep = None
    last_from_seq = None

    for row in links_df.itertuples(index=False):
        start_new = False

        if row.from_seq == start_seq:
            start_new = True

        if current:
            if last_dep is not None:
                gap = (row.departure_ts - last_dep).total_seconds()
                if gap > max_gap_s:
                    start_new = True

            if last_from_seq is not None and (last_from_seq - row.from_seq) >= seq_drop:
                start_new = True

            if row.speed_kmh < stuck_speed_kmh and row.dist_m <= stuck_dist_m:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= stuck_k:
                start_new = True
                stuck_count = 0

        if start_new:
            if current:
                trips.append(current)
            current = [row]
        else:
            current.append(row)

        last_dep = row.departure_ts
        last_from_seq = row.from_seq

    if current:
        trips.append(current)

    return trips

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
    parser.add_argument('--sample_n', type=int, default=50, help="Number of trips/vehicles to sample for plotting")
    parser.add_argument('--corridor_m', type=float, default=5000, help="Spatial corridor length in meters")
    parser.add_argument('--seed', type=int, default=7, help="Random seed for sampling")
    parser.add_argument('--mode', choices=['single_trip', 'multi'], default='single_trip',
                        help="single_trip for N=1 sampling-rate sanity check, multi for trajectory texture comparison")
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

    corridor_max_m = args.corridor_m
    if not df_sim_high.empty:
        df_sim_high = df_sim_high[(df_sim_high['dist'] >= 0) & (df_sim_high['dist'] <= corridor_max_m)]
    if not df_sim_low.empty:
        df_sim_low = df_sim_low[(df_sim_low['dist'] >= 0) & (df_sim_low['dist'] <= corridor_max_m)]

    rng = np.random.default_rng(args.seed)
    sim_vehicle_ids = df_sim_high['vehicle_id'].unique().tolist()

    if args.mode == 'single_trip' and sim_vehicle_ids:
        sim_durations = df_sim_high.groupby('vehicle_id')['time_rel'].max()
        median_dur = sim_durations.median()
        sim_keep = sim_durations.sub(median_dur).abs().idxmin()
        df_sim_high = df_sim_high[df_sim_high['vehicle_id'] == sim_keep]
        df_sim_low = df_sim_low[df_sim_low['vehicle_id'] == sim_keep]
    elif args.sample_n > 0 and len(sim_vehicle_ids) > args.sample_n:
        sim_keep = rng.choice(sim_vehicle_ids, size=args.sample_n, replace=False).tolist()
        df_sim_high = df_sim_high[df_sim_high['vehicle_id'].isin(sim_keep)]
        df_sim_low = df_sim_low[df_sim_low['vehicle_id'].isin(sim_keep)]
        
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
    
    clean_links['departure_ts'] = pd.to_datetime(clean_links['departure_ts'], errors='coerce')
    clean_links['arrival_ts'] = pd.to_datetime(clean_links['arrival_ts'], errors='coerce')
    clean_links = clean_links.dropna(subset=['departure_ts', 'arrival_ts'])

    if clean_links['departure_ts'].dt.tz is None:
        clean_links['departure_ts'] = clean_links['departure_ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
        clean_links['arrival_ts'] = clean_links['arrival_ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
    else:
        clean_links['departure_ts'] = clean_links['departure_ts'].dt.tz_convert('Asia/Hong_Kong')
        clean_links['arrival_ts'] = clean_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')

    clean_links = clean_links.sort_values(['departure_ts', 'from_seq'])
    trips = _build_trips_from_origin(clean_links, start_seq=start_seq)

    real_plot_segments = []
    all_trip_segments = []

    sorted_seqs = sorted([s for s in dist_df['seq'] if s >= start_seq])
    corridor_seqs = [
        s for s in sorted_seqs
        if dist_map.get(s, start_dist_abs) - start_dist_abs <= corridor_max_m
    ]
    total_seq_count = max(len(corridor_seqs), 1)
    min_coverage = 0.6
    min_dist_m = 3000

    for trip in trips:
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
            t0 = start_row.iloc[0]['departure_ts']
        else:
            # Trip doesn't contain start_seq link?
            # Maybe it started later? Or data missing?
            # Try to find closest preceeding or succeeding?
            # If we only keep trips that pass through start_seq:
            continue
            
        trip_segments = []
        covered = set()
        last_dist = -np.inf
        for _, row in df_trip.iterrows():
            u, v = row['from_seq'], row['to_seq']
            d_u = dist_map.get(u)
            d_v = dist_map.get(v)
            if d_u is None or d_v is None: continue
            
            d_u_rel = d_u - start_dist_abs
            d_v_rel = d_v - start_dist_abs
            
            # Crop Spatial
            if d_u_rel > corridor_max_m:
                break
            
            t_dep_rel = (row['departure_ts'] - t0).total_seconds()
            t_arr_rel = t_dep_rel + row['travel_time_s']

            if d_u_rel < 0:
                continue

            if d_v_rel < last_dist:
                continue
            last_dist = d_v_rel
            covered.add(u)

            if d_v_rel > corridor_max_m and d_v_rel > d_u_rel:
                ratio = (corridor_max_m - d_u_rel) / (d_v_rel - d_u_rel)
                t_arr_rel = t_dep_rel + ratio * (t_arr_rel - t_dep_rel)
                d_v_rel = corridor_max_m
                trip_segments.append({
                    'x0': t_dep_rel, 'x1': t_arr_rel,
                    'y0': d_u_rel, 'y1': d_v_rel
                })
                break

            trip_segments.append({
                'x0': t_dep_rel, 'x1': t_arr_rel,
                'y0': d_u_rel, 'y1': d_v_rel
            })

        if trip_segments:
            all_trip_segments.append(trip_segments)
            coverage_ratio = len(covered) / total_seq_count
            last_dist = max(seg['y1'] for seg in trip_segments)
            if coverage_ratio < min_coverage and last_dist < min_dist_m:
                continue
            real_plot_segments.append(trip_segments)

    if not real_plot_segments and all_trip_segments:
        print("No trips met coverage threshold; falling back to all aligned trips.")
        real_plot_segments = all_trip_segments

    if not real_plot_segments:
        print(f"No clean real trips aligned to start_seq {start_seq}.")
    else:
        print(f"Reconstructed {len(real_plot_segments)} clean real trips aligned to start_seq {start_seq}")

    if args.mode == 'single_trip' and real_plot_segments:
        trip_durations = [max(seg['x1'] for seg in trip) for trip in real_plot_segments]
        median_dur = np.median(trip_durations)
        best_idx = int(np.argmin([abs(d - median_dur) for d in trip_durations]))
        real_plot_segments = [real_plot_segments[best_idx]]
    elif args.sample_n > 0 and len(real_plot_segments) > args.sample_n:
        keep_idx = rng.choice(len(real_plot_segments), size=args.sample_n, replace=False)
        real_plot_segments = [real_plot_segments[i] for i in keep_idx]

    # --- 5. Plotting 3-Panel (IEEE Double-Column Width: 7.16 inches) ---
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    # This matches the style of B2_phase_comparison.png for consistency
    
    # Common limits
    real_x_vals = [seg['x1'] for trip in real_plot_segments for seg in trip]
    all_x = real_x_vals + df_sim_high['time_rel'].tolist()
    maxx = np.percentile(all_x, 98) if all_x else 3000
    
    def _panel_stats(time_vals):
        if len(time_vals) < 2:
            return "n/a"
        diffs = np.diff(np.sort(time_vals))
        median_dt = np.median(diffs)
        return f"{median_dt:.0f}"

    # Panel 1: Sim High Res
    ax0 = axes[0]
    if not df_sim_high.empty:
        for vid, group in df_sim_high.groupby('vehicle_id'):
            ax0.plot(group['time_rel'], group['dist'], color='#ff7f0e', linewidth=1.2, alpha=0.8) # Back to Orange
            
    ax0.set_title("Simulated (Raw 1s)", fontsize=8)
    ax0.set_xlabel("Relative Time (s)", fontsize=8)
    ax0.set_ylabel("Relative Distance (m)", fontsize=8)
    ax0.set_xlim(0, maxx)
    ax0.set_ylim(0, corridor_max_m)
    ax0.grid(True, alpha=0.4)

    sim_high_times = df_sim_high['time_rel'].tolist()
    sim_trip_count = df_sim_high['vehicle_id'].nunique()
    ax0.text(0.98, 0.98, f"Trips={sim_trip_count}, Points={len(sim_high_times)}\nMedian Δt={_panel_stats(sim_high_times)} s",
             transform=ax0.transAxes, ha='right', va='top', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
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

    sim_low_times = df_sim_low['time_rel'].tolist()
    sim_low_trip_count = df_sim_low['vehicle_id'].nunique()
    ax1.text(0.98, 0.98, f"Trips={sim_low_trip_count}, Points={len(sim_low_times)}\nMedian Δt={_panel_stats(sim_low_times)} s",
             transform=ax1.transAxes, ha='right', va='top', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Panel 3: Real Clean
    ax2 = axes[2]
    # Plot scatter points (single-trip sanity check)
    real_points = []
    for trip in real_plot_segments:
        for seg in trip:
            real_points.append((seg['x0'], seg['y0']))
            real_points.append((seg['x1'], seg['y1']))

    if real_points:
        real_times = [p[0] for p in real_points]
        real_dists = [p[1] for p in real_points]
        ax2.scatter(real_times, real_dists, color='#1f77b4', s=10, alpha=0.8)
    else:
        real_times = []
        
    ax2.set_title(f"Real World (Clean)", fontsize=8)
    ax2.set_xlabel("Relative Time (s)", fontsize=8)
    ax2.set_xlim(0, maxx)
    ax2.grid(True, alpha=0.4)

    real_trip_count = len(real_plot_segments)
    ax2.text(0.98, 0.98, f"Trips={real_trip_count}, Points={len(real_points)}\nMedian Δt={_panel_stats(real_times)} s",
             transform=ax2.transAxes, ha='right', va='top', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    
    # Apply tight_layout (matching B2_phase_comparison style)
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Comparison Spacetime to {args.out}")

if __name__ == "__main__":
    main()
