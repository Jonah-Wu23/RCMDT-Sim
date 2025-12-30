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

def _clip_trajectory(points, max_dist):
    if not points:
        return []

    clipped = [points[0]]
    for prev, curr in zip(points[:-1], points[1:]):
        x0, y0 = prev
        x1, y1 = curr
        if y1 <= max_dist:
            clipped.append((x1, y1))
            continue
        if y0 < max_dist and y1 > y0:
            ratio = (max_dist - y0) / (y1 - y0)
            x_cross = x0 + ratio * (x1 - x0)
            clipped.append((x_cross, max_dist))
        break
    return clipped

def main():
    parser = argparse.ArgumentParser(description="Plot Stepped Trajectory (Distance vs Time) to show Dwell Mechanics")
    parser.add_argument('--real_links', required=True, help="Path to real link stats CSV")
    parser.add_argument('--real_dist', required=True, help="Path to route stop distance CSV")
    parser.add_argument('--sim', required=True, help="Path to simulation stopinfo XML")
    parser.add_argument('--out', required=True, help="Output image path")
    parser.add_argument('--route', default='68X', help="Route ID (e.g. 68X, 960)")
    parser.add_argument('--bound', default=None, help="Direction (inbound/outbound), optional auto-detect")
    parser.add_argument('--title_suffix', default="", help="Suffix for plot title")
    parser.add_argument('--t_critical', type=float, default=325, help="Ghost filter threshold for Real Data (Rule C)")
    parser.add_argument('--speed_kmh', type=float, default=5, help="Speed threshold for Rule C (km/h)")
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
            if d_rel < -50 or d_rel > max_sim_dist + 50:
                continue # Loose crop
            
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

    if rep_trip['points']:
        sim_end_dist = max(pt['dist'] for pt in rep_trip['points'])
        corridor_max_m = sim_end_dist if np.isfinite(sim_end_dist) else max_sim_dist
    else:
        corridor_max_m = max_sim_dist if np.isfinite(max_sim_dist) else 5000
    
    # Construct Stepped Line Points for Sim (Full)
    sim_full_points = []  # (time, dist)
    for pt in rep_trip['points']:
        sim_full_points.append((pt['time_arr'], pt['dist']))
        sim_full_points.append((pt['time_dep'], pt['dist']))

    sim_full_points = _clip_trajectory(sim_full_points, corridor_max_m)
    sim_full_x = [p[0] for p in sim_full_points]
    sim_full_y = [p[1] for p in sim_full_points]

    # Construct Traffic-Only Sim Trajectory (exclude low-speed segments)
    sim_move_points = []
    move_time = 0.0
    if rep_trip['points']:
        sim_move_points.append((0.0, rep_trip['points'][0]['dist']))
    for curr, nxt in zip(rep_trip['points'][:-1], rep_trip['points'][1:]):
        dist_diff = nxt['dist'] - curr['dist']
        travel_time = nxt['time_arr'] - curr['time_dep']
        if travel_time <= 0:
            continue
        speed_kmh = (dist_diff / travel_time) * 3.6 if travel_time > 0 else 0.0
        if speed_kmh >= args.speed_kmh:
            move_time += travel_time
        sim_move_points.append((move_time, nxt['dist']))

    sim_move_points = _clip_trajectory(sim_move_points, corridor_max_m)
    sim_move_x = [p[0] for p in sim_move_points]
    sim_move_y = [p[1] for p in sim_move_points]
        
    # --- 3. Construct Real World Representative Trajectory ---
    print(f"Loading real link speeds from {args.real_links}...")
    real_links = load_real_link_speeds(args.real_links)
    real_links = real_links[(real_links['route'] == args.route) & (real_links['bound'] == target_bound)]

    if 'dist_m' in real_links.columns:
        real_links['speed_kmh'] = (real_links['dist_m'] / real_links['travel_time_s']) * 3.6
    else:
        real_links['speed_kmh'] = 0.0

    real_links = real_links[
        ~((real_links['travel_time_s'] > args.t_critical) & (real_links['speed_kmh'] < args.speed_kmh))
    ]

    real_links['departure_ts'] = pd.to_datetime(real_links['departure_ts'], errors='coerce')
    real_links['arrival_ts'] = pd.to_datetime(real_links['arrival_ts'], errors='coerce')
    real_links = real_links.dropna(subset=['departure_ts', 'arrival_ts'])

    if real_links['departure_ts'].dt.tz is None:
        real_links['departure_ts'] = real_links['departure_ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
        real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
    else:
        real_links['departure_ts'] = real_links['departure_ts'].dt.tz_convert('Asia/Hong_Kong')
        real_links['arrival_ts'] = real_links['arrival_ts'].dt.tz_convert('Asia/Hong_Kong')

    real_links = real_links.sort_values('departure_ts')

    sorted_seqs = sorted([s for s in dist_df['seq'] if s >= start_seq])
    corridor_seqs = [
        s for s in sorted_seqs
        if dist_map.get(s, start_dist_abs) - start_dist_abs <= corridor_max_m
    ]
    total_seq_count = max(len(corridor_seqs), 1)
    trips = _build_trips_from_origin(real_links, start_seq=start_seq)

    def collect_candidates(min_coverage, min_dist_m, min_segments):
        candidates = []
        for trip in trips:
            trip_df = pd.DataFrame(trip).sort_values('from_seq')
            start_row = trip_df[trip_df['from_seq'] == start_seq]
            if start_row.empty:
                continue
            segments = []
            covered = set()
            monotone = True
            last_dist = -np.inf
            current_t = 0.0
            current_t_move = 0.0

            for _, row in trip_df.iterrows():
                u, v = row['from_seq'], row['to_seq']
                d_u = dist_map.get(u)
                d_v = dist_map.get(v)
                if d_u is None or d_v is None:
                    continue

                d_u_rel = d_u - start_dist_abs
                d_v_rel = d_v - start_dist_abs

                if d_u_rel < 0:
                    continue
                if d_u_rel > corridor_max_m:
                    break

                dt = row['travel_time_s']
                if dt <= 0:
                    continue
                current_t += dt
                if row['speed_kmh'] >= args.speed_kmh:
                    current_t_move += dt

                if d_v_rel < last_dist:
                    monotone = False
                    break

                last_dist = d_v_rel
                covered.add(u)

                segments.append({
                    't_arr': current_t,
                    't_move': current_t_move,
                    'd_v': d_v_rel
                })

            if not monotone or len(segments) < min_segments:
                continue

            coverage_ratio = len(covered) / total_seq_count
            if coverage_ratio < min_coverage and last_dist < min_dist_m:
                continue

            duration = segments[-1]['t_arr']
            candidates.append({'duration': duration, 'segments': segments, 'last_dist': last_dist})
        return candidates

    candidate_trips = collect_candidates(min_coverage=0.6, min_dist_m=3000, min_segments=2)
    if not candidate_trips:
        candidate_trips = collect_candidates(min_coverage=0.3, min_dist_m=1500, min_segments=2)

    if not candidate_trips:
        print("No valid representative real trips found after relaxed thresholds.")
        return

    df_real = pd.DataFrame(candidate_trips)
    median_dur_real = df_real['duration'].median()
    best_real_idx = (df_real['duration'] - median_dur_real).abs().idxmin()
    rep_real = candidate_trips[best_real_idx]

    real_points = [(0, 0)]
    for seg in rep_real['segments']:
        real_points.append((seg['t_move'], seg['d_v']))

    real_points = _clip_trajectory(real_points, corridor_max_m)
    real_x = [p[0] for p in real_points]
    real_y = [p[1] for p in real_points]
        
    # --- 4. Plotting ---
    plt.figure(figsize=(3.5, 3.0)) # IEEE Small column width
    
    # Plot Real (Traffic-Only)
    plt.plot(real_x, real_y, label='Real world (traffic-only)', color='#1f77b4', linestyle='-', linewidth=2.0, alpha=0.9)

    # Plot Sim (Full, Stepped)
    plt.plot(sim_full_x, sim_full_y, label='Simulation (full time)', color='#ff7f0e', linewidth=2.5, alpha=0.9)

    # Plot Sim (Traffic-Only)
    plt.plot(sim_move_x, sim_move_y, label='Simulation (traffic-only)', color='#ff7f0e', linestyle='--', linewidth=2.0, alpha=0.9)
    
    # Style
    plt.xlabel("Cumulative Travel Time (s)", fontsize=8)
    plt.ylabel("Cumulative Distance (m)", fontsize=8)
    plt.title(f"{args.route} Trajectory (Full vs Traffic-only){args.title_suffix}")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=7, loc='lower right')

    max_time = max(sim_full_x + sim_move_x + real_x) if (sim_full_x and real_x) else max(sim_full_x or sim_move_x or real_x or [0])

    real_points_count = len(real_points)
    plt.text(0.02, 0.98, f"Real-world observations are sparse (Trips=1, Points={real_points_count}).",
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Formatting
    plt.ylim(0, corridor_max_m)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved Stepped Trajectory plot to {args.out}")

if __name__ == "__main__":
    main()
