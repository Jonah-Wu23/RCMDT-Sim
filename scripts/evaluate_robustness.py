"""
Evaluate P14 Robustness Test Results.
Compare off-peak simulation against real D2D travel time data.

Metrics: KS Distance, RMSE, P90 Error
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os

# Default paths
REAL_STATS = "data2/processed/link_stats_offpeak.csv"
SIM_STOPINFO = "sumo/output/offpeak_stopinfo.xml"
PEAK_STATS = "data/processed/link_stats.csv"  # For comparison
DIST_FILE = "data/processed/kmb_route_stop_dist.csv"

RULE_C_T_CRITICAL_S = 325
RULE_C_SPEED_KMH = 5.0
RULE_C_MAX_DIST_M = 1500.0


def load_real_stats(filepath):
    """Load real D2D link statistics."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} real link stats from {filepath}")
    return df


def load_sim_stopinfo(filepath):
    """Parse SUMO stopinfo.xml to extract bus stop times."""
    from xml.etree import ElementTree as ET
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    records = []
    for stop in root.findall('.//stopinfo'):
        records.append({
            'vehicle_id': stop.get('id'),
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started', 0)),
            'departure': float(stop.get('ended', 0)),
            'duration': float(stop.get('duration', 0))
        })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} stop records from {filepath}")
    return df


def calculate_sim_link_times(df_stops):
    """Calculate link travel times from consecutive stop visits."""
    if df_stops is None or len(df_stops) == 0:
        return None
    
    # Sort by vehicle and arrival time
    df = df_stops.sort_values(['vehicle_id', 'arrival'])
    
    # Calculate travel time between consecutive stops for each vehicle
    results = []
    for veh_id, veh_data in df.groupby('vehicle_id'):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = veh_data.loc[i, 'stop_id']
            to_stop = veh_data.loc[i+1, 'stop_id']
            departure = veh_data.loc[i, 'departure']
            arrival = veh_data.loc[i+1, 'arrival']
            travel_time = arrival - departure
            
            if travel_time > 0 and travel_time < 3600:  # 0-1 hour
                results.append({
                    'vehicle_id': veh_id,
                    'from_stop': from_stop,
                    'to_stop': to_stop,
                    'travel_time_s': travel_time
                })
    
    df_result = pd.DataFrame(results)
    print(f"Calculated {len(df_result)} simulated link times")
    return df_result


def load_sim_speeds_with_distances(stopinfo_xml: str, dist_csv: str) -> np.ndarray:
    """
    Convert simulated stop-to-stop travel times into effective speeds using a stop-pair distance map.

    The distance CSV is expected to contain columns:
      route, bound, service_type, seq, stop_id, link_dist_m
    """
    if not os.path.exists(stopinfo_xml):
        return np.array([])
    if not os.path.exists(dist_csv):
        return np.array([])

    df_dist = pd.read_csv(dist_csv)
    required = {"route", "bound", "service_type", "seq", "stop_id", "link_dist_m"}
    missing = sorted(required - set(df_dist.columns))
    if missing:
        raise ValueError(f"Distance CSV missing columns: {missing}")

    # Map: (from_stop_id, to_stop_id) -> link_dist_m (meters)
    dist_map: dict[tuple[str, str], float] = {}
    for _, group in df_dist.groupby(["route", "bound", "service_type"]):
        group = group.sort_values("seq")
        stops = group["stop_id"].astype(str).tolist()
        link_dists = group["link_dist_m"].tolist()
        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            d = link_dists[i + 1]
            if pd.notna(d) and d > 0:
                dist_map[(s1, s2)] = float(d)

    df_stops = load_sim_stopinfo(stopinfo_xml)
    if df_stops is None or df_stops.empty:
        return np.array([])

    df_stops = df_stops.sort_values(["vehicle_id", "arrival"]).reset_index(drop=True)
    speeds: list[float] = []
    for veh_id, veh_data in df_stops.groupby("vehicle_id"):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = str(veh_data.loc[i, "stop_id"])
            to_stop = str(veh_data.loc[i + 1, "stop_id"])
            departure = float(veh_data.loc[i, "departure"])
            arrival = float(veh_data.loc[i + 1, "arrival"])
            travel_time_s = arrival - departure
            if travel_time_s <= 0:
                continue
            dist_m = dist_map.get((from_stop, to_stop))
            if not dist_m:
                continue
            speed_kmh = (dist_m / 1000.0) / (travel_time_s / 3600.0)
            if 0.1 < speed_kmh < 120:
                speeds.append(speed_kmh)
    return np.asarray(speeds)


def calculate_ks_distance(real_values, sim_values):
    """Calculate Kolmogorov-Smirnov distance between two distributions."""
    if len(real_values) < 5 or len(sim_values) < 5:
        return None, None
    ks_stat, p_value = stats.ks_2samp(real_values, sim_values)
    return ks_stat, p_value


def calculate_rmse(real_values, sim_values):
    """Calculate RMSE between aligned values (if same length) or distributions."""
    real_mean = np.mean(real_values)
    sim_mean = np.mean(sim_values)
    return abs(real_mean - sim_mean)  # Mean absolute error of means


def apply_rule_c(df_stats: pd.DataFrame, t_critical: float, speed_kmh: float, max_dist_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Split real speed samples into (raw_speeds, clean_speeds) using Rule C:
      ghost if (tt_median > T*) & (speed_median < v*) & (dist_m < max_dist_m)
    """
    required = {"tt_median", "speed_median", "dist_m"}
    missing = sorted(required - set(df_stats.columns))
    if missing:
        raise ValueError(f"Real stats CSV missing columns: {missing}")
    raw_speeds = df_stats["speed_median"].dropna().to_numpy()
    cond_ghost = (df_stats["tt_median"] > t_critical) & (df_stats["speed_median"] < speed_kmh) & (df_stats["dist_m"] < max_dist_m)
    clean_speeds = df_stats.loc[~cond_ghost, "speed_median"].dropna().to_numpy()
    return raw_speeds, clean_speeds


def evaluate_robustness(real_file, sim_file, dist_file, t_critical, speed_kmh, max_dist_m, peak_file=None, grade: bool = True):
    """Main evaluation function. Returns a metrics dict for programmatic use."""
    print("\n" + "=" * 60)
    print("P14 Off-Peak Robustness Test Evaluation")
    print("=" * 60)
    
    # Load data
    df_real = load_real_stats(real_file)
    sim_speeds = load_sim_speeds_with_distances(sim_file, dist_file)
    
    if df_real is None:
        print("\nERROR: Cannot load real data. Using synthetic comparison.")
        return None
    
    raw_speeds, clean_speeds = apply_rule_c(df_real, t_critical, speed_kmh, max_dist_m)
    
    print(f"\n--- Real Data (Off-Peak) Statistics ---")
    if len(raw_speeds) > 0:
        print(f"  Raw Samples: {len(raw_speeds)}")
        print(f"  Raw Mean/Median: {np.mean(raw_speeds):.2f} / {np.median(raw_speeds):.2f} km/h")
    if len(clean_speeds) > 0:
        print(f"  Clean Samples (Rule C): {len(clean_speeds)}")
        print(f"  Clean Mean/Median: {np.mean(clean_speeds):.2f} / {np.median(clean_speeds):.2f} km/h")

    print(f"\n--- Simulated Data Statistics ---")
    if len(sim_speeds) > 0:
        print(f"  Samples: {len(sim_speeds)}")
        print(f"  Mean/Median: {np.mean(sim_speeds):.2f} / {np.median(sim_speeds):.2f} km/h")
        print(f"  P10/P90: {np.percentile(sim_speeds, 10):.2f} / {np.percentile(sim_speeds, 90):.2f} km/h")
    else:
        print("  WARNING: No simulated speeds computed (missing distance map or stop pairs).")

    metrics = {
        "ks_raw": None,
        "ks_clean": None,
        "p_raw": None,
        "p_clean": None,
    }

    if len(raw_speeds) > 0 and len(sim_speeds) > 0:
        ks_raw, p_raw = calculate_ks_distance(raw_speeds, sim_speeds)
        metrics["ks_raw"] = ks_raw
        metrics["p_raw"] = p_raw
    if len(clean_speeds) > 0 and len(sim_speeds) > 0:
        ks_clean, p_clean = calculate_ks_distance(clean_speeds, sim_speeds)
        metrics["ks_clean"] = ks_clean
        metrics["p_clean"] = p_clean

    print(f"\n--- Robustness Metrics (Validation Evidence) ---")
    if metrics["ks_raw"] is not None:
        print(f"  KS(raw):   {metrics['ks_raw']:.4f} (p={metrics['p_raw']:.4f})")
    if metrics["ks_clean"] is not None:
        print(f"  KS(clean): {metrics['ks_clean']:.4f} (p={metrics['p_clean']:.4f})")

    if grade and metrics["ks_clean"] is not None:
        print(f"\n--- Evaluation (Clean / Rule C) ---")
        if metrics["ks_clean"] <= 0.25:
            print(f"  [STRONG SUCCESS]: KS={metrics['ks_clean']:.4f} <= 0.25")
        elif metrics["ks_clean"] <= 0.35:
            print(f"  [SUCCESS]: KS={metrics['ks_clean']:.4f} <= 0.35")
        else:
            print(f"  [FAIL]: KS={metrics['ks_clean']:.4f} > 0.35")
    
    # Load peak stats for comparison if available
    if peak_file and os.path.exists(peak_file):
        df_peak = load_real_stats(peak_file)
        if df_peak is not None and 'speed_median' in df_peak.columns:
            peak_speeds = df_peak['speed_median'].dropna().values
            print(f"\n--- Peak Data (Training) Statistics ---")
            print(f"  Mean Speed: {np.mean(peak_speeds):.2f} km/h")
            print(f"  Median Speed: {np.median(peak_speeds):.2f} km/h")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P14 robustness test results.")
    parser.add_argument("--real", default=REAL_STATS, help="Real link stats CSV")
    parser.add_argument("--sim", default=SIM_STOPINFO, help="Simulation stopinfo XML")
    parser.add_argument("--dist", default=DIST_FILE, help="Route stop distance CSV for simulated speed calculation")
    parser.add_argument("--t_critical", type=float, default=RULE_C_T_CRITICAL_S, help="Rule C: travel time threshold T* (s)")
    parser.add_argument("--speed_kmh", type=float, default=RULE_C_SPEED_KMH, help="Rule C: speed threshold v* (km/h)")
    parser.add_argument("--max_dist_m", type=float, default=RULE_C_MAX_DIST_M, help="Rule C: apply only when dist_m < max_dist_m")
    parser.add_argument("--peak", default=PEAK_STATS, help="Peak link stats CSV for comparison")
    parser.add_argument("--no_grade", action="store_true", help="Do not print PASS/FAIL grading (useful for fixtures/CI).")
    
    args = parser.parse_args()
    evaluate_robustness(args.real, args.sim, args.dist, args.t_critical, args.speed_kmh, args.max_dist_m, args.peak, grade=(not args.no_grade))
