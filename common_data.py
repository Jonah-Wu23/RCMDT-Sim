
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

def load_sim_data(xml_path):
    """
    Parses SUMO stopinfo.xml output.
    Returns: DataFrame with [vehicle_id, bus_stop_id, started, ended, delay, pos]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        data = []
        for stop in root.findall('stopinfo'):
            # Check mandatory fields
            s_started = stop.get('started')
            s_ended = stop.get('ended')
            
            if s_started is None or s_ended is None:
                continue
                
            data.append({
                'vehicle_id': stop.get('id'),
                'bus_stop_id': stop.get('busStop'), 
                'started': float(s_started), 
                'ended': float(s_ended),     
                'delay': float(stop.get('delay', 0)),
                'pos': float(stop.get('pos', 0))
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"Warning: No valid stopinfo records found in {xml_path}")
            return pd.DataFrame(columns=['vehicle_id', 'bus_stop_id', 'started', 'ended', 'delay', 'pos'])

        if 'bus_stop_id' not in df.columns:
            df['bus_stop_id'] = None
            
        return df
    except Exception as e:
        print(f"Error loading SIM XML {xml_path}: {e}")
        # Return empty with correct columns to avoid KeyError downstream
        return pd.DataFrame(columns=['vehicle_id', 'bus_stop_id', 'started', 'ended', 'delay', 'pos'])

def load_real_link_speeds(csv_path):
    df = pd.read_csv(csv_path)
    if 'departure_ts' in df.columns:
        df['departure_ts'] = pd.to_datetime(df['departure_ts'], errors='coerce')
    if 'arrival_ts' in df.columns:
        df['arrival_ts'] = pd.to_datetime(df['arrival_ts'], errors='coerce')
    if 'travel_time_s' not in df.columns:
        if 'tt_mean' in df.columns:
            df['travel_time_s'] = df['tt_mean']
        elif 'tt_median' in df.columns:
            df['travel_time_s'] = df['tt_median']
    return df

def load_route_stop_dist(csv_path):
    return pd.read_csv(csv_path)

def get_dist_map(dist_df, key_col='seq'):
    """
    获取距离映射，优先使用 cum_dist_m_dir，NaN 时回退到 cum_dist_m
    """
    if 'cum_dist_m_dir' in dist_df.columns:
        # 用 cum_dist_m 填充 cum_dist_m_dir 的 NaN
        dist_vals = dist_df['cum_dist_m_dir'].fillna(dist_df['cum_dist_m'])
    else:
        dist_vals = dist_df['cum_dist_m']
    return dict(zip(dist_df[key_col], dist_vals))

def build_sim_trajectory(sim_df, dist_df, start_time_offset=0):
    if sim_df.empty: return pd.DataFrame()
    
    # Filter valid stops
    if 'bus_stop_id' not in sim_df.columns:
        return pd.DataFrame()
        
    sim_df = sim_df.dropna(subset=['bus_stop_id'])
    
    # 使用有向距离口径 (cum_dist_m_dir)，若不存在则回退到 cum_dist_m
    # 当 cum_dist_m_dir 存在但某些行为 NaN 时，用 cum_dist_m 填充
    if 'cum_dist_m_dir' in dist_df.columns:
        dist_df = dist_df.copy()
        dist_df['_dist_merged'] = dist_df['cum_dist_m_dir'].fillna(dist_df['cum_dist_m'])
        dist_col = '_dist_merged'
    else:
        dist_col = 'cum_dist_m'
    
    merge_cols = ['stop_id', 'seq', dist_col]
    
    merged = pd.merge(sim_df, dist_df[merge_cols], 
                      left_on='bus_stop_id', right_on='stop_id', how='inner')
    
    # 统一列名为 cum_dist_m 以保持下游兼容性
    merged['cum_dist_m'] = merged[dist_col]
    
    merged = merged.sort_values(['vehicle_id', 'started'])
    merged['arrival_time'] = merged['started'] + start_time_offset
    merged['departure_time'] = merged['ended'] + start_time_offset
    return merged

def get_sim_coverage(sim_df, dist_df):
    """
    Determines the min and max sequence numbers present in the simulation data
    for alignment with real world data.
    """
    if sim_df.empty:
        return None, None
        
    # We need to map stop_id to seq
    # dist_df might have multiple routes, filtering usually happens before
    # But stop_id is usually unique.
    stop_to_seq = dist_df.set_index('stop_id')['seq'].to_dict()
    
    sim_seqs = []
    # Only iterate stops that exist in dist_df
    for sid in sim_df['bus_stop_id'].unique():
        if sid in stop_to_seq:
            sim_seqs.append(stop_to_seq[sid])
            
    if not sim_seqs:
        return None, None
        
    return min(sim_seqs), max(sim_seqs)
