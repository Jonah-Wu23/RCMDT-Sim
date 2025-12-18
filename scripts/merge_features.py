import os
import json
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

def parse_filename_ts(filename):
    basename = os.path.basename(filename)
    parts = basename.replace('.json', '').replace('.xml', '').replace('.csv', '').split('-')
    if len(parts) >= 2:
        ts_str = f"{parts[-2]}-{parts[-1]}"
        try:
            return datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
        except ValueError:
            return None
    return None

def load_hko_data(hko_dir):
    data = []
    files = glob.glob(os.path.join(hko_dir, "*.json"))
    for f in files:
        ts = parse_filename_ts(f)
        if not ts: continue
        with open(f, 'r', encoding='utf-8') as jf:
            content = json.load(jf)
            # Get avg temperature if available
            temps = content.get('temperature', {}).get('data', [])
            avg_temp = None
            if temps:
                avg_temp = sum([t['value'] for t in temps]) / len(temps)
            
            # Get max rainfall
            rainfall = content.get('rainfall', {}).get('data', [])
            max_rain = 0
            if rainfall:
                max_rain = max([r.get('max', 0) for r in rainfall])
            
            data.append({'timestamp': ts, 'temp': avg_temp, 'rainfall': max_rain})
    return pd.DataFrame(data).sort_values('timestamp')

def load_stn_data(stn_dir):
    data = []
    files = glob.glob(os.path.join(stn_dir, "*.xml"))
    for f in files:
        ts = parse_filename_ts(f)
        if not ts: continue
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            count = len(root.findall('message'))
            data.append({'timestamp': ts, 'incident_count': count})
        except Exception as e:
            print(f"Error parsing STN {f}: {e}")
    return pd.DataFrame(data).sort_values('timestamp')

def load_jti_data(jti_dir):
    data = []
    files = glob.glob(os.path.join(jti_dir, "*.xml"))
    # Namespace handling is tricky with ElementTree if not exact.
    # The sample shows xmlns="http://data.one.gov.hk/td".
    # We can skip namespace by searching for local tags or using the full URI.
    ns = {'ns': 'http://data.one.gov.hk/td'}
    
    for f in files:
        ts = parse_filename_ts(f)
        if not ts: continue
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            jts = []
            
            # Find all jtis_journey_time elements. 
            # Try with namespace first, then without if empty (fallback).
            items = root.findall('ns:jtis_journey_time', ns)
            if not items:
                items = root.findall('jtis_journey_time')
                
            for item in items:
                # The sample shows <JOURNEY_DATA>11</JOURNEY_DATA>
                # It is NOT JOURNEY_TIME, it is JOURNEY_DATA.
                
                # Check for JOURNEY_DATA
                jd = item.find('ns:JOURNEY_DATA', ns)
                if jd is None:
                    jd = item.find('JOURNEY_DATA')
                    
                if jd is not None and jd.text and jd.text.isdigit():
                    jts.append(float(jd.text))
            
            avg_jti = sum(jts) / len(jts) if jts else None
            data.append({'timestamp': ts, 'avg_journey_time': avg_jti})
        except Exception as e:
            print(f"Error parsing JTI {f}: {e}")
    return pd.DataFrame(data).sort_values('timestamp')

def main():
    base_file = "data/processed/link_times.csv"
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found.")
        return

    df = pd.read_csv(base_file)
    df['departure_ts'] = pd.to_datetime(df['departure_ts']).dt.tz_localize(None)

    print("Loading HKO data...")
    df_hko = load_hko_data("data/raw/hko")
    if not df_hko.empty:
        df_hko['timestamp'] = pd.to_datetime(df_hko['timestamp']).dt.tz_localize(None)

    print("Loading STN data...")
    df_stn = load_stn_data("data/raw/stn")
    if not df_stn.empty:
        df_stn['timestamp'] = pd.to_datetime(df_stn['timestamp']).dt.tz_localize(None)

    print("Loading JTI data...")
    df_jti = load_jti_data("data/raw/jti")
    if not df_jti.empty:
        df_jti['timestamp'] = pd.to_datetime(df_jti['timestamp']).dt.tz_localize(None)

    # Use merge_asof for timestamp matching
    # Requires sorted dataframes
    df = df.sort_values('departure_ts')
    
    print("Merging features...")
    if not df_hko.empty:
        df = pd.merge_asof(df, df_hko, left_on='departure_ts', right_on='timestamp', direction='nearest')
        df = df.drop(columns=['timestamp'])
    
    if not df_stn.empty:
        df = pd.merge_asof(df, df_stn, left_on='departure_ts', right_on='timestamp', direction='nearest')
        df = df.drop(columns=['timestamp'])

    if not df_jti.empty:
        df = pd.merge_asof(df, df_jti, left_on='departure_ts', right_on='timestamp', direction='nearest')
        df = df.drop(columns=['timestamp'])

    output_file = "data/processed/enriched_link_stats.csv"
    df.to_csv(output_file, index=False)
    print(f"Enriched data saved to {output_file}")
    print(df.head())

if __name__ == "__main__":
    main()
