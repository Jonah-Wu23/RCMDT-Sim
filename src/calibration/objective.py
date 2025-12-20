import pandas as pd
import numpy as np
import sys
import os

# 确保可以导入 scripts 目录下的 common_data
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts'))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds

def calculate_l1_rmse(sim_xml_path, real_links_csv, route_stop_dist_csv, route='68X', bound='I'):
    """
    计算站点级行程时间的 RMSE (J1 指标)
    """
    # 1. 加载数据
    sim_raw = load_sim_data(sim_xml_path)
    if sim_raw.empty:
        return 1e6 # 如果没有仿真数据，返回一个极大的惩罚值
        
    dist_df = load_route_stop_dist(route_stop_dist_csv)
    # 转换方向标识以匹配
    bound_map = {'I': 'inbound', 'O': 'outbound'}
    target_bound = bound_map.get(bound, bound)
    
    dist_df = dist_df[(dist_df['route'] == route) & (dist_df['bound'] == target_bound)]
    if dist_df.empty:
        raise ValueError(f"No stops found for route {route} and bound {bound} in {route_stop_dist_csv}")

    real_links = load_real_link_speeds(real_links_csv)
    real_links = real_links[(real_links['route'] == route) & (real_links['bound'] == target_bound)]
    
    # 2. 计算真实值 (累积行程时间)
    real_link_stats = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    real_link_stats = real_link_stats.sort_values('from_seq')
    real_cum_time = {1: 0}
    for _, row in real_link_stats.iterrows():
        f, t = row['from_seq'], row['to_seq']
        if f in real_cum_time:
            real_cum_time[t] = real_cum_time[f] + row['travel_time_s']
    
    real_time_df = pd.DataFrame(list(real_cum_time.items()), columns=['seq', 'real_time_s'])

    # 3. 处理仿真数据 (累积行程时间)
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    if sim_traj.empty:
        return 1e6
        
    sim_trips = []
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        min_seq = group['seq'].min()
        start_time = group.loc[group['seq'] == min_seq, 'arrival_time'].values[0]
        group['rel_time_s'] = group['arrival_time'] - start_time
        sim_trips.append(group[['seq', 'rel_time_s']])
    
    if not sim_trips:
        return 1e6

    sim_all = pd.concat(sim_trips)
    sim_stats = sim_all.groupby('seq')['rel_time_s'].mean().reset_index().rename(columns={'rel_time_s': 'sim_time_s'})
    
    # 4. 合并并计算 RMSE
    comparison = pd.merge(real_time_df, sim_stats, on='seq', how='inner')
    
    # 我们只关注序列大于 1 的点（seq=1 时 error 恒为 0）
    valid_comp = comparison[comparison['seq'] > 1]
    if valid_comp.empty:
        return 1e6
        
    rmse = np.sqrt(((valid_comp['sim_time_s'] - valid_comp['real_time_s']) ** 2).mean())
    return rmse

if __name__ == "__main__":
    # 简单的冒烟测试逻辑
    pass
