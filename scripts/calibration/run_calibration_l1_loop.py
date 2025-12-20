import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
import subprocess
import time
import shutil
from datetime import datetime

# 导入核心模块
# 假设脚本在 scripts/calibration/，src 在项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.calibration.surrogate import KrigingSurrogate
from src.calibration.objective import calculate_l1_rmse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class L1CalibrationLoop:
    def __init__(self, config_path, project_root):
        self.root = project_root
        with open(os.path.join(self.root, config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.params_meta = self.config['parameters']
        self.param_names = [p['name'] for p in self.params_meta]
        self.bounds = np.array([[p['min'], p['max']] for p in self.params_meta])
        
        # 结果记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.root, f'data/calibration/l1_calibration_log_{timestamp}.csv')
        ensure_dir(os.path.dirname(self.log_file))
        
        # 初始化代理模型
        self.surrogate = KrigingSurrogate(random_state=self.config['sampling'].get('seed', 42))
        
        # SUMO 配置路径
        self.sumocfg = os.path.join(self.root, 'sumo/config/experiment2_cropped.sumocfg')
        self.vtype_xml = os.path.join(self.root, 'sumo/routes/calibration_vtypes.add.xml')
        
        # 真实数据路径 (根据 experiment2.3 记录)
        self.real_links = os.path.join(self.root, 'data/processed/link_speeds.csv')
        self.route_dist = os.path.join(self.root, 'data/processed/kmb_route_stop_dist.csv')

    def update_route_xml(self, params_dict):
        """
        生成包含校准参数的路由文件，包括 vType 参数和停站 duration
        """
        t_board = params_dict.get('t_board', 2.0)
        # 基论 route 文件 (使用已验证适用于裁剪路网的版本)
        base_route = os.path.join(self.root, 'sumo/routes/fixed_routes_cropped.rou.xml')
        calib_route = os.path.join(self.root, 'sumo/routes/calibration.rou.xml')
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(base_route)
        root = tree.getroot()
        
        # 1. 更新 vType 参数
        for vtype in root.iter('vType'):
            if vtype.get('id') == 'kmb_double_decker':
                vtype.set('accel', f"{params_dict.get('accel', 2.6):.2f}")
                vtype.set('decel', f"{params_dict.get('decel', 4.5):.2f}")
                vtype.set('sigma', f"{params_dict.get('sigma', 0.5):.2f}")
                vtype.set('tau', f"{params_dict.get('tau', 1.0):.2f}")
                vtype.set('minGap', f"{params_dict.get('minGap', 2.5):.2f}")
        
        # 2. 更新停站 duration
        # 简化模型：dwell = t_board * N，N 从 17:00 晚高峰数据中采样或取常数
        # 实验记录显示平均停站约数十秒，假设平均每站上车 15 人
        passengers = 15 
        for stop in root.iter('stop'):
            if 'busStop' in stop.attrib:
                duration = t_board * passengers
                stop.set('duration', f"{duration:.2f}")
        
        tree.write(calib_route, encoding='utf-8', xml_declaration=True)

    def run_simulation(self):
        """
        运行 SUMO 仿真
        """
        calib_route = os.path.join(self.root, 'sumo/routes/calibration.rou.xml')
        # 背景交通流文件
        bg_route = os.path.join(self.root, 'sumo/routes/background_cropped.rou.xml')
        cmd = [
            "sumo", "-c", self.sumocfg,
            "--route-files", f"{calib_route},{bg_route}",
            "--additional-files", "sumo/additional/bus_stops_cropped.add.xml",
            "--stop-output", "sumo/output/stopinfo_calibration.xml",
            "--no-warnings",
            "--no-step-log",
            "--end", "3600"
        ]
        # 如果需要 t_board 影响，可以在这里通过脚本预处理 rou.xml
        # 示例：如果 t_board 变化，重新生成包含新 duration 的 .rou.xml
        
        start_time = time.time()
        subprocess.run(cmd, check=True, cwd=self.root)
        return time.time() - start_time

    def get_objective(self):
        """
        计算 RMSE
        """
        sim_xml = os.path.join(self.root, 'sumo/output/stopinfo_calibration.xml')
        # 目前主要校准 68X Inbound
        return calculate_l1_rmse(sim_xml, self.real_links, self.route_dist, route='68X', bound='I')

    def run(self, max_iters=20, n_init=10):
        # 1. 加载初始样本 (LHS)
        initial_csv = os.path.join(self.root, 'data/calibration/l1_initial_samples.csv')
        
        # 如果文件不存在，或者样本数量不符，自动调用生成脚本
        should_generate = False
        if not os.path.exists(initial_csv):
            print(f"Initial samples file not found. Generating {n_init} samples...")
            should_generate = True
        else:
            df_init = pd.read_csv(initial_csv)
            if len(df_init) != n_init:
                print(f"Sample count mismatch (found {len(df_init)}, expected {n_init}). Regenerating...")
                should_generate = True
        
        if should_generate:
            gen_script = os.path.join(self.root, 'scripts/calibration/generate_l1_samples.py')
            subprocess.run([sys.executable, gen_script, "--n_samples", str(n_init)], check=True)
             
        history = pd.read_csv(initial_csv)
        results = []
        
        # 处理初始样本 (确保只取前 n_init 个，防止手动修改后文件变长)
        history = history.head(n_init)
        
        for i, row in history.iterrows():
            print(f"--- Iteration {i+1}/{max_iters + n_init} (Initial Sample) ---")
            params = row[self.param_names].to_dict()
            self.update_route_xml(params)
            
            sim_time = self.run_simulation()
            rmse = self.get_objective()
            
            results.append({**params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'initial'})
            print(f"Result: RMSE = {rmse:.4f}, Time = {sim_time:.1f}s")
            
            # 实时保存
            pd.DataFrame(results).to_csv(self.log_file, index=False)

        # 2. BO 循环
        for i in range(n_init, max_iters + n_init):
            print(f"--- Iteration {i+1}/{max_iters + n_init} (BO Optimization) ---")
            
            # 使用所有现有数据训练代理模型
            # 重新读取以确保获取最新数据
            df_curr = pd.read_csv(self.log_file)
            X = df_curr[self.param_names].values
            y = df_curr['rmse'].values
            
            self.surrogate.fit(X, y)
            
            # 在参数空间内随机采样候选点以寻找 EI 最大值
            n_candidates = 2000
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_candidates, len(self.param_names)))
            
            best_y = np.min(y)
            ei = self.surrogate.expected_improvement(candidates, best_y)
            next_params_arr = candidates[np.argmax(ei)]
            
            next_params = dict(zip(self.param_names, next_params_arr))
            print(f"Suggested Params: {next_params}")
            
            # 运行并记录
            self.update_route_xml(next_params)
            sim_time = self.run_simulation()
            rmse = self.get_objective()
            
            results.append({**next_params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'bo'})
            print(f"Result: RMSE = {rmse:.4f}, Time = {sim_time:.1f}s")
            
            pd.DataFrame(results).to_csv(self.log_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--init_samples", type=int, default=10, help="Number of initial LHS samples")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    loop = L1CalibrationLoop("config/calibration/l1_parameter_config.json", project_root)
    loop.run(max_iters=args.iters, n_init=args.init_samples)
