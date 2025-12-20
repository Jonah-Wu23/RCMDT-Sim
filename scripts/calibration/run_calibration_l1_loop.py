import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 确保导入路径正确：脚本在 scripts/calibration/，项目根目录在向上三级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.calibration.surrogate import KrigingSurrogate
from src.calibration.objective import calculate_l1_rmse

def ensure_dir(file_path: str):
    """确保文件所在的目录存在"""
    directory = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

class L1CalibrationLoop:
    def __init__(self, config_path: str, project_root: str):
        self.root = project_root
        
        # 加载参数配置
        full_config_path = os.path.join(self.root, config_path)
        with open(full_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.params_meta = self.config['parameters']
        self.param_names = [p['name'] for p in self.params_meta]
        self.bounds = np.array([[p['min'], p['max']] for p in self.params_meta])
        
        # 结果记录：使用时间戳区分实验，防止覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.root, f'data/calibration/l1_calibration_log_{timestamp}.csv')
        ensure_dir(self.log_file)
        
        # 初始化代理模型 (Kriging / Gaussian Process)
        self.surrogate = KrigingSurrogate(random_state=self.config.get('sampling', {}).get('seed', 42))
        
        # SUMO 相关路径 (使用绝对路径提高鲁棒性)
        self.sumocfg = os.path.join(self.root, 'sumo/config/experiment2_cropped.sumocfg')
        self.base_route = os.path.join(self.root, 'sumo/routes/fixed_routes_cropped.rou.xml')
        self.calib_route = os.path.join(self.root, 'sumo/routes/calibration.rou.xml')
        self.bg_route = os.path.join(self.root, 'sumo/routes/background_cropped.rou.xml')
        self.bus_stops = os.path.join(self.root, 'sumo/additional/bus_stops_cropped.add.xml')
        self.sim_output = os.path.join(self.root, 'sumo/output/stopinfo_calibration.xml')
        
        # 真实数据与路线辅助数据
        self.real_links = os.path.join(self.root, 'data/processed/link_speeds.csv')
        self.route_dist = os.path.join(self.root, 'data/processed/kmb_route_stop_dist.csv')

    def update_route_xml(self, params_dict: Dict[str, float]):
        """
        生成包含最新校准参数的路由文件。
        包括修改 vType (跟驰/变道参数) 和动态计算停站 duration。
        """
        import xml.etree.ElementTree as ET
        
        if not os.path.exists(self.base_route):
            raise FileNotFoundError(f"Base route file not found: {self.base_route}")
            
        tree = ET.parse(self.base_route)
        root = tree.getroot()
        
        # 1. 更新 vType 参数
        for vtype in root.iter('vType'):
            if vtype.get('id') == 'kmb_double_decker':
                # Krauss 核心模型参数
                if 'accel' in params_dict: vtype.set('accel', f"{params_dict['accel']:.2f}")
                if 'decel' in params_dict: vtype.set('decel', f"{params_dict['decel']:.2f}")
                if 'sigma' in params_dict: vtype.set('sigma', f"{params_dict['sigma']:.2f}")
                if 'tau' in params_dict: vtype.set('tau', f"{params_dict['tau']:.2f}")
                if 'minGap' in params_dict: vtype.set('minGap', f"{params_dict['minGap']:.2f}")
        
        # 2. 更新停站 duration (J1 校准核心：t_board * passengers)
        # TODO: 优化硬编码的乘客数逻辑
        passengers = 15 
        t_board = params_dict.get('t_board', 2.0)
        
        for stop in root.iter('stop'):
            if 'busStop' in stop.attrib:
                duration = t_board * passengers
                stop.set('duration', f"{duration:.2f}")
        
        tree.write(self.calib_route, encoding='utf-8', xml_declaration=True)

    def run_simulation(self):
        """执行 SUMO 仿真，返回运行耗时"""
        cmd = [
            "sumo", "-c", self.sumocfg,
            "--route-files", f"{self.calib_route},{self.bg_route}",
            "--additional-files", self.bus_stops,
            "--stop-output", self.sim_output,
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--end", "3600"
        ]
        
        start_time = time.time()
        # 捕获输出以防失败时无法定位原因
        result = subprocess.run(cmd, check=True, cwd=self.root, capture_output=True, text=True)
        return time.time() - start_time

    def get_objective(self) -> float:
        """从仿真输出中计算 RMSE"""
        if not os.path.exists(self.sim_output):
            print(f"[ERROR] Simulation output missing: {self.sim_output}")
            return 1e6
            
        try:
            # calculate_l1_rmse 返回站点级累积行程时间的 RMSE
            return calculate_l1_rmse(self.sim_output, self.real_links, self.route_dist, route='68X', bound='I')
        except Exception as e:
            print(f"[ERROR] Objective calculation failed: {e}")
            return 1e6

    def run(self, max_iters: int = 10, n_init: int = 10):
        """
        开始完整校准循环：LHS 探索 -> BO 优化
        """
        # 检查/生成初始样本
        initial_csv = os.path.join(self.root, 'data/calibration/l1_initial_samples.csv')
        if not os.path.exists(initial_csv):
            print(f"[INFO] Initial samples file missing. Triggering RHS sampling (N={n_init})...")
            gen_script = os.path.join(self.root, 'scripts/calibration/generate_l1_samples.py')
            subprocess.run([sys.executable, gen_script, "--n_samples", str(n_init)], check=True)
            
        df_init = pd.read_csv(initial_csv).head(n_init)
        results = []
        
        # --- 第一阶段：初始化评估 ---
        print(f"\n[PHASE 1] Evaluating {n_init} Initial Samples...")
        for i, row in df_init.iterrows():
            params = row[self.param_names].to_dict()
            self.update_route_xml(params)
            
            print(f"  > Iter {i+1}/{n_init + max_iters} [Initial]: ", end="", flush=True)
            sim_time = self.run_simulation()
            rmse = self.get_objective()
            
            results.append({**params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'initial'})
            print(f"RMSE={rmse:.4f} ({sim_time:.1f}s)")
            
            # 增量保存，防止中途断电
            pd.DataFrame(results).to_csv(self.log_file, index=False)

        # --- 第二阶段：贝叶斯优化 ---
        print(f"\n[PHASE 2] Starting Bayesian Optimization ({max_iters} iterations)...")
        for i in range(n_init, n_init + max_iters):
            # 获取最新数据训练代理模型
            df_curr = pd.read_csv(self.log_file)
            X = df_curr[self.param_names].values
            y = df_curr['rmse'].values
            
            self.surrogate.fit(X, y)
            
            # 使用 Expected Improvement (EI) 寻找下一个候选点
            n_candidates = 2000
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_candidates, len(self.param_names)))
            best_y_so_far = np.min(y)
            ei = self.surrogate.expected_improvement(candidates, best_y_so_far)
            
            next_params_arr = candidates[np.argmax(ei)]
            next_params = dict(zip(self.param_names, next_params_arr))
            
            print(f"  > Iter {i+1}/{n_init + max_iters} [BO]: ", end="", flush=True)
            self.update_route_xml(next_params)
            sim_time = self.run_simulation()
            rmse = self.get_objective()
            
            results.append({**next_params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'bo'})
            print(f"RMSE={rmse:.4f} ({sim_time:.1f}s)")
            
            pd.DataFrame(results).to_csv(self.log_file, index=False)

        print(f"\n[FINISH] Calibration complete. Log: {self.log_file}")
        return self.log_file

def main():
    parser = argparse.ArgumentParser(description="L1 Micro-parameters Calibration Loop")
    parser.add_argument("--iters", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--init_samples", type=int, default=10, help="Number of initial LHS samples")
    parser.add_argument("--no_plot", action="store_true", help="Do not generate plot automatically")
    args = parser.parse_args()
    
    loop = L1CalibrationLoop("config/calibration/l1_parameter_config.json", PROJECT_ROOT)
    
    try:
        log_path = loop.run(max_iters=args.iters, n_init=args.init_samples)
        
        # 自动调用可视化脚本
        if not args.no_plot:
            print("\n[INFO] Generating publication-quality plots...")
            plot_script = os.path.join(PROJECT_ROOT, "scripts/calibration/plot_calibration_results.py")
            # 调用 shell 命令以确保所有环境配置生效
            subprocess.run([sys.executable, plot_script, "--log", log_path, "--save-pdf"], check=False)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Loop execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
