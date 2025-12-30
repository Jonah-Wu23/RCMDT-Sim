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
from scipy.stats import qmc  # Quasi-Monte Carlo for Sobol sequence

# 确保导入路径正确：脚本在 scripts/calibration/，项目根目录在向上三级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.calibration.surrogate import KrigingSurrogate
from src.calibration.rbf_surrogate import RBFSurrogate
from src.calibration.dual_surrogate import DualSurrogate
from src.calibration.objective import calculate_l1_rmse, calculate_l1_robust_objective

def ensure_dir(file_path: str):
    """确保文件所在的目录存在"""
    directory = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

class L1CalibrationLoop:
    def __init__(self, config_path: str, project_root: str, label: str = "l1", seed: int = 42):
        self.root = project_root
        self.label = label
        
        # 加载参数配置
        full_config_path = os.path.join(self.root, config_path)
        with open(full_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.params_meta = self.config['parameters']
        self.param_names = [p['name'] for p in self.params_meta]
        self.bounds = np.array([[p['min'], p['max']] for p in self.params_meta])
        
        # 结果记录：使用 label 区分实验（无时间戳，便于覆盖重跑）
        self.log_file = os.path.join(self.root, f'data/calibration/{label}_log.csv')
        ensure_dir(self.log_file)
        
        # 初始化代理模型 (默认 Kriging，可通过参数切换)
        # 注意：代理模型类型在 run() 方法中根据 surrogate_type 参数动态设置
        self.surrogate = None  # 延迟初始化
        # 使用命令行传入的 seed，允许不同实验使用不同初始采样
        self.random_state = seed
        
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
        采用 Physics-Informed Model: Dwell = T_fixed + t_board * (N_base * W_stop)
        """
        import xml.etree.ElementTree as ET
        
        # 加载站点权重字典
        weights_path = os.path.join(self.root, 'config/calibration/bus_stop_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r', encoding='utf-8') as f:
                stop_weights = json.load(f)
        else:
            print(f"[WARN] Stop weights file not found, falling back to uniform weights.")
            stop_weights = {}

        if not os.path.exists(self.base_route):
            raise FileNotFoundError(f"Base route file not found: {self.base_route}")
            
        tree = ET.parse(self.base_route)
        root = tree.getroot()
        
        # 1. 更新 vType 参数 (Krauss 核心模型)
        for vtype in root.iter('vType'):
            if vtype.get('id') == 'kmb_double_decker':
                if 'accel' in params_dict: vtype.set('accel', f"{params_dict['accel']:.2f}")
                if 'decel' in params_dict: vtype.set('decel', f"{params_dict['decel']:.2f}")
                if 'sigma' in params_dict: vtype.set('sigma', f"{params_dict['sigma']:.2f}")
                if 'tau' in params_dict: vtype.set('tau', f"{params_dict['tau']:.2f}")
                if 'minGap' in params_dict: vtype.set('minGap', f"{params_dict['minGap']:.2f}")
        
        # 2. 更新停站 duration (物理一致性建模)
        # T_fixed: 固定开销 (开门/起步/安全确认)
        # N_base: 基准客流 (15人)
        T_fixed = params_dict.get('t_fixed', 5.0) 
        N_base = 15.0
        t_board = params_dict.get('t_board', 2.0)
        
        for stop in root.iter('stop'):
            stop_id = stop.get('busStop')
            if stop_id:
                # 获取该站的启发式客流权重, 默认为 1.0
                w_stop = stop_weights.get(stop_id, {}).get('weight', 1.0)
                # 核心公式
                duration = T_fixed + t_board * (N_base * w_stop)
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

    def get_objective(self) -> Dict[str, float]:
        """
        从仿真输出中计算损失函数 (Loss Function)
        策略: Constraint-Optimization
        1. Anchor: 960 误差不能恶化太多 (Threshold: +15% error relative to reality)
        2. Target: 全力优化 68X 的 RMSE
        """
        if not os.path.exists(self.sim_output):
            print(f"[ERROR] Simulation output missing: {self.sim_output}")
            return {'rmse': 1e6, 'rmse_68x': 1e6, 'rmse_960': 1e6}
            
        try:
            # 分别计算两条线路的 RMSE
            # calculate_l1_rmse 返回的是绝对秒数误差 (Seconds)
            rmse_68x = calculate_l1_rmse(self.sim_output, self.real_links, self.route_dist, route='68X', bound='I')
            rmse_960 = calculate_l1_rmse(self.sim_output, self.real_links, self.route_dist, route='960', bound='I')
            
            # 960 的容忍阈值 (Soft Constraint with Smooth Penalty)
            THRESHOLD_960 = 350.0 
            
            # 方案 B2 (Experiment 3.3): Priority-based Loss Function
            # KMB 68X: Target, KMB 960: Constraint Anchor (Threshold 350s)
            
            if rmse_960 <= THRESHOLD_960:
                loss = rmse_68x
                penalty = 0.0
            else:
                # 惩罚函数: 2000 + (Over-shoot) * 10
                penalty = 2000.0 + (rmse_960 - THRESHOLD_960) * 10.0
                loss = penalty # Note: The doc says loss = 2000 + ..., implied replacing rmse_68x or adding? 
                               # Doc code block: loss = 2000 + ... 
                               # This means when violated, loss is dominated effectively by penalty.
                               # Let's stick to the doc exactly: "loss = 2000 + (rmse_960 - 350) * 10.0"
            
            return {
                'rmse': loss,     # 这是 BO 看到并试图最小化的值
                'rmse_68x': rmse_68x,
                'rmse_960': rmse_960,
                'penalty': penalty  # 记录惩罚项便于调试
            }
        except Exception as e:
            print(f"[ERROR] Objective calculation failed: {e}")
            return {'rmse': 1e6, 'rmse_68x': 1e6, 'rmse_960': 1e6, 'penalty': 1e6}

    def run(self, max_iters: int = 10, n_init: int = 10, warm_start_log: Optional[str] = None,
            surrogate_type: str = 'kriging', objective_type: str = 'rmse',
            lambda_std: float = 0.5, ks_weight: float = 50.0):
        """
        开始完整校准循环：LHS 探索 -> BO 优化
        
        Args:
            max_iters: 贝叶斯优化迭代次数
            n_init: 初始样本数量（仅在非热启动模式下使用）
            warm_start_log: 热启动日志文件路径
            surrogate_type: 代理模型类型 ('kriging', 'rbf', 'dual')
            objective_type: 目标函数类型 ('rmse', 'robust')
            lambda_std: 鲁棒损失的 λ 参数
            ks_weight: K-S 统计量权重
        """
        # 初始化代理模型
        if surrogate_type == 'dual':
            self.surrogate = DualSurrogate(kriging_random_state=self.random_state)
            print(f"[CONFIG] 使用双代理融合模型 (Kriging + RBF 反比方差加权)")
        elif surrogate_type == 'rbf':
            self.surrogate = RBFSurrogate(kernel='gaussian', smoothing=0.1)
            print(f"[CONFIG] 使用 RBF 代理模型 (高斯核)")
        else:
            self.surrogate = KrigingSurrogate(random_state=self.random_state)
            print(f"[CONFIG] 使用 Kriging 代理模型 (Matern 核)")
        
        # 记录目标函数类型
        self.objective_type = objective_type
        self.lambda_std = lambda_std
        self.ks_weight = ks_weight
        if objective_type == 'robust':
            print(f"[CONFIG] 使用鲁棒目标函数 (λ={lambda_std}, K-S权重={ks_weight})")
        else:
            print(f"[CONFIG] 使用 RMSE 目标函数")
        
        results = []
        optimization_logs = []  # List to store GP variance logs per iteration
        
        if warm_start_log and os.path.exists(warm_start_log):
            # --- 热启动模式：复用历史数据 ---
            print(f"\n[WARM START] 加载历史校准日志: {warm_start_log}")
            df_warm = pd.read_csv(warm_start_log)
            
            # 复制所有历史数据点到 results（保留原始 type 标记）
            for _, row in df_warm.iterrows():
                record = row.to_dict()
                # 标记为热启动数据，便于后续区分
                if 'type' in record:
                    record['type'] = f"warm_{record['type']}"
                else:
                    record['type'] = 'warm_initial'
                results.append(record)
            
            # 保存到新日志文件（热启动数据作为基础）
            pd.DataFrame(results).to_csv(self.log_file, index=False)
            
            n_warm = len(results)
            print(f"[WARM START] 已加载 {n_warm} 个历史数据点，跳过初始仿真阶段")
            print(f"[WARM START] 历史最优 RMSE: {df_warm['rmse'].min():.2f}")
            
            # BO 阶段的起始迭代编号
            bo_start_iter = n_warm
            
        else:
            # --- 标准模式：动态生成 LHS 采样 ---
            # 使用 self.random_state (来自 --seed 参数) 控制随机性
            # 不同 seed 将产生不同的初始样本，确保消融实验独立性
            print(f"[PHASE 1] Generating {n_init} initial LHS samples (seed={self.random_state})...")
            
            n_dims = len(self.param_names)
            sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_state)
            samples_unit = sampler.random(n_init)  # [0, 1]^d
            samples = qmc.scale(samples_unit, self.bounds[:, 0], self.bounds[:, 1])
            df_init = pd.DataFrame(samples, columns=self.param_names)
            
            # --- 第一阶段：初始化评估 ---
            print(f"\n[PHASE 1] Evaluating {n_init} Initial Samples...")
            for i, row in df_init.iterrows():
                params = row[self.param_names].to_dict()
                self.update_route_xml(params)
                
                print(f"  > Iter {i+1}/{n_init + max_iters} [Initial]: ", end="", flush=True)
                sim_time = self.run_simulation()
                obj_metrics = self.get_objective()
                rmse = obj_metrics['rmse']
                
                # 记录详细结果
                record = {**params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'initial'}
                record.update(obj_metrics) # 添加 rmse_68x, rmse_960
                results.append(record)
                
                print(f"RMSE={rmse:.4f} (68X={obj_metrics['rmse_68x']:.1f}, 960={obj_metrics['rmse_960']:.1f})")
                
                # 增量保存，防止中途断电
                pd.DataFrame(results).to_csv(self.log_file, index=False)
            
            bo_start_iter = n_init

        # --- 第二阶段：贝叶斯优化 ---
        total_iters = bo_start_iter + max_iters
        print(f"\n[PHASE 2] Starting Bayesian Optimization ({max_iters} iterations)...")
        for i in range(bo_start_iter, total_iters):
            # 获取最新数据训练代理模型
            df_curr = pd.read_csv(self.log_file)
            X = df_curr[self.param_names].values
            y = df_curr['rmse'].values
            
            self.surrogate.fit(X, y)
            
            # 方案B：使用 Sobol 准蒙特卡洛序列 + 更多候选点
            # Sobol 序列比纯随机采样有更好的空间填充性
            n_candidates = 10000  # 从 2000 增加到 10000
            n_dims = len(self.param_names)
            
            # 使用 Sobol 序列生成均匀分布的候选点
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=42 + i)  # seed 随迭代变化增加多样性
            samples_unit = sampler.random(n_candidates)  # [0, 1]^d
            
            # 缩放到实际参数边界
            candidates = qmc.scale(samples_unit, self.bounds[:, 0], self.bounds[:, 1])
            
            best_y_so_far = np.min(y)
            ei = self.surrogate.expected_improvement(candidates, best_y_so_far)
            
            next_params_arr = candidates[np.argmax(ei)]
            next_params = dict(zip(self.param_names, next_params_arr))
            
            print(f"  > Iter {i+1}/{total_iters} [BO]: ", end="", flush=True)

            # Record GP variance statistics
            mu, sigma = self.surrogate.predict(candidates)
            sigma_mean = float(np.mean(sigma))
            sigma_max = float(np.max(sigma))
            sigma_min = float(np.min(sigma))
            
            opt_log_entry = {
                "iteration": i,
                "sigma_mean": sigma_mean,
                "sigma_max": sigma_max,
                "sigma_min": sigma_min,
                "best_rmse_so_far": float(best_y_so_far)
            }
            optimization_logs.append(opt_log_entry)

            self.update_route_xml(next_params)
            sim_time = self.run_simulation()
            obj_metrics = self.get_objective()
            rmse = obj_metrics['rmse']
            
            record = {**next_params, 'rmse': rmse, 'sim_time': sim_time, 'iter': i, 'type': 'bo'}
            record.update(obj_metrics)
            results.append(record)
            
            print(f"RMSE={rmse:.4f} (68X={obj_metrics['rmse_68x']:.1f}, 960={obj_metrics['rmse_960']:.1f})")
            
            pd.DataFrame(results).to_csv(self.log_file, index=False)
            
            # Save optimization logs incrementally
            opt_log_file = os.path.join(self.root, f'data/calibration/{self.label}_optimization_logs.json')
            with open(opt_log_file, 'w') as f:
                json.dump(optimization_logs, f, indent=4)

        print(f"\n[FINISH] Calibration complete. Log: {self.log_file}")
        return self.log_file

def main():
    parser = argparse.ArgumentParser(description="L1 Micro-parameters Calibration Loop (Week 3 Enhanced)")
    parser.add_argument("--iters", type=int, default=10, help="Number of BO iterations")
    parser.add_argument("--init_samples", type=int, default=10, help="Number of initial LHS samples")
    parser.add_argument("--no_plot", action="store_true", help="Do not generate plot automatically")
    parser.add_argument("--warm_start", type=str, default=None, 
                        help="Path to historical calibration log CSV for warm start.")
    
    # 实验标识与随机种子控制
    parser.add_argument("--label", type=str, default="l1",
                        help="Experiment label for output files (e.g., B2, B3). Affects log and plot filenames.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initial LHS sampling (default: 42). Different seeds produce different initial samples.")
    
    # Week 3 新增：代理模型选择
    parser.add_argument("--surrogate", type=str, default="kriging",
                        choices=["kriging", "rbf", "dual"],
                        help="Surrogate model type: kriging (default), rbf, or dual (inverse-variance weighted fusion)")
    
    # Week 3 新增：目标函数类型
    parser.add_argument("--objective", type=str, default="rmse",
                        choices=["rmse", "robust"],
                        help="Objective function type: rmse (default) or robust (mean + lambda*std + KS)")
    parser.add_argument("--lambda_std", type=float, default=0.5,
                        help="Lambda parameter for robust loss (default: 0.5)")
    parser.add_argument("--ks_weight", type=float, default=50.0,
                        help="Weight for K-S statistic in robust objective (default: 50.0)")
    
    args = parser.parse_args()
    
    loop = L1CalibrationLoop("config/calibration/l1_parameter_config.json", PROJECT_ROOT, 
                             label=args.label, seed=args.seed)
    
    try:
        log_path = loop.run(
            max_iters=args.iters, 
            n_init=args.init_samples, 
            warm_start_log=args.warm_start,
            surrogate_type=args.surrogate,
            objective_type=args.objective,
            lambda_std=args.lambda_std,
            ks_weight=args.ks_weight
        )
        
        # 自动调用可视化脚本，传入 label 参数
        if not args.no_plot:
            print("\n[INFO] Generating publication-quality plots...")
            plot_script = os.path.join(PROJECT_ROOT, "scripts/calibration/plot_calibration_results.py")
            subprocess.run([sys.executable, plot_script, "--log", log_path, "--label", args.label, "--save-pdf"], check=False)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Loop execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
