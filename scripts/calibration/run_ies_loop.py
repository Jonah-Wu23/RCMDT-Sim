#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ies_loop.py
===============
B4 迭代式系综平滑 (Iterative Ensemble Smoother, IES) 循环

用于 L2 宏观参数校准：通过并行批量仿真，利用协方差信息更新参数分布。

核心流程:
    批量仿真 → 计算协方差 → 更新参数分布 → 下一轮批量

Author: Auto-generated for RMBC-Sim project
Date: 2025-12-22
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from scipy import stats

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 动态导入同目录下的模块
import importlib.util
def _import_from_file(module_name: str, file_path: Path):
    """从文件路径动态导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_parse_module = _import_from_file(
    "parse_edgedata_output",
    Path(__file__).parent / "parse_edgedata_output.py"
)
parse_edgedata_xml = _parse_module.parse_edgedata_xml

# 导入同口径仿真向量构建模块
_sim_vector_module = _import_from_file(
    "build_l2_simulation_vector",
    Path(__file__).parent / "build_l2_simulation_vector.py"
)
build_simulation_vector = _sim_vector_module.build_simulation_vector
parse_net_edge_lengths = _sim_vector_module.parse_net_edge_lengths


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def parse_summary_insertion_rate(summary_path: str) -> float:
    """从 summary.xml 解析插入率 (inserted / loaded)"""
    try:
        tree = ET.parse(summary_path)
        root = tree.getroot()
        steps = root.findall('step')
        if not steps:
            return 0.0
        last_step = steps[-1]
        inserted = int(last_step.get('inserted', 0))
        loaded = int(last_step.get('loaded', 0))
        return inserted / loaded if loaded > 0 else 0.0
    except Exception:
        return 0.0



class IESLoop:
    """
    迭代式系综平滑 (IES) 校准循环
    
    Attributes:
        ensemble_size (int): 系综规模 N
        max_iters (int): 最大迭代轮数 K
        label (str): 实验标签，用于输出文件命名
    """
    
    def __init__(
        self,
        project_root: str,
        label: str = "B4",
        ensemble_size: int = 20,
        max_iters: int = 5,
        seed: int = 42,
        use_baseline: bool = True,  # True: 使用默认 L1 参数（B4）, False: 使用 B2 优化参数（B5）
        # === IES 稳定性增强参数 ===
        es_mda_alpha: float = None,  # ES-MDA 噪声放大因子，默认 = max_iters
        update_damping: float = 0.3,  # 更新阻尼系数 β
        use_group_weights: bool = True,  # 使用组归一权重
        cyy_nugget_ratio: float = 0.05,  # Cyy nugget 正则化比率
        adaptive_damping: bool = True,  # 自适应阻尼（clip 后降低 β）
        obs_var_floor: float = 1.0,  # obs_var 下限 (km/h)^2
        t_min: float = 61200.0,  # 时间窗起始 (17:00)
        t_max: float = 64800.0,   # 时间窗结束 (18:00)
        tt_mode: str = 'moving',  # T Mode: moving vs door
        net_file: str = None  # 自定义网络文件路径
    ):
        self.root = Path(project_root)
        self.label = label
        self.ensemble_size = ensemble_size
        self.max_iters = max_iters
        self.seed = seed
        self.use_baseline = use_baseline
        
        # 时间窗配置
        self.t_min = t_min
        self.t_max = t_max
        self.tt_mode = tt_mode
        
        # IES 稳定性增强配置
        self.es_mda_alpha = es_mda_alpha if es_mda_alpha is not None else max_iters
        self.update_damping = update_damping
        self.use_group_weights = use_group_weights
        self.cyy_nugget_ratio = cyy_nugget_ratio
        self.adaptive_damping = adaptive_damping
        self.obs_var_floor = obs_var_floor ** 2  # 转换为方差
        self.current_damping = update_damping  # 运行时阻尼（可自适应调整）
        
        self.base_seed = seed  # 保存基础 seed
        np.random.seed(seed)  # 初始化用于 __init__ 阶段

        
        # 加载配置
        self._load_priors()
        self._load_l1_frozen_params()
        self._load_observation_vector()
        self._load_link_edge_mapping()
        
        # 输出目录
        self.output_dir = self.root / "sumo" / "output" / "ies_runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        self.log_path = self.root / "data" / "calibration" / f"{label}_ies_log.csv"
        ensure_dir(str(self.log_path))
        
        # SUMO 相关路径 (B4-Sanity: 统一使用研究区域口径)
        self.base_sumocfg = self.root / "sumo" / "config" / "experiment2_calibrated.sumocfg"
        if net_file:
            self.net_file = Path(net_file)
            print(f"[IES] 使用自定义网络文件: {self.net_file}")
        else:
            self.net_file = self.root / "sumo" / "net" / "hk_cropped.net.xml"  # 研究区域网络
        
        self.bus_stops = self.root / "sumo" / "additional" / "bus_stops_cropped.add.xml"  # cropped 站点
        self.bus_route = self.root / "sumo" / "routes" / "fixed_routes_cropped.rou.xml"  # cropped 路由
        self.bg_route_base = self.root / "sumo" / "routes" / "background_cropped.rou.xml"  # cropped 背景
        
        # 计算组归一权重
        self._compute_group_weights()
        
        print(f"[IES] 初始化完成")
        print(f"  - 系综规模 N = {self.ensemble_size}")
        print(f"  - 最大迭代轮数 K = {self.max_iters}")
        print(f"  - 随机种子 = {self.seed}")
        print(f"  - 参数维度 = {self.param_dim}")
        print(f"  - 观测维度 M = {len(self.obs_df)}")
        print(f"\n[IES] 稳定性增强配置:")
        print(f"  - ES-MDA alpha = {self.es_mda_alpha}")
        print(f"  - 更新阻尼 β = {self.update_damping}")
        print(f"  - 组归一权重 = {self.use_group_weights}")
        print(f"  - Cyy nugget ratio = {self.cyy_nugget_ratio}")
        print(f"  - 自适应阻尼 = {self.adaptive_damping}")
        print(f"  - obs_var 下限 = {np.sqrt(self.obs_var_floor):.1f} km/h")
        print(f"  - 时间窗过滤 = [{self.t_min:.1f}, {self.t_max:.1f}]")
        print(f"  - TT Mode = {self.tt_mode}")

        
        # Corridor 详情打印
        if 'route' in self.obs_df.columns and 'bound' in self.obs_df.columns:
            print(f"\n[IES] Corridor 详情:")
            for (route, bound), g in self.obs_df.groupby(['route', 'bound']):
                ids = g['observation_id'].tolist()
                w_total = self.group_weights[g.index].sum() if self.use_group_weights else len(g)
                print(f"  - {route} {bound}: {len(g)} links, 总权重={w_total:.2f} -> {ids[:5]}{'...' if len(ids) > 5 else ''}")

    def _load_priors(self) -> None:
        """加载 L2 先验参数分布"""
        priors_path = self.root / "config" / "calibration" / "l2_priors.json"
        with open(priors_path, 'r', encoding='utf-8') as f:
            priors = json.load(f)
        
        self.param_names = []
        self.mu = []
        self.sigma = []
        self.bounds = []
        
        for p in priors['parameters']:
            self.param_names.append(p['name'])
            self.mu.append(p['mu'])
            self.sigma.append(p['sigma'])
            self.bounds.append((p['min'], p['max']))
        
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)
        self.param_dim = len(self.param_names)
        
        # 构建对角协方差矩阵
        self.P = np.diag(self.sigma ** 2)
        
        # 观测噪声 (从配置读取)
        self.obs_noise_std = priors.get('ensemble_config', {}).get('observation_noise_std', 2.0)
        
        print(f"[IES] 加载先验分布: {self.param_names}")
        print(f"  - μ = {self.mu}")
        print(f"  - σ = {self.sigma}")

    def _load_l1_frozen_params(self) -> None:
        """
        加载冻结的 L1 微观参数
        
        use_baseline=True: 使用 baseline_parameters.json（B4 实验）
        use_baseline=False: 使用 best_l1_parameters.json（B5/最终实验）
        """
        if self.use_baseline:
            # B4: 使用默认 L1 参数
            l1_path = self.root / "config" / "calibration" / "baseline_parameters.json"
            with open(l1_path, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
            
            # 从 baseline 格式提取参数
            bus_params = baseline['micro_parameters']['kmb_double_decker']
            dwell = baseline['dwell_time_model']
            
            self.l1_params = {
                't_board': dwell['mean_s'] / 10,  # 估算每乘客上车时间
                't_fixed': dwell['mean_s'] * 0.1,  # 固定站停时间
                'tau': bus_params['tau'],
                'sigma': bus_params['sigma'],
                'minGap': bus_params['minGap'],
                'accel': bus_params['accel'],
                'decel': bus_params['decel']
            }
            print(f"[IES] 加载默认 L1 参数 (B1 基线): {self.l1_params}")
        else:
            # B5/最终: 使用 B2 优化的 L1 参数
            l1_path = self.root / "config" / "calibration" / "best_l1_parameters.json"
            with open(l1_path, 'r', encoding='utf-8') as f:
                l1_config = json.load(f)
            
            self.l1_params = l1_config['best_parameters']
            print(f"[IES] 加载优化 L1 参数 (B2): {self.l1_params}")

    def _load_observation_vector(self) -> None:
        """加载观测向量（优先使用 M11 / corridor 版本）"""
        # 优先级: M11 > corridor > full
        obs_path_m11 = self.root / "data" / "calibration" / "l2_observation_vector_corridor_M11.csv"
        obs_path_corridor = self.root / "data" / "calibration" / "l2_observation_vector_corridor.csv"
        obs_path_full = self.root / "data" / "calibration" / "l2_observation_vector.csv"
        
        if obs_path_m11.exists():
            obs_path = obs_path_m11
            print(f"[IES] 使用 M11 观测向量 (11点正式口径): {obs_path}")
        elif obs_path_corridor.exists():
            obs_path = obs_path_corridor
            print(f"[IES] 使用 corridor 观测向量: {obs_path}")
        else:
            obs_path = obs_path_full
            print(f"[IES] Fallback 到完整观测向量: {obs_path}")
        
        self.obs_csv_path = str(obs_path)  # 保存路径供后续使用
        self.obs_df = pd.read_csv(obs_path)
        
        # 关键：重置索引，确保 0..M-1 连续（避免权重索引错位）
        self.obs_df = self.obs_df.reset_index(drop=True)
        
        self.Y_obs = self.obs_df['mean_speed_kmh'].values
        self.obs_variance = self.obs_df['std_speed_kmh'].values ** 2
        
        # 处理方差为 0 或 NaN 的情况
        self.obs_variance = np.where(
            (self.obs_variance == 0) | np.isnan(self.obs_variance),
            self.obs_noise_std ** 2,
            self.obs_variance
        )
        
        print(f"[IES] 加载观测向量: {len(self.Y_obs)} 个观测点")

    def _load_link_edge_mapping(self) -> None:
        """加载链路-边映射表（优先使用 M11 / corridor 版本）"""
        # 优先级: M11 > corridor > full
        mapping_path_m11 = self.root / "config" / "calibration" / "link_edge_mapping_corridor_M11.csv"
        mapping_path_corridor = self.root / "config" / "calibration" / "link_edge_mapping_corridor.csv"
        mapping_path_full = self.root / "config" / "calibration" / "link_edge_mapping.csv"
        
        if mapping_path_m11.exists():
            mapping_path = mapping_path_m11
            print(f"[IES] 使用 M11 映射表 (11点正式口径): {mapping_path}")
        elif mapping_path_corridor.exists():
            mapping_path = mapping_path_corridor
            print(f"[IES] 使用 corridor 映射表: {mapping_path}")
        else:
            mapping_path = mapping_path_full
            print(f"[IES] Fallback 到完整映射表: {mapping_path}")
        
        self.mapping_csv_path = str(mapping_path)  # 保存路径供后续使用
        self.link_edge_mapping = {}  # Dict[observation_id, List[edge_id]]
        
        if not mapping_path.exists():
            print(f"[WARN] 链路-边映射表不存在: {mapping_path}，使用全网平均速度")
            return
        
        mapping_df = pd.read_csv(mapping_path)
        
        for _, row in mapping_df.iterrows():
            obs_id = int(row['observation_id'])
            edge_ids_str = row['edge_ids']
            
            try:
                # 解析 JSON 格式的边列表
                edge_ids = json.loads(edge_ids_str)
            except (json.JSONDecodeError, TypeError):
                edge_ids = []
            
            self.link_edge_mapping[obs_id] = edge_ids
        
        matched_count = sum(1 for v in self.link_edge_mapping.values() if v)
        total = len(self.link_edge_mapping)
        print(f"[IES] 加载链路-边映射: {matched_count}/{total} ({100*matched_count/total:.1f}%)")

    def _compute_group_weights(self) -> None:
        """
        计算组归一权重：每个 (route, bound) 组总权重相等
        
        解决 68X inbound (10 维) 霸权问题：
        - 4 个组各分配 25% 的总权重
        - 组内均分到每个 link
        """
        M = len(self.obs_df)
        self.group_weights = np.ones(M)
        
        if not self.use_group_weights:
            print(f"[IES] 组归一权重未启用，使用等权")
            return
        
        if 'route' not in self.obs_df.columns or 'bound' not in self.obs_df.columns:
            print(f"[WARN] 观测数据缺少 route/bound 列，使用等权")
            return
        
        # 计算每个组的权重
        n_groups = self.obs_df.groupby(['route', 'bound']).ngroups
        weight_per_group = 1.0 / n_groups  # 每组总权重（如 4 组则各 0.25）
        
        for (route, bound), g in self.obs_df.groupby(['route', 'bound']):
            indices = g.index.tolist()
            n_links = len(indices)
            w_per_link = weight_per_group / n_links
            for idx in indices:
                self.group_weights[idx] = w_per_link
        
        # 归一化到均值为 1（保持原始量纲）
        self.group_weights = self.group_weights * (M / self.group_weights.sum())
        
        print(f"[IES] 组归一权重计算完成: {n_groups} 组, 权重范围 [{self.group_weights.min():.3f}, {self.group_weights.max():.3f}]")

    def generate_ensemble(self, mu: np.ndarray, P: np.ndarray, iteration: int = 0) -> np.ndarray:
        """
        生成系综样本
        
        Args:
            mu: 当前参数均值 (dim,)
            P: 当前协方差矩阵 (dim, dim)
            iteration: 当前迭代轮次（用于 seed 递增）
        
        Returns:
            X_ensemble: 形状 (N, dim) 的参数矩阵
        """
        # B4-Sanity 修复：每轮迭代使用不同的 seed
        iter_seed = self.base_seed + iteration * 1000
        np.random.seed(iter_seed)
        print(f"[IES] 迭代 {iteration} 使用 seed = {iter_seed}")
        
        X_ensemble = np.random.multivariate_normal(mu, P, self.ensemble_size)
        
        # 边界裁剪
        for i, (low, high) in enumerate(self.bounds):
            X_ensemble[:, i] = np.clip(X_ensemble[:, i], low, high)
        
        return X_ensemble

    def _create_bg_route_with_params(
        self,
        run_id: str,
        params: Dict[str, float]
    ) -> str:
        """
        创建带有 L2 参数的背景车辆路由文件
        
        Args:
            run_id: 运行标识符
            params: L2 参数字典
        
        Returns:
            生成的路由文件路径
        """
        output_path = self.output_dir / run_id / "background_l2.rou.xml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析原始背景路由
        tree = ET.parse(self.bg_route_base)
        root = tree.getroot()
        
        # 查找或创建背景车辆 vType
        bg_vtype = None
        for vtype in root.findall('vType'):
            if vtype.get('id') in ['passenger', 'car', 'background']:
                bg_vtype = vtype
                break
        
        if bg_vtype is None:
            # 创建新的 vType
            bg_vtype = ET.Element('vType')
            bg_vtype.set('id', 'background')
            root.insert(0, bg_vtype)
        
        # 应用 L2 参数
        if 'minGap_background' in params:
            bg_vtype.set('minGap', f"{params['minGap_background']:.2f}")
        if 'impatience' in params:
            bg_vtype.set('impatience', f"{params['impatience']:.2f}")
        
        tree.write(str(output_path), encoding='utf-8', xml_declaration=True)
        return str(output_path)

    def _create_sumocfg(
        self,
        run_id: str,
        params: Dict[str, float],
        bg_route_path: str
    ) -> Tuple[str, str]:
        """
        创建 SUMO 配置文件
        
        Args:
            run_id: 运行标识符
            params: L2 参数字典
            bg_route_path: 背景路由文件路径
        
        Returns:
            (sumocfg_path, edgedata_output_path)
        """
        output_dir = self.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sumocfg_path = output_dir / f"{run_id}.sumocfg"
        edgedata_path = output_dir / "edgedata.out.xml"
        stopinfo_path = output_dir / "stopinfo.xml"
        
        # capacityFactor 通过 --scale 参数实现 (在运行时传递)
        
        # 创建 sumocfg XML
        config = ET.Element('configuration')
        config.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        config.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/sumoConfiguration.xsd')
        
        # input
        inp = ET.SubElement(config, 'input')
        ET.SubElement(inp, 'net-file').set('value', str(self.net_file))
        ET.SubElement(inp, 'route-files').set('value', f"{self.bus_route},{bg_route_path}")
        ET.SubElement(inp, 'additional-files').set('value', str(self.bus_stops))
        
        # time
        time_elem = ET.SubElement(config, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', '3600')
        
        # processing
        proc = ET.SubElement(config, 'processing')
        ET.SubElement(proc, 'ignore-route-errors').set('value', 'true')
        ET.SubElement(proc, 'time-to-teleport').set('value', '300')
        
        # report
        report = ET.SubElement(config, 'report')
        ET.SubElement(report, 'verbose').set('value', 'false')
        ET.SubElement(report, 'no-step-log').set('value', 'true')
        ET.SubElement(report, 'no-warnings').set('value', 'true')
        
        # output
        output = ET.SubElement(config, 'output')
        ET.SubElement(output, 'stop-output').set('value', str(stopinfo_path))
        
        # edgedata 通过 additional 文件配置
        edgedata_add_path = output_dir / "edgedata.add.xml"
        self._create_edgedata_additional(str(edgedata_add_path), str(edgedata_path))
        
        # 添加 edgedata additional
        inp_elem = config.find('input')
        add_files = inp_elem.find('additional-files')
        current_add = add_files.get('value')
        add_files.set('value', f"{current_add},{edgedata_add_path}")
        
        tree = ET.ElementTree(config)
        tree.write(str(sumocfg_path), encoding='utf-8', xml_declaration=True)
        
        return str(sumocfg_path), str(edgedata_path)

    def _create_edgedata_additional(self, add_path: str, output_path: str) -> None:
        """创建 edgedata 输出的 additional 文件 (只统计公交车)"""
        additional = ET.Element('additional')
        
        edgedata = ET.SubElement(additional, 'edgeData')
        edgedata.set('id', 'ies_edgedata')
        edgedata.set('file', output_path)
        edgedata.set('begin', '0')
        edgedata.set('end', '3600')
        edgedata.set('freq', '3600')  # 整个时段一个 interval
        # 关键修复：只统计公交车的速度（而不是全交通流）
        edgedata.set('vTypes', 'kmb_double_decker')
        edgedata.set('excludeEmpty', 'true')  # 排除公交没经过的边
        
        tree = ET.ElementTree(additional)
        tree.write(add_path, encoding='utf-8', xml_declaration=True)

    def generate_sumo_configs(
        self,
        iteration: int,
        X_ensemble: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        """
        为系综中的每个样本生成 SUMO 配置
        
        Args:
            iteration: 当前迭代轮次
            X_ensemble: 参数矩阵 (N, dim)
        
        Returns:
            List of (sumocfg_path, edgedata_path, scale_factor)
        """
        configs = []
        
        for i in range(self.ensemble_size):
            run_id = f"iter{iteration:02d}_run{i:02d}"
            params = {name: X_ensemble[i, j] for j, name in enumerate(self.param_names)}
            
            # 创建背景路由文件
            bg_route_path = self._create_bg_route_with_params(run_id, params)
            
            # 创建 sumocfg
            sumocfg_path, edgedata_path = self._create_sumocfg(run_id, params, bg_route_path)
            
            # capacityFactor 作为 scale 参数
            scale_factor = params.get('capacityFactor', 1.0)
            
            configs.append((sumocfg_path, edgedata_path, scale_factor))
        
        return configs

    def run_single_simulation(
        self,
        sumocfg_path: str,
        edgedata_path: str,
        scale_factor: float
    ) -> Tuple[Optional[str], float]:
        """
        运行单个 SUMO 仿真
        
        Args:
            sumocfg_path: 配置文件路径
            edgedata_path: edgedata 输出路径
            scale_factor: 流量缩放因子
        
        Returns:
            (edgedata_path, insertion_rate): 成功时返回路径和插入率，失败时返回 (None, 0.0)
        """
        # 推断 summary 路径
        run_dir = os.path.dirname(edgedata_path)
        summary_path = os.path.join(run_dir, "summary.xml")
        
        cmd = [
            "sumo",
            "-c", sumocfg_path,
            "--scale", f"{scale_factor:.3f}",
            "--summary-output", summary_path,
            "--stop-output", os.path.join(run_dir, "stopinfo.xml"),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 小时超时
            )
            
            if os.path.exists(edgedata_path):
                insertion_rate = parse_summary_insertion_rate(summary_path)
                return edgedata_path, insertion_rate
            else:
                print(f"[WARN] edgedata 文件未生成: {edgedata_path}")
                return None, 0.0
                
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 仿真失败: {sumocfg_path}")
            print(f"  stderr: {e.stderr[:500] if e.stderr else 'N/A'}")
            return None, 0.0
        except subprocess.TimeoutExpired:
            print(f"[ERROR] 仿真超时: {sumocfg_path}")
            return None, 0.0

    def run_parallel_simulations(
        self,
        configs: List[Tuple[str, str, float]]
    ) -> Tuple[List[Optional[str]], List[float]]:
        """
        并行运行所有仿真
        
        Args:
            configs: List of (sumocfg_path, edgedata_path, scale_factor)
        
        Returns:
            (edgedata_paths, insertion_rates)
        """
        # 限制最大并行度为 2 (适应 16G 内存)
        max_workers = min(2, self.ensemble_size)
        print(f"[IES] 启动并行仿真: {len(configs)} 个实例, {max_workers} 并行度")
        
        results = [(None, 0.0)] * len(configs)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.run_single_simulation,
                    cfg[0], cfg[1], cfg[2]
                ): i
                for i, cfg in enumerate(configs)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[ERROR] 仿真 {idx} 异常: {e}")
                    results[idx] = (None, 0.0)
        
        edgedata_paths = [r[0] for r in results]
        insertion_rates = [r[1] for r in results]
        
        success_count = sum(1 for p in edgedata_paths if p is not None)
        avg_insertion = np.mean([r for r in insertion_rates if r > 0]) if any(r > 0 for r in insertion_rates) else 0.0
        print(f"[IES] 仿真完成: {success_count}/{len(configs)} 成功, 平均插入率={avg_insertion:.2%}")
        
        return edgedata_paths, insertion_rates

    def collect_simulation_results(
        self,
        edgedata_paths: List[Optional[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集仿真结果，构建 Y_sim 矩阵（使用旅行时间加权调和平均，与观测同口径）
        
        Args:
            edgedata_paths: edgedata 文件路径列表
        
        Returns:
            (Y_sim, matched_masks): 
                Y_sim: 形状 (N, M) 的仿真结果矩阵
                matched_masks: 形状 (N, M) 的匹配掩码
        """
        M = len(self.obs_df)
        Y_sim = np.full((self.ensemble_size, M), np.nan)
        matched_masks = np.zeros((self.ensemble_size, M), dtype=bool)
        
        # 缓存边长度（只解析一次）
        if not hasattr(self, '_edge_lengths'):
            self._edge_lengths = parse_net_edge_lengths(str(self.net_file))
        
        for i, edgedata_path in enumerate(edgedata_paths):
            if edgedata_path is None or not os.path.exists(edgedata_path):
                continue
            
            try:
                # 强制使用 Travel Time 模式 (P12 D2D 验证)
                run_dir = os.path.dirname(edgedata_path)
                stopinfo_path = os.path.join(run_dir, "stopinfo.xml")
                
                # 使用同口径脚本构建仿真向量
                # 即使 obs 是 speed-based, build_sim_vector 会处理转换
                sim_df = build_simulation_vector(
                    edgedata_path=stopinfo_path, # 传 stopinfo 路径
                    observation_csv=self.obs_csv_path,
                    mapping_csv=self.mapping_csv_path,
                    net_file=str(self.net_file),
                    min_sampled_seconds=10.0,
                    t_min=self.t_min,
                    t_max=self.t_max,
                    verbose=False,
                    metric_type='traveltime', # 强制 Travel Time
                    tt_mode=self.tt_mode
                )
                
                # 提取速度和匹配状态
                Y_sim[i, :] = sim_df['sim_speed_kmh'].values
                matched_masks[i, :] = sim_df['matched'].values
                
                matched_count = matched_masks[i, :].sum()
                print(f"[IES] Run {i}: 匹配 {matched_count}/{M} 个观测点")
                
            except Exception as e:
                print(f"[WARN] 构建仿真向量失败 (run {i}): {e}")
        
        return Y_sim, matched_masks

    def ies_update(
        self,
        X_ensemble: np.ndarray,
        Y_sim: np.ndarray,
        matched_masks: np.ndarray,
        mu_old: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        执行 IES 更新步（使用 matched_mask 过滤 + 稳定性增强）
        
        稳定性增强:
        - ES-MDA: R_eff = alpha * R（分多口吃观测）
        - 组归一权重: 4 组各 25%
        - obs_var 下限: 防止某维度 R 过小
        - Cyy nugget: 数值稳定性
        - solve 代替 inv: 更稳定的求解
        - 更新阻尼: β * delta
        - 自适应阻尼: clip 后降低 β
        
        Args:
            X_ensemble: 参数矩阵 (N, dim)
            Y_sim: 仿真结果矩阵 (N, M)
            matched_masks: 匹配掩码 (N, M)
            mu_old: 旧参数均值 (dim,)
        
        Returns:
            (mu_new, stats_dict)
        """
        N = self.ensemble_size
        M = Y_sim.shape[1]
        
        # ============================================================
        # 维度固定策略：corridor 16 维不变，缺失值用 ensemble 均值填补
        # ============================================================
        
        # 1. 先对每个 observation 计算"有效 ensemble 数量"
        n_valid_per_obs = matched_masks.sum(axis=0)  # (M,) 每个 obs 有多少 ensemble 匹配
        
        # 2. 严格过滤：observation 必须有足够的 ensemble 匹配（至少 50%）
        min_ensemble_ratio = 0.5
        obs_usable = n_valid_per_obs >= int(self.ensemble_size * min_ensemble_ratio)
        n_usable_obs = obs_usable.sum()
        
        if n_usable_obs < int(M * 0.7):  # 至少 70% 的 corridor link 可用
            print(f"[WARN] 可用观测点不足 ({n_usable_obs}/{M}, 需要 {int(M*0.7)}), 跳过更新")
            return mu_old, {'rmse': np.nan, 'ks_distance': np.nan, 'n_valid_obs': n_usable_obs, 'clip_count': 0}
        
        print(f"[IES] 使用 {n_usable_obs}/{M} 个观测点进行更新 (维度固定)")
        
        # 3. 对 Y_sim 做缺失值填补：NaN 或 unmatched 用该 obs 的 ensemble 均值填补
        Y_sim_filled = Y_sim.copy()
        for j in range(M):
            if not obs_usable[j]:
                # 不可用的 obs 全部填补为 Y_obs（中性处理）
                Y_sim_filled[:, j] = self.Y_obs[j]
            else:
                # 可用的 obs：unmatched 的 ensemble 用 matched ensemble 均值填补
                valid_mask_j = matched_masks[:, j] & ~np.isnan(Y_sim[:, j])
                if valid_mask_j.sum() > 0:
                    fill_val = Y_sim[valid_mask_j, j].mean()
                    invalid_mask_j = ~valid_mask_j
                    Y_sim_filled[invalid_mask_j, j] = fill_val
        
        # 4. 处理 NaN: 使用有效样本（仿真成功的 ensemble）
        sample_valid_mask = ~np.any(np.isnan(Y_sim_filled), axis=1)
        if sample_valid_mask.sum() < 3:
            print(f"[WARN] 有效样本不足 ({sample_valid_mask.sum()}), 跳过更新")
            return mu_old, {'rmse': np.nan, 'ks_distance': np.nan, 'n_valid_obs': n_usable_obs, 'clip_count': 0}
        
        X_valid = X_ensemble[sample_valid_mask]
        Y_valid = Y_sim_filled[sample_valid_mask]
        N_valid = len(X_valid)
        
        # 5. 只在可用 obs 上计算统计量（但维度结构不变）
        Y_valid_usable = Y_valid[:, obs_usable]
        Y_obs_usable = self.Y_obs[obs_usable]
        obs_var_usable = self.obs_variance[obs_usable].copy()
        
        # ============================================================
        # 稳定性增强 (1): obs_var 下限
        # ============================================================
        obs_var_usable = np.maximum(obs_var_usable, self.obs_var_floor)
        
        # ============================================================
        # 稳定性增强 (2): 组归一权重
        # ============================================================
        if self.use_group_weights:
            weights_usable = self.group_weights[obs_usable]
            # 权重大 → 噪声小 → 更"听它的"
            obs_var_weighted = obs_var_usable / weights_usable
        else:
            obs_var_weighted = obs_var_usable
        
        # 6. 计算系综均值
        Y_bar = np.mean(Y_valid_usable, axis=0)  # (n_usable,)
        X_bar = np.mean(X_valid, axis=0)  # (dim,)
        
        # 7. 计算协方差
        X_anom = X_valid - X_bar  # (N_valid, dim)
        Y_anom = Y_valid_usable - Y_bar  # (N_valid, n_usable)
        
        C_xf = (1 / (N_valid - 1)) * X_anom.T @ Y_anom  # (dim, n_usable)
        C_ff = (1 / (N_valid - 1)) * Y_anom.T @ Y_anom  # (n_usable, n_usable)
        
        # ============================================================
        # 稳定性增强 (3): Cyy nugget 正则化
        # ============================================================
        if self.cyy_nugget_ratio > 0:
            nugget = self.cyy_nugget_ratio * np.mean(np.diag(C_ff))
            C_ff_reg = C_ff + nugget * np.eye(C_ff.shape[0])
        else:
            C_ff_reg = C_ff
        
        # ============================================================
        # 稳定性增强 (4): ES-MDA 噪声放大
        # ============================================================
        R_eff = np.diag(obs_var_weighted) * self.es_mda_alpha  # R_eff = alpha * R
        
        # ============================================================
        # 稳定性增强 (5): 用 solve 代替 inv（更稳定）
        # ============================================================
        S = C_ff_reg + R_eff  # (n_usable, n_usable)
        innovation = Y_obs_usable - Y_bar  # (n_usable,)
        
        try:
            # delta = C_xf @ inv(S) @ innovation = C_xf @ solve(S, innovation)
            delta = C_xf @ np.linalg.solve(S, innovation)  # (dim,)
        except np.linalg.LinAlgError:
            print("[WARN] solve 失败，使用 lstsq")
            delta = C_xf @ np.linalg.lstsq(S, innovation, rcond=None)[0]
        
        # ============================================================
        # 稳定性增强 (6): 更新阻尼
        # ============================================================
        mu_new_raw = mu_old + self.current_damping * delta  # 带阻尼的更新
        
        # 记录更新前后的变化（用于诊断）
        delta_actual = mu_new_raw - mu_old
        print(f"[IES] 原始更新 (damping={self.current_damping:.2f}):")
        for i, name in enumerate(self.param_names):
            print(f"      {name}: {mu_old[i]:.4f} + {delta_actual[i]:.4f} = {mu_new_raw[i]:.4f}")
        
        # 边界裁剪并计数
        mu_new = mu_new_raw.copy()
        clip_count = 0
        for i, (low, high) in enumerate(self.bounds):
            if mu_new[i] < low:
                mu_new[i] = low
                clip_count += 1
                print(f"[WARN] {self.param_names[i]} 被下界裁剪: {mu_new_raw[i]:.4f} -> {low}")
            elif mu_new[i] > high:
                mu_new[i] = high
                clip_count += 1
                print(f"[WARN] {self.param_names[i]} 被上界裁剪: {mu_new_raw[i]:.4f} -> {high}")
        
        # ============================================================
        # 稳定性增强 (7): 自适应阻尼（clip 后降低 β）
        # ============================================================
        if self.adaptive_damping and clip_count > 0:
            old_damping = self.current_damping
            self.current_damping = max(0.1, self.current_damping * 0.5)
            print(f"[IES] 自适应阻尼: {old_damping:.2f} -> {self.current_damping:.2f} (因 {clip_count} 次 clip)")
        
        # 计算统计量
        residuals = Y_obs_usable - Y_bar
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # ============================================================
        # 口径诊断打印（Step 0: 强制口径自检）
        # ============================================================
        n_diag = min(5, len(Y_obs_usable))
        print(f"\n[IES] ====== 口径诊断 ======")
        print(f"[IES] 有效观测点: {n_usable_obs}/{M}")
        
        # 缺失观测点 ID
        missing_ids = self.obs_df.loc[~obs_usable, 'observation_id'].tolist()
        if missing_ids:
            print(f"[IES] 缺失观测点 ID: {missing_ids}")
        
        # min/median/max 对比
        print(f"[IES] Y_obs: min={Y_obs_usable.min():.2f}, median={np.median(Y_obs_usable):.2f}, max={Y_obs_usable.max():.2f} km/h")
        print(f"[IES] Y_bar: min={Y_bar.min():.2f}, median={np.median(Y_bar):.2f}, max={Y_bar.max():.2f} km/h")
        
        # 比值诊断（关键！看数量级）
        ratio = np.median(Y_bar) / np.median(Y_obs_usable) if np.median(Y_obs_usable) > 0 else float('nan')
        print(f"[IES] Y_bar/Y_obs 中位数比值: {ratio:.2f}x")
        if ratio > 2 or ratio < 0.5:
            print(f"[WARN] ⚠️ 数量级差距过大！可能存在口径不一致（单位/统计对象）")
        
        print(f"[IES] Y_obs[:5] = {Y_obs_usable[:n_diag].round(2).tolist()}")
        print(f"[IES] Y_bar[:5] = {Y_bar[:n_diag].round(2).tolist()}")
        print(f"[IES] =======================\n")
        
        try:
            ks_stat, _ = stats.ks_2samp(Y_bar, Y_obs_usable)
        except Exception:
            ks_stat = np.nan
        
        return mu_new, {
            'rmse': rmse,
            'ks_distance': ks_stat,
            'n_valid_obs': n_usable_obs,
            'clip_count': clip_count,
            'damping_used': self.current_damping
        }


    def run(self) -> Dict[str, float]:
        """
        执行完整的 IES 循环
        
        Returns:
            最终校准参数字典
        """
        print(f"\n{'='*60}")
        print(f"[IES] 开始 IES 校准循环")
        print(f"{'='*60}\n")
        
        # 初始化日志
        log_records = []
        
        mu_current = self.mu.copy()
        P_current = self.P.copy()
        
        for k in range(self.max_iters):
            print(f"\n[IES] ========== 迭代 {k+1}/{self.max_iters} ==========")
            
            # 3.2 系综生成
            print(f"[IES] 当前参数均值: {dict(zip(self.param_names, mu_current))}")
            X_ensemble = self.generate_ensemble(mu_current, P_current, iteration=k+1)
            print(f"[IES] 生成 {self.ensemble_size} 个系综样本 (seed={self.base_seed + (k+1)*1000})")
            
            # 3.3 生成配置并并行仿真
            configs = self.generate_sumo_configs(k + 1, X_ensemble)
            edgedata_paths, insertion_rates = self.run_parallel_simulations(configs)
            
            # 护栏 A: 插入率惩罚 (低于 50% 标记为无效)
            for i, ir in enumerate(insertion_rates):
                if ir < 0.50:
                    print(f"[IES] Run {i}: 插入率 {ir:.2%} < 50%, 标记为无效")
                    edgedata_paths[i] = None  # 使该样本在 collect 时被跳过
            
            # 3.4 收集结果
            Y_sim, matched_masks = self.collect_simulation_results(edgedata_paths)
            
            # 3.5 IES 更新
            mu_new, stats = self.ies_update(X_ensemble, Y_sim, matched_masks, mu_current)
            
            # 日志记录
            record = {
                'iteration': k + 1,
                'rmse': stats['rmse'],
                'ks_distance': stats['ks_distance'],
                'n_valid_obs': stats.get('n_valid_obs', 16),
                'clip_count': stats.get('clip_count', 0),
                'damping_used': stats.get('damping_used', self.update_damping)
            }
            for j, name in enumerate(self.param_names):
                record[f'{name}_mu'] = mu_new[j]
            log_records.append(record)
            
            print(f"[IES] 更新后参数: {dict(zip(self.param_names, mu_new))}")
            print(f"[IES] RMSE = {stats['rmse']:.4f}, K-S = {stats['ks_distance']:.4f}, 有效观测 = {stats.get('n_valid_obs', 16)}, clip = {stats.get('clip_count', 0)}")
            
            mu_current = mu_new
            
            # 可选: 收缩协方差
            # P_current = P_current * 0.9
        
        # 保存日志
        log_df = pd.DataFrame(log_records)
        log_df.to_csv(self.log_path, index=False)
        print(f"\n[IES] 日志已保存: {self.log_path}")
        
        # 返回最终参数
        final_params = {name: mu_current[j] for j, name in enumerate(self.param_names)}
        print(f"\n[IES] 最终校准参数: {final_params}")
        
        return final_params


def main():
    parser = argparse.ArgumentParser(
        description='B4 IES 校准循环 - 迭代式系综平滑'
    )
    parser.add_argument(
        '--label', '-l',
        type=str,
        default='B4',
        help='实验标签 (默认: B4)'
    )
    parser.add_argument(
        '--ensemble-size', '-n',
        type=int,
        default=20,
        help='系综规模 (默认: 20)'
    )
    parser.add_argument(
        '--max-iters', '-k',
        type=int,
        default=5,
        help='最大迭代轮数 (默认: 5)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--use-baseline', '-b',
        action='store_true',
        help='使用默认 L1 参数 (B4 实验)，否则使用 B2 优化参数 (B5/最终)'
    )
    
    # === IES 稳定性增强参数 ===
    parser.add_argument(
        '--es-mda-alpha',
        type=float,
        default=None,
        help='ES-MDA 噪声放大因子 (默认: max_iters)'
    )
    parser.add_argument(
        '--damping',
        type=float,
        default=0.3,
        help='更新阻尼系数 β (默认: 0.3)'
    )
    parser.add_argument(
        '--no-group-weights',
        action='store_true',
        help='禁用组归一权重'
    )
    parser.add_argument(
        '--cyy-nugget',
        type=float,
        default=0.05,
        help='Cyy nugget 正则化比率 (默认: 0.05)'
    )
    parser.add_argument(
        '--no-adaptive-damping',
        action='store_true',
        help='禁用自适应阻尼'
    )
    parser.add_argument(
        '--obs-var-floor',
        type=float,
        default=1.0,
        help='obs_var 下限 (km/h) (默认: 1.0)'
    )
    parser.add_argument(
        '--tmin',
        type=float,
        default=61200.0,
        help='时间窗起始 (秒), 默认 61200 (17:00)'
    )
    parser.add_argument(
        '--tmax',
        type=float,
        default=64800.0,
        help='时间窗结束 (秒), 默认 64800 (18:00)'
    )
    
    parser.add_argument(
        '--tt-mode',
        type=str,
        default='moving',
        choices=['moving', 'door'],
        help='Travel Time 计算口径 (仅对 traveltime 指标有效)'
    )
    parser.add_argument(
        "--net-file",
        type=str,
        default=None,
        help="自定义网络文件路径"
    )
    
    args = parser.parse_args()
    
    ies = IESLoop(
        project_root=str(PROJECT_ROOT),
        label=args.label,
        ensemble_size=args.ensemble_size,
        max_iters=args.max_iters,
        seed=args.seed,
        use_baseline=args.use_baseline,
        # 稳定性增强参数
        es_mda_alpha=args.es_mda_alpha,
        update_damping=args.damping,
        use_group_weights=not args.no_group_weights,
        cyy_nugget_ratio=args.cyy_nugget,
        adaptive_damping=not args.no_adaptive_damping,
        obs_var_floor=args.obs_var_floor,
        t_min=args.tmin,
        t_max=args.tmax,
        tt_mode=args.tt_mode,  # Pass tt_mode from CLI args
        net_file=args.net_file # Pass net_file from CLI args
    )
    
    final_params = ies.run()
    
    print(f"\n{'='*60}")
    print(f"[IES] B4 校准完成!")
    print(f"{'='*60}")
    for name, value in final_params.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
