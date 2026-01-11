#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
l2_protocol_config.py
=====================
L2 口径统一配置文件

论文定义:
- 状态向量: x ∈ R³ = [capacityFactor, minGap, impatience]
- 观测向量: y ∈ R¹¹ = M11走廊 11条link 的 moving (traffic-only) 均速
- IES设置: Ne=10, K=3, damping β=0.3
- Rule C: T*=325s, v*=5km/h (跨线路/窗口固定)

所有实验必须复用此配置以保证口径一致性。

Author: RCMDT Project
Date: 2026-01-11
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

# =============================================================================
# 项目根目录
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# 观测向量配置 (y ∈ R¹¹)
# =============================================================================

@dataclass
class ObservationVectorConfig:
    """M11走廊观测向量配置"""
    name: str = "y_M11_moving"
    description: str = "M11走廊 11条link 的 moving (traffic-only) 均速"
    dimension: int = 11
    
    # 数据文件路径
    data_file: Path = field(default_factory=lambda: 
        PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11_moving_irn.csv")
    
    # link 定义 (10条68X + 1条960)
    links: List[Dict] = field(default_factory=lambda: [
        {"id": 1, "route": "68X", "bound": "inbound", "from_seq": 1, "to_seq": 2},
        {"id": 2, "route": "68X", "bound": "inbound", "from_seq": 3, "to_seq": 4},
        {"id": 3, "route": "68X", "bound": "inbound", "from_seq": 5, "to_seq": 6},
        {"id": 4, "route": "68X", "bound": "inbound", "from_seq": 6, "to_seq": 7},
        {"id": 5, "route": "68X", "bound": "inbound", "from_seq": 7, "to_seq": 8},
        {"id": 6, "route": "68X", "bound": "inbound", "from_seq": 8, "to_seq": 9},
        {"id": 7, "route": "68X", "bound": "inbound", "from_seq": 9, "to_seq": 10},
        {"id": 8, "route": "68X", "bound": "inbound", "from_seq": 10, "to_seq": 11},
        {"id": 9, "route": "68X", "bound": "inbound", "from_seq": 11, "to_seq": 12},
        {"id": 10, "route": "68X", "bound": "inbound", "from_seq": 12, "to_seq": 13},
        {"id": 22, "route": "960", "bound": "inbound", "from_seq": 1, "to_seq": 2},
    ])
    
    # 语义
    semantic: str = "moving"  # traffic-only, 排除运营停站


# =============================================================================
# 状态向量配置 (x ∈ R³)
# =============================================================================

@dataclass
class StateVectorConfig:
    """背景状态向量配置"""
    name: str = "x_corr"
    description: str = "Corridor background state vector"
    dimension: int = 3
    
    # 参数定义
    components: List[Dict] = field(default_factory=lambda: [
        {
            "name": "capacityFactor",
            "description": "路段通行能力因子",
            "bounds": [0.5, 3.0],
            "unit": "-",
            "prior_mean": 1.0,
            "prior_std": 0.3
        },
        {
            "name": "minGap",
            "description": "最小跟车间距 (背景车辆)",
            "bounds": [0.5, 5.0],
            "unit": "meters",
            "prior_mean": 2.5,
            "prior_std": 0.5
        },
        {
            "name": "impatience",
            "description": "驾驶员不耐烦度",
            "bounds": [0.0, 1.0],
            "unit": "-",
            "prior_mean": 0.5,
            "prior_std": 0.2
        }
    ])
    
    def get_prior_mean(self) -> np.ndarray:
        return np.array([c["prior_mean"] for c in self.components])
    
    def get_prior_std(self) -> np.ndarray:
        return np.array([c["prior_std"] for c in self.components])
    
    def get_bounds(self) -> np.ndarray:
        return np.array([c["bounds"] for c in self.components])


# =============================================================================
# IES 配置
# =============================================================================

@dataclass
class IESConfig:
    """Iterative Ensemble Smoother 配置"""
    ensemble_size: int = 10      # Ne = 10
    max_iterations: int = 3      # K = 3
    damping: float = 0.3         # β = 0.3
    
    # 观测误差
    obs_error_type: str = "diagonal"
    obs_error_source: str = "empirical"
    variance_floor: float = 1.0  # (km/h)²
    
    # 局部化
    localization: Optional[str] = "Patch-wise (L=16)"
    nugget_ratio: float = 0.05
    adaptive_damping: bool = True


# =============================================================================
# Rule C 审计配置
# =============================================================================

@dataclass
class RuleCConfig:
    """Rule C Ghost Jam 审计配置"""
    t_critical: float = 325.0    # T* = 325s
    speed_kmh: float = 5.0       # v* = 5 km/h
    max_dist_m: float = 1500.0   # 仅对短距离link生效
    
    description: str = "T*=325s, v*=5km/h (跨线路/窗口固定)"


# =============================================================================
# 场景配置
# =============================================================================

@dataclass
class ScenarioConfig:
    """实验场景配置"""
    name: str
    hkt_time: str
    utc8_start_sec: int
    utc8_end_sec: int
    duration_sec: int = 3600
    
    # 数据路径
    real_stats: Optional[Path] = None
    sim_stopinfo: Optional[Path] = None
    dist_csv: Optional[Path] = None


# PM Peak 场景 (论文主场景)
PM_PEAK_SCENARIO = ScenarioConfig(
    name="pm_peak",
    hkt_time="17:00-18:00",
    utc8_start_sec=61200,   # 17:00 in seconds from midnight
    utc8_end_sec=64800,     # 18:00 in seconds from midnight
    duration_sec=3600,
    real_stats=PROJECT_ROOT / "data" / "processed" / "link_stats.csv",
    dist_csv=PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
)

# Off-Peak 场景 (补充验证)
OFF_PEAK_SCENARIO = ScenarioConfig(
    name="off_peak",
    hkt_time="15:00-16:00",
    utc8_start_sec=54000,   # 15:00 in seconds from midnight
    utc8_end_sec=57600,     # 16:00 in seconds from midnight
    duration_sec=3600,
    real_stats=PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
    dist_csv=PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"
)


# =============================================================================
# 统一 L2 协议配置
# =============================================================================

@dataclass
class L2ProtocolConfig:
    """L2 口径统一配置 (论文定义)"""
    version: str = "1.0"
    
    # 核心组件
    observation: ObservationVectorConfig = field(default_factory=ObservationVectorConfig)
    state: StateVectorConfig = field(default_factory=StateVectorConfig)
    ies: IESConfig = field(default_factory=IESConfig)
    rule_c: RuleCConfig = field(default_factory=RuleCConfig)
    
    # 默认场景
    default_scenario: ScenarioConfig = field(default_factory=lambda: PM_PEAK_SCENARIO)
    
    def summary(self) -> str:
        """返回配置摘要"""
        return f"""
L2 Protocol Config v{self.version}
================================
观测向量: y ∈ R^{self.observation.dimension} ({self.observation.description})
状态向量: x ∈ R^{self.state.dimension} ({self.state.description})
IES: Ne={self.ies.ensemble_size}, K={self.ies.max_iterations}, β={self.ies.damping}
Rule C: {self.rule_c.description}
场景: {self.default_scenario.name} ({self.default_scenario.hkt_time})
"""


# =============================================================================
# 全局实例 (供其他模块导入)
# =============================================================================

L2_CONFIG = L2ProtocolConfig()
OBSERVATION_CONFIG = L2_CONFIG.observation
STATE_CONFIG = L2_CONFIG.state
IES_CONFIG = L2_CONFIG.ies
RULE_C_CONFIG = L2_CONFIG.rule_c


# =============================================================================
# 便捷函数
# =============================================================================

def load_observation_vector() -> np.ndarray:
    """加载 M11 走廊观测向量 (11维均速)"""
    import pandas as pd
    
    obs_file = OBSERVATION_CONFIG.data_file
    if not obs_file.exists():
        raise FileNotFoundError(f"观测向量文件不存在: {obs_file}")
    
    df = pd.read_csv(obs_file)
    return df['mean_speed_kmh'].values


def get_observation_links() -> List[Dict]:
    """获取观测向量对应的 link 列表"""
    return OBSERVATION_CONFIG.links


def get_state_bounds() -> np.ndarray:
    """获取状态向量边界"""
    return STATE_CONFIG.get_bounds()


def get_state_prior() -> tuple:
    """获取状态向量先验 (mean, std)"""
    return STATE_CONFIG.get_prior_mean(), STATE_CONFIG.get_prior_std()


if __name__ == "__main__":
    print(L2_CONFIG.summary())
    
    print("\n观测向量 (y):")
    try:
        y = load_observation_vector()
        print(f"  维度: {len(y)}")
        print(f"  值: {y}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
    
    print("\n状态向量 (x) 先验:")
    mean, std = get_state_prior()
    print(f"  mean: {mean}")
    print(f"  std: {std}")
    
    print("\n状态向量边界:")
    bounds = get_state_bounds()
    for i, c in enumerate(STATE_CONFIG.components):
        print(f"  {c['name']}: [{bounds[i, 0]}, {bounds[i, 1]}]")
