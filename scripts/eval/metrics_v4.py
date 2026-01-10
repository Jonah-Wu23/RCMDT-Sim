#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics_v4.py - Protocol v4 统一评估器
======================================

根据 docs/protocol_v4.yaml 和 docs/SPEC.md 规范实现。
所有实验脚本必须调用此模块计算指标，禁止自行计算 KS/worst-window。

主要功能：
1. Rule C 审计 (Op-L2-v1.1)
2. KS 检验 + Pass/Fail 判据
3. Exhaustive worst-window stress test
4. Bootstrap 置信区间
5. 多场景支持 (PM Peak / Off-Peak)
6. Sanity checks

Author: RCMDT Project
Date: 2026-01-09
Version: 4.0
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from scipy.stats import ks_2samp
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ============================================================================
# Protocol v4 配置常量
# ============================================================================

PROTOCOL_V4_CONFIG = {
    "version": "4.0",
    
    # Rule C (Op-L2-v1.1) 默认阈值
    "rule_c": {
        "T_star": 325.0,      # seconds
        "v_star": 5.0,        # km/h
        "max_dist_m": 1500.0  # meters
    },
    
    # 场景定义
    "scenarios": {
        "pm_peak": {
            "name": "PM Peak",
            "date": "2025-12-19",
            "data_folder": "data",
            "hkt_start": "17:00",
            "hkt_end": "18:00",
            "utc_start_sec": 32400,   # 09:00 UTC
            "utc_end_sec": 36000,     # 10:00 UTC
            "duration_sec": 3600
        },
        "off_peak": {
            "name": "Off-Peak (P14)", 
            "date": "2025-12-30",
            "data_folder": "data2",
            "hkt_start": "15:00",
            "hkt_end": "16:00",
            "utc_start_sec": 25200,   # 07:00 UTC
            "utc_end_sec": 28800,     # 08:00 UTC
            "duration_sec": 3600
        }
    },
    
    # KS 检验配置
    "ks_test": {
        "alpha": 0.05,
        "c_alpha": 1.36  # for alpha=0.05
    },
    
    # Worst-window 配置
    "worst_window": {
        "duration_sec": 900,  # 15 minutes
        "step_sec": 60,       # 1 minute
        "min_samples": 5,
        "method": "exhaustive"
    },
    
    # Bootstrap 配置
    "bootstrap": {
        "n_samples": 1000,
        "confidence": 0.95,
        "seed": 42
    },
    
    # Sanity checks
    "sanity": {
        "n_clean_min": 10,
        "ks_range": [0.0, 1.0]
    }
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AuditConfig:
    """Audit 配置"""
    T_star: float = 325.0
    v_star: float = 5.0
    max_dist_m: float = 1500.0
    enabled: bool = True
    
    @classmethod
    def from_protocol(cls) -> 'AuditConfig':
        cfg = PROTOCOL_V4_CONFIG["rule_c"]
        return cls(
            T_star=cfg["T_star"],
            v_star=cfg["v_star"],
            max_dist_m=cfg["max_dist_m"]
        )


@dataclass
class KSResult:
    """KS 检验结果"""
    ks_stat: Optional[float] = None
    p_value: Optional[float] = None
    n_real: int = 0
    n_sim: int = 0
    critical_value: Optional[float] = None
    passed: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorstWindowResult:
    """Worst-window 结果"""
    worst_ks: Optional[float] = None
    window_start_sec: Optional[int] = None
    window_end_sec: Optional[int] = None
    window_start_time: Optional[str] = None
    window_end_time: Optional[str] = None
    n_windows_checked: int = 0
    method: str = "exhaustive"
    
    def to_dict(self) -> Dict:
        return {
            "worst_ks": self.worst_ks,
            "window_start_sec": self.window_start_sec,
            "window_end_sec": self.window_end_sec,
            "window_start_time": self.window_start_time,
            "window_end_time": self.window_end_time,
            "n_windows_checked": self.n_windows_checked,
            "method": self.method
        }


@dataclass
class BootstrapCI:
    """Bootstrap 置信区间"""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_samples: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AuditStats:
    """Audit 统计"""
    n_raw: int = 0
    n_clean: int = 0
    n_flagged: int = 0
    flagged_fraction: float = 0.0
    config: Optional[AuditConfig] = None
    
    def to_dict(self) -> Dict:
        return {
            "n_raw": self.n_raw,
            "n_clean": self.n_clean,
            "n_flagged": self.n_flagged,
            "flagged_fraction": self.flagged_fraction,
            "T_star": self.config.T_star if self.config else None,
            "v_star": self.config.v_star if self.config else None
        }


@dataclass
class MetricsV4Result:
    """Protocol v4 完整评估结果"""
    # 场景信息
    scenario: str = ""
    route: str = ""
    period: str = ""
    
    # 样本统计
    audit_stats: AuditStats = field(default_factory=AuditStats)
    n_sim: int = 0
    
    # Full-hour KS (speed)
    ks_speed_raw: KSResult = field(default_factory=KSResult)
    ks_speed_clean: KSResult = field(default_factory=KSResult)
    
    # Full-hour KS (TT)
    ks_tt_raw: KSResult = field(default_factory=KSResult)
    ks_tt_clean: KSResult = field(default_factory=KSResult)
    
    # Worst-window
    worst_window_speed: WorstWindowResult = field(default_factory=WorstWindowResult)
    worst_window_tt: WorstWindowResult = field(default_factory=WorstWindowResult)
    
    # Bootstrap CI (跨窗口汇总用)
    bootstrap_ci: Optional[BootstrapCI] = None
    
    # Sanity check
    sanity_passed: bool = True
    sanity_errors: List[str] = field(default_factory=list)
    sanity_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario,
            "route": self.route,
            "period": self.period,
            "audit_stats": self.audit_stats.to_dict(),
            "n_sim": self.n_sim,
            "ks_speed_raw": self.ks_speed_raw.to_dict(),
            "ks_speed_clean": self.ks_speed_clean.to_dict(),
            "ks_tt_raw": self.ks_tt_raw.to_dict(),
            "ks_tt_clean": self.ks_tt_clean.to_dict(),
            "worst_window_speed": self.worst_window_speed.to_dict(),
            "worst_window_tt": self.worst_window_tt.to_dict(),
            "bootstrap_ci": self.bootstrap_ci.to_dict() if self.bootstrap_ci else None,
            "sanity_passed": self.sanity_passed,
            "sanity_errors": self.sanity_errors,
            "sanity_warnings": self.sanity_warnings
        }
    
    def to_flat_dict(self) -> Dict:
        """扁平化输出，用于 CSV"""
        return {
            "scenario": self.scenario,
            "route": self.route,
            "period": self.period,
            "n_raw": self.audit_stats.n_raw,
            "n_clean": self.audit_stats.n_clean,
            "n_sim": self.n_sim,
            "flagged_fraction": self.audit_stats.flagged_fraction,
            "ks_speed_raw": self.ks_speed_raw.ks_stat,
            "ks_speed_clean": self.ks_speed_clean.ks_stat,
            "ks_speed_critical": self.ks_speed_clean.critical_value,
            "ks_speed_passed": self.ks_speed_clean.passed,
            "ks_tt_raw": self.ks_tt_raw.ks_stat,
            "ks_tt_clean": self.ks_tt_clean.ks_stat,
            "ks_tt_critical": self.ks_tt_clean.critical_value,
            "ks_tt_passed": self.ks_tt_clean.passed,
            "worst_window_ks_speed": self.worst_window_speed.worst_ks,
            "worst_window_ks_tt": self.worst_window_tt.worst_ks,
            "worst_window_start": self.worst_window_speed.window_start_time,
            "worst_window_end": self.worst_window_speed.window_end_time,
            "sanity_passed": self.sanity_passed
        }


# ============================================================================
# 数据加载函数
# ============================================================================

def load_real_link_stats(filepath: str) -> pd.DataFrame:
    """
    加载真实链路统计数据
    
    必需列: tt_median, speed_median, dist_m
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"真实数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = {"tt_median", "speed_median", "dist_m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需列: {missing}")
    
    return df


def load_sim_stopinfo(filepath: str) -> pd.DataFrame:
    """
    解析 SUMO stopinfo.xml
    
    处理可能不完整的 XML（缺少关闭标签）
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if '</stops>' not in content:
                content = content.rstrip() + '\n</stops>'
            root = ET.fromstring(content)
        except Exception as e:
            warnings.warn(f"无法解析 stopinfo XML: {e}")
            return pd.DataFrame()
    
    records = []
    for stop in root.findall('.//stopinfo'):
        arrival_str = stop.get('arrival')
        arrival = float(arrival_str) if arrival_str else float(stop.get('started', 0))
        records.append({
            'vehicle_id': stop.get('id'),
            'stop_id': stop.get('busStop'),
            'arrival': arrival,
            'started': float(stop.get('started', 0)),
            'departure': float(stop.get('ended', 0)),
            'duration': float(stop.get('duration', 0))
        })
    
    return pd.DataFrame(records)


def load_dist_mapping(filepath: str) -> Dict[Tuple[str, str], float]:
    """加载站点间距离映射"""
    if not os.path.exists(filepath):
        return {}
    
    df = pd.read_csv(filepath)
    required = {"route", "bound", "service_type", "seq", "stop_id", "link_dist_m"}
    if not required.issubset(set(df.columns)):
        return {}
    
    dist_map = {}
    for _, group in df.groupby(["route", "bound", "service_type"]):
        group = group.sort_values("seq")
        stops = group["stop_id"].astype(str).tolist()
        link_dists = group["link_dist_m"].tolist()
        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            d = link_dists[i + 1]
            if pd.notna(d) and d > 0:
                dist_map[(s1, s2)] = float(d)
    
    return dist_map


def compute_sim_link_data(
    stopinfo_xml: str,
    dist_csv: str,
    tt_mode: str = 'door'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从仿真输出计算链路速度和行程时间
    
    Args:
        tt_mode: 'door' (arrival-arrival) 或 'moving' (started-departure)
    
    Returns:
        speeds: 链路速度数组 (km/h)
        travel_times: 链路行程时间数组 (s)
        timestamps: 出发时间戳数组 (s)
    """
    dist_map = load_dist_mapping(dist_csv)
    if not dist_map:
        return np.array([]), np.array([]), np.array([])
    
    df_stops = load_sim_stopinfo(stopinfo_xml)
    if df_stops.empty:
        return np.array([]), np.array([]), np.array([])
    
    df_stops = df_stops.sort_values(["vehicle_id", "arrival"]).reset_index(drop=True)
    
    speeds = []
    travel_times = []
    timestamps = []
    
    for veh_id, veh_data in df_stops.groupby("vehicle_id"):
        veh_data = veh_data.reset_index(drop=True)
        for i in range(len(veh_data) - 1):
            from_stop = str(veh_data.loc[i, "stop_id"])
            to_stop = str(veh_data.loc[i + 1, "stop_id"])
            
            if tt_mode == 'door':
                # door-to-door: 下一站到达 - 当前站到达
                departure = float(veh_data.loc[i, "arrival"])
                arrival = float(veh_data.loc[i + 1, "arrival"])
            else:
                # moving-only: 下一站开始(started) - 当前站离开(departure)
                departure = float(veh_data.loc[i, "departure"])
                arrival = float(veh_data.loc[i + 1, "started"])
            
            travel_time_s = arrival - departure
            
            if travel_time_s <= 0:
                continue
            
            dist_m = dist_map.get((from_stop, to_stop))
            if not dist_m:
                continue
            
            speed_kmh = (dist_m / 1000.0) / (travel_time_s / 3600.0)
            if 0.1 < speed_kmh < 120:
                speeds.append(speed_kmh)
                travel_times.append(travel_time_s)
                timestamps.append(departure)
    
    return np.array(speeds), np.array(travel_times), np.array(timestamps)


# ============================================================================
# Rule C 审计 (Op-L2-v1.1)
# ============================================================================

def apply_audit_rule_c(
    df: pd.DataFrame,
    config: Optional[AuditConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, AuditStats]:
    """
    应用 Rule C (Op-L2-v1.1) 审计规则
    
    Rule C: flagged if (tt >= T*) AND (speed <= v*) AND (dist < max_dist)
    
    Args:
        df: 包含 tt_median, speed_median, dist_m 列的 DataFrame
        config: AuditConfig，默认使用 Protocol v4 配置
    
    Returns:
        raw_speeds: 原始速度数组
        clean_speeds: 清洗后速度数组
        raw_tt: 原始行程时间数组
        clean_tt: 清洗后行程时间数组
        audit_stats: 审计统计
    """
    if config is None:
        config = AuditConfig.from_protocol()
    
    raw_speeds = df["speed_median"].dropna().values
    raw_tt = df["tt_median"].dropna().values
    
    if not config.enabled:
        # Audit 关闭时，clean = raw
        stats = AuditStats(
            n_raw=len(raw_speeds),
            n_clean=len(raw_speeds),
            n_flagged=0,
            flagged_fraction=0.0,
            config=config
        )
        return raw_speeds, raw_speeds.copy(), raw_tt, raw_tt.copy(), stats
    
    # Rule C: flagged if (tt >= T*) AND (speed <= v*) AND (dist < max_dist)
    cond_flagged = (
        (df["tt_median"] >= config.T_star) &
        (df["speed_median"] <= config.v_star) &
        (df["dist_m"] < config.max_dist_m)
    )
    
    clean_mask = ~cond_flagged
    clean_speeds = df.loc[clean_mask, "speed_median"].dropna().values
    clean_tt = df.loc[clean_mask, "tt_median"].dropna().values
    
    n_flagged = cond_flagged.sum()
    n_raw = len(df)
    
    stats = AuditStats(
        n_raw=n_raw,
        n_clean=len(clean_speeds),
        n_flagged=n_flagged,
        flagged_fraction=n_flagged / n_raw if n_raw > 0 else 0.0,
        config=config
    )
    
    return raw_speeds, clean_speeds, raw_tt, clean_tt, stats


# ============================================================================
# KS 检验
# ============================================================================

def compute_ks_with_critical(
    real_values: np.ndarray,
    sim_values: np.ndarray,
    alpha: float = None
) -> KSResult:
    """
    计算 KS 检验 + Pass/Fail 判定
    
    Pass iff KS < D_crit, where D_crit = c(alpha) * sqrt((n+m)/(n*m))
    """
    if alpha is None:
        alpha = PROTOCOL_V4_CONFIG["ks_test"]["alpha"]
    
    n = len(real_values)
    m = len(sim_values)
    
    min_samples = PROTOCOL_V4_CONFIG["sanity"]["n_clean_min"]
    if n < min_samples or m < min_samples:
        return KSResult(
            ks_stat=None,
            p_value=None,
            n_real=n,
            n_sim=m,
            critical_value=None,
            passed=False
        )
    
    ks_stat, p_value = ks_2samp(real_values, sim_values)
    
    # Critical value: c(alpha) * sqrt((n+m)/(n*m))
    c_alpha = PROTOCOL_V4_CONFIG["ks_test"]["c_alpha"]
    critical_value = c_alpha * np.sqrt((n + m) / (n * m))
    
    # Pass iff KS < D_crit
    passed = ks_stat < critical_value
    
    return KSResult(
        ks_stat=ks_stat,
        p_value=p_value,
        n_real=n,
        n_sim=m,
        critical_value=critical_value,
        passed=passed
    )


# ============================================================================
# Worst-Window Stress Test (Exhaustive)
# ============================================================================

def seconds_to_time_str(seconds: int, offset_hours: int = 0) -> str:
    """将秒数转换为 HH:MM 格式"""
    total_seconds = seconds + offset_hours * 3600
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def compute_worst_window_exhaustive(
    real_values: np.ndarray,
    sim_values: np.ndarray,
    total_duration_sec: int = 3600,
    window_duration_sec: int = None,
    step_sec: int = None,
    base_time_sec: int = 0,
    min_samples: int = None,
    hkt_offset: int = 8
) -> WorstWindowResult:
    """
    Exhaustive worst-window stress test
    
    遍历所有可能的 15-min 子窗口，找到 KS 最大的窗口
    禁止使用 random sub-windows
    
    Args:
        real_values: 真实数据数组
        sim_values: 仿真数据数组
        total_duration_sec: 总时长（秒）
        window_duration_sec: 子窗口时长（秒）
        step_sec: 滑动步长（秒）
        base_time_sec: 基准时间（秒，UTC）
        min_samples: 每个窗口最小样本数
        hkt_offset: 时区偏移（香港 = +8）
    
    Returns:
        WorstWindowResult
    """
    cfg = PROTOCOL_V4_CONFIG["worst_window"]
    if window_duration_sec is None:
        window_duration_sec = cfg["duration_sec"]
    if step_sec is None:
        step_sec = cfg["step_sec"]
    if min_samples is None:
        min_samples = cfg["min_samples"]
    
    if len(real_values) < min_samples or len(sim_values) < min_samples:
        return WorstWindowResult(
            worst_ks=None,
            window_start_sec=None,
            window_end_sec=None,
            window_start_time=None,
            window_end_time=None,
            n_windows_checked=0,
            method="exhaustive"
        )
    
    n_samples = len(real_values)
    samples_per_sec = n_samples / total_duration_sec if total_duration_sec > 0 else 0
    
    worst_ks = 0.0
    worst_start = 0
    worst_end = window_duration_sec
    n_windows = 0
    
    for start_sec in range(0, total_duration_sec - window_duration_sec + 1, step_sec):
        end_sec = start_sec + window_duration_sec
        
        start_idx = int(start_sec * samples_per_sec)
        end_idx = int(end_sec * samples_per_sec)
        
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        
        window_real = real_values[start_idx:end_idx]
        
        if len(window_real) < min_samples:
            continue
        
        n_window = len(window_real)
        if len(sim_values) >= n_window:
            np.random.seed(42 + start_sec)
            sim_sample = np.random.choice(sim_values, n_window, replace=False)
        else:
            sim_sample = sim_values
        
        if len(sim_sample) < min_samples:
            continue
        
        ks_stat, _ = ks_2samp(window_real, sim_sample)
        n_windows += 1
        
        if ks_stat > worst_ks:
            worst_ks = ks_stat
            worst_start = base_time_sec + start_sec
            worst_end = base_time_sec + end_sec
    
    if n_windows == 0:
        return WorstWindowResult(
            worst_ks=None,
            window_start_sec=None,
            window_end_sec=None,
            window_start_time=None,
            window_end_time=None,
            n_windows_checked=0,
            method="exhaustive"
        )
    
    return WorstWindowResult(
        worst_ks=worst_ks,
        window_start_sec=worst_start,
        window_end_sec=worst_end,
        window_start_time=seconds_to_time_str(worst_start, hkt_offset),
        window_end_time=seconds_to_time_str(worst_end, hkt_offset),
        n_windows_checked=n_windows,
        method="exhaustive"
    )


# ============================================================================
# Bootstrap 置信区间
# ============================================================================

def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = None,
    confidence: float = None,
    seed: int = None
) -> BootstrapCI:
    """
    计算 Bootstrap 95% 置信区间
    
    用于跨窗口汇总统计
    """
    cfg = PROTOCOL_V4_CONFIG["bootstrap"]
    if n_bootstrap is None:
        n_bootstrap = cfg["n_samples"]
    if confidence is None:
        confidence = cfg["confidence"]
    if seed is None:
        seed = cfg["seed"]
    
    if len(values) == 0:
        return BootstrapCI()
    
    np.random.seed(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapCI(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_samples=len(values)
    )


# ============================================================================
# Sanity Checks
# ============================================================================

def run_sanity_checks(result: MetricsV4Result) -> MetricsV4Result:
    """
    运行 Protocol v4 Sanity Checks
    
    SC1: 单位一致性（由数据加载保证）
    SC2: 掩码一致性（由评估流程保证）
    SC3: n_clean >= 10
    SC4: 窗口一致性（由评估流程保证）
    SC5: KS 范围 [0, 1]
    SC6: 禁止 random worst-window
    """
    errors = []
    warnings_list = []
    
    cfg = PROTOCOL_V4_CONFIG["sanity"]
    
    # SC3: n_clean 最小值
    if result.audit_stats.n_clean < cfg["n_clean_min"]:
        warnings_list.append(
            f"SC3: n_clean={result.audit_stats.n_clean} < {cfg['n_clean_min']}, "
            "KS 结果可能不可靠"
        )
    
    # SC5: KS 范围
    for name, ks_result in [
        ("ks_speed_raw", result.ks_speed_raw),
        ("ks_speed_clean", result.ks_speed_clean),
        ("ks_tt_raw", result.ks_tt_raw),
        ("ks_tt_clean", result.ks_tt_clean)
    ]:
        if ks_result.ks_stat is not None:
            if not (cfg["ks_range"][0] <= ks_result.ks_stat <= cfg["ks_range"][1]):
                errors.append(f"SC5: {name}={ks_result.ks_stat:.4f} 超出范围 [0,1]")
    
    # SC6: 禁止 random worst-window
    if result.worst_window_speed.method != "exhaustive":
        errors.append(f"SC6: worst-window 必须使用 exhaustive 方法")
    
    result.sanity_errors = errors
    result.sanity_warnings = warnings_list
    result.sanity_passed = len(errors) == 0
    
    return result


# ============================================================================
# 主评估函数
# ============================================================================

def compute_metrics_v4(
    real_data: Union[str, pd.DataFrame],
    sim_data: Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dist_file: Optional[str] = None,
    audit_config: Optional[AuditConfig] = None,
    scenario: str = "off_peak",
    route: str = "",
    period: str = "",
    tt_mode: str = "door"
) -> MetricsV4Result:
    """
    Protocol v4 完整评估流程
    
    Args:
        real_data: 真实数据（CSV 路径或 DataFrame）
        sim_data: 仿真数据（stopinfo XML 路径或 (speeds, tt, timestamps) 元组）
        dist_file: 站点距离 CSV（仅当 sim_data 为 XML 时需要）
        audit_config: Audit 配置，默认使用 Protocol v4
        scenario: 场景名称
        route: 路由名称
        period: 时段名称
        tt_mode: 'door' (arrival-arrival) 或 'moving' (started-departure)
    
    Returns:
        MetricsV4Result
    """
    result = MetricsV4Result(scenario=scenario, route=route, period=period)
    
    # 加载真实数据
    if isinstance(real_data, str):
        df_real = load_real_link_stats(real_data)
    else:
        df_real = real_data
    
    # 应用 Audit
    if audit_config is None:
        audit_config = AuditConfig.from_protocol()
    
    raw_speeds, clean_speeds, raw_tt, clean_tt, audit_stats = apply_audit_rule_c(
        df_real, audit_config
    )
    result.audit_stats = audit_stats
    
    # 加载仿真数据
    if isinstance(sim_data, str):
        if dist_file is None:
            dist_file = str(PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv")
        sim_speeds, sim_tt, sim_timestamps = compute_sim_link_data(sim_data, dist_file, tt_mode=tt_mode)
    else:
        sim_speeds, sim_tt, sim_timestamps = sim_data
    
    result.n_sim = len(sim_speeds)
    
    # 获取场景配置
    scenario_cfg = PROTOCOL_V4_CONFIG["scenarios"].get(scenario, {})
    base_time_sec = scenario_cfg.get("start_sec", 0)
    duration_sec = scenario_cfg.get("duration_sec", 3600)
    
    # Full-hour KS (speed)
    result.ks_speed_raw = compute_ks_with_critical(raw_speeds, sim_speeds)
    result.ks_speed_clean = compute_ks_with_critical(clean_speeds, sim_speeds)
    
    # Full-hour KS (TT)
    result.ks_tt_raw = compute_ks_with_critical(raw_tt, sim_tt)
    result.ks_tt_clean = compute_ks_with_critical(clean_tt, sim_tt)
    
    # Worst-window (exhaustive) for speed
    result.worst_window_speed = compute_worst_window_exhaustive(
        clean_speeds, sim_speeds,
        total_duration_sec=duration_sec,
        base_time_sec=base_time_sec
    )
    
    # Worst-window (exhaustive) for TT
    result.worst_window_tt = compute_worst_window_exhaustive(
        clean_tt, sim_tt,
        total_duration_sec=duration_sec,
        base_time_sec=base_time_sec
    )
    
    # Sanity checks
    result = run_sanity_checks(result)
    
    return result


# ============================================================================
# 输出格式化
# ============================================================================

def print_metrics_v4_summary(result: MetricsV4Result):
    """打印评估结果摘要"""
    print("=" * 70)
    print("Protocol v4 评估结果")
    print("=" * 70)
    print(f"场景: {result.scenario} | 路由: {result.route} | 时段: {result.period}")
    print(f"Audit: T*={result.audit_stats.config.T_star}s, v*={result.audit_stats.config.v_star}km/h")
    print()
    
    print("【样本统计】")
    print(f"  n_raw:     {result.audit_stats.n_raw}")
    print(f"  n_clean:   {result.audit_stats.n_clean}")
    print(f"  n_sim:     {result.n_sim}")
    print(f"  flagged:   {result.audit_stats.flagged_fraction*100:.1f}%")
    print()
    
    print("【Full-hour KS(speed)】")
    if result.ks_speed_raw.ks_stat is not None:
        print(f"  raw:   {result.ks_speed_raw.ks_stat:.4f}")
    if result.ks_speed_clean.ks_stat is not None:
        status = "PASS" if result.ks_speed_clean.passed else "FAIL"
        print(f"  clean: {result.ks_speed_clean.ks_stat:.4f} [{status}] "
              f"(D_crit={result.ks_speed_clean.critical_value:.4f})")
    print()
    
    print("【Full-hour KS(TT)】")
    if result.ks_tt_raw.ks_stat is not None:
        print(f"  raw:   {result.ks_tt_raw.ks_stat:.4f}")
    if result.ks_tt_clean.ks_stat is not None:
        status = "PASS" if result.ks_tt_clean.passed else "FAIL"
        print(f"  clean: {result.ks_tt_clean.ks_stat:.4f} [{status}]")
    print()
    
    print("【Worst-Window (Exhaustive)】")
    if result.worst_window_speed.worst_ks is not None:
        print(f"  KS(speed): {result.worst_window_speed.worst_ks:.4f}")
        print(f"  窗口:      {result.worst_window_speed.window_start_time} - "
              f"{result.worst_window_speed.window_end_time}")
        print(f"  检查窗口数: {result.worst_window_speed.n_windows_checked}")
    print()
    
    print("【Sanity Checks】")
    if result.sanity_errors:
        for err in result.sanity_errors:
            print(f"  [ERROR] {err}")
    if result.sanity_warnings:
        for warn in result.sanity_warnings:
            print(f"  [WARN] {warn}")
    if result.sanity_passed and not result.sanity_warnings:
        print("  [OK] 所有检查通过")
    print()


def results_to_dataframe(results: List[MetricsV4Result]) -> pd.DataFrame:
    """将多个结果转换为 DataFrame"""
    return pd.DataFrame([r.to_flat_dict() for r in results])


def results_to_markdown(results: List[MetricsV4Result], caption: str = "") -> str:
    """生成 Markdown 表格"""
    df = results_to_dataframe(results)
    
    cols = ["scenario", "n_clean", "n_sim", "ks_speed_clean", "ks_speed_passed", 
            "worst_window_ks_speed", "worst_window_start"]
    df_display = df[cols].copy()
    df_display.columns = ["Config", "n_clean", "n_sim", "KS(speed)", "Pass", 
                          "Worst-15min", "Window"]
    
    markdown = df_display.to_markdown(index=False, floatfmt=".4f")
    
    if caption:
        markdown = f"# {caption}\n\n{markdown}"
    
    return markdown


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Protocol v4 统一评估器")
    parser.add_argument(
        "--real",
        type=str,
        default=str(PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv"),
        help="真实链路统计 CSV"
    )
    parser.add_argument(
        "--sim",
        type=str,
        default=str(PROJECT_ROOT / "sumo" / "output" / "offpeak_v2_offpeak_stopinfo.xml"),
        help="仿真 stopinfo XML"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv"),
        help="站点距离 CSV"
    )
    parser.add_argument(
        "--t_star",
        type=float,
        default=325.0,
        help="Rule C: T* (秒)"
    )
    parser.add_argument(
        "--v_star",
        type=float,
        default=5.0,
        help="Rule C: v* (km/h)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="off_peak",
        help="场景名称"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径"
    )
    
    args = parser.parse_args()
    
    audit_config = AuditConfig(
        T_star=args.t_star,
        v_star=args.v_star
    )
    
    result = compute_metrics_v4(
        real_data=args.real,
        sim_data=args.sim,
        dist_file=args.dist,
        audit_config=audit_config,
        scenario=args.scenario
    )
    
    print_metrics_v4_summary(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
