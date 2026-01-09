#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics_v3.py
=============
统一评估器 v3 - 论文主线口径

口径说明：
- P14 Off-Peak transfer (15:00–16:00)
- Op-L2-v1.1 (D2D + Decont.), Rule C: Time > T*=325s & Speed < 5 km/h
- KS 指标明确区分：KS(speed) vs KS(TT)
- worst-window stress test: exhaustive all 15-min subwindows max

关键修正：
1. worst-15min 使用 exhaustive（遍历所有可能的 15-min 子窗口），而非 4 random sub-windows
2. 输出 worst-window 的起止时间
3. Pass/Fail: alpha=0.05, Dcrit=1.36*sqrt((n+m)/(nm)), Pass iff KS < Dcrit

Author: RCMDT Project
Date: 2026-01-09
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy.stats import ks_2samp
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ============================================================================
# 配置常量 - 论文主线口径
# ============================================================================

# Rule C (Op-L2-v1.1): ghost jam detection
RULE_C_T_CRITICAL = 325.0   # T* = 325 seconds
RULE_C_SPEED_KMH = 5.0      # v* = 5 km/h
RULE_C_MAX_DIST_M = 1500.0  # 最大距离阈值

# P14 Off-Peak 时间窗口
OFFPEAK_START_SEC = 54000   # 15:00 = 15*3600
OFFPEAK_END_SEC = 57600     # 16:00 = 16*3600
OFFPEAK_DURATION_SEC = 3600 # 1 hour

# Worst-window 配置
SUBWINDOW_DURATION_SEC = 900  # 15 minutes
SUBWINDOW_STEP_SEC = 60       # 1 minute step for exhaustive search

# KS 检验配置
KS_ALPHA = 0.05
KS_C_ALPHA = 1.36  # critical value coefficient for alpha=0.05

# Sanity check 容许误差
SANITY_KS_RAW_EXPECTED = 0.54
SANITY_KS_CLEAN_EXPECTED = 0.2618
SANITY_WORST_KS_EXPECTED = 0.3337
SANITY_N_CLEAN_EXPECTED = 37
SANITY_KS_TOLERANCE = 0.02
SANITY_WORST_TOLERANCE = 0.03
SANITY_N_TOLERANCE = 2


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class KSResult:
    """KS 检验结果"""
    ks_stat: Optional[float]
    p_value: Optional[float]
    n_real: int
    n_sim: int
    critical_value: Optional[float]
    passed: bool
    
    def to_dict(self) -> Dict:
        return {
            "ks_stat": self.ks_stat,
            "p_value": self.p_value,
            "n_real": self.n_real,
            "n_sim": self.n_sim,
            "critical_value": self.critical_value,
            "passed": self.passed
        }


@dataclass
class WorstWindowResult:
    """Worst-window 结果"""
    worst_ks: Optional[float]
    window_start_sec: Optional[int]
    window_end_sec: Optional[int]
    window_start_time: Optional[str]
    window_end_time: Optional[str]
    n_windows_checked: int
    all_window_ks: List[Tuple[int, int, float]]  # (start, end, ks)
    
    def to_dict(self) -> Dict:
        return {
            "worst_ks": self.worst_ks,
            "window_start_sec": self.window_start_sec,
            "window_end_sec": self.window_end_sec,
            "window_start_time": self.window_start_time,
            "window_end_time": self.window_end_time,
            "n_windows_checked": self.n_windows_checked
        }


@dataclass
class MetricsV3Result:
    """完整评估结果"""
    # 样本信息
    n_raw: int
    n_clean: int
    n_sim: int
    flagged_fraction: float
    
    # Full-hour KS
    ks_speed_raw: KSResult
    ks_speed_clean: KSResult
    ks_tt_raw: KSResult
    ks_tt_clean: KSResult
    
    # Worst-window (exhaustive)
    worst_window_speed: WorstWindowResult
    worst_window_tt: WorstWindowResult
    
    # Sanity check
    sanity_passed: bool
    sanity_messages: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "n_raw": self.n_raw,
            "n_clean": self.n_clean,
            "n_sim": self.n_sim,
            "flagged_fraction": self.flagged_fraction,
            "ks_speed_raw": self.ks_speed_raw.to_dict(),
            "ks_speed_clean": self.ks_speed_clean.to_dict(),
            "ks_tt_raw": self.ks_tt_raw.to_dict(),
            "ks_tt_clean": self.ks_tt_clean.to_dict(),
            "worst_window_speed": self.worst_window_speed.to_dict(),
            "worst_window_tt": self.worst_window_tt.to_dict(),
            "sanity_passed": self.sanity_passed,
            "sanity_messages": self.sanity_messages
        }


# ============================================================================
# 数据加载
# ============================================================================

def load_real_link_stats(filepath: str) -> pd.DataFrame:
    """加载真实链路统计数据 (link_stats_offpeak.csv)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"真实数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = {"tt_median", "speed_median", "dist_m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需列: {missing}")
    
    return df


def load_sim_stopinfo(filepath: str) -> pd.DataFrame:
    """解析 SUMO stopinfo.xml，处理可能不完整的 XML"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        # 尝试修复不完整的 XML（缺少关闭标签）
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            # 添加缺失的关闭标签
            if '</stops>' not in content:
                content = content.rstrip() + '\n</stops>'
            root = ET.fromstring(content)
        except Exception as e:
            print(f"[WARN] 无法解析 stopinfo XML: {e}")
            return pd.DataFrame()
    
    records = []
    for stop in root.findall('.//stopinfo'):
        records.append({
            'vehicle_id': stop.get('id'),
            'stop_id': stop.get('busStop'),
            'arrival': float(stop.get('started', 0)),
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
    dist_csv: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从仿真输出计算链路速度和行程时间
    
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
                travel_times.append(travel_time_s)
                timestamps.append(departure)
    
    return np.array(speeds), np.array(travel_times), np.array(timestamps)


# ============================================================================
# Rule C 清洗 (Op-L2-v1.1)
# ============================================================================

def apply_rule_c_audit(
    df: pd.DataFrame,
    t_critical: float = RULE_C_T_CRITICAL,
    speed_kmh: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    应用 Rule C (Op-L2-v1.1) 审计规则
    
    Rule C: ghost if (tt > T*) & (speed < v*) & (dist < max_dist)
    
    Returns:
        raw_speeds: 原始速度数组
        clean_speeds: 清洗后速度数组
        raw_tt: 原始行程时间数组
        clean_tt: 清洗后行程时间数组
        flagged_fraction: 被标记为 ghost 的比例
        n_clean: 清洗后样本数
    """
    raw_speeds = df["speed_median"].dropna().values
    raw_tt = df["tt_median"].dropna().values
    
    # Rule C: ghost if (tt > T*) & (speed < v*) & (dist < max_dist)
    cond_ghost = (
        (df["tt_median"] > t_critical) & 
        (df["speed_median"] < speed_kmh) & 
        (df["dist_m"] < max_dist_m)
    )
    
    clean_mask = ~cond_ghost
    clean_speeds = df.loc[clean_mask, "speed_median"].dropna().values
    clean_tt = df.loc[clean_mask, "tt_median"].dropna().values
    
    flagged_fraction = cond_ghost.sum() / len(df) if len(df) > 0 else 0.0
    n_clean = len(clean_speeds)
    
    return raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_fraction, n_clean


# ============================================================================
# KS 检验
# ============================================================================

def compute_ks_with_stats(
    real_values: np.ndarray, 
    sim_values: np.ndarray,
    alpha: float = KS_ALPHA
) -> KSResult:
    """
    计算 KS 检验并返回完整统计信息
    
    Pass/Fail 判定: KS < Dcrit, where Dcrit = c(alpha) * sqrt((n+m)/(n*m))
    """
    n = len(real_values)
    m = len(sim_values)
    
    if n < 5 or m < 5:
        return KSResult(
            ks_stat=None,
            p_value=None,
            n_real=n,
            n_sim=m,
            critical_value=None,
            passed=False
        )
    
    ks_stat, p_value = ks_2samp(real_values, sim_values)
    
    # KS 临界值：c(alpha) * sqrt((n+m)/(n*m))
    c_alpha = KS_C_ALPHA if alpha == 0.05 else 1.22
    critical_value = c_alpha * np.sqrt((n + m) / (n * m))
    
    # Pass iff KS < Dcrit
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

def seconds_to_time_str(seconds: int) -> str:
    """将秒数转换为 HH:MM 格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def compute_worst_window_exhaustive(
    real_values: np.ndarray,
    sim_values: np.ndarray,
    total_duration_sec: int = OFFPEAK_DURATION_SEC,
    window_duration_sec: int = SUBWINDOW_DURATION_SEC,
    step_sec: int = SUBWINDOW_STEP_SEC,
    base_time_sec: int = OFFPEAK_START_SEC,
    min_samples: int = 5
) -> WorstWindowResult:
    """
    Exhaustive worst-window stress test
    
    遍历所有可能的 15-min 子窗口，找到 KS 最大的窗口
    
    Args:
        real_values: 真实数据数组
        sim_values: 仿真数据数组
        total_duration_sec: 总时长（秒）
        window_duration_sec: 子窗口时长（秒）
        step_sec: 滑动步长（秒）
        base_time_sec: 基准时间（秒）
        min_samples: 每个窗口最小样本数
    
    Returns:
        WorstWindowResult
    """
    if len(real_values) < min_samples or len(sim_values) < min_samples:
        return WorstWindowResult(
            worst_ks=None,
            window_start_sec=None,
            window_end_sec=None,
            window_start_time=None,
            window_end_time=None,
            n_windows_checked=0,
            all_window_ks=[]
        )
    
    # 将数据按时间索引分组（假设数据已按时间排序）
    n_samples = len(real_values)
    samples_per_sec = n_samples / total_duration_sec if total_duration_sec > 0 else 0
    
    all_window_ks = []
    worst_ks = 0.0
    worst_start = 0
    worst_end = window_duration_sec
    
    # 遍历所有可能的起始位置
    n_windows = 0
    for start_sec in range(0, total_duration_sec - window_duration_sec + 1, step_sec):
        end_sec = start_sec + window_duration_sec
        
        # 计算窗口内的样本索引范围
        start_idx = int(start_sec * samples_per_sec)
        end_idx = int(end_sec * samples_per_sec)
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        
        window_real = real_values[start_idx:end_idx]
        
        if len(window_real) < min_samples:
            continue
        
        # 从仿真数据中采样相同数量
        n_window = len(window_real)
        if len(sim_values) >= n_window:
            # 使用固定种子确保可复现
            np.random.seed(42 + start_sec)
            sim_sample = np.random.choice(sim_values, n_window, replace=False)
        else:
            sim_sample = sim_values
        
        if len(sim_sample) < min_samples:
            continue
        
        ks_stat, _ = ks_2samp(window_real, sim_sample)
        
        abs_start = base_time_sec + start_sec
        abs_end = base_time_sec + end_sec
        all_window_ks.append((abs_start, abs_end, ks_stat))
        n_windows += 1
        
        if ks_stat > worst_ks:
            worst_ks = ks_stat
            worst_start = abs_start
            worst_end = abs_end
    
    if n_windows == 0:
        return WorstWindowResult(
            worst_ks=None,
            window_start_sec=None,
            window_end_sec=None,
            window_start_time=None,
            window_end_time=None,
            n_windows_checked=0,
            all_window_ks=[]
        )
    
    return WorstWindowResult(
        worst_ks=worst_ks,
        window_start_sec=worst_start,
        window_end_sec=worst_end,
        window_start_time=seconds_to_time_str(worst_start),
        window_end_time=seconds_to_time_str(worst_end),
        n_windows_checked=n_windows,
        all_window_ks=all_window_ks
    )


# ============================================================================
# Sanity Checks
# ============================================================================

def run_sanity_checks(
    ks_raw: float,
    ks_clean: float,
    worst_ks: float,
    n_clean: int
) -> Tuple[bool, List[str]]:
    """
    运行 sanity checks
    
    Expected values (P14 Off-Peak, Rule-C 325,5):
    - KS(raw) ≈ 0.54 (±0.02)
    - KS(clean) ≈ 0.2618 (±0.02)
    - worst-15min KS ≈ 0.3337 (±0.03)
    - n_clean ≈ 37 (±2)
    """
    messages = []
    all_passed = True
    
    # Check KS(raw)
    if ks_raw is not None:
        if abs(ks_raw - SANITY_KS_RAW_EXPECTED) > SANITY_KS_TOLERANCE:
            messages.append(
                f"[WARN] KS(raw)={ks_raw:.4f}, expected ≈{SANITY_KS_RAW_EXPECTED} "
                f"(diff={abs(ks_raw - SANITY_KS_RAW_EXPECTED):.4f})"
            )
        else:
            messages.append(f"[OK] KS(raw)={ks_raw:.4f} ✓")
    
    # Check KS(clean)
    if ks_clean is not None:
        if abs(ks_clean - SANITY_KS_CLEAN_EXPECTED) > SANITY_KS_TOLERANCE:
            messages.append(
                f"[WARN] KS(clean)={ks_clean:.4f}, expected ≈{SANITY_KS_CLEAN_EXPECTED} "
                f"(diff={abs(ks_clean - SANITY_KS_CLEAN_EXPECTED):.4f})"
            )
        else:
            messages.append(f"[OK] KS(clean)={ks_clean:.4f} ✓")
    
    # Check worst-15min KS
    if worst_ks is not None:
        if abs(worst_ks - SANITY_WORST_KS_EXPECTED) > SANITY_WORST_TOLERANCE:
            messages.append(
                f"[WARN] worst-15min KS={worst_ks:.4f}, expected ≈{SANITY_WORST_KS_EXPECTED} "
                f"(diff={abs(worst_ks - SANITY_WORST_KS_EXPECTED):.4f})"
            )
        else:
            messages.append(f"[OK] worst-15min KS={worst_ks:.4f} ✓")
    
    # Check n_clean
    if abs(n_clean - SANITY_N_CLEAN_EXPECTED) > SANITY_N_TOLERANCE:
        messages.append(
            f"[WARN] n_clean={n_clean}, expected ≈{SANITY_N_CLEAN_EXPECTED} "
            f"(diff={abs(n_clean - SANITY_N_CLEAN_EXPECTED)})"
        )
    else:
        messages.append(f"[OK] n_clean={n_clean} ✓")
    
    # Determine overall pass/fail
    # 不强制失败，只是警告
    all_passed = not any("[WARN]" in m for m in messages)
    
    return all_passed, messages


# ============================================================================
# 主评估函数
# ============================================================================

def evaluate_metrics_v3(
    real_stats_file: str,
    sim_stopinfo_file: str,
    dist_file: str,
    t_critical: float = RULE_C_T_CRITICAL,
    speed_kmh: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M,
    run_sanity: bool = True
) -> MetricsV3Result:
    """
    完整评估流程 v3
    
    Args:
        real_stats_file: 真实链路统计 CSV (link_stats_offpeak.csv)
        sim_stopinfo_file: 仿真 stopinfo XML
        dist_file: 站点距离 CSV
        t_critical: Rule C T* 阈值
        speed_kmh: Rule C v* 阈值
        max_dist_m: Rule C 最大距离
        run_sanity: 是否运行 sanity checks
    
    Returns:
        MetricsV3Result
    """
    # 加载真实数据
    df_real = load_real_link_stats(real_stats_file)
    
    # 应用 Rule C 清洗
    raw_speeds, clean_speeds, raw_tt, clean_tt, flagged_frac, n_clean = apply_rule_c_audit(
        df_real, t_critical, speed_kmh, max_dist_m
    )
    
    # 加载仿真数据
    sim_speeds, sim_tt, sim_timestamps = compute_sim_link_data(sim_stopinfo_file, dist_file)
    
    # Full-hour KS
    ks_speed_raw = compute_ks_with_stats(raw_speeds, sim_speeds)
    ks_speed_clean = compute_ks_with_stats(clean_speeds, sim_speeds)
    ks_tt_raw = compute_ks_with_stats(raw_tt, sim_tt)
    ks_tt_clean = compute_ks_with_stats(clean_tt, sim_tt)
    
    # Worst-window (exhaustive) for speed
    worst_window_speed = compute_worst_window_exhaustive(
        clean_speeds, sim_speeds,
        total_duration_sec=OFFPEAK_DURATION_SEC,
        window_duration_sec=SUBWINDOW_DURATION_SEC,
        step_sec=SUBWINDOW_STEP_SEC,
        base_time_sec=OFFPEAK_START_SEC
    )
    
    # Worst-window (exhaustive) for TT
    worst_window_tt = compute_worst_window_exhaustive(
        clean_tt, sim_tt,
        total_duration_sec=OFFPEAK_DURATION_SEC,
        window_duration_sec=SUBWINDOW_DURATION_SEC,
        step_sec=SUBWINDOW_STEP_SEC,
        base_time_sec=OFFPEAK_START_SEC
    )
    
    # Sanity checks
    sanity_passed = True
    sanity_messages = []
    if run_sanity:
        sanity_passed, sanity_messages = run_sanity_checks(
            ks_raw=ks_speed_raw.ks_stat,
            ks_clean=ks_speed_clean.ks_stat,
            worst_ks=worst_window_speed.worst_ks,
            n_clean=n_clean
        )
    
    return MetricsV3Result(
        n_raw=len(raw_speeds),
        n_clean=n_clean,
        n_sim=len(sim_speeds),
        flagged_fraction=flagged_frac,
        ks_speed_raw=ks_speed_raw,
        ks_speed_clean=ks_speed_clean,
        ks_tt_raw=ks_tt_raw,
        ks_tt_clean=ks_tt_clean,
        worst_window_speed=worst_window_speed,
        worst_window_tt=worst_window_tt,
        sanity_passed=sanity_passed,
        sanity_messages=sanity_messages
    )


def print_metrics_summary(result: MetricsV3Result):
    """打印评估结果摘要"""
    print("=" * 70)
    print("Metrics V3 评估结果 (论文主线口径)")
    print("=" * 70)
    print(f"口径: P14 Off-Peak (15:00-16:00), Op-L2-v1.1, Rule C (T*=325s, v*=5km/h)")
    print()
    
    print("【样本信息】")
    print(f"  真实样本 (raw):   {result.n_raw}")
    print(f"  真实样本 (clean): {result.n_clean}")
    print(f"  仿真样本:         {result.n_sim}")
    print(f"  Flagged 比例:     {result.flagged_fraction*100:.1f}%")
    print()
    
    print("【Full-hour KS(speed)】")
    print(f"  KS(raw):   {result.ks_speed_raw.ks_stat:.4f}" if result.ks_speed_raw.ks_stat else "  KS(raw):   N/A")
    print(f"  KS(clean): {result.ks_speed_clean.ks_stat:.4f} "
          f"({'PASS' if result.ks_speed_clean.passed else 'FAIL'})" 
          if result.ks_speed_clean.ks_stat else "  KS(clean): N/A")
    print()
    
    print("【Full-hour KS(TT)】")
    print(f"  KS(raw):   {result.ks_tt_raw.ks_stat:.4f}" if result.ks_tt_raw.ks_stat else "  KS(raw):   N/A")
    print(f"  KS(clean): {result.ks_tt_clean.ks_stat:.4f} "
          f"({'PASS' if result.ks_tt_clean.passed else 'FAIL'})" 
          if result.ks_tt_clean.ks_stat else "  KS(clean): N/A")
    print()
    
    print("【Worst-Window Stress Test (Exhaustive)】")
    if result.worst_window_speed.worst_ks is not None:
        print(f"  worst KS(speed):    {result.worst_window_speed.worst_ks:.4f}")
        print(f"  worst window:       {result.worst_window_speed.window_start_time} - "
              f"{result.worst_window_speed.window_end_time}")
        print(f"  windows checked:    {result.worst_window_speed.n_windows_checked}")
    else:
        print("  worst KS(speed):    N/A")
    print()
    
    print("【Sanity Checks】")
    for msg in result.sanity_messages:
        print(f"  {msg}")
    print(f"  Overall: {'PASS' if result.sanity_passed else 'WARN'}")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Metrics V3 评估器")
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
        "--t_critical", 
        type=float, 
        default=RULE_C_T_CRITICAL,
        help="Rule C: T* (秒)"
    )
    parser.add_argument(
        "--speed_kmh", 
        type=float, 
        default=RULE_C_SPEED_KMH,
        help="Rule C: v* (km/h)"
    )
    
    args = parser.parse_args()
    
    result = evaluate_metrics_v3(
        real_stats_file=args.real,
        sim_stopinfo_file=args.sim,
        dist_file=args.dist,
        t_critical=args.t_critical,
        speed_kmh=args.speed_kmh
    )
    
    print_metrics_summary(result)
    
    return result


if __name__ == "__main__":
    main()
