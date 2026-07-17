import pandas as pd
import numpy as np
import sys
import os
from typing import Any, Dict, Iterable, Optional, Tuple

# 确保可以导入 scripts 目录下的 common_data
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts'))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds

MIN_MATCHED_DOWNSTREAM_STOPS = 3
L1_CONSTRAINT_RMSE_960_S = 350.0
L1_CONSTRAINT_PENALTY_BASE = 2000.0
L1_CONSTRAINT_PENALTY_SCALE = 10.0


class L1ObjectiveError(RuntimeError):
    """L1 目标函数无法生成有效候选观测。"""


class L1InfrastructureError(L1ObjectiveError):
    """仿真输出缺失、损坏或无法解析。"""


class L1DataContractError(L1ObjectiveError):
    """输入数据不满足 L1 字段或拓扑契约。"""


class L1UnevaluableError(L1ObjectiveError):
    """有效匹配点不足，当前候选不可评估。"""


def normalize_bound(bound: str) -> str:
    """将历史方向缩写归一化为数据文件使用的名称。"""
    value = str(bound).strip()
    aliases = {
        "i": "inbound",
        "inbound": "inbound",
        "o": "outbound",
        "outbound": "outbound",
    }
    return aliases.get(value.lower(), value)


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    label: str,
    error_type: type[L1ObjectiveError] = L1DataContractError,
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise error_type(f"{label} missing required columns: {missing}")


def _filter_route_bound(
    df: pd.DataFrame,
    route: str,
    bound: str,
    label: str,
    error_type: type[L1ObjectiveError] = L1DataContractError,
) -> pd.DataFrame:
    _require_columns(df, ("route", "bound"), label, error_type)
    target_route = str(route)
    target_bound = normalize_bound(bound)
    work = df.copy()
    normalized_bounds = work["bound"].map(normalize_bound)
    mask = work["route"].astype(str).eq(target_route) & normalized_bounds.eq(target_bound)
    work = work.loc[mask].copy()
    if work.empty:
        raise error_type(f"{label} has no rows for route={target_route}, bound={target_bound}")
    work["route"] = target_route
    work["bound"] = target_bound
    return work


def _coerce_sequence_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    error_type: type[L1ObjectiveError] = L1DataContractError,
) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        values = pd.to_numeric(work[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise error_type(f"{label}.{column} contains missing or non-numeric values")
        if not np.allclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float))):
            raise error_type(f"{label}.{column} must contain integer stop indices")
        work[column] = values.astype(int)
    return work


def build_observed_cumulative_times(
    real_links: pd.DataFrame,
    route: str,
    bound: str,
    origin_seq: int = 1,
) -> pd.DataFrame:
    """按路线、方向和链路键构造从序列 1 开始的累计观察时间。"""
    _require_columns(
        real_links,
        ("route", "bound", "from_seq", "to_seq", "travel_time_s"),
        "real_links",
    )
    work = _filter_route_bound(real_links, route, bound, "real_links")
    work = _coerce_sequence_columns(work, ("from_seq", "to_seq"), "real_links")
    travel_times = pd.to_numeric(work["travel_time_s"], errors="coerce")
    if travel_times.isna().any() or not np.isfinite(travel_times.to_numpy(dtype=float)).all():
        raise L1DataContractError("real_links.travel_time_s contains missing or non-numeric values")
    if (travel_times <= 0).any():
        raise L1DataContractError("real_links.travel_time_s must be positive")
    work["travel_time_s"] = travel_times.astype(float)

    grouped = (
        work.groupby(["route", "bound", "from_seq", "to_seq"], as_index=False)["travel_time_s"]
        .mean()
        .sort_values(["from_seq", "to_seq"], kind="stable")
        .reset_index(drop=True)
    )

    cumulative: Dict[int, float] = {int(origin_seq): 0.0}
    unresolved = list(grouped.itertuples(index=False))
    while unresolved:
        progressed = False
        remaining = []
        for row in unresolved:
            from_seq = int(row.from_seq)
            to_seq = int(row.to_seq)
            if from_seq not in cumulative:
                remaining.append(row)
                continue
            candidate = cumulative[from_seq] + float(row.travel_time_s)
            if to_seq in cumulative and not np.isclose(cumulative[to_seq], candidate):
                raise L1DataContractError(
                    f"real_links has conflicting cumulative paths to sequence {to_seq}"
                )
            cumulative[to_seq] = candidate
            progressed = True
        if not progressed:
            break
        unresolved = remaining

    downstream = sorted(seq for seq in cumulative if seq != int(origin_seq))
    if not downstream:
        raise L1UnevaluableError(
            f"real_links has no chain reachable from origin sequence {origin_seq}"
        )
    target_route = str(route)
    target_bound = normalize_bound(bound)
    return pd.DataFrame(
        {
            "route": target_route,
            "bound": target_bound,
            "to_seq": downstream,
            "observed_cumulative_time_s": [cumulative[seq] for seq in downstream],
        }
    )


def build_simulated_cumulative_times(
    sim_trajectory: pd.DataFrame,
    matched_stop_seqs: Iterable[int],
    route: str,
    bound: str,
    origin_seq: int = 1,
) -> pd.DataFrame:
    """按车辆首个匹配站归零，再按下游站序列平均累计到达时间。"""
    _require_columns(
        sim_trajectory,
        ("route", "bound", "vehicle_id", "seq", "arrival_time"),
        "sim_trajectory",
        L1InfrastructureError,
    )
    work = _filter_route_bound(
        sim_trajectory,
        route,
        bound,
        "sim_trajectory",
        L1InfrastructureError,
    )
    work = _coerce_sequence_columns(
        work,
        ("seq",),
        "sim_trajectory",
        L1InfrastructureError,
    )
    arrivals = pd.to_numeric(work["arrival_time"], errors="coerce")
    if arrivals.isna().any() or not np.isfinite(arrivals.to_numpy(dtype=float)).all():
        raise L1InfrastructureError("sim_trajectory.arrival_time contains missing or malformed values")
    if work["vehicle_id"].isna().any():
        raise L1InfrastructureError("sim_trajectory.vehicle_id contains missing values")
    work["arrival_time"] = arrivals.astype(float)

    matched = {int(seq) for seq in matched_stop_seqs}
    matched.add(int(origin_seq))
    work = work.loc[work["seq"].isin(matched)].copy()
    if work.empty:
        raise L1InfrastructureError("simulation output has no stops matching the observation chain")

    group_keys = ["route", "bound", "vehicle_id"]
    sequence_ordered = work.sort_values([*group_keys, "seq"], kind="stable")
    arrival_deltas = sequence_ordered.groupby(group_keys, sort=False)["arrival_time"].diff()
    invalid_arrivals = arrival_deltas.notna() & arrival_deltas.le(0.0)
    if invalid_arrivals.any():
        invalid_row = sequence_ordered.loc[invalid_arrivals].iloc[0]
        raise L1InfrastructureError(
            "sim_trajectory.arrival_time must be strictly increasing along stop seq "
            f"for vehicle_id={invalid_row['vehicle_id']}"
        )

    work = work.sort_values([*group_keys, "arrival_time", "seq"], kind="stable")
    first_arrival = work.groupby(group_keys, sort=False)["arrival_time"].transform("first")
    work["relative_arrival_time_s"] = work["arrival_time"] - first_arrival
    simulated = (
        work.groupby(["route", "bound", "seq"], as_index=False)["relative_arrival_time_s"]
        .mean()
        .rename(
            columns={
                "seq": "to_seq",
                "relative_arrival_time_s": "simulated_cumulative_time_s",
            }
        )
        .sort_values("to_seq", kind="stable")
        .reset_index(drop=True)
    )
    return simulated


def compute_l1_error_table(
    sim_trajectory: pd.DataFrame,
    real_links: pd.DataFrame,
    route: str,
    bound: str,
    origin_seq: int = 1,
    min_downstream_stops: int = MIN_MATCHED_DOWNSTREAM_STOPS,
) -> pd.DataFrame:
    """执行第 6.3 节匹配契约并返回逐下游站误差。"""
    observed = build_observed_cumulative_times(real_links, route, bound, origin_seq)
    simulated = build_simulated_cumulative_times(
        sim_trajectory,
        observed["to_seq"].tolist(),
        route,
        bound,
        origin_seq,
    )
    comparison = observed.merge(
        simulated,
        on=["route", "bound", "to_seq"],
        how="inner",
        validate="one_to_one",
    )
    comparison = comparison.loc[comparison["to_seq"] != int(origin_seq)].copy()
    comparison = comparison.sort_values("to_seq", kind="stable").reset_index(drop=True)
    if len(comparison) < int(min_downstream_stops):
        raise L1UnevaluableError(
            f"route={route}, bound={normalize_bound(bound)} has {len(comparison)} matched "
            f"downstream stops; at least {min_downstream_stops} are required"
        )
    comparison["error_s"] = (
        comparison["simulated_cumulative_time_s"]
        - comparison["observed_cumulative_time_s"]
    )
    return comparison


def calculate_jl1_from_errors(
    errors: Iterable[float],
    alpha: float = 1.0,
    lambda_std: float = 0.5,
    beta: float = 0.3,
) -> Dict[str, Any]:
    """从一个候选的下游累计时间误差计算 JL1 及其分项。"""
    values = np.asarray(list(errors), dtype=float)
    if values.ndim != 1 or len(values) < MIN_MATCHED_DOWNSTREAM_STOPS:
        raise L1UnevaluableError(
            f"JL1 requires at least {MIN_MATCHED_DOWNSTREAM_STOPS} downstream errors"
        )
    if not np.isfinite(values).all():
        raise L1DataContractError("JL1 errors contain missing or non-finite values")
    abs_errors = np.abs(values)
    rmse_term = float(np.sqrt(np.mean(values ** 2)))
    mae_term = float(np.mean(abs_errors))
    std_term = float(np.std(abs_errors))
    dispersion_term = mae_term + float(lambda_std) * std_term
    tail_term = float(np.quantile(abs_errors, 0.9))
    jl1 = rmse_term + float(alpha) * dispersion_term + float(beta) * tail_term
    return {
        "status": "succeeded",
        "jl1": jl1,
        "rmse_term": rmse_term,
        "mae_term": mae_term,
        "std_term": std_term,
        "dispersion_term": dispersion_term,
        "tail_term": tail_term,
        "alpha": float(alpha),
        "lambda_std": float(lambda_std),
        "beta": float(beta),
        "n_errors": int(len(values)),
        "errors": values.tolist(),
    }


def calculate_l1_candidate_score_from_frames(
    sim_trajectory: pd.DataFrame,
    real_links: pd.DataFrame,
    bound: str = "I",
    rmse_960_limit_s: float = L1_CONSTRAINT_RMSE_960_S,
) -> Dict[str, Any]:
    """计算固定协议下的 68X JL1、960 约束和候选得分。"""
    errors_68x = compute_l1_error_table(
        sim_trajectory, real_links, route="68X", bound=bound
    )["error_s"].to_numpy()
    metrics_68x = calculate_jl1_from_errors(errors_68x)
    errors_960 = compute_l1_error_table(
        sim_trajectory, real_links, route="960", bound=bound
    )["error_s"].to_numpy()
    rmse_960 = float(np.sqrt(np.mean(errors_960 ** 2)))
    feasible = bool(rmse_960 <= float(rmse_960_limit_s))
    violation_s = max(0.0, rmse_960 - float(rmse_960_limit_s))
    score = (
        float(metrics_68x["jl1"])
        if feasible
        else L1_CONSTRAINT_PENALTY_BASE + L1_CONSTRAINT_PENALTY_SCALE * violation_s
    )
    return {
        "status": "succeeded",
        "score": float(score),
        "feasible": feasible,
        "jl1_68x": float(metrics_68x["jl1"]),
        "rmse_68x": float(metrics_68x["rmse_term"]),
        "mae_68x": float(metrics_68x["mae_term"]),
        "std_abs_68x": float(metrics_68x["std_term"]),
        "q90_abs_68x": float(metrics_68x["tail_term"]),
        "rmse_960": rmse_960,
        "rmse_960_limit_s": float(rmse_960_limit_s),
        "constraint_violation_s": float(violation_s),
        "penalty": 0.0 if feasible else float(score),
        "n_errors_68x": int(metrics_68x["n_errors"]),
        "n_errors_960": int(len(errors_960)),
    }


def _resolve_legacy_route_bound(
    sim_traj: pd.DataFrame,
    real_links: pd.DataFrame,
    route: Optional[str],
    bound: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """为历史内部两参数调用补入已在上游筛选的路线和方向。"""
    resolved_route = route
    resolved_bound = bound
    if resolved_route is None and "route" in real_links.columns:
        routes = real_links["route"].dropna().astype(str).unique().tolist()
        if len(routes) == 1:
            resolved_route = routes[0]
    if resolved_bound is None and "bound" in real_links.columns:
        bounds = real_links["bound"].dropna().map(normalize_bound).unique().tolist()
        if len(bounds) == 1:
            resolved_bound = bounds[0]
    if resolved_route is None:
        resolved_route = "__legacy_route__"
    if resolved_bound is None:
        resolved_bound = "__legacy_bound__"
    resolved_bound = normalize_bound(resolved_bound)

    sim_work = sim_traj.copy()
    real_work = real_links.copy()
    if "route" not in sim_work.columns:
        sim_work["route"] = str(resolved_route)
    if "bound" not in sim_work.columns:
        sim_work["bound"] = resolved_bound
    if "route" not in real_work.columns:
        real_work["route"] = str(resolved_route)
    if "bound" not in real_work.columns:
        real_work["bound"] = resolved_bound
    return sim_work, real_work, str(resolved_route), resolved_bound


def _compute_cumulative_time_errors(
    sim_traj: pd.DataFrame,
    real_links: pd.DataFrame,
    route: Optional[str] = None,
    bound: Optional[str] = None,
) -> np.ndarray:
    """保留历史调用形式并转发到统一匹配契约。"""
    sim_work, real_work, resolved_route, resolved_bound = _resolve_legacy_route_bound(
        sim_traj, real_links, route, bound
    )
    return compute_l1_error_table(
        sim_work,
        real_work,
        resolved_route,
        resolved_bound,
    )["error_s"].to_numpy()


def _load_l1_route_frames(
    sim_xml_path: str,
    real_links_csv: str,
    route_stop_dist_csv: str,
    route: str,
    bound: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载文件接口并将任何缺失或畸形仿真输出提升为显式错误。"""
    target_route = str(route)
    target_bound = normalize_bound(bound)
    try:
        sim_raw = load_sim_data(sim_xml_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise L1InfrastructureError(f"failed to load simulation output: {exc}") from exc
    if sim_raw.empty:
        raise L1InfrastructureError(f"simulation output is missing or malformed: {sim_xml_path}")
    try:
        real_links = load_real_link_speeds(real_links_csv)
        dist_df = load_route_stop_dist(route_stop_dist_csv)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise L1InfrastructureError(f"failed to load L1 reference inputs: {exc}") from exc
    _require_columns(dist_df, ("route", "bound", "stop_id", "seq"), "route_stop_dist")
    route_dist = _filter_route_bound(dist_df, target_route, target_bound, "route_stop_dist")
    try:
        sim_traj = build_sim_trajectory(sim_raw, route_dist)
    except Exception as exc:
        raise L1InfrastructureError(f"failed to build simulation trajectory: {exc}") from exc
    if sim_traj.empty:
        raise L1InfrastructureError(
            f"simulation output has no matched stops for route={target_route}, bound={target_bound}"
        )
    sim_traj = sim_traj.copy()
    sim_traj["route"] = target_route
    sim_traj["bound"] = target_bound
    return sim_traj, real_links


def calculate_l1_rmse(sim_xml_path, real_links_csv, route_stop_dist_csv, route='68X', bound='I'):
    """按统一累计时间契约计算站点级 RMSE。"""
    sim_traj, real_links = _load_l1_route_frames(
        sim_xml_path,
        real_links_csv,
        route_stop_dist_csv,
        route,
        bound,
    )
    errors = compute_l1_error_table(sim_traj, real_links, route, bound)["error_s"].to_numpy()
    return float(np.sqrt(np.mean(errors ** 2)))


def calculate_l1_candidate_score(
    sim_xml_path: str,
    real_links_csv: str,
    route_stop_dist_csv: str,
    bound: str = "I",
    rmse_960_limit_s: float = L1_CONSTRAINT_RMSE_960_S,
) -> Dict[str, Any]:
    """文件路径兼容接口：计算一个候选的完整 L1 固定协议得分。"""
    sim_68x, real_links = _load_l1_route_frames(
        sim_xml_path, real_links_csv, route_stop_dist_csv, "68X", bound
    )
    sim_960, _ = _load_l1_route_frames(
        sim_xml_path, real_links_csv, route_stop_dist_csv, "960", bound
    )
    sim_trajectory = pd.concat([sim_68x, sim_960], ignore_index=True)
    return calculate_l1_candidate_score_from_frames(
        sim_trajectory,
        real_links,
        bound=bound,
        rmse_960_limit_s=rmse_960_limit_s,
    )


# =============================================================================
# 鲁棒性目标函数升级 (Week 3)
# =============================================================================

from scipy.stats import ks_2samp, wasserstein_distance as scipy_wasserstein


def calculate_ks_distance(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    """
    计算两组样本的 K-S (Kolmogorov-Smirnov) 统计量
    
    K-S 统计量衡量两个经验分布函数的最大垂直距离，
    值越小表示分布越相似。
    
    Args:
        sim_values: 仿真值数组
        real_values: 真实值数组
        
    Returns:
        ks_stat: K-S 统计量 [0, 1]，越小越好
    """
    if len(sim_values) < 2 or len(real_values) < 2:
        raise L1UnevaluableError("K-S distance requires at least two values per sample")
    
    stat, _ = ks_2samp(sim_values, real_values)
    return stat


def calculate_wasserstein_distance(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    """
    计算 Wasserstein-1 距离 (Earth Mover's Distance)
    
    Wasserstein 距离衡量将一个分布"搬运"到另一个分布所需的最小代价，
    对分布形状和位置都敏感。
    
    Args:
        sim_values: 仿真值数组
        real_values: 真实值数组
        
    Returns:
        wasserstein: Wasserstein-1 距离，越小越好
    """
    if len(sim_values) < 1 or len(real_values) < 1:
        raise L1UnevaluableError("Wasserstein distance requires non-empty samples")
    
    return scipy_wasserstein(sim_values, real_values)


def robust_loss(errors: np.ndarray, lambda_std: float = 0.5) -> float:
    """
    鲁棒综合损失：J = mean(E) + λ * std(E)
    
    通过惩罚误差标准差，鼓励参数在不同时段/场景下表现稳定，
    而非仅优化平均值。
    
    Args:
        errors: 误差数组（绝对误差）
        lambda_std: 标准差惩罚系数，典型值 0.3-0.7
        
    Returns:
        loss: 鲁棒损失值
    """
    if len(errors) == 0:
        raise L1UnevaluableError("robust loss requires at least one error")
    
    mean_e = np.mean(np.abs(errors))
    std_e = np.std(np.abs(errors))
    return mean_e + lambda_std * std_e


def quantile_loss(errors: np.ndarray, quantile: float = 0.9) -> float:
    """
    分位数损失（如 P90）
    
    关注误差分布的尾部（最差情况），
    确保参数在极端情况下也有可接受的表现。
    
    Args:
        errors: 误差数组（绝对误差）
        quantile: 分位数，如 0.9 表示 P90
        
    Returns:
        loss: 分位数损失值
    """
    if len(errors) == 0:
        raise L1UnevaluableError("quantile loss requires at least one error")
    
    return np.quantile(np.abs(errors), quantile)


def calculate_l1_robust_objective(
    sim_xml_path: str,
    real_links_csv: str,
    route_stop_dist_csv: str,
    route: str = '68X',
    bound: str = 'I',
    use_ks: bool = True,
    use_robust: bool = True,
    lambda_std: float = 0.5,
    ks_weight: float = 50.0,
    quantile: float = 0.9
) -> Dict[str, float]:
    """
    综合鲁棒性 L1 目标函数
    
    整合 RMSE、K-S 分布距离、鲁棒损失等多个指标，
    用于多目标优化或加权单目标优化。
    
    Args:
        sim_xml_path: SUMO 仿真输出 XML 路径
        real_links_csv: 真实链路速度 CSV 路径
        route_stop_dist_csv: 路线站点距离 CSV 路径
        route: 线路名称
        bound: 方向 ('I' or 'O')
        use_ks: 是否计算 K-S 统计量
        use_robust: 是否使用 mean+λ*std 鲁棒损失
        lambda_std: 鲁棒损失中的 λ 参数
        ks_weight: K-S 项的权重系数
        quantile: 分位数损失的分位点
        
    Returns:
        dict: 包含各项指标的字典
            - rmse: 传统 RMSE
            - ks_stat: K-S 统计量（站间时间分布）
            - wasserstein: Wasserstein 距离
            - robust_loss: mean + λ*std
            - quantile_loss: P90 分位损失
            - combined: 加权综合目标
    """
    sim_traj, real_links = _load_l1_route_frames(
        sim_xml_path,
        real_links_csv,
        route_stop_dist_csv,
        route,
        bound,
    )
    target_bound = normalize_bound(bound)
    real_links = _filter_route_bound(real_links, route, target_bound, "real_links")
    cum_errors = compute_l1_error_table(
        sim_traj,
        real_links,
        route,
        target_bound,
    )["error_s"].to_numpy()
    rmse = np.sqrt(np.mean(cum_errors ** 2))

    # 计算链路时间分布与链路误差 (用于 KS / Wasserstein / Tail)
    real_link_times = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].apply(list).to_dict()

    sim_link_times = {}
    for _, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        for i in range(len(group) - 1):
            row1 = group.iloc[i]
            row2 = group.iloc[i + 1]
            key = (int(row1['seq']), int(row2['seq']))
            travel_time = row2['arrival_time'] - row1['departure_time']
            if travel_time > 0:
                sim_link_times.setdefault(key, []).append(travel_time)

    all_real_times = []
    all_sim_times = []
    link_errors = []
    for key, sim_times in sim_link_times.items():
        if key in real_link_times:
            real_times = real_link_times[key]
            all_real_times.extend(real_times)
            all_sim_times.extend(sim_times)
            sim_mean = np.mean(sim_times)
            real_mean = np.mean(real_times)
            link_errors.append(sim_mean - real_mean)

    all_real_times = np.array(all_real_times)
    all_sim_times = np.array(all_sim_times)
    if len(link_errors) == 0:
        raise L1UnevaluableError(
            f"route={route}, bound={target_bound} has no matched link-time distributions"
        )

    # K-S 统计量
    ks_stat = calculate_ks_distance(all_sim_times, all_real_times) if use_ks else 0.0

    # Wasserstein 距离
    wasserstein = calculate_wasserstein_distance(all_sim_times, all_real_times)

    # 误差统计 (基于链路误差分布)
    link_errors = np.array(link_errors)
    abs_errors = np.abs(link_errors)
    mean_abs = float(np.mean(abs_errors))
    std_abs = float(np.std(abs_errors))

    # 鲁棒损失
    robust = mean_abs + lambda_std * std_abs if use_robust else rmse

    # 分位数损失
    q_loss = quantile_loss(link_errors, quantile)
    
    # 综合目标（加权组合）
    combined = robust + ks_weight * ks_stat if use_ks else robust
    
    return {
        'rmse': rmse,
        'ks_stat': ks_stat,
        'wasserstein': wasserstein,
        'robust_loss': robust,
        'mean_abs': mean_abs,
        'std_abs': std_abs,
        'quantile_loss': q_loss,
        'combined': combined
    }


def calculate_jl1_loss(
    sim_xml_path: str,
    real_links_csv: str,
    route_stop_dist_csv: str,
    route: str = '68X',
    bound: str = 'I',
    alpha: float = 1.0,
    lambda_std: float = 0.5,
    beta: float = 0.3
) -> Dict[str, Any]:
    """
    计算论文 Eq.(6) 的完整 JL1 复合损失
    
    JL1 = RMSE + α(MAE + λ·std(|ei|)) + β·Q0.9(|e|)
        = √(1/n∑ei²) + α(1/n∑|ei| + λ·std(|ei|)) + β·Q0.9(|e|)
    
    Args:
        sim_xml_path: SUMO 仿真输出 XML 路径
        real_links_csv: 真实链路速度 CSV 路径
        route_stop_dist_csv: 路线站点距离 CSV 路径
        route: 线路名称
        bound: 方向 ('I' or 'O')
        alpha: MAE+dispersion 项权重（论文默认 1.0）
        lambda_std: dispersion 权重（论文默认 0.5）
        beta: 尾部风险权重（论文默认 0.3）
        
    Returns:
        dict: 包含 JL1 各分项的字典
            - rmse_term: RMSE 项
            - mae_term: MAE (mean absolute error)
            - std_term: std(|ei|) 分散度项
            - dispersion_term: MAE + λ·std
            - tail_term: P90 尾部风险项
            - jl1: 完整复合损失
            - errors: 原始误差向量 (用于后续分析)
    """
    sim_traj, real_links = _load_l1_route_frames(
        sim_xml_path,
        real_links_csv,
        route_stop_dist_csv,
        route,
        bound,
    )
    errors = compute_l1_error_table(
        sim_traj,
        real_links,
        route,
        bound,
    )["error_s"].to_numpy()
    metrics = calculate_jl1_from_errors(
        errors,
        alpha=alpha,
        lambda_std=lambda_std,
        beta=beta,
    )
    metrics["route"] = str(route)
    metrics["bound"] = normalize_bound(bound)
    return metrics


if __name__ == "__main__":
    # 简单的冒烟测试逻辑
    print("=== Objective Functions Smoke Test ===")
    
    # 测试基础损失函数
    test_errors = np.array([10, 15, 8, 12, 20, 5])
    print(f"测试误差: {test_errors}")
    print(f"  RMSE: {np.sqrt(np.mean(test_errors**2)):.2f}")
    print(f"  Robust (λ=0.5): {robust_loss(test_errors, 0.5):.2f}")
    print(f"  P90: {quantile_loss(test_errors, 0.9):.2f}")
    
    # 测试分布距离
    sim_vals = np.random.normal(100, 15, 50)
    real_vals = np.random.normal(105, 12, 50)
    print(f"\n分布距离测试:")
    print(f"  K-S: {calculate_ks_distance(sim_vals, real_vals):.4f}")
    print(f"  Wasserstein: {calculate_wasserstein_distance(sim_vals, real_vals):.2f}")
    
    print("\n✓ 目标函数测试通过")

