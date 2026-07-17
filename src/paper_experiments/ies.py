"""Small, deterministic iterative ensemble smoother used by the paper runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class IESIteration:
    iteration: int
    ensemble_seed: int
    mean_before: tuple[float, ...]
    mean_after: tuple[float, ...]
    rmse: float
    clipped_components: int


@dataclass(frozen=True)
class IESResult:
    final_mean: np.ndarray
    iterations: tuple[IESIteration, ...]
    ensembles: tuple[np.ndarray, ...]
    simulations: tuple[np.ndarray, ...]


def ies_mean_update(
    ensemble: np.ndarray,
    simulated: np.ndarray,
    observation: np.ndarray,
    observation_variance: np.ndarray,
    mean_before: np.ndarray,
    bounds: np.ndarray,
    *,
    damping: float,
    es_mda_alpha: float,
    nugget_ratio: float = 0.05,
) -> tuple[np.ndarray, float, int]:
    if ensemble.ndim != 2 or simulated.ndim != 2:
        raise ValueError("ensemble and simulated arrays must be two-dimensional")
    if len(ensemble) != len(simulated) or len(ensemble) < 3:
        raise ValueError("at least three paired ensemble simulations are required")
    if simulated.shape[1] != len(observation):
        raise ValueError("simulated and observation dimensions differ")
    if not np.isfinite(ensemble).all() or not np.isfinite(simulated).all():
        raise ValueError("IES inputs contain non-finite values")

    x_mean = ensemble.mean(axis=0)
    y_mean = simulated.mean(axis=0)
    x_anomaly = ensemble - x_mean
    y_anomaly = simulated - y_mean
    divisor = len(ensemble) - 1
    c_xy = x_anomaly.T @ y_anomaly / divisor
    c_yy = y_anomaly.T @ y_anomaly / divisor
    diagonal_mean = float(np.mean(np.diag(c_yy))) if c_yy.size else 0.0
    nugget = nugget_ratio * diagonal_mean
    system = c_yy + nugget * np.eye(c_yy.shape[0]) + np.diag(observation_variance) * es_mda_alpha
    innovation = observation - y_mean
    try:
        delta = c_xy @ np.linalg.solve(system, innovation)
    except np.linalg.LinAlgError:
        delta = c_xy @ np.linalg.lstsq(system, innovation, rcond=None)[0]
    updated_raw = mean_before + damping * delta
    updated = np.clip(updated_raw, bounds[:, 0], bounds[:, 1])
    clipped = int(np.count_nonzero(updated != updated_raw))
    rmse = float(np.sqrt(np.mean((observation - y_mean) ** 2)))
    return updated, rmse, clipped


def run_ies(
    *,
    prior_mean: Sequence[float],
    prior_std: Sequence[float],
    bounds: Sequence[Sequence[float]],
    observation: Sequence[float],
    observation_std: Sequence[float],
    seed: int,
    simulate: Callable[[np.ndarray, int, int], np.ndarray],
    ensemble_size: int = 10,
    iterations: int = 3,
    damping: float = 0.3,
    variance_floor: float = 1.0,
) -> IESResult:
    if ensemble_size < 3 or iterations < 1:
        raise ValueError("IES requires at least three members and one iteration")
    mean = np.asarray(prior_mean, dtype=float)
    std = np.asarray(prior_std, dtype=float)
    bounds_array = np.asarray(bounds, dtype=float)
    observed = np.asarray(observation, dtype=float)
    observed_std = np.asarray(observation_std, dtype=float)
    if bounds_array.shape != (len(mean), 2) or len(std) != len(mean):
        raise ValueError("Prior and bound dimensions differ")
    if len(observed_std) != len(observed):
        raise ValueError("Observation standard deviations have the wrong dimension")
    covariance = np.diag(std**2)
    observation_variance = np.maximum(observed_std**2, variance_floor**2)
    iteration_records: list[IESIteration] = []
    ensemble_records: list[np.ndarray] = []
    simulation_records: list[np.ndarray] = []

    for iteration in range(1, iterations + 1):
        ensemble_seed = 10000 * int(seed) + 100 * iteration
        rng = np.random.default_rng(ensemble_seed)
        ensemble = rng.multivariate_normal(mean, covariance, size=ensemble_size)
        ensemble = np.clip(ensemble, bounds_array[:, 0], bounds_array[:, 1])
        simulated = np.vstack(
            [np.asarray(simulate(member, iteration, index), dtype=float) for index, member in enumerate(ensemble)]
        )
        mean_before = mean.copy()
        mean, rmse, clipped = ies_mean_update(
            ensemble,
            simulated,
            observed,
            observation_variance,
            mean,
            bounds_array,
            damping=damping,
            es_mda_alpha=float(iterations),
        )
        iteration_records.append(
            IESIteration(
                iteration=iteration,
                ensemble_seed=ensemble_seed,
                mean_before=tuple(float(value) for value in mean_before),
                mean_after=tuple(float(value) for value in mean),
                rmse=rmse,
                clipped_components=clipped,
            )
        )
        ensemble_records.append(ensemble)
        simulation_records.append(simulated)

    return IESResult(
        final_mean=mean,
        iterations=tuple(iteration_records),
        ensembles=tuple(ensemble_records),
        simulations=tuple(simulation_records),
    )

