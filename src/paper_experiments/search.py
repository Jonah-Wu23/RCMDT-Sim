"""Deterministic equal-budget BO and continued-LHS search primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc

from src.calibration.surrogate import KrigingSurrogate


@dataclass(frozen=True)
class TargetResult:
    target: float
    evaluation_reached: int | None


def _bounds_array(bounds: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(bounds, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("bounds must have shape (dimensions, 2)")
    if not np.isfinite(array).all() or np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("bounds must be finite and strictly increasing")
    return array


def shared_initial_lhs(
    bounds: Sequence[Sequence[float]],
    *,
    optimization_seed: int,
    evaluations: int = 15,
) -> np.ndarray:
    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    limits = _bounds_array(bounds)
    sampler = qmc.LatinHypercube(d=len(limits), seed=int(optimization_seed))
    return qmc.scale(sampler.random(evaluations), limits[:, 0], limits[:, 1])


def continued_lhs(
    bounds: Sequence[Sequence[float]],
    *,
    optimization_seed: int,
    evaluations: int = 25,
) -> np.ndarray:
    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    limits = _bounds_array(bounds)
    sampler = qmc.LatinHypercube(d=len(limits), seed=50000 + int(optimization_seed))
    return qmc.scale(sampler.random(evaluations), limits[:, 0], limits[:, 1])


def select_bo_candidate(
    evaluated_parameters: np.ndarray,
    evaluated_scores: np.ndarray,
    bounds: Sequence[Sequence[float]],
    *,
    optimization_seed: int,
    evaluation_index: int,
    candidate_pool_size: int = 16384,
) -> np.ndarray:
    parameters = np.asarray(evaluated_parameters, dtype=float)
    scores = np.asarray(evaluated_scores, dtype=float)
    limits = _bounds_array(bounds)
    if parameters.ndim != 2 or parameters.shape[1] != len(limits):
        raise ValueError("evaluated_parameters have the wrong shape")
    if scores.shape != (len(parameters),) or len(scores) < 3:
        raise ValueError("at least three paired evaluated scores are required")
    if not np.isfinite(parameters).all() or not np.isfinite(scores).all():
        raise ValueError("BO training data contain non-finite values")
    power = int(np.log2(candidate_pool_size))
    if 2**power != candidate_pool_size:
        raise ValueError("candidate_pool_size must be a power of two")

    sampler = qmc.Sobol(
        d=len(limits),
        scramble=True,
        seed=500000 + 100 * int(optimization_seed) + int(evaluation_index),
    )
    candidates = qmc.scale(
        sampler.random_base2(power),
        limits[:, 0],
        limits[:, 1],
    )
    duplicate = np.any(
        np.all(np.isclose(candidates[:, None, :], parameters[None, :, :], rtol=0.0, atol=1e-12), axis=2),
        axis=1,
    )
    candidates = candidates[~duplicate]
    if not len(candidates):
        raise ValueError("candidate pool contains no unevaluated point")

    surrogate = KrigingSurrogate(random_state=int(optimization_seed))
    surrogate.fit(parameters, scores)
    expected_improvement = surrogate.expected_improvement(candidates, float(scores.min()))
    return candidates[int(np.argmax(expected_improvement))]


def cumulative_best(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    return np.minimum.accumulate(values)


def predeclared_target(initial_scores: Sequence[float], initial_feasible: Sequence[bool]) -> float:
    scores = np.asarray(initial_scores, dtype=float)
    feasible = np.asarray(initial_feasible, dtype=bool)
    if scores.shape != feasible.shape or scores.ndim != 1:
        raise ValueError("initial score and feasibility vectors differ")
    valid = scores[feasible & np.isfinite(scores)]
    if not len(valid):
        raise ValueError("shared initial design has no feasible candidate")
    return float(0.95 * valid.min())


def target_reach(scores: Sequence[float], target: float) -> TargetResult:
    values = cumulative_best(scores)
    reached = np.flatnonzero(values <= float(target))
    return TargetResult(
        target=float(target),
        evaluation_reached=int(reached[0] + 1) if len(reached) else None,
    )

