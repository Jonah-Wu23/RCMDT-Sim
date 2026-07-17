from __future__ import annotations

import numpy as np
import pytest

from src.paper_experiments.search import (
    continued_lhs,
    cumulative_best,
    predeclared_target,
    select_bo_candidate,
    shared_initial_lhs,
    target_reach,
)


def test_lhs_designs_are_deterministic_distinct_and_in_bounds() -> None:
    bounds = [[0.0, 1.0], [10.0, 20.0]]
    first = shared_initial_lhs(bounds, optimization_seed=3)
    repeated = shared_initial_lhs(bounds, optimization_seed=3)
    continued = continued_lhs(bounds, optimization_seed=3)
    assert first.shape == (15, 2)
    assert continued.shape == (25, 2)
    assert np.allclose(first, repeated)
    assert not np.allclose(first[:10], continued[:10])
    assert np.all((first[:, 0] >= 0) & (first[:, 0] <= 1))
    assert np.all((first[:, 1] >= 10) & (first[:, 1] <= 20))


def test_target_uses_only_feasible_initial_scores_and_reach_is_one_based() -> None:
    target = predeclared_target([100.0, 80.0, 1.0], [True, True, False])
    assert target == pytest.approx(76.0)
    assert target_reach([100.0, 90.0, 75.0, 70.0], target).evaluation_reached == 3
    assert target_reach([100.0, 90.0], target).evaluation_reached is None
    np.testing.assert_allclose(cumulative_best([5.0, 6.0, 4.0]), [5.0, 5.0, 4.0])


def test_initial_design_without_feasible_candidate_is_invalid() -> None:
    with pytest.raises(ValueError, match="no feasible candidate"):
        predeclared_target([10.0, 20.0], [False, False])


def test_bo_candidate_selection_is_deterministic_and_bounded() -> None:
    parameters = np.array([[0.0], [0.5], [1.0]])
    scores = np.array([1.0, 0.2, 1.1])
    first = select_bo_candidate(
        parameters,
        scores,
        [[0.0, 1.0]],
        optimization_seed=2,
        evaluation_index=15,
        candidate_pool_size=16,
    )
    repeated = select_bo_candidate(
        parameters,
        scores,
        [[0.0, 1.0]],
        optimization_seed=2,
        evaluation_index=15,
        candidate_pool_size=16,
    )
    assert np.allclose(first, repeated)
    assert 0.0 <= first[0] <= 1.0
