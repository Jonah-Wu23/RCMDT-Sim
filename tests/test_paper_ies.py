from __future__ import annotations

import numpy as np

from src.paper_experiments.ies import run_ies


def test_ies_is_reproducible_and_moves_toward_observation() -> None:
    def simulate(member: np.ndarray, iteration: int, index: int) -> np.ndarray:
        del iteration, index
        return np.array([member[0] + 2.0 * member[1]])

    kwargs = dict(
        prior_mean=[0.0, 0.0],
        prior_std=[1.0, 1.0],
        bounds=[[-3.0, 3.0], [-3.0, 3.0]],
        observation=[2.0],
        observation_std=[0.2],
        seed=4,
        simulate=simulate,
        ensemble_size=10,
        iterations=3,
        damping=0.3,
    )
    first = run_ies(**kwargs)
    second = run_ies(**kwargs)
    assert np.allclose(first.final_mean, second.final_mean)
    assert first.final_mean[0] + 2 * first.final_mean[1] > 0
    assert len(first.iterations) == 3

