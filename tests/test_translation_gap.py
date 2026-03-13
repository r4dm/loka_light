import pytest
import numpy as np

from loka_light.physics.translation_gap import translation_gap


@pytest.mark.parametrize("n", [3, 5, 7])
def test_translation_gap_match_vs_mismatch_visibility_for_odd_n(n: int) -> None:
    result, _x, _p_match, _p_mismatch, _y_match, _y_mismatch = translation_gap(n, seed=0, sigma_clean=True)
    assert np.isclose(abs(result.sigma_after_purify), 0.0, atol=1e-12)
    assert np.isclose(result.visibility_match, 1.0, atol=1e-12)
    assert np.isclose(result.visibility_mismatch, 0.0, atol=1e-12)


def test_translation_gap_unclean_branch_preserves_projection_artifact() -> None:
    result, _x, _p_match, _p_mismatch, _y_match, _y_mismatch = translation_gap(5, seed=0, sigma_clean=False)

    assert abs(result.sigma_after_purify) > 1e-6
    assert np.isclose(result.visibility_match, 1.0, atol=1e-12)
    assert np.isclose(result.visibility_mismatch, 0.0, atol=1e-12)

