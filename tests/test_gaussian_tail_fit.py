import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fitTool import gaussianFit


def test_tail_only_fit_retains_baseline_and_sharpens_profile():
    # Generate a Gaussian profile with a flattened (saturated) core so the
    # standard fit overestimates the width and underestimates the amplitude.
    x = np.linspace(-6.0, 6.0, 401)
    true_center = 0.0
    true_sigma = 1.2
    baseline = 0.1
    true_amplitude = 1.0

    gaussian = true_amplitude * np.exp(-((x - true_center) ** 2) / (2 * true_sigma**2)) + baseline
    clipped = np.minimum(gaussian, baseline + 0.7)

    # Full-profile fit should succeed but be biased by the flattened top.
    full_fit = gaussianFit(
        x,
        clipped,
        ((0, 0), (0, 0)),
        axis="x",
        fit_tails_only=False,
        use_analytic_jacobian=False,
    )

    # Tail-only refit should leverage the same baseline and recover a sharper,
    # higher-amplitude Gaussian from the wings.
    tail_fit = gaussianFit(
        x,
        clipped,
        ((0, 0), (0, 0)),
        axis="x",
        fit_tails_only=True,
        tail_fraction=0.3,
        tail_sigma_multiplier=1.0,
        min_tail_points=20,
        use_analytic_jacobian=False,
    )

    assert full_fit[5] is True
    assert tail_fit[5] is True

    # The tail-informed amplitude should be higher (closer to the true peak).
    assert tail_fit[3] > full_fit[3]

    # The tail-informed width should be narrower than the flattened estimate.
    assert tail_fit[1] < full_fit[1]


def test_side_selected_tail_fit_mirrors_data_and_preserves_baseline():
    x = np.linspace(-6.0, 6.0, 401)
    true_center = 0.0
    true_sigma = 1.2
    baseline = 0.1
    true_amplitude = 1.0

    gaussian = true_amplitude * np.exp(-((x - true_center) ** 2) / (2 * true_sigma**2)) + baseline
    clipped = np.minimum(gaussian, baseline + 0.7)

    full_fit = gaussianFit(
        x,
        clipped,
        ((0, 0), (0, 0)),
        axis="x",
        fit_tails_only=False,
        use_analytic_jacobian=False,
    )

    left_fit = gaussianFit(
        x,
        clipped,
        ((0, 0), (0, 0)),
        axis="x",
        fit_tails_only=True,
        tail_fraction=0.3,
        tail_sigma_multiplier=1.0,
        min_tail_points=20,
        tail_side="left",
        use_analytic_jacobian=False,
    )

    right_fit = gaussianFit(
        x,
        clipped,
        ((0, 0), (0, 0)),
        axis="x",
        fit_tails_only=True,
        tail_fraction=0.3,
        tail_sigma_multiplier=1.0,
        min_tail_points=20,
        tail_side="right",
        use_analytic_jacobian=False,
    )

    assert full_fit[5] is True
    assert left_fit[5] is True
    assert right_fit[5] is True

    # Baseline parameters should remain anchored to the full-profile solution.
    assert np.isclose(left_fit[2], full_fit[2])
    assert np.isclose(left_fit[6], full_fit[6])
    assert np.isclose(right_fit[2], full_fit[2])
    assert np.isclose(right_fit[6], full_fit[6])

    # Mirroring the retained samples keeps the centers aligned with the true peak.
    assert abs(left_fit[0] - full_fit[0]) < 0.05
    assert abs(right_fit[0] - full_fit[0]) < 0.05

    # Single-sided selections still sharpen the fitted width relative to the
    # flattened full-profile solution.
    assert left_fit[1] < full_fit[1]
    assert right_fit[1] < full_fit[1]

    # Left and right selections should converge to similar widths when mirrored.
    assert np.isclose(left_fit[1], right_fit[1], rtol=0.1)
