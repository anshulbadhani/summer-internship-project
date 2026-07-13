"""
Quadratic polynomial distance (Section 3.1.1.1 of the paper).

Models each cluster's expected intensity as a smooth quadratic function
of spatial position (x, y), to account for MRI intensity inhomogeneity
(bias field), rather than comparing raw pixel intensity to a single
fixed cluster center value.

    f(x, y) = a0*x^2 + a1*y^2 + a2*x*y + a3*x + a4*y + a5
    distance(pixel, Vk) = (f(x, y) - Vk)^2

ASSUMPTION FLAG: the paper does not specify how a0..a5 are fit. This
implements the natural interpretation -- weighted least squares per
cluster, using fuzzy membership u_ik^m as weights and pixel intensity
as the regression target, refit every outer clustering iteration. Flag
for cross-check against Person A once the full ISSARKFCM loop is wired
together, since this determines how often/where fitting happens.
"""

from __future__ import annotations

import numpy as np


def _design_matrix(coords: np.ndarray) -> np.ndarray:
    """Build the [x^2, y^2, xy, x, y, 1] design matrix for a set of (x, y) coords.

    coords: (N, 2) array of (x, y) spatial positions.
    Returns: (N, 6) design matrix.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return np.stack([x**2, y**2, x * y, x, y, np.ones_like(x)], axis=1)


def fit_quadratic_coeffs(
    coords: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted least-squares fit of a0..a5.

    coords:  (N, 2) spatial (x, y) positions.
    values:  (N,) target intensities (e.g. raw pixel values).
    weights: (N,) optional weights (e.g. u_ik^m, fuzzy membership to
             cluster k raised to the fuzzifier power). If None, an
             unweighted fit is performed.

    Returns: (6,) array [a0, a1, a2, a3, a4, a5].
    """
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords and values must have the same number of rows")

    A = _design_matrix(coords)

    if weights is None:
        coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
        return coeffs

    if weights.shape[0] != values.shape[0]:
        raise ValueError("weights must match values in length")

    # Weighted least squares: scale rows by sqrt(weight) before solving,
    # equivalent to minimizing sum(w_i * (A_i @ coeffs - value_i)^2).
    sqrt_w = np.sqrt(np.clip(weights, a_min=0.0, a_max=None))
    A_weighted = A * sqrt_w[:, None]
    values_weighted = values * sqrt_w

    coeffs, *_ = np.linalg.lstsq(A_weighted, values_weighted, rcond=None)
    return coeffs


def quadratic_transform(coords: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate f(x, y) at each coordinate given fitted coefficients."""
    A = _design_matrix(coords)
    return A @ coeffs


def quadratic_distance(
    coords: np.ndarray,
    Vk: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Squared distance (Eq. 18): (f(x,y) - Vk)^2, per pixel.

    coords: (N, 2) spatial positions for the pixels being scored against
            cluster k.
    Vk:     scalar cluster center value for cluster k.
    coeffs: (6,) fitted [a0..a5] for cluster k.

    Returns: (N,) squared distances.
    """
    f_xy = quadratic_transform(coords, coeffs)
    return (f_xy - Vk) ** 2


if __name__ == "__main__":
    # Hand-check: a known linear surface f(x,y) = x + y (a3=1, a4=1, rest 0)
    # over a 3x3 coordinate grid. Fitting should recover these exactly.
    xs, ys = np.meshgrid(np.arange(3), np.arange(3))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    true_coeffs = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    values = quadratic_transform(coords, true_coeffs)  # exact, no noise

    fitted_unweighted = fit_quadratic_coeffs(coords, values)
    assert np.allclose(fitted_unweighted, true_coeffs, atol=1e-8), (
        f"expected {true_coeffs}, got {fitted_unweighted}"
    )
    print(f"unweighted fit recovered coeffs: {fitted_unweighted}")

    # Weighted fit: give one point an outlier value with near-zero weight,
    # verify the fit still recovers the true surface (outlier essentially ignored).
    values_with_outlier = values.copy()
    values_with_outlier[4] = 999.0  # center pixel corrupted
    weights = np.ones(9)
    weights[4] = 1e-6  # near-zero trust in the corrupted pixel

    fitted_weighted = fit_quadratic_coeffs(coords, values_with_outlier, weights=weights)
    assert np.allclose(fitted_weighted, true_coeffs, atol=1e-2), (
        f"expected {true_coeffs}, got {fitted_weighted}"
    )
    print(f"weighted fit (outlier down-weighted) recovered coeffs: {fitted_weighted}")

    # quadratic_distance sanity check: distance to the true value at each
    # point should be ~0 when Vk equals f(x,y) exactly, and grow when Vk is off.
    Vk_correct = values[0]  # f(0,0) = 0
    dist_correct = quadratic_distance(coords[:1], Vk_correct, true_coeffs)
    assert np.isclose(dist_correct[0], 0.0)

    Vk_wrong = values[0] + 5.0
    dist_wrong = quadratic_distance(coords[:1], Vk_wrong, true_coeffs)
    assert np.isclose(dist_wrong[0], 25.0)

    print("quadratic_dist.py: all self-tests passed.")