"""
Outlier detection (Section 3.1 of the paper): LVC, omega, phi.

These feed directly into Person A's ARKFCM/ISSARKFCM update rule as the
per-pixel adaptive regularization term (phi). Nothing here depends on
Person C's CNN/PSO work.

IMPORTANT ASSUMPTION FLAG:
The source PDF's equations for omega/alpha were partially garbled by OCR
extraction. What's implemented below is a best-faith reconstruction of
the stated form:
    LVC_i = sum_{k in N_i} (x_k - xbar_i)^2 / (NR * xbar_i^2)
    alpha_i = exp(-LVC_i)
    omega_i = alpha_i / sum_{k in N_i, k != i} alpha_k
    phi_i   = 2 + omega_i  if  s_i * xbar_i < x_i
            = 2 - omega_i  if  s_i * xbar_i > x_i
            = 0            if  s_i * xbar_i == x_i
This should be cross-checked against the original paper's equation
numbering before being treated as final -- flagging explicitly rather
than silently guessing.
"""

from __future__ import annotations

import numpy as np

from src.neighborhood import compute_neighbor_map

_EPS = 1e-8  # guards against division by zero in flat/zero-mean regions


def compute_lvc(image: np.ndarray, window: int = 3) -> np.ndarray:
    """Local Coefficient of Variation per pixel.

    High LVC => noisy/heterogeneous neighborhood (e.g. near a noise
    spike or a real edge). Low LVC => smooth, homogeneous neighborhood.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")

    local_mean = compute_neighbor_map(image, window=window)  # xbar_i, reuse Person B's own module
    NR = window * window

    # Local variance: E[(x - mean)^2] over the window, computed via a
    # second uniform filter on squared deviations. We approximate
    # sum_{k in N_i}(x_k - xbar_i)^2 as NR * local_variance_i.
    sq_image = image.astype(np.float64) ** 2
    local_mean_sq = compute_neighbor_map(sq_image, window=window)  # E[x^2]
    local_variance = local_mean_sq - local_mean ** 2  # Var = E[x^2] - E[x]^2
    local_variance = np.clip(local_variance, a_min=0.0, a_max=None)  # guard tiny negatives from fp error

    numerator = NR * local_variance
    denominator = NR * (local_mean ** 2) + _EPS

    return numerator / denominator


def compute_alpha(lvc: np.ndarray) -> np.ndarray:
    """alpha_i = exp(-LVC_i): high LVC (noisy) -> alpha near 0 (low trust)."""
    return np.exp(-lvc)


def compute_omega(image: np.ndarray, window: int = 3) -> np.ndarray:
    """omega_i: alpha_i normalized against its neighbors' alpha values.

    Per-pixel weight used in phi. Relies on the same 3x3 neighborhood
    convention as neighborhood.py.
    """
    if window != 3:
        raise NotImplementedError("only window=3 supported; neighbor offsets are hardcoded")

    lvc = compute_lvc(image, window=window)
    alpha = compute_alpha(lvc)

    H, W = image.shape
    pad = window // 2
    padded_alpha = np.pad(alpha, pad, mode="reflect")

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    neighbor_alpha_sum = np.zeros((H, W), dtype=np.float64)
    for di, dj in offsets:
        neighbor_alpha_sum += padded_alpha[pad + di: pad + di + H, pad + dj: pad + dj + W]

    return alpha / (neighbor_alpha_sum + _EPS)


def compute_phi(image: np.ndarray, s: float = 1.0, window: int = 3) -> np.ndarray:
    """Adaptive regularization phi_i, the piecewise term consumed by ARKFCM.

    s: scaling factor applied to the local mean before comparing against
    the raw pixel value (s_i in the paper -- treated here as a single
    scalar hyperparameter rather than a per-pixel value, since the paper
    doesn't specify how s_i itself is derived. Flag this if it turns out
    to need per-pixel tuning.)
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")

    local_mean = compute_neighbor_map(image, window=window)
    omega = compute_omega(image, window=window)

    scaled_mean = s * local_mean
    phi = np.zeros_like(image, dtype=np.float64)

    below = scaled_mean < image  # s_i * xbar_i < x_i
    above = scaled_mean > image  # s_i * xbar_i > x_i
    # equal case left at the initialized 0.0

    phi[below] = 2.0 + omega[below]
    phi[above] = 2.0 - omega[above]

    return phi


if __name__ == "__main__":
    # Same 5x5 toy image, noise spike at (1, 2).
    TOY_IMAGE = np.array([
        [10, 12, 11, 13, 12],
        [11, 13, 200, 14, 13],
        [12, 14, 15, 16, 14],
        [13, 15, 16, 17, 15],
        [12, 13, 14, 15, 13],
    ], dtype=np.float64)

    lvc = compute_lvc(TOY_IMAGE, window=3)
    assert lvc.shape == (5, 5)
    # The noisy pixel's neighborhood should have far higher LVC than a
    # flat region elsewhere in the image.
    flat_region_lvc = lvc[3, 3]  # deep in a smooth gradient area
    noisy_region_lvc = lvc[1, 2]
    assert noisy_region_lvc > flat_region_lvc, (
        f"expected noisy region LVC ({noisy_region_lvc}) > flat region LVC ({flat_region_lvc})"
    )
    print(f"LVC at noisy pixel (1,2): {noisy_region_lvc:.4f}")
    print(f"LVC at flat pixel  (3,3): {flat_region_lvc:.4f}")

    alpha = compute_alpha(lvc)
    assert np.all(alpha > 0) and np.all(alpha <= 1.0)
    # Noisy pixel should have lower alpha (less trust) than the flat pixel.
    assert alpha[1, 2] < alpha[3, 3]

    omega = compute_omega(TOY_IMAGE, window=3)
    assert omega.shape == (5, 5)
    assert np.all(np.isfinite(omega))

    phi = compute_phi(TOY_IMAGE, s=1.0, window=3)
    assert phi.shape == (5, 5)
    # phi should only ever take values in {0} union (2-omega, 2+omega) range,
    # i.e. never wildly outside [0, 3] given omega is a normalized weight <= 1.
    assert np.all(phi >= 0.0)
    assert np.all(phi <= 3.0 + 1e-6)

    print("outlier.py: all self-tests passed.")