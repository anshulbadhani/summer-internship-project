"""
Neighborhood vectorization (Section 3.1.1.3a of the paper).

Turns each pixel into a robust local representation:
  - center value  -> median of the 3x3 window (outlier rejection)
  - 8 neighbors   -> average-filtered values at the 8 surrounding
                     positions (spatial smoothing / context)

Output contract (agreed with Person A):
    flatten_pixel_features(image) -> X of shape (N, 9)
        column 0        = median-filtered center value
        columns 1..8    = average-filtered neighbor values
    N = H * W, flattened in row-major (C) order, so pixel index
    i in X corresponds to (row = i // W, col = i % W) in the
    original image. Person A's FCM/ARKFCM must index cluster
    membership arrays with this same flattening convention.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

# Relative (row, col) offsets of the 8 neighbors, center excluded.
# Order is fixed and must stay consistent everywhere we use it.
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def compute_center_map(image: np.ndarray, window: int = 3) -> np.ndarray:
    """Median-filtered image; replaces each pixel's own value.

    Uses reflect padding at borders (mirrors edge pixels rather than
    inventing zeros), matching scipy's default 'reflect' mode.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")
    return median_filter(image.astype(np.float64), size=window, mode="reflect")


def compute_neighbor_map(image: np.ndarray, window: int = 3) -> np.ndarray:
    """Average-filtered image; source for the 8 neighbor-vector values."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")
    return uniform_filter(image.astype(np.float64), size=window, mode="reflect")


def neighborhood_vectors(image: np.ndarray, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-pixel representation.

    Returns:
        center_map: (H, W) float array, median-filtered center values.
        neighbor_vecs: (H, W, 8) float array, average-filtered values
            at each of the 8 neighbor offsets, in _NEIGHBOR_OFFSETS order.
    """
    if window != 3:
        raise NotImplementedError(
            "only window=3 is supported for now; the 8-neighbor offset "
            "table is hardcoded for a 3x3 window. Generalize _NEIGHBOR_OFFSETS "
            "before using a larger window."
        )

    H, W = image.shape
    center_map = compute_center_map(image, window=window)
    smoothed = compute_neighbor_map(image, window=window)

    pad = window // 2
    padded = np.pad(smoothed, pad, mode="reflect")

    neighbor_vecs = np.zeros((H, W, 8), dtype=np.float64)
    for k, (di, dj) in enumerate(_NEIGHBOR_OFFSETS):
        neighbor_vecs[:, :, k] = padded[pad + di: pad + di + H, pad + dj: pad + dj + W]

    return center_map, neighbor_vecs


def flatten_pixel_features(image: np.ndarray, window: int = 3) -> np.ndarray:
    """Build the final (N, 9) feature matrix handed off to Person A's FCM/ARKFCM.

    Column 0 is the median center value, columns 1-8 are the average
    neighbor values. Row-major flattening: row i of X corresponds to
    pixel (i // W, i % W) in the original image.
    """
    center_map, neighbor_vecs = neighborhood_vectors(image, window=window)
    H, W = image.shape

    center_flat = center_map.reshape(H * W, 1)
    neighbor_flat = neighbor_vecs.reshape(H * W, 8)

    return np.concatenate([center_flat, neighbor_flat], axis=1)


if __name__ == "__main__":
    # Same 5x5 toy image we worked through by hand, noise spike at (1, 2).
    TOY_IMAGE = np.array([
        [10, 12, 11, 13, 12],
        [11, 13, 200, 14, 13],
        [12, 14, 15, 16, 14],
        [13, 15, 16, 17, 15],
        [12, 13, 14, 15, 13],
    ], dtype=np.float64)

    center_map = compute_center_map(TOY_IMAGE, window=3)
    assert center_map[1, 2] == 14.0, "median at noisy pixel should be 14"

    neighbor_map = compute_neighbor_map(TOY_IMAGE, window=3)
    expected_mean = (12 + 11 + 13 + 13 + 200 + 14 + 14 + 15 + 16) / 9
    assert np.isclose(neighbor_map[1, 2], expected_mean)

    c_map, n_vecs = neighborhood_vectors(TOY_IMAGE, window=3)
    assert c_map.shape == (5, 5)
    assert n_vecs.shape == (5, 5, 8)
    assert c_map[1, 2] == 14.0
    for k, (di, dj) in enumerate(_NEIGHBOR_OFFSETS):
        assert np.isclose(n_vecs[1, 2, k], neighbor_map[1 + di, 2 + dj])

    X = flatten_pixel_features(TOY_IMAGE, window=3)
    H, W = TOY_IMAGE.shape
    assert X.shape == (H * W, 9)
    flat_idx = 1 * W + 2
    assert X[flat_idx, 0] == c_map[1, 2]
    assert np.allclose(X[flat_idx, 1:], n_vecs[1, 2, :])

    # Border pixel must not crash / produce NaNs under reflect padding.
    assert np.isfinite(c_map[0, 0])
    assert np.all(np.isfinite(n_vecs[0, 0, :]))

    print("neighborhood.py: all self-tests passed.")