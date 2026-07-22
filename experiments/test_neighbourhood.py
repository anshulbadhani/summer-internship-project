import numpy as np

from src.neighborhood import (
    compute_center_map,
    compute_neighbor_map,
    flatten_pixel_features,
    neighborhood_vectors,
)

# Same 5x5 toy image we worked through by hand, with a noise spike at (1, 2).
TOY_IMAGE = np.array([
    [10, 12, 11, 13, 12],
    [11, 13, 200, 14, 13],
    [12, 14, 15, 16, 14],
    [13, 15, 16, 17, 15],
    [12, 13, 14, 15, 13],
], dtype=np.float64)


def test_center_map_rejects_outlier_via_median():
    center_map = compute_center_map(TOY_IMAGE, window=3)
    # Hand-computed: window around (1,2) sorted -> median = 14
    assert center_map[1, 2] == 14.0


def test_neighbor_map_is_smoothed_average():
    neighbor_map = compute_neighbor_map(TOY_IMAGE, window=3)
    # Hand-computed: mean of the 9 values around (1,2) = 308/9
    expected = (12 + 11 + 13 + 13 + 200 + 14 + 14 + 15 + 16) / 9
    assert np.isclose(neighbor_map[1, 2], expected)


def test_neighborhood_vectors_shape_and_values():
    center_map, neighbor_vecs = neighborhood_vectors(TOY_IMAGE, window=3)
    assert center_map.shape == (5, 5)
    assert neighbor_vecs.shape == (5, 5, 8)

    # Center of (1,2) should be the rejected-outlier median, not the raw 200.
    assert center_map[1, 2] == 14.0

    # Every neighbor-vector entry should come from the *smoothed* map,
    # i.e. neighbor_vecs[1, 2, k] should equal compute_neighbor_map at
    # the corresponding offset position.
    smoothed = compute_neighbor_map(TOY_IMAGE, window=3)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]
    for k, (di, dj) in enumerate(offsets):
        assert np.isclose(neighbor_vecs[1, 2, k], smoothed[1 + di, 2 + dj])


def test_flatten_pixel_features_shape_and_flattening_order():
    X = flatten_pixel_features(TOY_IMAGE, window=3)
    H, W = TOY_IMAGE.shape
    assert X.shape == (H * W, 9)

    # Row-major flattening: pixel (1, 2) is at flat index 1*5 + 2 = 7.
    center_map, neighbor_vecs = neighborhood_vectors(TOY_IMAGE, window=3)
    flat_idx = 1 * W + 2
    assert X[flat_idx, 0] == center_map[1, 2]
    assert np.allclose(X[flat_idx, 1:], neighbor_vecs[1, 2, :])


def test_border_pixel_does_not_crash_and_uses_reflect_padding():
    # Corner pixel (0, 0) has no real neighbors above/left; reflect
    # padding should mirror in-bounds values rather than raising or
    # silently using zeros.
    center_map, neighbor_vecs = neighborhood_vectors(TOY_IMAGE, window=3)
    assert np.isfinite(center_map[0, 0])
    assert np.all(np.isfinite(neighbor_vecs[0, 0, :]))