"""
GHM multiwavelet decomposition (Section 3.1.1.1 / Figs 1-3 of the paper).

GHM (Geronimo-Hardin-Massopust) is a multiwavelet with multiplicity 2:
two scaling functions and two wavelet functions, using 2x2 matrix-valued
filter taps instead of scalar taps. This means it operates on 2-vector
samples, not scalars -- hence the "prefilter" step below.

Filter matrices verified against two independent published sources
(a MATLAB File Exchange multiwavelet-tools implementation, and a
peer-reviewed multiwavelet paper) that agree exactly:
    H0=[3/(5*sqrt(2)),4/5;-1/20,-3/(10*sqrt(2))]
    H1=[3/(5*sqrt(2)),0;9/20,1/sqrt(2)]
    H2=[0,0;9/20,-3/(10*sqrt(2))]
    H3=[0,0;-1/20,0]
    G0=[-1/20,-3/(10*sqrt(2));1/(10*sqrt(2)),3/10]
    G1=[9/20,-1/sqrt(2);-9/(10*sqrt(2)),0]
    G2=[9/20,-3/(10*sqrt(2));9/(10*sqrt(2)),-3/10]
    G3=[-1/20,0;-1/(10*sqrt(2)),0]

ASSUMPTION FLAG: prefiltering uses the simplest "repeated row" scheme
(duplicate each scalar into both vector components), not the more
accurate "approximation" prefilter used in some multiwavelet image-
processing literature. Documented tradeoff, not a hidden shortcut --
revisit if reconstruction fidelity on real images turns out to matter
more than pipeline simplicity.
"""

from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)

H0 = np.array([[3 / (5 * SQRT2), 4 / 5], [-1 / 20, -3 / (10 * SQRT2)]])
H1 = np.array([[3 / (5 * SQRT2), 0], [9 / 20, 1 / SQRT2]])
H2 = np.array([[0, 0], [9 / 20, -3 / (10 * SQRT2)]])
H3 = np.array([[0, 0], [-1 / 20, 0]])
H_TAPS = [H0, H1, H2, H3]

G0 = np.array([[-1 / 20, -3 / (10 * SQRT2)], [1 / (10 * SQRT2), 3 / 10]])
G1 = np.array([[9 / 20, -1 / SQRT2], [-9 / (10 * SQRT2), 0]])
G2 = np.array([[9 / 20, -3 / (10 * SQRT2)], [9 / (10 * SQRT2), -3 / 10]])
G3 = np.array([[-1 / 20, 0], [-1 / (10 * SQRT2), 0]])
G_TAPS = [G0, G1, G2, G3]


def prefilter_repeated_row(x: np.ndarray) -> np.ndarray:
    """Scalar signal (N,) -> vector signal (2, N) by duplicating each sample."""
    return np.vstack([x, x])


def postfilter_average(V: np.ndarray) -> np.ndarray:
    """Vector signal (2, N) -> scalar signal (N,) by averaging the two rows.

    Approximate inverse of the repeated-row prefilter.
    """
    return V.mean(axis=0)


def ghm_forward_1d(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-level GHM analysis on a 2-channel vector signal.

    V: (2, N) vector signal, N must be even. Circular (periodic) boundary.
    Returns:
        C: (2, N//2) approximation coefficients.
        D: (2, N//2) detail coefficients.
    """
    if V.ndim != 2 or V.shape[0] != 2:
        raise ValueError(f"expected V of shape (2, N), got {V.shape}")
    N = V.shape[1]
    if N % 2 != 0:
        raise ValueError(f"N must be even, got {N}")

    half = N // 2
    C = np.zeros((2, half))
    D = np.zeros((2, half))

    for n in range(half):
        idxs = [(2 * n + k) % N for k in range(4)]
        c = sum(H_TAPS[k] @ V[:, idxs[k]] for k in range(4))
        d = sum(G_TAPS[k] @ V[:, idxs[k]] for k in range(4))
        C[:, n] = c
        D[:, n] = d

    return C, D


def ghm_inverse_1d(C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Inverse of ghm_forward_1d, via transpose (overlap-add) synthesis.

    Valid because GHM is an orthogonal (paraunitary) multiwavelet --
    the synthesis filters are the transposes of the analysis filters.
    """
    half = C.shape[1]
    N = half * 2
    V = np.zeros((2, N))

    for n in range(half):
        for k in range(4):
            idx = (2 * n + k) % N
            V[:, idx] += H_TAPS[k].T @ C[:, n] + G_TAPS[k].T @ D[:, n]

    return V


def ghm_decompose_2d(image: np.ndarray) -> np.ndarray:
    """Single-level 2D low-frequency subband, via separable row+column transform.

    Simplification (flagged): keeps only the approximation (C) band at
    each stage, collapsing back to scalar via postfilter_average before
    the next stage, rather than propagating the full 2-channel detail
    structure through a true 2D multiwavelet tree. This matches what's
    actually needed downstream: a coarse low-resolution image to run
    initial FCM on (Section 3.1.1.1's "apply conventional FCM to this
    [low-frequency] image").

    image: (H, W), H and W must be even.
    Returns: (H//2, W//2) low-frequency subband.
    """
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    H, W = image.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"H and W must both be even, got ({H}, {W})")

    # Row pass: transform each row, keep only the approximation band.
    row_low = np.zeros((H, W // 2))
    for i in range(H):
        V = prefilter_repeated_row(image[i, :])
        C, _ = ghm_forward_1d(V)
        row_low[i, :] = postfilter_average(C)

    # Column pass: transform each column of the row-reduced image.
    low = np.zeros((H // 2, W // 2))
    for j in range(W // 2):
        V = prefilter_repeated_row(row_low[:, j])
        C, _ = ghm_forward_1d(V)
        low[:, j] = postfilter_average(C)

    return low


def upsample_labels(labels: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample of low-resolution FCM labels back to full resolution.

    Used to bring the low-frequency subband's initial segmentation labels
    back up for the beta_ij similarity term -- exact wavelet-domain label
    reconstruction isn't meaningful for discrete cluster labels, so
    nearest-neighbor (block repeat) is the standard, simplest choice here.
    """
    H_target, W_target = target_shape
    H_low, W_low = labels.shape
    row_scale = H_target // H_low
    col_scale = W_target // W_low
    return np.repeat(np.repeat(labels, row_scale, axis=0), col_scale, axis=1)


if __name__ == "__main__":
    # --- 1D round-trip: the real correctness test for the filter matrices ---
    rng = np.random.default_rng(42)
    x = rng.normal(size=8)
    V = prefilter_repeated_row(x)
    assert V.shape == (2, 8)

    C, D = ghm_forward_1d(V)
    assert C.shape == (2, 4) and D.shape == (2, 4)

    V_reconstructed = ghm_inverse_1d(C, D)
    assert np.allclose(V, V_reconstructed, atol=1e-10), (
        "GHM forward+inverse did not perfectly reconstruct -- filter "
        "matrices or synthesis formula likely wrong"
    )
    print("1D GHM round-trip: perfect reconstruction confirmed.")

    x_reconstructed = postfilter_average(V_reconstructed)
    assert np.allclose(x, x_reconstructed, atol=1e-10)
    print("1D scalar round-trip (prefilter -> transform -> inverse -> postfilter): confirmed.")

    # --- 2D low-frequency subband: shape and smoothing sanity checks ---
    toy_image = np.array([
        [10, 12, 11, 13, 12, 11, 10, 12],
        [11, 13, 200, 14, 13, 12, 11, 13],
        [12, 14, 15, 16, 14, 13, 12, 14],
        [13, 15, 16, 17, 15, 14, 13, 15],
        [12, 13, 14, 15, 13, 12, 11, 13],
        [11, 12, 13, 14, 12, 11, 10, 12],
        [10, 11, 12, 13, 11, 10, 9, 11],
        [9, 10, 11, 12, 10, 9, 8, 10],
    ], dtype=np.float64)

    low = ghm_decompose_2d(toy_image)
    assert low.shape == (4, 4)
    assert np.all(np.isfinite(low))
    print(f"2D low-frequency subband shape: {low.shape}")

    # --- label upsampling sanity check ---
    toy_labels = np.array([[0, 1], [1, 0]])
    upsampled = upsample_labels(toy_labels, target_shape=(8, 8))
    assert upsampled.shape == (8, 8)
    # top-left 4x4 block should all be label 0, matching toy_labels[0,0]
    assert np.all(upsampled[:4, :4] == 0)
    assert np.all(upsampled[:4, 4:] == 1)
    print("label upsampling: block structure confirmed.")

    print("ghm_wavelet.py: all self-tests passed.")