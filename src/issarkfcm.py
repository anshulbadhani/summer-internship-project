import numpy as np

from src.fcm import fcm
from src.metrics import vpc, vpe, vxb

from src.ghm_wavelet import ghm_decompose_2d
from src.quadratic_dist import quadratic_distance, fit_quadratic_coeffs
from src.outlier import compute_lvc, compute_phi

from src.init_centers import init_centers_cnn_pso


def issarkfcm(image, c, m=2.0, max_iter=100, eps=1e-5, alpha=0.5):
    """
    Improved Semi-Supervised Adaptive Regularized Kernel FCM (ISSARKFCM)
    """
    H, W = image.shape
    N = H * W
    
    # =========================================================================
    # [1/4] Initializing Cluster Centers via CNN + PSO
    # =========================================================================
    print("[1/4] Initializing Cluster Centers via CNN + PSO...")
    # Person C's logic outputs initial high-quality cluster centers
    V_init = init_centers_cnn_pso(image, c)
    
    # =========================================================================
    # [2/4] Running GHM Multiwavelet & Spatial Extraction
    # =========================================================================
    print("[2/4] Running GHM Multiwavelet & Spatial Extraction...")
    # Person B's modules clean high-frequency noise and build the spatial penalty maps
    low_freq_subband = ghm_decompose_2d(image)
    lvc = compute_lvc(image, window=3)
    
    # Use window=3 directly; Person B's function manages omega internally
    phi = compute_phi(image, lvc, window=3)  
    phi_flat = phi.flatten()
    
    # =========================================================================
    # [3/4] Generating Semi-Supervised Pseudo-Labels
    # =========================================================================
    print("[3/4] Generating Semi-Supervised Pseudo-Labels...")
    # Person A's baseline FCM runs on the downsampled clean wavelet domain
    H_low, W_low = low_freq_subband.shape
    low_freq_flat = low_freq_subband.flatten().reshape(-1, 1)
    U0_low, _ = fcm(low_freq_flat, c=c, m=m, eps=eps, max_iter=30)
    
    # UP-SAMPLE STEP: Map low-res pseudo-labels back to full resolution (N, c)
    U0_spatial = U0_low.reshape(H_low, W_low, c)
    row_scale = H // H_low
    col_scale = W // W_low
    U0_full_spatial = np.repeat(np.repeat(U0_spatial, row_scale, axis=0), col_scale, axis=1)
    U0_full = U0_full_spatial.reshape(N, c)
    
    # =========================================================================
    # [4/4] Executing Main ISSARKFCM Optimization Loop
    # =========================================================================
    print("[4/4] Executing Main ISSARKFCM Optimization Loop...")
    
    # 1. Generate the spatial (x, y) coordinate matrix for all pixels
    X_grid, Y_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1) # Shape: (N, 2)
    values = image.flatten()                                        # Shape: (N,)

    # 2. Initialize a coefficient matrix for all 'c' clusters (6 taps per cluster)
    coeffs = np.zeros((c, 6))
    
    # Pre-fit initial coefficients using the full-resolution pseudo-labels
    for k in range(c):
        coeffs[k] = fit_quadratic_coeffs(coords, values, weights=U0_full[:, k]**m)

    # Initialize membership matrix U and cluster centers V from the CNN+PSO step
    V = V_init.copy()
    U = U0_full.copy()

    # Optimization Loop
    for iteration in range(max_iter):
        V_old = V.copy()
        
        # --- Membership (U) Matrix Update (Fully Vectorized!) ---
        dist_matrix = np.zeros((N, c))
        
        for k in range(c):
            # Compute the quadratic surface distance for ALL pixels at once
            d_qk = quadratic_distance(coords, V[k], coeffs[k]) # Returns shape (N,)
            
            # Combine surface distance, spatial penalization (phi), and semi-supervision (alpha)
            dist_matrix[:, k] = d_qk * (1.0 + phi_flat) + alpha * (1.0 - U0_full[:, k])
        
        # Prevent division by zero
        dist_matrix = np.fmax(dist_matrix, 1e-10)
        
        # Standard Fuzzy C-Means membership recalculation
        inv_dist = dist_matrix ** (-2.0 / (m - 1.0))
        U = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        # --- Cluster Center (V) & Surface Coefficients (Coeffs) Update ---
        for k in range(c):
            weights = U[:, k] ** m
            sum_weights = np.sum(weights)
            
            if sum_weights > 1e-10:
                # Update the cluster center value
                V[k] = np.sum(weights * values) / sum_weights
                
                # Update the bias-field surface coefficients using Person B's least-squares solver
                coeffs[k] = fit_quadratic_coeffs(coords, values, weights=weights)

        # Check convergence
        if np.linalg.norm(V - V_old) < eps:
            print(f"ISSARKFCM converged early at iteration {iteration + 1}")
            break
            
    return U, V