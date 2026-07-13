# src/arkfcm.py
import numpy as np
from scipy.ndimage import uniform_filter

def arkfcm(X: np.ndarray, c: int, m: float = 2.0, sigma: float = 15.0, eps: float = 1e-6, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    # 1. Precompute spatial structural neighborhood configurations
    X_orig = X.astype(np.float32)
    # 3x3 window mean filter to get x_bar
    X_bar = uniform_filter(X_orig, size=3)
    
    # Local Coefficient of Variation (LVC)
    mean_sq = X_bar ** 2
    local_var = uniform_filter(X_orig ** 2, size=3) - mean_sq
    local_var = np.maximum(local_var, 1e-10)
    
    LVC = local_var / (9.0 * mean_sq + 1e-10)
    omega = np.exp(-LVC) 
    
    # Define adaptive parameter phi
    phi = np.zeros_like(X_orig)
    phi[X_bar < X_orig] = 2.0 + omega[X_bar < X_orig]
    phi[X_bar > X_orig] = 2.0 - omega[X_bar > X_orig]
    phi[X_bar == X_orig] = 0.0
    
    # Flatten for clustering operations
    X_flat = X_orig.flatten()
    X_bar_flat = X_bar.flatten()
    phi_flat = phi.flatten()
    N = X_flat.shape[0]
    
    # Initialize U and V
    U = np.random.rand(N, c)
    U /= np.sum(U, axis=1, keepdims=True)
    V = np.linspace(np.min(X_flat), np.max(X_flat), c)
    
    for _ in range(max_iter):
        U_prev = U.copy()
        Um = U ** m
        
        # Compute kernels
        Q_x = np.zeros((N, c))
        Q_bar = np.zeros((N, c))
        for j in range(c):
            Q_x[:, j] = np.exp(-((X_flat - V[j]) ** 2) / (sigma ** 2))
            Q_bar[:, j] = np.exp(-((X_bar_flat - V[j]) ** 2) / (sigma ** 2))
            
        # Update Cluster Centers V
        for j in range(c):
            num = np.sum(Um[:, j] * (Q_x[:, j] * X_flat + phi_flat * Q_bar[:, j] * X_bar_flat))
            den = np.sum(Um[:, j] * (Q_x[:, j] + phi_flat * Q_bar[:, j]))
            V[j] = num / (den + 1e-10)
            
        # Update Membership Matrix U
        dist = (1.0 - Q_x) + phi_flat[:, None] * (1.0 - Q_bar)
        dist = np.maximum(dist, 1e-10)
        
        power = -1.0 / (m - 1)
        inv_dist = dist ** power
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        
        if np.linalg.norm(U - U_prev) < eps:
            break
            
    return U, V