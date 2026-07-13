import numpy as np

def vpc(U: np.ndarray) -> float:
    N = U.shape[0]
    return float(np.sum(U ** 2) / N)

def vpe(U: np.ndarray) -> float:
    N = U.shape[0]
  
    # Avoid log(0) by adding a tiny epsilon value
  
    eps = 1e-10
    return float(-np.sum(U * np.log(U + eps)) / N)

def vxb(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    N, C = U.shape
    X_flat = X.reshape(-1, 1) if X.ndim > 1 else X.reshape(-1, 1)
    V_flat = V.reshape(-1, 1)
    
    # Calculate numerator: sum of squared distances weighted by U^2
    num = 0.0
    for i in range(C):
        dist_sq = np.sum((X_flat - V_flat[i]) ** 2, axis=1)
        num += np.sum((U[:, i] ** 2) * dist_sq)
        
    # Calculate denominator minimum distance between different cluster centers
    v_dist = pdist_sq = np.sum((V_flat[:, None, :] - V_flat[None, :, :]) ** 2, axis=-1)
    
    np.fill_diagonal(v_dist, np.inf) # Ignore self-distance
    min_v_dist = np.min(v_dist)
    
    return float(num / (N * min_v_dist))