import numpy as np

def kfcm(X: np.ndarray, c: int, m: float = 2.0, sigma: float = 15.0, eps: float = 1e-6, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    X_flat = X.flatten().reshape(-1, 1)
    N = X_flat.shape[0]
    
    U = np.random.rand(N, c)
    U /= np.sum(U, axis=1, keepdims=True)
    V = np.linspace(np.min(X_flat), np.max(X_flat), c)
    
    for _ in range(max_iter):
        U_prev = U.copy()
        Um = U ** m
        
        # Compute Gaussian Kernel matrix Q between X and V
        Q = np.zeros((N, c))
        for j in range(c):
            Q[:, j] = np.exp(-((X_flat.flatten() - V[j]) ** 2) / (sigma ** 2))
            
        # Update Cluster Centers V
        for j in range(c):
            num = np.sum(Um[:, j] * Q[:, j] * X_flat.flatten())
            den = np.sum(Um[:, j] * Q[:, j])
            V[j] = num / (den + 1e-10)
            
        # Update Membership Matrix U
        # Objective utilizes 1 - Q(x, v) as the distance metric
        dist = 2.0 * (1.0 - Q)
        dist = np.maximum(dist, 1e-10)
        
        power = -1.0 / (m - 1)
        inv_dist = dist ** power
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        
        if np.linalg.norm(U - U_prev) < eps:
            break
            
    return U, V