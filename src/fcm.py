import numpy as np

def fcm(X: np.ndarray, c: int, m: float = 2.0, eps: float = 1e-6, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    X_flat = X.flatten().reshape(-1, 1)
    N = X_flat.shape[0]
    
    # Initialize membership matrix randomly and normalize rows
    U = np.random.rand(N, c)
    U /= np.sum(U, axis=1, keepdims=True)
    
    for _ in range(max_iter):
        U_prev = U.copy()
        Um = U ** m
        
        # Update cluster centers
        V = (Um.T @ X_flat) / np.sum(Um, axis=0, keepdims=True).T
        
        # Compute distances
        D = np.zeros((N, c))
        for j in range(c):
            D[:, j] = np.linalg.norm(X_flat - V[j], axis=1)
            
        # Avoid division by zero
        D = np.maximum(D, 1e-10)
        
        # Update memberships
        for i in range(N):
            for j in range(c):
                U[i, j] = 1.0 / np.sum((D[i, j] / D[i, :]) ** (2 / (m - 1)))
                
        # Check Frobenius norm convergence
        if np.linalg.norm(U - U_prev) < eps:
            break
            
    return U, V.flatten()