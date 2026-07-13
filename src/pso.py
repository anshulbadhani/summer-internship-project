import numpy as np

def pso_optimize(
    fitness_fn,
    bounds,
    n_particles=12,
    c1=1.70,
    c2=1.70,
    nbmaxiter=300,
    nerp=10,
    eps=1e-6,
    verbose=True
) -> np.ndarray:
    """
    Particle Swarm Optimization (PSO) loop.
    
    Implements parameters from Table 1 and Section 3.1.1.2:
    - w: inertia weight linearly decreasing from 0.9 to 0.4 over nbmaxiter iterations.
    - c1: cognitive parameter (default: 1.70).
    - c2: social parameter (default: 1.70).
    - nbmaxiter: maximum iterations (default: 300).
    - nerp: iterations window to check for non-significant gains (default: 10).
    - eps: threshold for non-significant gains (default: 1e-6).
    - n_particles: number of particles in the swarm (default: 12).
    
    Parameters:
    - fitness_fn: callable, takes a 1D numpy array representing a particle's position
                 and returns a scalar fitness value (to be minimized).
    - bounds: tuple (low, high), where low and high are 1D numpy arrays of shape (D,)
              or scalars (in which case they will be broadcasted to D dimensions).
    - n_particles: int, number of particles in the swarm.
    - c1: float, cognitive factor.
    - c2: float, social factor.
    - nbmaxiter: int, max iterations.
    - nerp: int, convergence window.
    - eps: float, convergence threshold.
    - verbose: bool, print progress messages.
    
    Returns:
    - best_position: 1D numpy array of shape (D,) representing the global best solution.
    """
    low, high = bounds
    
    # Ensure bounds are numpy arrays
    low = np.atleast_1d(low)
    high = np.atleast_1d(high)
    
    if len(low) != len(high):
        raise ValueError("Bounds 'low' and 'high' must have the same dimension.")
        
    D = len(low)
    
    # Initialize particle positions uniformly within the bounds
    X = np.random.uniform(low, high, size=(n_particles, D))
    
    # Initialize particle velocities within a small range (20% of search space width)
    v_limit = 0.2 * np.abs(high - low)
    # Ensure velocity limit has no zero elements to avoid freezing dimensions
    v_limit = np.clip(v_limit, 1e-5, None)
    V = np.random.uniform(-v_limit, v_limit, size=(n_particles, D))
    
    # Evaluate initial positions to establish personal bests
    pbest = np.copy(X)
    pbest_fitness = np.zeros(n_particles)
    for i in range(n_particles):
        pbest_fitness[i] = fitness_fn(X[i])
        
    # Find the global best particle
    gbest_idx = np.argmin(pbest_fitness)
    gbest = np.copy(pbest[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    
    # Keep track of global best fitness history to check for convergence
    gbest_history = [gbest_fitness]
    
    if verbose:
        print(f"PSO Initial Global Best Fitness: {gbest_fitness:.8f}")
        
    # Iterative swarm optimization
    for k in range(nbmaxiter):
        # Linear decay of inertia weight w from 0.9 to 0.4
        w = 0.9 - (k / max(1, nbmaxiter - 1)) * (0.9 - 0.4)
        
        for i in range(n_particles):
            # Generate random numbers for stochastic updates
            r1 = np.random.uniform(0.0, 1.0, D)
            r2 = np.random.uniform(0.0, 1.0, D)
            
            # Velocity update equation:
            # V(k+1) = w*V(k) + c1*r1*(pbest - X) + c2*r2*(gbest - X)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            # Apply velocity clamping
            V[i] = np.clip(V[i], -v_limit, v_limit)
            
            # Position update equation:
            # X(k+1) = X(k) + V(k+1)
            X[i] = X[i] + V[i]
            # Apply position clamping to keep particles in the valid search space
            X[i] = np.clip(X[i], low, high)
            
            # Evaluate fitness of the new position
            fit = fitness_fn(X[i])
            
            # Update personal best if the new position is better (lower objective value)
            if fit < pbest_fitness[i]:
                pbest[i] = np.copy(X[i])
                pbest_fitness[i] = fit
                
                # Update global best if personal best is better than the global best
                if fit < gbest_fitness:
                    gbest = np.copy(X[i])
                    gbest_fitness = fit
                    
        gbest_history.append(gbest_fitness)
        
        if verbose and (k + 1) % 10 == 0:
            print(f"Iteration {k+1:3d}/{nbmaxiter} - Best Fitness: {gbest_fitness:.8f}")
            
        # Convergence Check:
        # Stop if change in global best fitness over the last 'nerp' iterations is less than 'eps'
        if len(gbest_history) > nerp:
            gain = gbest_history[-nerp-1] - gbest_history[-1]
            if gain < eps:
                if verbose:
                    print(f"PSO converged at iteration {k+1} because gain ({gain:.2e}) < eps ({eps:.2e})")
                break
                
    return gbest
