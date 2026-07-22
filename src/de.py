import numpy as np

def de_optimize(
    fitness_fn,
    bounds,
    pop_size=12,
    F=0.8,
    CR=0.7,
    maxiter=300,
    nerp=10,
    eps=1e-6,
    verbose=True
) -> np.ndarray:
    """
    Differential Evolution (DE/rand/1/bin) optimization loop.
    
    Parameters:
    - fitness_fn: callable, takes a 1D numpy array representing an individual's position
                 and returns a scalar fitness value (to be minimized).
    - bounds: tuple (low, high), where low and high are 1D numpy arrays of shape (D,)
              or scalars (in which case they will be broadcasted to D dimensions).
    - pop_size: int, number of individuals in the population (default: 12).
    - F: float, differential weight / mutation factor (default: 0.8).
    - CR: float, crossover probability (default: 0.7).
    - maxiter: int, maximum generations (default: 300).
    - nerp: int, iterations window to check for non-significant gains (default: 10).
    - eps: float, threshold for non-significant gains (default: 1e-6).
    - verbose: bool, print progress messages.
    
    Returns:
    - gbest: 1D numpy array of shape (D,) representing the global best solution vector.
    """
    low, high = bounds
    
    # Ensure bounds are 1D numpy arrays
    low = np.atleast_1d(low)
    high = np.atleast_1d(high)
    
    if len(low) != len(high):
        raise ValueError("Bounds 'low' and 'high' must have the same dimension.")
        
    D = len(low)
    
    if pop_size < 4:
        raise ValueError("Population size 'pop_size' must be at least 4 for DE/rand/1/bin mutation.")
        
    # Initialize population uniformly within search bounds
    population = np.random.uniform(low, high, size=(pop_size, D))
    
    # Evaluate initial fitness for each candidate solution
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = fitness_fn(population[i])
        
    # Track the global best solution
    best_idx = np.argmin(fitness)
    gbest = np.copy(population[best_idx])
    gbest_fitness = fitness[best_idx]
    
    gbest_history = [gbest_fitness]
    
    if verbose:
        print(f"DE Initial Global Best Fitness: {gbest_fitness:.8f}")
        
    # Main Differential Evolution generation loop
    for gen in range(maxiter):
        for i in range(pop_size):
            # 1. Mutation (DE/rand/1): Select 3 distinct random individuals excluding target i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, size=3, replace=False)
            
            x_a = population[a_idx]
            x_b = population[b_idx]
            x_c = population[c_idx]
            
            # Compute mutant vector and clamp to domain bounds
            mutant = x_a + F * (x_b - x_c)
            mutant = np.clip(mutant, low, high)
            
            # 2. Binomial Crossover: Generate trial vector
            # Force at least one dimension from mutant vector
            rand_j = np.random.randint(0, D)
            crossover_mask = np.random.uniform(0.0, 1.0, size=D) < CR
            crossover_mask[rand_j] = True
            
            trial = np.where(crossover_mask, mutant, population[i])
            
            # 3. Selection: Evaluate trial vector and perform greedy replacement
            trial_fitness = fitness_fn(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < gbest_fitness:
                    gbest = np.copy(trial)
                    gbest_fitness = trial_fitness
                    
        gbest_history.append(gbest_fitness)
        
        if verbose and (gen + 1) % 10 == 0:
            print(f"Iteration {gen+1:3d}/{maxiter} - Best Fitness: {gbest_fitness:.8f}")
            
        # Convergence Check:
        # Stop if gain over the last 'nerp' iterations is less than 'eps'
        if len(gbest_history) > nerp:
            gain = gbest_history[-nerp-1] - gbest_history[-1]
            if gain < eps:
                if verbose:
                    print(f"DE converged at iteration {gen+1} because gain ({gain:.2e}) < eps ({eps:.2e})")
                break
                
    return gbest
