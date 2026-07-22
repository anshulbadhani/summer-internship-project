"""
Particle Swarm Optimization module replaced by Differential Evolution (DE).
This module re-exports `de_optimize` as `pso_optimize` for backward compatibility.
"""
from src.de import de_optimize

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
):
    """
    Deprecated alias for `de_optimize`.
    Replaces PSO with Differential Evolution optimization.
    """
    return de_optimize(
        fitness_fn=fitness_fn,
        bounds=bounds,
        pop_size=n_particles,
        F=0.8,
        CR=0.7,
        maxiter=nbmaxiter,
        nerp=nerp,
        eps=eps,
        verbose=verbose
    )
