"""
Optimisation strategies and infill criteria.

Provides query point selection strategies for guided exploration.
"""

import numpy as np
from typing import Callable, Tuple


def random_search(
    bounds: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 100,
    dim: int = 2,
    seed: int = None
) -> np.ndarray:
    """
    Random search: sample uniformly from input space.
    
    Parameters
    ----------
    bounds : Tuple[float, float], default=(0.0, 1.0)
        Lower and upper bounds
    n_points : int, default=100
        Number of random points to generate
    dim : int, default=2
        Dimensionality
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points, dim) with random samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = bounds
    samples = np.random.uniform(lower, upper, size=(n_points, dim))
    return samples


def grid_search(
    bounds: Tuple[float, float] = (0.0, 1.0),
    n_per_axis: int = 10,
    dim: int = 2
) -> np.ndarray:
    """
    Grid search: systematically sample along each dimension.
    
    Parameters
    ----------
    bounds : Tuple[float, float], default=(0.0, 1.0)
        Lower and upper bounds
    n_per_axis : int, default=10
        Number of points per dimension
    dim : int, default=2
        Dimensionality
    
    Returns
    -------
    np.ndarray
        Array of shape (n_per_axis ** dim, dim) with grid points
        
    Warning
    -------
    Grid search becomes exponentially expensive for high dimensions.
    For dim > 4, prefer random or Latin hypercube sampling.
    """
    if dim > 4:
        print(f"Warning: Grid search for {dim}D may create {n_per_axis**dim} points. "
              f"Consider random_search or latin_hypercube_search instead.")
    
    lower, upper = bounds
    axis_samples = np.linspace(lower, upper, n_per_axis)
    grid = np.meshgrid(*[axis_samples] * dim, indexing='ij')
    points = np.stack([g.ravel() for g in grid], axis=1)
    return points


def latin_hypercube_search(
    bounds: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 100,
    dim: int = 2,
    seed: int = None
) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS): stratified random sampling.
    
    Ensures better space-filling properties than pure random search.
    Preferred for exploratory search in high dimensions.
    
    Parameters
    ----------
    bounds : Tuple[float, float], default=(0.0, 1.0)
        Lower and upper bounds
    n_points : int, default=100
        Number of samples
    dim : int, default=2
        Dimensionality
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points, dim) with LHS samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = bounds
    
    # Generate stratified samples
    samples = np.zeros((n_points, dim))
    for d in range(dim):
        # Divide [0, 1] into n_points bins, sample one point per bin
        bin_edges = np.linspace(0, 1, n_points + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Perturb sample within each bin
        perturbations = np.random.uniform(-0.5, 0.5, n_points) / n_points
        samples[:, d] = bin_centers + perturbations
        # Shuffle to randomize assignment
        np.random.shuffle(samples[:, d])
    
    # Scale to [lower, upper]
    samples = lower + samples * (upper - lower)
    return samples


def expected_improvement(
    X_candidates: np.ndarray,
    f_best: float,
    predict_func: Callable,
    predict_std_func: Callable = None,
    xi: float = 0.01
) -> np.ndarray:
    """
    Expected Improvement (EI) infill criterion.
    
    Balances exploitation (high predicted mean) and exploration (high uncertainty).
    
    Parameters
    ----------
    X_candidates : np.ndarray
        Candidate points, shape (n_candidates, dim)
    f_best : float
        Current best observed output
    predict_func : Callable
        Function that returns predicted mean for X
    predict_std_func : Callable, optional
        Function that returns predicted std for X. If None, assumes std=0.
    xi : float, default=0.01
        Exploration trade-off parameter (higher = more exploration)
    
    Returns
    -------
    np.ndarray
        EI value for each candidate, shape (n_candidates,)
    
    References
    ----------
    Jones et al. (1998). Efficient Global Optimization of Expensive Black-Box
    Functions. Journal of Global Optimization.
    """
    from scipy.stats import norm
    
    mu = predict_func(X_candidates)
    
    if predict_std_func is None:
        # No uncertainty: EI reduces to improvement
        ei = np.maximum(mu - f_best - xi, 0.0)
    else:
        std = predict_std_func(X_candidates)
        
        # Avoid division by zero
        with np.errstate(divide='warn'):
            imp = mu - f_best - xi
            Z = imp / (std + 1e-8)
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
    
    return ei


def upper_confidence_bound(
    X_candidates: np.ndarray,
    predict_func: Callable,
    predict_std_func: Callable = None,
    beta: float = 2.0
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) infill criterion.
    
    Parameters
    ----------
    X_candidates : np.ndarray
        Candidate points, shape (n_candidates, dim)
    predict_func : Callable
        Function that returns predicted mean
    predict_std_func : Callable, optional
        Function that returns predicted std
    beta : float, default=2.0
        Exploration trade-off parameter (higher = more exploration)
    
    Returns
    -------
    np.ndarray
        UCB value for each candidate, shape (n_candidates,)
    """
    mu = predict_func(X_candidates)
    
    if predict_std_func is None:
        return mu
    else:
        std = predict_std_func(X_candidates)
        return mu + beta * std


# TODO: Implement Entropy Search and other advanced infill criteria
# TODO: Add constraint handling for bounded search spaces
