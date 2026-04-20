"""
Visualization utilities for exploratory analysis.

Functions for plotting function outputs, exploration-exploitation balance,
and other diagnostic visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


def plot_function_outputs(
    outputs: np.ndarray,
    function_ids: Optional[list] = None,
    title: str = "Function Outputs Summary"
) -> None:
    """
    Plot histogram of function outputs for quick visualization.
    
    Parameters
    ----------
    outputs : np.ndarray
        Array of output values from all functions
    function_ids : list, optional
        Function IDs (1-8). If None, auto-generates.
    title : str
        Plot title
    """
    if function_ids is None:
        function_ids = [f"F{i}" for i in range(1, len(outputs) + 1)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if y > 0 else 'red' for y in outputs]
    ax.bar(range(len(outputs)), outputs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(outputs)))
    ax.set_xticklabels(function_ids)
    ax.set_ylabel("Output Value")
    ax.set_title(title)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_exploration_exploitation_balance(
    exploration_percentages: Dict[int, float],
    function_ids: Optional[list] = None,
    title: str = "Exploration vs Exploitation Balance"
) -> None:
    """
    Visualize exploration-exploitation trade-off per function.
    
    Parameters
    ----------
    exploration_percentages : Dict[int, float]
        Dictionary mapping function_id -> exploration_percentage (0-100)
    function_ids : list, optional
        Function IDs. If None, auto-generates.
    title : str
        Plot title
    """
    if function_ids is None:
        function_ids = [f"F{i}" for i in range(1, len(exploration_percentages) + 1)]
    
    exploration = list(exploration_percentages.values())
    exploitation = [100 - e for e in exploration]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(exploration))
    width = 0.6
    
    ax.bar(x, exploration, width, label='Exploration', color='skyblue', edgecolor='black')
    ax.bar(x, exploitation, width, bottom=exploration, label='Exploitation', 
           color='orange', edgecolor='black')
    
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(function_ids)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# TODO: Add more advanced visualizations:
# - 2D/3D scatter plots for low-dimensional functions
# - Convergence plots across weeks
# - Surrogate model fit visualization
# - Uncertainty estimates (for Bayesian approaches)
