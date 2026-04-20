"""
Detailed landscape visualization with surrogate model predictions.
Estimates function landscape using linear regression surrogates trained on initial data + submissions.

Run: python3 notebooks/visualize_surrogates.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loading import load_all_functions
from src.models.linear_models import LinearRegressionSurrogate
from submissions.week_01.queries import week1_queries, week1_results
from submissions.week_02.queries import week2_queries, week2_results
from submissions.week_03.queries import week3_queries, week3_results


def train_surrogates():
    """Train linear regression surrogates on combined data."""
    print("🤖 Training surrogates on combined data...")
    
    all_data = load_all_functions()
    surrogates = {}
    
    for func_id in range(1, 9):
        # Load initial data
        X_initial, y_initial = all_data[func_id]
        
        # Add Week 1-2 submissions
        X_w1 = week1_queries[func_id].reshape(1, -1)
        y_w1 = np.array([week1_results[func_id]])
        
        X_w2 = week2_queries[func_id].reshape(1, -1)
        y_w2 = np.array([week2_results[func_id]])
        
        # Combine
        X_train = np.vstack([X_initial, X_w1, X_w2])
        y_train = np.hstack([y_initial, y_w1, y_w2])
        
        # Train surrogate
        surrogate = LinearRegressionSurrogate()
        surrogate.fit(X_train, y_train)
        surrogates[func_id] = (surrogate, X_train, y_train)
        
        print(f"  ✓ F{func_id}: Trained on {X_train.shape[0]} points, dim={X_train.shape[1]}")
    
    return surrogates


def plot_2d_landscape_with_surrogate(surrogates):
    """Plot 2D function landscape estimates with surrogate predictions."""
    print("🗺️  Generating 2D landscape plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, func_id in enumerate([1, 2]):
        ax = axes[idx]
        
        surrogate, X_train, y_train = surrogates[func_id]
        
        # Create grid
        x_range = np.linspace(0, 1, 50)
        y_range = np.linspace(0, 1, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        X_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Predict on grid
        Z_pred = surrogate.predict(X_flat).reshape(X_grid.shape)
        
        # Plot contours
        contour = ax.contourf(X_grid, Y_grid, Z_pred, levels=20, cmap='RdYlGn', alpha=0.8)
        contour_lines = ax.contour(X_grid, Y_grid, Z_pred, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        # Plot training data
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                           cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', 
                           linewidth=1, label='Training Data', zorder=3)
        
        # Plot submissions
        w1 = week1_queries[func_id]
        w2 = week2_queries[func_id]
        w3 = week3_queries[func_id]
        
        ax.plot(w1[0], w1[1], 'b*', markersize=18, label='Week 1', 
               markeredgecolor='darkblue', markeredgewidth=2, zorder=4)
        ax.plot(w2[0], w2[1], 'o', color='orange', markersize=12, label='Week 2', 
               markeredgecolor='darkorange', markeredgewidth=2, zorder=4)
        ax.plot(w3[0], w3[1], 's', color='green', markersize=12, label='Week 3', 
               markeredgecolor='darkgreen', markeredgewidth=2, zorder=4)
        
        ax.set_xlabel('Dimension 1', fontweight='bold', fontsize=11)
        ax.set_ylabel('Dimension 2', fontweight='bold', fontsize=11)
        ax.set_title(f'F{func_id}: Linear Surrogate Landscape Estimate', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.2)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Output', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/surrogate_landscape_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/surrogate_landscape_2d.png")


def plot_1d_slices(surrogates):
    """Plot 1D slices of high-dimensional surrogates."""
    print("📏 Generating 1D surrogate slices...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, func_id in enumerate(range(3, 9)):  # F3-F8 (3D-8D)
        ax = axes[idx]
        
        surrogate, X_train, y_train = surrogates[func_id]
        
        # Create slice: vary first dimension, fix others at 0.5
        x_slice = np.linspace(0, 1, 100)
        X_slice = np.tile([0.5] * X_train.shape[1], (100, 1))
        X_slice[:, 0] = x_slice
        
        # Predict
        y_slice = surrogate.predict(X_slice)
        
        # Plot surrogate prediction
        ax.plot(x_slice, y_slice, 'b-', linewidth=2.5, label='Surrogate', zorder=3)
        ax.fill_between(x_slice, y_slice - 0.1*np.abs(y_slice), 
                       y_slice + 0.1*np.abs(y_slice), alpha=0.2, color='blue')
        
        # Plot training points projected onto 1D
        X_train_dim0 = X_train[:, 0]
        ax.scatter(X_train_dim0, y_train, s=100, alpha=0.6, 
                  color='red', edgecolors='darkred', linewidth=1, 
                  label='Training Data', zorder=4)
        
        # Plot submission points
        for week_num, queries, marker, color in [
            (1, week1_queries, '*', '#1f77b4'),
            (2, week2_queries, 'o', '#ff7f0e'),
            (3, week3_queries, 's', '#2ca02c'),
        ]:
            query = queries[func_id]
            ax.plot(query[0], 0, marker=marker, markersize=14, color=color, 
                   markeredgecolor='black', markeredgewidth=1.5, label=f'W{week_num}', zorder=5)
        
        ax.set_xlabel('Dimension 1 (others fixed at 0.5)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Predicted Output', fontweight='bold', fontsize=10)
        ax.set_title(f'F{func_id} ({X_train.shape[1]}D): 1D Slice of Surrogate', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig('results/surrogate_1d_slices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/surrogate_1d_slices.png")


def plot_surrogate_uncertainty():
    """Visualize surrogate model uncertainty (residuals from training data)."""
    print("📊 Generating surrogate uncertainty analysis...")
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    all_data = load_all_functions()
    
    predictions_by_func = {}
    errors_by_func = {}
    
    for func_id in range(1, 9):
        surrogate, X_train, y_train = surrogates[func_id]
        
        # Predict on training data
        y_pred = surrogate.predict(X_train)
        errors = np.abs(y_pred - y_train)
        
        predictions_by_func[func_id] = y_pred
        errors_by_func[func_id] = errors
    
    # Box plot of prediction errors
    error_lists = [errors_by_func[i] for i in range(1, 9)]
    bp = ax.boxplot(error_lists, labels=[f'F{i}' for i in range(1, 9)],
                    patch_artist=True, showmeans=True)
    
    # Customize colors
    for patch in bp['boxes']:
        patch.set_facecolor('#8dd3c7')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Absolute Prediction Error', fontweight='bold', fontsize=12)
    ax.set_xlabel('Function', fontweight='bold', fontsize=12)
    ax.set_title('Surrogate Model Prediction Errors on Training Data', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/surrogate_uncertainty.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/surrogate_uncertainty.png")


def plot_feature_importance(surrogates):
    """Plot feature importance (coefficients) for linear regression surrogates."""
    print("🎯 Generating feature importance analysis...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for func_id in range(1, 9):
        ax = axes[func_id - 1]
        
        surrogate, X_train, y_train = surrogates[func_id]
        
        # Get coefficients
        coefs = surrogate.model.coef_
        dims = [f'D{i+1}' for i in range(len(coefs))]
        
        # Plot
        colors = ['green' if c > 0 else 'red' for c in coefs]
        bars = ax.bar(dims, coefs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, coefs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', 
                   va='bottom' if val > 0 else 'top', fontsize=9)
        
        ax.set_ylabel('Coefficient Value', fontweight='bold', fontsize=10)
        ax.set_title(f'F{func_id} ({len(coefs)}D)', fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/feature_importance.png")


def main():
    print("\n" + "="*80)
    print("SURROGATE MODEL VISUALIZATION & LANDSCAPE ANALYSIS".center(80))
    print("="*80 + "\n")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Train surrogates
    global surrogates
    surrogates = train_surrogates()
    
    print()
    
    # Generate visualizations
    plot_2d_landscape_with_surrogate(surrogates)
    plot_1d_slices(surrogates)
    plot_surrogate_uncertainty()
    plot_feature_importance(surrogates)
    
    print("\n" + "="*80)
    print("✓ SURROGATE VISUALIZATIONS COMPLETE".center(80))
    print("="*80)
    print("\nGenerated files:")
    print("  🗺️  results/surrogate_landscape_2d.png")
    print("  📏 results/surrogate_1d_slices.png")
    print("  📊 results/surrogate_uncertainty.png")
    print("  🎯 results/feature_importance.png\n")


if __name__ == "__main__":
    main()
