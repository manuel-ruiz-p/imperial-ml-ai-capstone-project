"""
Comprehensive visualization of Week 1-3 submissions and results.
Plots output progression, input space exploration, and function landscape estimates.

Run: python3 notebooks/visualize_progress.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loading import load_all_functions
from submissions.week_01.queries import week1_queries, week1_results
from submissions.week_02.queries import week2_queries, week2_results
from submissions.week_03.queries import week3_queries, week3_results


def plot_output_progression():
    """Plot output values across Week 1, 2, 3 for each function."""
    print("📊 Generating output progression plot...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for func_id in range(1, 9):
        ax = axes[func_id - 1]
        
        # Collect results across weeks
        weeks = ['Week 1', 'Week 2', 'Week 3']
        results = [
            week1_results[func_id],
            week2_results[func_id],
            week3_results[func_id] if week3_results[func_id] is not None else 0
        ]
        
        # Plot bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(weeks, results, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, results):
            if val != 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3e}' if abs(val) < 0.01 else f'{val:.3f}',
                       ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
        
        ax.set_ylabel("Output Value")
        ax.set_title(f"F{func_id} ({[2, 2, 3, 4, 4, 5, 6, 8][func_id-1]}D)", fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/output_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/output_progression.png")


def plot_improvement_trajectory():
    """Plot improvement from Week 1 → Week 2 → Week 3."""
    print("📈 Generating improvement trajectory plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    week1_vals = np.array([week1_results[i] for i in range(1, 9)])
    week2_vals = np.array([week2_results[i] for i in range(1, 9)])
    week3_vals = np.array([week3_results[i] if week3_results[i] is not None else week2_vals[i-1] 
                           for i in range(1, 9)])
    
    x = np.arange(1, 9)
    width = 0.25
    
    ax.bar(x - width, week1_vals, width, label='Week 1', alpha=0.8, color='#1f77b4')
    ax.bar(x, week2_vals, width, label='Week 2', alpha=0.8, color='#ff7f0e')
    ax.bar(x + width, week3_vals, width, label='Week 3', alpha=0.8, color='#2ca02c')
    
    ax.set_xlabel('Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Output Value', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Progress: Week 1 → Week 2 → Week 3', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'F{i}' for i in range(1, 9)])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/improvement_trajectory.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/improvement_trajectory.png")


def plot_2d_exploration():
    """For 2D functions (F1, F2), visualize exploration in input space."""
    print("📍 Generating 2D input space exploration plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    all_data = load_all_functions()
    
    for idx, func_id in enumerate([1, 2]):
        ax = axes[idx]
        
        # Load initial data
        X_initial, y_initial = all_data[func_id]
        
        # Normalize colors by output value
        scatter = ax.scatter(X_initial[:, 0], X_initial[:, 1], 
                           c=y_initial, cmap='RdYlGn', s=100, 
                           alpha=0.5, label='Initial Data', edgecolors='black', linewidth=0.5)
        
        # Plot Week 1-3 queries
        w1_query = week1_queries[func_id]
        w2_query = week2_queries[func_id]
        w3_query = week3_queries[func_id]
        
        ax.plot(w1_query[0], w1_query[1], 'b*', markersize=15, label='Week 1', markeredgecolor='black', markeredgewidth=1)
        ax.plot(w2_query[0], w2_query[1], 'o', color='orange', markersize=10, label='Week 2', markeredgecolor='black', markeredgewidth=1)
        ax.plot(w3_query[0], w3_query[1], 's', color='green', markersize=10, label='Week 3', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Dimension 1', fontweight='bold')
        ax.set_ylabel('Dimension 2', fontweight='bold')
        ax.set_title(f'F{func_id}: Input Space Exploration', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Output Value', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/2d_exploration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/2d_exploration.png")


def plot_high_dim_summary():
    """For high-dimensional functions, show projection summary."""
    print("🔍 Generating high-dimensional function summary...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    all_data = load_all_functions()
    dims_to_plot = [3, 4, 4, 5, 6, 8]  # F3-F8
    
    for idx, func_id in enumerate(range(3, 9)):
        ax = axes[idx]
        
        X_initial, y_initial = all_data[func_id]
        
        # Plot initial data as 2D projection (first two dims)
        scatter = ax.scatter(X_initial[:, 0], X_initial[:, 1], 
                           c=y_initial, cmap='RdYlGn', s=80,
                           alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot submitted queries (first 2 dims only)
        w1 = week1_queries[func_id]
        w2 = week2_queries[func_id]
        w3 = week3_queries[func_id]
        
        ax.plot(w1[0], w1[1], 'b*', markersize=14, label='W1', markeredgecolor='black', markeredgewidth=1)
        ax.plot(w2[0], w2[1], 'o', color='orange', markersize=9, label='W2', markeredgecolor='black', markeredgewidth=1)
        ax.plot(w3[0], w3[1], 's', color='green', markersize=9, label='W3', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Dim 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('Dim 2', fontsize=10, fontweight='bold')
        ax.set_title(f'F{func_id} ({dims_to_plot[idx-3]}D) - First 2 Dims', fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/high_dim_projection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/high_dim_projection.png")


def plot_output_range_boundaries():
    """Show output ranges for all functions to understand landscape boundaries."""
    print("📏 Generating output range boundaries plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    all_data = load_all_functions()
    
    output_ranges = []
    output_means = []
    function_labels = []
    
    for func_id in range(1, 9):
        X, y = all_data[func_id]
        output_ranges.append((y.min(), y.max()))
        output_means.append(y.mean())
        function_labels.append(f'F{func_id}')
    
    # Plot ranges as error bars
    mins = [r[0] for r in output_ranges]
    maxs = [r[1] for r in output_ranges]
    means = output_means
    
    x_pos = np.arange(len(function_labels))
    
    # Draw range lines
    for i, (mn, mx, mean) in enumerate(zip(mins, maxs, means)):
        ax.plot([i, i], [mn, mx], 'k-', linewidth=3, alpha=0.3)
        ax.scatter([i], [mean], s=200, marker='D', color='red', zorder=5, edgecolors='black', linewidth=1.5)
    
    # Add Week 1-3 results as points
    for week_num, week_queries, week_results, marker, color in [
        (1, week1_queries, week1_results, '*', '#1f77b4'),
        (2, week2_queries, week2_results, 'o', '#ff7f0e'),
        (3, week3_queries, week3_results, 's', '#2ca02c'),
    ]:
        results = [week_results[i] for i in range(1, 9)]
        offsets = np.random.normal(0, 0.02, len(results))  # Jitter for visibility
        ax.scatter(x_pos + offsets, results, s=120, marker=marker, 
                  color=color, alpha=0.7, label=f'Week {week_num}', 
                  edgecolors='black', linewidth=1, zorder=4)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(function_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Output Value', fontsize=12, fontweight='bold')
    ax.set_title('Function Landscape Boundaries: Output Ranges vs Submissions', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/output_boundaries.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/output_boundaries.png")


def generate_summary_table():
    """Generate text summary table of submissions and results."""
    print("📋 Generating summary table...")
    
    all_data = load_all_functions()
    
    summary = []
    summary.append("="*120)
    summary.append("WEEK 1-3 SUBMISSIONS & RESULTS SUMMARY".center(120))
    summary.append("="*120)
    
    for func_id in range(1, 9):
        X_init, y_init = all_data[func_id]
        dims = X_init.shape[1]
        
        w1_result = week1_results[func_id]
        w2_result = week2_results[func_id]
        w3_result = week3_results[func_id] if week3_results[func_id] is not None else 0
        
        improvement_1_2 = ((w2_result - w1_result) / (abs(w1_result) + 1e-10)) * 100
        improvement_2_3 = ((w3_result - w2_result) / (abs(w2_result) + 1e-10)) * 100 if w3_result != 0 else 0
        
        summary.append(f"\n📊 Function {func_id} ({dims}D)")
        summary.append(f"  Initial Data Range: [{y_init.min():.3e}, {y_init.max():.3e}] (n={len(y_init)})")
        summary.append(f"  Week 1: {w1_result:12.6e}  |  Week 2: {w2_result:12.6e} ({improvement_1_2:+6.1f}%)  |  Week 3: {w3_result:12.6e}" + 
                     (f" ({improvement_2_3:+6.1f}%)" if w3_result != 0 else " (pending)"))
    
    summary.append("\n" + "="*120)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save to file
    with open('results/summary_table.txt', 'w') as f:
        f.write(summary_text)
    print("✓ Saved: results/summary_table.txt")


def main():
    """Generate all visualizations."""
    print("\n" + "="*80)
    print("WEEK 1-3 SUBMISSIONS & PROGRESS VISUALIZATION".center(80))
    print("="*80 + "\n")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Generate all plots
    plot_output_progression()
    plot_improvement_trajectory()
    plot_2d_exploration()
    plot_high_dim_summary()
    plot_output_range_boundaries()
    generate_summary_table()
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE".center(80))
    print("="*80)
    print("\nGenerated files:")
    print("  📊 results/output_progression.png")
    print("  📈 results/improvement_trajectory.png")
    print("  📍 results/2d_exploration.png")
    print("  🔍 results/high_dim_projection.png")
    print("  📏 results/output_boundaries.png")
    print("  📋 results/summary_table.txt\n")


if __name__ == "__main__":
    main()
