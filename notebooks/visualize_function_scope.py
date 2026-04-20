"""
Function Scope & Coverage Visualization
Visualizes the input space boundaries and output ranges discovered.
Shows explored vs unexplored regions.

Run: python3 notebooks/visualize_function_scope.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loading import load_all_functions
from submissions.week_01.queries import week1_queries, week1_results
from submissions.week_02.queries import week2_queries, week2_results
from submissions.week_03.queries import week3_queries, week3_results


def plot_input_space_coverage_2d():
    """Visualize input space coverage for 2D functions (F1, F2)."""
    print("📍 Generating 2D input space coverage plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    all_data = load_all_functions()
    
    for idx, func_id in enumerate([1, 2]):
        ax = axes[idx]
        
        # Load initial data
        X_initial, y_initial = all_data[func_id]
        
        # Plot initial data cloud
        scatter = ax.scatter(X_initial[:, 0], X_initial[:, 1], 
                           c=y_initial, cmap='RdYlGn', s=80, 
                           alpha=0.4, label='Initial Data', edgecolors='gray', linewidth=0.5)
        
        # Plot boundary box
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', 
                                   linewidth=2, linestyle='--', label='Input Space [0,1]²'))
        
        # Plot queries with trajectory
        w1 = week1_queries[func_id]
        w2 = week2_queries[func_id]
        w3 = week3_queries[func_id]
        
        # Draw trajectory line
        trajectory = np.array([w1, w2, w3])
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Plot queries
        ax.scatter([w1[0]], [w1[1]], s=200, marker='*', color='#1f77b4', 
                  label='Week 1', edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter([w2[0]], [w2[1]], s=150, marker='o', color='#ff7f0e', 
                  label='Week 2', edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter([w3[0]], [w3[1]], s=150, marker='s', color='#2ca02c', 
                  label='Week 3', edgecolors='black', linewidth=1.5, zorder=5)
        
        # Annotations
        ax.text(w1[0], w1[1]-0.08, 'W1', ha='center', fontsize=9, fontweight='bold')
        ax.text(w2[0], w2[1]+0.08, 'W2', ha='center', fontsize=9, fontweight='bold')
        ax.text(w3[0], w3[1]-0.08, 'W3', ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dimension 1', fontweight='bold', fontsize=11)
        ax.set_ylabel('Dimension 2', fontweight='bold', fontsize=11)
        ax.set_title(f'F{func_id}: Input Space Coverage & Trajectory', fontweight='bold', fontsize=12)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Output Value', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/input_space_coverage_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/input_space_coverage_2d.png")


def plot_dimension_wise_coverage():
    """Show per-dimension input ranges explored for each function."""
    print("📊 Generating dimension-wise coverage plot...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    all_data = load_all_functions()
    
    dims_per_func = [2, 2, 3, 4, 4, 5, 6, 8]
    
    for func_id in range(1, 9):
        ax = axes[func_id - 1]
        
        X_initial, y_initial = all_data[func_id]
        dim = X_initial.shape[1]
        
        # Compute per-dimension min/max from initial data
        mins = X_initial.min(axis=0)
        maxs = X_initial.max(axis=0)
        means = X_initial.mean(axis=0)
        
        # Get all queries for this function
        w1 = week1_queries[func_id]
        w2 = week2_queries[func_id]
        w3 = week3_queries[func_id]
        queries = np.array([w1, w2, w3])
        query_mins = queries.min(axis=0)
        query_maxs = queries.max(axis=0)
        
        # Plot dimension-wise ranges
        dim_range = np.arange(1, dim + 1)
        
        # Initial data range (gray)
        ax.barh(dim_range - 0.2, maxs - mins, left=mins, height=0.4, 
               color='gray', alpha=0.4, label='Initial Data Range', edgecolor='black', linewidth=0.5)
        
        # Query range (colored)
        ax.barh(dim_range + 0.2, query_maxs - query_mins, left=query_mins, height=0.4,
               color='#ff7f0e', alpha=0.6, label='Query Coverage', edgecolor='black', linewidth=0.5)
        
        # Add boundary lines
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.5, dim + 0.5)
        ax.set_yticks(dim_range)
        ax.set_yticklabels([f'Dim {i}' for i in range(1, dim + 1)], fontsize=9)
        ax.set_xlabel('Input Value [0, 1]', fontweight='bold', fontsize=9)
        ax.set_title(f'F{func_id} ({dim}D)', fontweight='bold', fontsize=11)
        ax.grid(axis='x', alpha=0.2)
        
        if func_id == 1:
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/dimension_wise_coverage.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/dimension_wise_coverage.png")


def plot_output_range_spectrum():
    """Visualize output ranges as a continuous spectrum."""
    print("📈 Generating output range spectrum plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_data = load_all_functions()
    
    y_pos = 0
    function_ranges = []
    function_labels = []
    
    for func_id in range(1, 9):
        X, y = all_data[func_id]
        
        y_min = y.min()
        y_max = y.max()
        y_mean = y.mean()
        y_std = y.std()
        
        # Draw range as horizontal bar
        range_width = y_max - y_min
        ax.barh(y_pos, range_width, left=y_min, height=0.6, 
               color='lightblue', alpha=0.5, edgecolor='black', linewidth=1.5)
        
        # Mark mean and std
        ax.plot([y_mean], [y_pos], 'ro', markersize=8, zorder=5, label='Mean' if func_id == 1 else '')
        ax.plot([y_mean - y_std, y_mean + y_std], [y_pos, y_pos], 'g-', linewidth=3, zorder=4, label='±1 Std' if func_id == 1 else '')
        
        # Add query results
        w1_result = week1_results[func_id]
        w2_result = week2_results[func_id]
        w3_result = week3_results[func_id]
        
        ax.scatter([w1_result], [y_pos - 0.15], s=100, marker='*', color='#1f77b4', 
                  edgecolors='black', linewidth=1, zorder=5)
        ax.scatter([w2_result], [y_pos], s=80, marker='o', color='#ff7f0e', 
                  edgecolors='black', linewidth=1, zorder=5)
        ax.scatter([w3_result], [y_pos + 0.15], s=80, marker='s', color='#2ca02c', 
                  edgecolors='black', linewidth=1, zorder=5)
        
        # Label
        ax.text(y_min - 0.05 * range_width, y_pos, f'F{func_id}', 
               ha='right', va='center', fontweight='bold', fontsize=10)
        
        y_pos += 1
        function_labels.append(f'F{func_id}')
    
    ax.set_ylim(-0.5, 8.5)
    ax.set_ylabel('Function', fontweight='bold', fontsize=12)
    ax.set_xlabel('Output Value', fontweight='bold', fontsize=12)
    ax.set_title('Output Space Ranges: Initial Data Distribution vs Query Results', 
                fontweight='bold', fontsize=13)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.2)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f77b4', markersize=10, label='Week 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label='Week 2'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=8, label='Week 3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Mean'),
        Line2D([0], [0], color='green', linewidth=3, label='±1 Std Dev'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/output_range_spectrum.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/output_range_spectrum.png")


def plot_query_density_heatmap():
    """Show query density and coverage patterns per dimension."""
    print("🔥 Generating query density heatmap...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    all_data = load_all_functions()
    
    for func_id in range(1, 9):
        ax = axes[func_id - 1]
        
        X_initial, y_initial = all_data[func_id]
        dim = X_initial.shape[1]
        
        # For 2D functions, create a 2D histogram
        if dim == 2:
            # Create 2D histogram of initial data
            hist, xedges, yedges = np.histogram2d(X_initial[:, 0], X_initial[:, 1], 
                                                   bins=10, range=[[0, 1], [0, 1]])
            
            im = ax.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='YlOrRd', aspect='auto', alpha=0.6)
            
            # Overlay queries
            w1 = week1_queries[func_id]
            w2 = week2_queries[func_id]
            w3 = week3_queries[func_id]
            
            ax.scatter([w1[0]], [w1[1]], s=150, marker='*', color='blue', 
                      edgecolors='black', linewidth=1.5, zorder=5, label='W1')
            ax.scatter([w2[0]], [w2[1]], s=120, marker='o', color='orange', 
                      edgecolors='black', linewidth=1.5, zorder=5, label='W2')
            ax.scatter([w3[0]], [w3[1]], s=120, marker='s', color='green', 
                      edgecolors='black', linewidth=1.5, zorder=5, label='W3')
            
            ax.set_xlabel('Dimension 1', fontweight='bold')
            ax.set_ylabel('Dimension 2', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # For higher dimensions, show first 2 dims only
            hist, xedges, yedges = np.histogram2d(X_initial[:, 0], X_initial[:, 1] if dim > 1 else np.zeros_like(X_initial[:, 0]), 
                                                   bins=8, range=[[0, 1], [0, 1]])
            
            im = ax.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='YlOrRd', aspect='auto', alpha=0.6)
            
            w1 = week1_queries[func_id]
            w2 = week2_queries[func_id]
            w3 = week3_queries[func_id]
            
            ax.scatter([w1[0]], [w1[1]], s=150, marker='*', color='blue', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            ax.scatter([w2[0]], [w2[1]], s=120, marker='o', color='orange', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            ax.scatter([w3[0]], [w3[1]], s=120, marker='s', color='green', 
                      edgecolors='black', linewidth=1.5, zorder=5)
            
            ax.set_xlabel('Dimension 1', fontweight='bold', fontsize=9)
            ax.set_ylabel('Dimension 2', fontweight='bold', fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.set_title(f'F{func_id} ({dim}D) - Data Density', fontweight='bold', fontsize=11)
        ax.grid(alpha=0.2)
        
        if func_id == 1:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/query_density_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/query_density_heatmap.png")


def generate_scope_summary():
    """Generate text summary of explored scope."""
    print("📋 Generating scope summary...")
    
    all_data = load_all_functions()
    
    summary = []
    summary.append("="*120)
    summary.append("FUNCTION SCOPE & COVERAGE ANALYSIS".center(120))
    summary.append("="*120)
    
    for func_id in range(1, 9):
        X_init, y_init = all_data[func_id]
        dim = X_init.shape[1]
        
        w1 = week1_queries[func_id]
        w2 = week2_queries[func_id]
        w3 = week3_queries[func_id]
        
        summary.append(f"\n📍 Function {func_id} ({dim}D)")
        summary.append(f"   Initial Data Samples: {len(X_init)}")
        summary.append(f"   Output Range: [{y_init.min():.6e}, {y_init.max():.6e}]")
        summary.append(f"   Output Mean: {y_init.mean():.6e} ± {y_init.std():.6e}")
        
        summary.append(f"\n   Input Space Coverage (per dimension):")
        for d in range(dim):
            init_min = X_init[:, d].min()
            init_max = X_init[:, d].max()
            init_range = init_max - init_min
            init_coverage = (init_range / 1.0) * 100
            
            query_vals = np.array([w1[d], w2[d], w3[d]])
            query_min = query_vals.min()
            query_max = query_vals.max()
            
            summary.append(f"      Dim {d+1}: Initial [{init_min:.3f}, {init_max:.3f}] (coverage: {init_coverage:.1f}%)")
            summary.append(f"             Queries [{query_min:.3f}, {query_max:.3f}]")
        
        summary.append(f"\n   Queries Trajectory:")
        for week, query, result in [(1, w1, week1_results[func_id]), 
                                     (2, w2, week2_results[func_id]),
                                     (3, w3, week3_results[func_id])]:
            summary.append(f"      Week {week}: {query} → {result:.6e}")
    
    summary.append("\n" + "="*120)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open('results/scope_summary.txt', 'w') as f:
        f.write(summary_text)
    print("✓ Saved: results/scope_summary.txt")


def main():
    """Generate all scope visualizations."""
    print("\n" + "="*80)
    print("FUNCTION SCOPE & INPUT COVERAGE VISUALIZATION".center(80))
    print("="*80 + "\n")
    
    Path('results').mkdir(exist_ok=True)
    
    plot_input_space_coverage_2d()
    plot_dimension_wise_coverage()
    plot_output_range_spectrum()
    plot_query_density_heatmap()
    generate_scope_summary()
    
    print("\n" + "="*80)
    print("✓ ALL SCOPE VISUALIZATIONS COMPLETE".center(80))
    print("="*80)
    print("\nGenerated files:")
    print("  📍 results/input_space_coverage_2d.png")
    print("  📊 results/dimension_wise_coverage.png")
    print("  📈 results/output_range_spectrum.png")
    print("  🔥 results/query_density_heatmap.png")
    print("  📋 results/scope_summary.txt\n")


if __name__ == "__main__":
    main()
