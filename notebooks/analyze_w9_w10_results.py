"""
Week 9-10 Results Analysis & PCA Visualization
================================================
Analyzes full W1-W9 history, generates PCA insights, and produces
progress plots updated through Week 9.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# FULL W1-W9 HISTORY
# ============================================================================

WEEKS = list(range(1, 10))
DIMS  = {1:2, 2:2, 3:3, 4:4, 5:4, 6:5, 7:6, 8:8}

weekly_outputs = {
    1: [2.61e-96,   7.57e-193, -5.38e-16,  -1.56e-117, 3.44e-131,
        -2.74e-103, -1.47e-21,  -1.21e-112, -2.34e-192],
    2: [0.3692,  0.8474,  0.4074, -0.0581,  0.0538,
       -0.0301,  0.1429,  0.0329,  0.4809],
    3: [-0.01025, -0.01045, -0.07883, -0.01232, -0.13592,
        -0.08010, -0.10580, -0.13830, -0.00514],
    4: [-13.072, -13.072, -28.648, -12.608, -27.441,
        -14.197, -17.894,  -5.556,  -4.635],
    5: [ 5.273,  4.049, 34.983, 32.966, 25.576,
        79.327,  9.247,  1.149, 77.553],
    6: [-0.6996, -1.9120, -1.5524, -1.4792, -1.2937,
        -1.8080, -1.5940, -1.5700, -0.9876],
    7: [0.1196, 0.1413, 0.2197, 0.2290, 0.1934,
        0.3705, 0.3448, 0.3185, 0.4174],
    8: [8.6945, 8.7377, 9.4489, 9.4330, 9.3981,
        7.4160, 8.0010, 7.8230, 9.5286],
}

all_inputs = {
    1: np.array([[0.250000,0.750000],[0.050000,0.050000],[0.754891,0.704403],
                 [0.374540,0.950714],[0.929616,0.316376],[0.369879,0.911559],
                 [0.524103,0.765891],[0.312456,0.876543],[0.199816,0.930286]]),
    2: np.array([[0.750000,0.250000],[0.500000,0.500000],[0.686831,0.530211],
                 [0.173199,0.159866],[0.984082,0.997991],[0.197124,0.134901],
                 [0.287456,0.321654],[0.317841,0.368804],[0.521679,0.544560]]),
    3: np.array([[0.333333,0.666667,0.500000],[0.350000,0.650000,0.500000],
                 [0.039713,0.302029,0.315311],[0.594963,0.644959,0.529293],
                 [0.094455,0.311399,0.225967],[0.196910,0.578417,0.518415],
                 [0.412789,0.534612,0.678901],[0.517589,0.451612,0.728901],
                 [0.444135,0.690017,0.469704]]),
    4: np.array([[0.200000,0.800000,0.400000,0.600000],[0.800000,0.200000,0.600000,0.400000],
                 [0.728602,0.982928,0.708406,0.027707],[0.208588,0.216178,0.533292,0.773294],
                 [0.674055,0.965114,0.741781,0.048580],[0.544877,0.212012,0.505766,0.142121],
                 [0.123456,0.876543,0.345678,0.654321],[0.456789,0.567890,0.567890,0.512345],
                 [0.456255,0.549210,0.507284,0.534446]]),
    5: np.array([[0.700000,0.300000,0.600000,0.200000],[0.720000,0.280000,0.580000,0.220000],
                 [0.014688,0.641578,0.349456,0.493352],[0.033484,0.654876,0.337950,0.480625],
                 [0.000000,0.653906,0.374032,0.519541],[0.035061,0.936953,0.433456,0.231451],
                 [0.612345,0.234567,0.789012,0.456789],[0.661034,0.311567,0.738512,0.456789],
                 [0.014371,0.939107,0.406653,0.158417]]),
    6: np.array([[0.200000,0.400000,0.600000,0.800000,0.500000],
                 [0.800000,0.600000,0.400000,0.200000,0.500000],
                 [0.575333,0.108777,0.034359,0.840559,0.517247],
                 [0.543673,0.089201,0.036835,0.833754,0.496370],
                 [0.447812,0.116655,0.108676,0.805596,0.481036],
                 [0.305865,0.067273,0.099217,0.188707,0.380951],
                 [0.234567,0.123456,0.987654,0.456789,0.678901],
                 [0.246789,0.141234,0.912345,0.385678,0.612345],
                 [0.130251,0.389162,0.512184,0.882463,0.463840]]),
    7: np.array([[0.150000,0.350000,0.550000,0.750000,0.950000,0.450000],
                 [0.250000,0.400000,0.500000,0.700000,0.850000,0.500000],
                 [0.102635,0.201553,0.788679,0.155646,0.990262,0.833759],
                 [0.109346,0.179923,0.776208,0.147628,0.987626,0.850870],
                 [0.070161,0.171326,0.805916,0.183311,0.953336,0.821749],
                 [0.144382,0.158557,0.675855,0.356036,0.978113,0.783174],
                 [0.187654,0.234567,0.698765,0.345678,0.987654,0.765432],
                 [0.201654,0.244567,0.708765,0.316678,0.967654,0.776432],
                 [0.165884,0.080835,0.634684,0.334935,1.000000,0.789440]]),
    8: np.array([[0.125000,0.250000,0.375000,0.500000,0.625000,0.750000,0.875000,0.437500],
                 [0.150000,0.300000,0.400000,0.480000,0.600000,0.700000,0.850000,0.450000],
                 [0.018659,0.622726,0.428889,0.224671,0.701438,0.385308,0.247735,0.172798],
                 [0.000000,0.623865,0.436282,0.188387,0.710042,0.358950,0.212939,0.208709],
                 [0.235697,0.815314,0.215750,0.128421,0.651928,0.386742,0.366773,0.147227],
                 [0.577976,0.625124,0.913285,0.349672,0.774003,0.455605,0.153573,0.983399],
                 [0.456789,0.345678,0.876543,0.234567,0.654321,0.567890,0.234567,0.876543],
                 [0.488769,0.396678,0.896543,0.275867,0.665321,0.597890,0.254567,0.886543],
                 [0.000000,0.640759,0.358863,0.177693,0.622064,0.382017,0.241987,0.160305]]),
}

# ============================================================================
# PLOT 1: W1-W9 Progress Trajectories
# ============================================================================

def plot_progress(save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Function Optimization Progress: Weeks 1–9 (Complete)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 9))

    for fid in range(1, 9):
        ax = axes[(fid-1)//4, (fid-1)%4]
        y = np.array(weekly_outputs[fid], dtype=float)
        # Clip near-zero to 0 for display
        y_display = np.where(np.abs(y) < 1e-10, 0.0, y)
        ax.plot(WEEKS, y_display, 'o-', linewidth=2, markersize=7, color='steelblue')
        # Highlight W9 if it's a new best
        best_prev = np.max(y_display[:-1])
        if y_display[-1] > best_prev:
            ax.plot(9, y_display[-1], '*', markersize=14, color='gold',
                    markeredgecolor='darkorange', label='New Best!')
            ax.legend(fontsize=8)
        ax.set_title(f'F{fid} ({DIMS[fid]}D)', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Output')
        ax.set_xticks(WEEKS)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=9, color='green', linestyle='--', alpha=0.4, linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close()


# ============================================================================
# PLOT 2: PCA Analysis of High-Value Regions
# ============================================================================

def plot_pca_analysis(save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('PCA of Top-3 Historical Inputs per Function (W1–W9)',
                 fontsize=13, fontweight='bold')

    for fid in range(1, 9):
        ax = axes[(fid-1)//4, (fid-1)%4]
        X = all_inputs[fid]
        y = np.array(weekly_outputs[fid])
        d = DIMS[fid]

        top_k = 3
        top_idx = np.argsort(y)[-top_k:]
        X_top  = X[top_idx]

        # Project all points onto PC1 of top-k
        if len(X_top) >= 2:
            pca = PCA(n_components=1)
            pca.fit(X_top)
            X_proj = pca.transform(X).flatten()
            var_exp = pca.explained_variance_ratio_[0]
        else:
            X_proj = np.zeros(len(X))
            var_exp = 0.0

        sc = ax.scatter(X_proj, y, c=WEEKS, cmap='RdYlGn', s=80,
                        edgecolors='grey', linewidths=0.5, zorder=3)
        ax.scatter(X_proj[top_idx], y[top_idx], s=150, marker='*',
                   color='gold', edgecolors='darkorange', linewidths=1, zorder=4, label='Top-3')
        ax.set_title(f'F{fid} ({d}D) — PC1 {var_exp*100:.0f}% var', fontweight='bold')
        ax.set_xlabel('PC1 projection')
        ax.set_ylabel('Output')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.colorbar(sc, ax=axes.ravel().tolist(), label='Week', shrink=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close()


# ============================================================================
# PLOT 3: Portfolio Trajectory W1-W9
# ============================================================================

def plot_portfolio(save_path=None):
    portfolios = []
    for w in range(9):
        total = sum(weekly_outputs[f][w] for f in range(1, 9))
        portfolios.append(total)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(WEEKS, portfolios,
                  color=['#d73027' if v < 10 else '#1a9850' for v in portfolios],
                  edgecolor='black', linewidth=0.7, alpha=0.85)
    ax.plot(WEEKS, portfolios, 'ko-', linewidth=1.5, markersize=5, zorder=5)
    ax.axhline(0, color='black', linewidth=0.8)
    for w, v in zip(WEEKS, portfolios):
        ax.text(w, v + (2 if v >= 0 else -5), f'{v:.1f}', ha='center',
                fontsize=8, fontweight='bold')
    ax.set_title('Portfolio Value W1–W9', fontsize=13, fontweight='bold')
    ax.set_xlabel('Week')
    ax.set_ylabel('Portfolio Sum')
    ax.set_xticks(WEEKS)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close()


# ============================================================================
# PCA SUMMARY TABLE
# ============================================================================

def print_pca_summary():
    print("\n" + "="*65)
    print("PCA SUMMARY — Top-3 Historical Inputs per Function (W1–W9)")
    print("="*65)
    for fid in range(1, 9):
        X = all_inputs[fid]
        y = np.array(weekly_outputs[fid])
        top_idx = np.argsort(y)[-3:]
        X_top = X[top_idx]
        if len(X_top) >= 2:
            pca = PCA(n_components=min(DIMS[fid], len(X_top)-1))
            pca.fit(X_top)
            print(f"\nF{fid} ({DIMS[fid]}D):")
            print(f"  Top-3 outputs: {np.round(y[top_idx], 4)}")
            print(f"  PC1 variance:  {pca.explained_variance_ratio_[0]*100:.1f}%")
            print(f"  PC1 direction: {np.round(pca.components_[0], 3)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    base = '/Users/ruiz.m.20/Documents/repos/imperial-ml-ai-capstone-project/results'

    print("Generating W1-W9 progress plots...")
    plot_progress(f'{base}/progress_w1_w9.png')

    print("Generating PCA analysis plots...")
    plot_pca_analysis(f'{base}/pca_analysis_w9.png')

    print("Generating portfolio trajectory...")
    plot_portfolio(f'{base}/portfolio_trajectory_w9.png')

    print_pca_summary()

    print("\nPortfolio values W1-W9:")
    for w in range(9):
        total = sum(weekly_outputs[f][w] for f in range(1, 9))
        print(f"  W{w+1}: {total:.4f}")
