"""
WEEK 10 QUERIES — PCA-Guided GP/EI Optimization
================================================
Date: March 30, 2026
Strategy: PCA over top-k historical inputs → GP Expected Improvement filtering
Full history: W1–W9 (9 observations per function)

Method:
1. Identify top-k performing input points per function
2. Apply PCA to extract principal directions of the high-value region
3. Generate candidates along PC directions + centroid neighbourhood
4. Select final query via Gaussian Process Expected Improvement

This approach explicitly reduces the effective search dimensionality to the
directions that produced historical improvements, mirroring how PCA separates
meaningful variance from noise in high-dimensional datasets.

W9 Results (actual — informed these queries):
  F1: -2.34e-192  (noise floor, no change)
  F2:  0.4809     (+1362% vs W8)
  F3: -0.0051     (ALL-TIME BEST)
  F4: -4.635      (ALL-TIME BEST — 3rd consecutive improvement)
  F5: 77.553      (near W6 peak of 79.327)
  F6: -0.9876     (improved from -1.570)
  F7:  0.4174     (ALL-TIME BEST)
  F8:  9.529      (ALL-TIME BEST — surpassed W3 peak of 9.449)

Portfolio W9: 82.35  (+3,899% vs W8)
"""

import numpy as np

# ============================================================================
# WEEK 10 QUERIES — PCA-Guided GP/EI
# ============================================================================

week10_queries = {
    1: np.array([0.101613, 0.728058]),
    2: np.array([0.501828, 0.474203]),
    3: np.array([0.483270, 0.740795, 0.558480]),
    4: np.array([0.555318, 0.550912, 0.531991, 0.670623]),
    5: np.array([0.000000, 1.000000, 0.319539, 0.210119]),
    6: np.array([0.264033, 0.460343, 0.695113, 0.856664, 0.528116]),
    7: np.array([0.228555, 0.060712, 0.586883, 0.448725, 1.000000, 0.867749]),
    8: np.array([0.019493, 0.553542, 0.280020, 0.100114, 0.630907, 0.331912, 0.281512, 0.000000]),
}

# ============================================================================
# RESULTS (to be filled after submission)
# ============================================================================

week10_results = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
    8: None,
}

# ============================================================================
# PCA ANALYSIS SUMMARY
# ============================================================================

pca_insights = """
PCA PRINCIPAL DIRECTIONS (top-3 performers per function)
=========================================================

F4 (4D): PC1 explains 98.8% of variance
  Direction: [0.504, 0.697, 0.014, -0.510]
  → Dims 1, 2, 4 dominate. Dim 3 irrelevant.
  W10 query targets movement along this axis from W9 best.

F5 (4D): PC1 explains 98.3% of variance
  Direction: [-0.019, -0.689, -0.159, 0.707]
  → Low dim-2, high dim-4 consistently near peak.
  W10 query: [0.0, 1.0, 0.32, 0.21] — extreme values in dims 2/4.

F8 (8D): PC1 explains 81.0% of variance
  Direction: [0.058, -0.140, 0.603, 0.198, 0.684, -0.105, -0.127, 0.280]
  → Dims 3, 5, 8 carry signal. Most other dims near-noise.
  W10 query targets PC1 neighbourhood of W9 best.
"""

# ============================================================================
# STRATEGY RATIONALE
# ============================================================================

strategy = """
WEEK 10 STRATEGY: PCA-Guided Dimensionality Reduction
======================================================

Principle:
  With 9 observations per function, the dataset is large enough to reveal
  structure via PCA. The top-k historical inputs span a subspace — the
  principal components of that subspace point toward the most exploitable
  directions in the landscape.

  This parallels how PCA separates signal from noise in high-dimensional
  datasets: most variance is captured in 1-2 principal components, meaning
  most of the input space is irrelevant to output improvement.

Per-function approach:
  F1: No exploitable signal confirmed. Random query in unexplored region.
  F2: GP/EI targeting W2 peak neighbourhood (0.847 still unreached).
  F3: Refinement near W9 all-time best (-0.0051). PC1 exploration.
  F4: Continue 3-week improvement streak. PC1 = [0.50, 0.70, 0.01, -0.51].
  F5: Target extreme region dim-2=0, dim-4=1 (W6 peak mechanism).
  F6: Target historical best region (-0.700 at W1). PC-guided search.
  F7: Continue momentum from W9 all-time best (0.4174). PC1 refinement.
  F8: W9 new record (9.529). Target same cluster with PC1 perturbation.

Expected outcomes:
  Conservative portfolio: ~15–20 (F5 + F8 carry weight)
  Optimistic portfolio:   ~85–95 (F5 near peak + F8 new record)
"""

# ============================================================================
# PORTAL FORMAT
# ============================================================================

def print_portal_format():
    print("WEEK 10 PORTAL SUBMISSION")
    print("=" * 50)
    for f, q in week10_queries.items():
        fmt = '-'.join(f'{v:.6f}' for v in q)
        print(f"F{f}: {fmt}")

if __name__ == "__main__":
    print_portal_format()
    print(pca_insights)
