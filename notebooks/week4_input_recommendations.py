#!/usr/bin/env python3
"""Week 4 Input Strategy Analysis - Based on W1-W3 Performance"""

import numpy as np

# Week 1-3 Results Data
week1_results = {1: 2.6e-96, 2: 0.369, 3: -0.0103, 4: -13.07, 5: 5.273, 6: -0.700, 7: 0.120, 8: 8.694}
week2_results = {1: 7.57e-193, 2: 0.847, 3: -0.0105, 4: -13.07, 5: 4.049, 6: -1.912, 7: 0.141, 8: 8.738}
week3_results = {1: -5.38e-16, 2: 0.407, 3: -0.0788, 4: -28.65, 5: 34.98, 6: -1.552, 7: 0.220, 8: 9.449}

# Week 1-3 Query Points
week1_queries = {
    1: np.array([0.5, 0.5]),
    2: np.array([0.5, 0.5]),
    3: np.array([0.0, 0.0, 0.0]),
    4: np.array([0.5, 0.5, 0.5, 0.5]),
    5: np.array([0.5, 0.5, 0.5, 0.5]),
    6: np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    7: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    8: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
}

week2_queries = {
    1: np.array([0.0, 1.0]),
    2: np.array([0.0, 1.0]),
    3: np.array([1.0, 1.0, 1.0]),
    4: np.array([0.0, 1.0, 0.5, 0.5]),
    5: np.array([0.0, 0.0, 0.0, 1.0]),
    6: np.array([0.0, 1.0, 0.0, 1.0, 0.5]),
    7: np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    8: np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
}

week3_queries = {
    1: np.array([0.754891, 0.704403]),
    2: np.array([0.686831, 0.530211]),
    3: np.array([0.039713, 0.302029, 0.315311]),
    4: np.array([0.728602, 0.982928, 0.708406, 0.027707]),
    5: np.array([0.014688, 0.641578, 0.349456, 0.493352]),
    6: np.array([0.575333, 0.108777, 0.034359, 0.840559, 0.517247]),
    7: np.array([0.102635, 0.201553, 0.788679, 0.155646, 0.990262, 0.833759]),
    8: np.array([0.018659, 0.622726, 0.428889, 0.224671, 0.701438, 0.385308, 0.247735, 0.172798]),
}

print("\n" + "="*110)
print("WEEK 4 STRATEGIC INPUT ANALYSIS".center(110))
print("="*110 + "\n")

print(f"{'Func':<6} {'W2→W3 Change':<18} {'Status':<16} {'Strategy':<20} {'Recommended Action':<30}")
print("-" * 110)

for func_id in range(1, 9):
    w1_r = week1_results[func_id]
    w2_r = week2_results[func_id]
    w3_r = week3_results[func_id]
    
    # Calculate W2→W3 change
    if abs(w2_r) > 1e-10:
        change_pct = ((w3_r - w2_r) / abs(w2_r)) * 100
    else:
        change_pct = (1e6 if w3_r > 0 else -1e6)
    
    # Categorize by trend
    if change_pct > 20:
        status = "EXCELLENT ✅"
        strategy = "EXPLOITATION"
        action = "Refine near W3"
    elif 0 < change_pct <= 20:
        status = "IMPROVING ✅"
        strategy = "BALANCED"
        action = "Small step + exploration"
    elif -10 <= change_pct <= 0:
        status = "STABLE →"
        strategy = "SHIFT"
        action = "Try new region"
    else:  # < -10
        status = "DECLINING ⚠️"
        strategy = "EXPLORATION"
        action = "Jump to new area"
    
    print(f"F{func_id:<5} {change_pct:+.1f}%{'':<10} {status:<16} {strategy:<20} {action:<30}")

print("\n" + "="*110)
print("WEEK 4 QUERY RECOMMENDATIONS BY GROUP".center(110))
print("="*110 + "\n")

# Group 1: Excellent performers
print("📈 GROUP 1: WINNERS (F5, F7) - Continue Exploitation Strategy")
print("-" * 110)
print("   F5: {:.6e} to {:.6e} [+763.9%]".format(week2_results[5], week3_results[5]))
print("       Week 4 Action: Local refinement around W3 = [0.014688, 0.641578, 0.349456, 0.493352]")
print("       Suggested W4 ≈ [0.01±0.02, 0.64±0.05, 0.35±0.05, 0.49±0.05]")
print()
print("   F7: {:.6e} to {:.6e} [+55.5%]".format(week2_results[7], week3_results[7]))
print("       Week 4 Action: Local refinement around W3")
print("       Suggested W4 ≈ [0.10±0.05, 0.20±0.05, 0.79±0.05, 0.15±0.05, 0.99±0.02, 0.83±0.05]")
print()

# Group 2: Improving
print("\n✅ GROUP 2: IMPROVING (F6, F8) - Balanced Approach")
print("-" * 110)
print("   F6: {:.6e} to {:.6e} [+18.8%]".format(week2_results[6], week3_results[6]))
print("       Week 4 Action: Continue trend with exploration")
print("       Suggested W4 ≈ [0.55±0.08, 0.10±0.08, 0.03±0.05, 0.84±0.08, 0.52±0.08]")
print()
print("   F8: {:.6e} to {:.6e} [+8.1%]".format(week2_results[8], week3_results[8]))
print("       Week 4 Action: Continue gradient movement")
print("       Suggested W4 ≈ [0.02±0.05, 0.62±0.08, 0.43±0.08, 0.22±0.08, 0.70±0.08, 0.39±0.08, 0.25±0.08, 0.17±0.05]")
print()

# Group 3: Declining - needs adjustment
print("\n⚠️  GROUP 3: DECLINING (F2, F3, F4) - Increase Exploration")
print("-" * 110)
print("   F2: {:.6e} to {:.6e} [-51.9%]".format(week2_results[2], week3_results[2]))
print("       Week 4 Action: ESCAPE current region, try opposite corner or boundary")
print("       Suggested W4 ≈ [0.3-0.7, 0.0-0.3] or [0.0-0.3, 0.7-1.0] (explore edges)")
print()
print("   F3: {:.6e} to {:.6e} [-654.3%]".format(week2_results[3], week3_results[3]))
print("       Week 4 Action: MAJOR shift - try opposite region from W2")
print("       W2 was [1,1,1], W3 was [0.04, 0.30, 0.32] → Try [0.5-1.0, 0.5-1.0, 0.5-1.0]")
print()
print("   F4: {:.6e} to {:.6e} [-119.2%]".format(week2_results[4], week3_results[4]))
print("       Week 4 Action: Explore corners/edges, non-linear behavior suspected")
print("       Suggested W4 ≈ [0.0-0.3, 0.0-0.3, 0.3-0.7, 0.7-1.0] (edge exploration)")
print()

# Group 4: Sparse
print("\n❓ GROUP 4: SPARSE (F1) - High Uncertainty")
print("-" * 110)
print("   F1: {:.6e} to {:.6e} [Nearly zero - unreliable surrogate]".format(week2_results[1], week3_results[1]))
print("       Week 4 Action: Function likely ~0 everywhere. Try random exploration")
print("       Suggested W4 ≈ Random point or quasi-random (Sobol sequence)")
print()

print("\n" + "="*110)
print("WEEK 4 PARAMETER RECOMMENDATIONS FOR BAYESIAN OPTIMIZATION".center(110))
print("="*110 + "\n")

print("UCB (Upper Confidence Bound) β Parameters by Function Group:\n")
print("   F1 (SPARSE)     → β = 5.0  (Maximum exploration, surrogates unreliable)")
print("   F2, F3, F4 (DECLINING) → β = 3.0  (High exploration, escape local optima)")
print("   F6, F8 (IMPROVING)     → β = 1.5  (Balanced: some exploration, some exploitation)")
print("   F5, F7 (WINNING)       → β = 0.5  (Low exploration, focus on best regions)")
print()

print("="*110)
print("SUMMARY: W4 EXPECTED OUTCOMES\n")
print("✅ If strategy works:")
print("   • F5, F7: Continue +20-50% improvements (moving toward local optima)")
print("   • F6, F8: +5-15% improvements (maintaining gradient)")
print("   • F2, F3, F4: Recover or stabilize (found new peaks)\n")

print("⚠️  If current strategy fails:")
print("   • F2, F3, F4 stay negative → Need Week 5 SVM/NN surrogates (non-linearity)")
print("   • F1 stays at ~0 → Function is constant or has extreme sparsity\n")

print("="*110)
