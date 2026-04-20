#!/usr/bin/env python3
"""Week 4 Input Strategy Analysis"""

import numpy as np
from src.utils.data_loading import load_all_functions
from submissions.week_01.queries import week1_queries, week1_results
from submissions.week_02.queries import week2_queries, week2_results
from submissions.week_03.queries import week3_queries, week3_results

print("\n" + "="*100)
print("WEEK 4 STRATEGIC INPUT RECOMMENDATIONS".center(100))
print("="*100)

all_data = load_all_functions()

print("\n🎯 PERFORMANCE-BASED STRATEGIES:\n")

strategies = {}

for func_id in range(1, 9):
    X_init, y_init = all_data[func_id]
    
    w1 = week1_queries[func_id]
    w2 = week2_queries[func_id]
    w3 = week3_queries[func_id]
    
    w1_r = week1_results[func_id]
    w2_r = week2_results[func_id]
    w3_r = week3_results[func_id]
    
    # Calculate improvement trend
    imp_2_3 = ((w3_r - w2_r) / (abs(w2_r) + 1e-10)) * 100
    dim = w1.shape[0]
    
    print(f"\n📍 FUNCTION {func_id} ({dim}D)")
    print(f"   W1: {w1_r:+.6e} | W2: {w2_r:+.6e} | W3: {w3_r:+.6e}")
    print(f"   Trend W2→W3: {imp_2_3:+.1f}%")
    
    # Categorize and recommend
    if imp_2_3 > 20:
        status = "EXCELLENT"
        strategy = "EXPLOITATION"
        print(f"   ✅ {status} - {strategy}")
        print(f"      Action: Continue refining near W3")
        
        # Exploit: move slightly toward the best point
        best_idx = np.argmax(y_init) if y_init.max() > 0 else np.argmin(np.abs(y_init))
        best_point = X_init[best_idx]
        w4 = 0.95 * w3 + 0.05 * best_point
        w4 = np.clip(w4, 0, 1)
        
    elif 0 < imp_2_3 <= 20:
        status = "IMPROVING"
        strategy = "BALANCED"
        print(f"   ✅ {status} - {strategy}")
        print(f"      Action: Small step + exploration")
        
        # Keep moving in same direction with slight noise
        direction = w3 - w2
        w4 = w3 + 0.3 * direction + np.random.uniform(-0.03, 0.03, dim)
        w4 = np.clip(w4, 0, 1)
        
    elif -10 <= imp_2_3 <= 0:
        status = "STABLE"
        strategy = "SHIFT"
        print(f"   → {status} - {strategy}")
        print(f"      Action: Move to underexplored region")
        
        # Move away from recent trajectory
        initial_center = X_init.mean(axis=0)
        w4 = 0.6 * initial_center + 0.4 * np.random.rand(dim)
        w4 = np.clip(w4, 0, 1)
        
    else:  # imp_2_3 < -10
        status = "DECLINING"
        strategy = "EXPLORATION"
        print(f"   ⚠️ {status} - {strategy}")
        print(f"      Action: Switch to new region entirely")
        
        # High exploration: sample from corners/edges
        if func_id in [2, 3, 4]:  # Struggling functions
            # Try opposite corner from recent queries
            avg_point = (w1 + w2 + w3) / 3
            w4 = 1 - avg_point + np.random.uniform(-0.1, 0.1, dim)
        else:
            w4 = np.random.rand(dim)
        w4 = np.clip(w4, 0, 1)
    
    print(f"   💡 W4 Suggestion: {np.round(w4, 3)}")
    
    strategies[func_id] = {
        'status': status,
        'strategy': strategy,
        'w4': w4,
        'trend': imp_2_3
    }

# Summary
print("\n" + "="*100)
print("WEEK 4 INPUT STRATEGY SUMMARY".center(100))
print("="*100 + "\n")

print("GROUP 1: WINNERS (F5, F7) → EXPLOITATION")
print("   Strategy: Expect Improvement with exploitation bias")
print("   β parameter: Low (0.5) for UCB - focus on best regions")
print("   Action: Refine around best points found\n")

print("GROUP 2: IMPROVING (F6, F8) → BALANCED")
print("   Strategy: Moderate exploration + exploitation")
print("   β parameter: Medium (1.5) for UCB")
print("   Action: Continue gradient while exploring nearby\n")

print("GROUP 3: DECLINING (F2, F3, F4) → EXPLORATION")
print("   Strategy: Increase exploration weight significantly")
print("   β parameter: High (3.0) for UCB")
print("   Action: Jump to underexplored regions, escape local optima\n")

print("GROUP 4: SPARSE (F1) → RANDOM EXPLORATION")
print("   Strategy: Surrogate may be unreliable with sparse results")
print("   β parameter: Maximum (5.0) for UCB - pure exploration")
print("   Action: Random or quasi-random sampling (Sobol sequence)\n")

print("="*100 + "\n")

print("DETAILED W4 QUERY RECOMMENDATIONS:\n")
for func_id in range(1, 9):
    w4 = strategies[func_id]['w4']
    status = strategies[func_id]['status']
    print(f"F{func_id}: {list(np.round(w4, 3))} ({status})")
