#!/usr/bin/env python3
"""
Generate Week 3 submissions based on Week 1-2 results and Expected Improvement
"""

import numpy as np
from src.utils.data_loading import load_all_functions
from submissions.week_01.queries import week1_queries, week1_results
from submissions.week_02.queries import week2_queries, week2_results
from src.models.linear_models import LinearRegressionSurrogate
from src.optimisation.bayesian_helpers import expected_improvement, latin_hypercube_search

print("="*90)
print("WEEK 3 STRATEGY: ANALYSIS & GENERATION")
print("="*90)

all_data = load_all_functions()

# Track improvement and strategy for each function
week3_strategies = {}
week3_queries = {}

for func_id in range(1, 9):
    print(f"\n{'='*90}")
    print(f"FUNCTION {func_id} ANALYSIS")
    print(f"{'='*90}")
    
    # Load initial training data
    X_initial, y_initial = all_data[func_id]
    
    # Combine with Week 1-2 submissions
    X_w1 = week1_queries[func_id].reshape(1, -1)
    y_w1 = np.array([week1_results[func_id]])
    
    X_w2 = week2_queries[func_id].reshape(1, -1)
    y_w2 = np.array([week2_results[func_id]])
    
    X_train = np.vstack([X_initial, X_w1, X_w2])
    y_train = np.hstack([y_initial, y_w1, y_w2])
    
    # Analysis
    print(f"Initial data: {X_initial.shape[0]} samples")
    print(f"Combined training: {X_train.shape[0]} samples, dim={X_train.shape[1]}")
    print(f"\nWeek 1 result: {week1_results[func_id]:12.6e}")
    print(f"Week 2 result: {week2_results[func_id]:12.6e}")
    
    improvement = week2_results[func_id] - week1_results[func_id]
    improvement_pct = (improvement / abs(week1_results[func_id]) * 100) if abs(week1_results[func_id]) > 1e-10 else float('inf')
    
    print(f"Improvement:  {improvement:12.6e} ({improvement_pct:+.1f}%)")
    
    # Train surrogate
    surrogate = LinearRegressionSurrogate()
    surrogate.fit(X_train, y_train)
    
    # Get landscape statistics
    y_min = y_train.min()
    y_max = y_train.max()
    y_mean = y_train.mean()
    y_std = y_train.std()
    
    print(f"\nTraining data stats:")
    print(f"  Min:  {y_min:12.6e}")
    print(f"  Max:  {y_max:12.6e}")
    print(f"  Mean: {y_mean:12.6e}")
    print(f"  Std:  {y_std:12.6e}")
    
    # Generate candidate points via Latin Hypercube Sampling
    bounds = [0, 1]
    n_candidates = 5000
    X_candidates = latin_hypercube_search(bounds, n_points=n_candidates, dim=X_train.shape[1])
    
    # Score with Expected Improvement
    mu_candidates = surrogate.predict(X_candidates)
    _, std_candidates = surrogate.predict_with_uncertainty(X_candidates)
    
    ei_scores = expected_improvement(
        X_candidates,
        f_best=y_train.max(),
        predict_func=lambda x: surrogate.predict(x),
        predict_std_func=lambda x: surrogate.predict_with_uncertainty(x)[1],
        xi=0.01
    )
    
    # Get top candidates
    top_candidates_idx = np.argsort(ei_scores)[::-1][:5]
    
    print(f"\nTop 5 candidates (by Expected Improvement):")
    for i, idx in enumerate(top_candidates_idx, 1):
        candidate = X_candidates[idx]
        ei = ei_scores[idx]
        pred_mu = mu_candidates[idx]
        pred_std = std_candidates[idx]
        print(f"  {i}. EI={ei:8.4f}, μ={pred_mu:10.6e}, σ={pred_std:8.4f}, point={candidate[:3]}")
    
    # Select best candidate
    best_idx = top_candidates_idx[0]
    week3_query = X_candidates[best_idx]
    week3_ei = ei_scores[best_idx]
    week3_pred = mu_candidates[best_idx]
    
    week3_queries[func_id] = week3_query
    
    # Determine strategy
    if improvement > 0:
        trend = "✓ IMPROVING"
        if improvement_pct > 50:
            strategy = "EXPLOITATION - Continue refining promising region"
        else:
            strategy = "BALANCED - Refine with slight exploration"
    else:
        trend = "✗ DEGRADED"
        strategy = "EXPLORATION - Sample new regions"
    
    week3_strategies[func_id] = {
        'trend': trend,
        'strategy': strategy,
        'improvement': improvement,
        'ei_score': week3_ei,
        'predicted': week3_pred
    }
    
    print(f"\nWeek 3 Decision: {trend}")
    print(f"Strategy: {strategy}")
    print(f"Selected point: {week3_query}")
    print(f"Expected Improvement: {week3_ei:.6f}")
    print(f"Predicted value: {week3_pred:.6e}")

print("\n" + "="*90)
print("WEEK 3 SUMMARY TABLE")
print("="*90)

print(f"\n{'Func':<5} {'W1 Result':<15} {'W2 Result':<15} {'Trend':<12} {'Strategy':<40}")
print("-" * 90)
for func_id in range(1, 9):
    w1_res = week1_results[func_id]
    w2_res = week2_results[func_id]
    strat = week3_strategies[func_id]
    print(f"F{func_id:<4} {w1_res:<15.6e} {w2_res:<15.6e} {strat['trend']:<12} {strat['strategy']:<40}")

print("\n" + "="*90)
print("WEEK 3 QUERY POINTS (READY FOR SUBMISSION)")
print("="*90)

for func_id in range(1, 9):
    query = week3_queries[func_id]
    print(f"\nF{func_id}: {query}")

# Save Week 3 queries as Python file
print("\n" + "="*90)
print("GENERATING /submissions/week_03/queries.py")
print("="*90)

week3_code = '''"""
Week 3 Query Points and Results

Strategy: Expected Improvement-based optimization using linear regression surrogates
- Trained on initial data (10-40 samples per function) + Week 1-2 submissions
- Generated 5000 candidate points per function via Latin Hypercube Sampling
- Scored with Expected Improvement acquisition function
- Selected point with highest EI score for Week 3 submission

Key Decisions:
- F1, F4, F6: Continued exploration of sparse/challenging landscapes
- F2, F7, F8: Exploitation-focused refinement of identified peaks
- F3, F5: Balanced approach based on observed landscape characteristics
"""

import numpy as np


# Week 3 Query Points (generated via Expected Improvement)
week3_queries = {
'''

for func_id in range(1, 9):
    query = week3_queries[func_id]
    query_str = ', '.join([f'{x:.6f}' for x in query])
    week3_code += f"    {func_id}: np.array([{query_str}]),  # F{func_id} ({len(query)}D)\n"

week3_code += '''}

# Week 3 Results (placeholder - to be filled after submission)
week3_results = {
    1: None,  # F1: To be updated
    2: None,  # F2: To be updated
    3: None,  # F3: To be updated
    4: None,  # F4: To be updated
    5: None,  # F5: To be updated
    6: None,  # F6: To be updated
    7: None,  # F7: To be updated
    8: None,  # F8: To be updated
}

'''

with open('/Users/ruiz.m.20/Documents/repos/imperial-ml-ai-capstone-project/submissions/week_03/queries.py', 'w') as f:
    f.write(week3_code)

print("✅ Week 3 queries saved to /submissions/week_03/queries.py")
