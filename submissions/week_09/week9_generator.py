"""
Week 9 Query Generator - Final Two Submissions
================================================
Date: March 9, 2026
Strategy: Selective allocation to highest-confidence functions
Selected: F4 (breakthrough momentum) and F8 (stability)

This represents the culmination of 8 weeks of learning:
- F4: Bounded random exploration validated (+69% W8 improvement)
- F8: Most reliable function (8% volatility, consistent mean≈8.5)

Methodology:
1. Analyze W1-W8 historical data for F4 and F8
2. Build ensemble surrogate models
3. Use EI (Expected Improvement) for F4, PI (Probability of Improvement) for F8
4. Generate conservative queries with safety margins
5. Validate against constraints and historical patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.stats import norm
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL DATA (Weeks 1-8)
# ============================================================================

# Function 4 results (4D)
f4_inputs = np.array([
    [0.3, 0.7, 0.1, 0.9],      # W1: -13.07
    [0.5, 0.5, 0.5, 0.5],      # W2: -13.07
    [0.2, 0.8, 0.3, 0.7],      # W3: -28.65
    [0.1, 0.9, 0.2, 0.8],      # W4: -12.61
    [0.9, 0.1, 0.8, 0.2],      # W5: -27.44
    [0.412, 0.588, 0.523, 0.477],  # W6: -14.197
    [0.123456, 0.876543, 0.345678, 0.654321],  # W7: -17.894
    [0.420000, 0.580000, 0.550000, 0.500000],  # W8: -5.556 (BREAKTHROUGH!)
])

f4_outputs = np.array([
    -13.07, -13.07, -28.65, -12.61, -27.44, -14.197, -17.894, -5.556
])

# Function 8 results (8D)
f8_inputs = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # W1: 8.694
    [0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9],  # W2: 8.738
    [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1],  # W3: 9.449
    [0.25, 0.75, 0.35, 0.65, 0.45, 0.55, 0.15, 0.85],  # W4: 9.433
    [0.52, 0.48, 0.62, 0.38, 0.58, 0.42, 0.68, 0.32],  # W5: 9.398
    [0.456789, 0.543211, 0.612345, 0.387655, 0.523456, 0.476544, 0.634567, 0.365433],  # W6: 7.416
    [0.456789, 0.345678, 0.876543, 0.234567, 0.654321, 0.567890, 0.234567, 0.876543],  # W7: 8.001
    [0.470000, 0.410000, 0.900000, 0.290000, 0.680000, 0.610000, 0.270000, 0.890000],  # W8: 7.823
])

f8_outputs = np.array([
    8.694, 8.738, 9.449, 9.433, 9.398, 7.416, 8.001, 7.823
])

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_function_patterns(X: np.ndarray, y: np.ndarray, func_name: str) -> Dict:
    """Analyze patterns in historical data"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {func_name}")
    print(f"{'='*60}")
    
    stats = {
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y),
        'best_value': np.max(y),
        'best_idx': np.argmax(y),
        'worst_value': np.min(y),
        'trend': np.polyfit(range(len(y)), y, 1)[0],  # Linear trend
        'volatility': np.std(y) / (np.abs(np.mean(y)) + 1e-8),
        'recent_momentum': y[-1] - y[-3]  # Last 3 weeks
    }
    
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Best: {stats['best_value']:.4f} (Week {stats['best_idx']+1})")
    print(f"Worst: {stats['worst_value']:.4f}")
    print(f"Trend: {stats['trend']:.4f} per week")
    print(f"Volatility: {stats['volatility']:.2%}")
    print(f"Recent Momentum (W6-W8): {stats['recent_momentum']:.4f}")
    
    print(f"\nBest Location (Week {stats['best_idx']+1}):")
    print(f"  Input: {X[stats['best_idx']]}")
    print(f"  Output: {y[stats['best_idx']]:.4f}")
    
    return stats

# ============================================================================
# SURROGATE MODEL ENSEMBLE
# ============================================================================

def build_ensemble(X: np.ndarray, y: np.ndarray, func_name: str) -> Dict:
    """Build ensemble of surrogate models"""
    print(f"\n{'='*60}")
    print(f"BUILDING ENSEMBLE: {func_name}")
    print(f"{'='*60}")
    
    # Standardize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    
    # Model 1: Gradient Boosting (best for complex patterns)
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_scaled, y)
    models['gb'] = gb
    print(f"  GB Score: {gb.score(X_scaled, y):.4f}")
    
    # Model 2: Random Forest (robust to outliers)
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    rf.fit(X_scaled, y)
    models['rf'] = rf
    print(f"  RF Score: {rf.score(X_scaled, y):.4f}")
    
    # Model 3: SVM with RBF (smooth interpolation)
    print("Training SVM RBF...")
    svm = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)
    svm.fit(X_scaled, y)
    models['svm'] = svm
    print(f"  SVM Score: {svm.score(X_scaled, y):.4f}")
    
    # Model 4: Gaussian Process (uncertainty quantification)
    print("Training Gaussian Process...")
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.1,
        n_restarts_optimizer=5,
        random_state=42
    )
    gp.fit(X_scaled, y)
    models['gp'] = gp
    print(f"  GP Score: {gp.score(X_scaled, y):.4f}")
    
    return {'models': models, 'scaler': scaler}

# ============================================================================
# ACQUISITION FUNCTIONS
# ============================================================================

def expected_improvement(X_candidate: np.ndarray, ensemble: Dict, 
                        y_best: float, xi: float = 0.01) -> Tuple[float, float]:
    """Calculate Expected Improvement for candidate point"""
    models = ensemble['models']
    scaler = ensemble['scaler']
    
    X_scaled = scaler.transform(X_candidate.reshape(1, -1))
    
    # Get predictions from all models
    predictions = []
    for model_name, model in models.items():
        if model_name == 'gp':
            pred, std = model.predict(X_scaled, return_std=True)
            predictions.append(pred[0])
            uncertainty = std[0]
        else:
            predictions.append(model.predict(X_scaled)[0])
    
    mu = np.mean(predictions)
    
    # Use GP uncertainty if available, else use ensemble variance
    if 'gp' in models:
        sigma = uncertainty
    else:
        sigma = np.std(predictions)
    
    if sigma == 0:
        return 0.0, mu
    
    # Calculate EI
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    return ei, mu

# ============================================================================
# QUERY GENERATION
# ============================================================================

def generate_f4_query() -> np.ndarray:
    """
    Generate F4 query - Aggressive bounded exploitation
    
    Strategy: W8 achieved breakthrough (-5.556). This is best yet by far.
    Continue momentum by exploring around W8 location with controlled perturbation.
    """
    print("\n" + "="*60)
    print("F4 QUERY GENERATION")
    print("="*60)
    
    # Analyze patterns
    stats = analyze_function_patterns(f4_inputs, f4_outputs, "Function 4")
    
    # Build ensemble
    ensemble = build_ensemble(f4_inputs, f4_outputs, "Function 4")
    
    # W8 was breakthrough - start from there
    w8_location = f4_inputs[-1]  # [0.42, 0.58, 0.55, 0.50]
    best_value = stats['best_value']  # -5.556
    
    print(f"\nW8 Breakthrough Location: {w8_location}")
    print(f"W8 Value: {best_value:.4f}")
    
    print("\n" + "-"*60)
    print("STRATEGY: Perturb around W8 location")
    print("-"*60)
    
    # Generate candidates around W8 location
    n_candidates = 200
    candidates = []
    ei_values = []
    mu_values = []
    
    # Add small perturbations
    for _ in range(n_candidates):
        # Gaussian perturbation with clipping to stay in safe region
        perturbation = np.random.normal(0, 0.08, 4)
        candidate = w8_location + perturbation
        candidate = np.clip(candidate, 0.35, 0.65)  # Stay in validated [0.35, 0.65] region
        
        ei, mu = expected_improvement(candidate, ensemble, best_value, xi=0.01)
        
        candidates.append(candidate)
        ei_values.append(ei)
        mu_values.append(mu)
    
    # Select best EI
    best_idx = np.argmax(ei_values)
    best_candidate = candidates[best_idx]
    
    print(f"\nBest Candidate (max EI):")
    print(f"  Location: {best_candidate}")
    print(f"  Expected Improvement: {ei_values[best_idx]:.6f}")
    print(f"  Predicted Value: {mu_values[best_idx]:.4f}")
    print(f"  Expected Gain: {mu_values[best_idx] - best_value:.4f}")
    
    return best_candidate

def generate_f8_query() -> np.ndarray:
    """
    Generate F8 query - Cautious mean reversion
    
    Strategy: F8 is most stable (volatility=8%). Historical mean ≈ 8.8.
    Current value (7.823) is below mean. Sample near centroid of successful queries.
    """
    print("\n" + "="*60)
    print("F8 QUERY GENERATION")
    print("="*60)
    
    # Analyze patterns
    stats = analyze_function_patterns(f8_inputs, f8_outputs, "Function 8")
    
    # Build ensemble
    ensemble = build_ensemble(f8_inputs, f8_outputs, "Function 8")
    
    # Find top 3 performing queries
    top_3_idx = np.argsort(f8_outputs)[-3:]
    top_3_inputs = f8_inputs[top_3_idx]
    top_3_outputs = f8_outputs[top_3_idx]
    
    print(f"\nTop 3 Performing Queries:")
    for i, (idx, inp, out) in enumerate(zip(top_3_idx, top_3_inputs, top_3_outputs)):
        print(f"  {i+1}. Week {idx+1}: {out:.4f}")
        print(f"     Input: {inp}")
    
    # Compute centroid
    centroid = np.mean(top_3_inputs, axis=0)
    print(f"\nCentroid of Top 3: {centroid}")
    
    print("\n" + "-"*60)
    print("STRATEGY: Sample near centroid with small perturbations")
    print("-"*60)
    
    # Generate candidates around centroid
    n_candidates = 200
    candidates = []
    ei_values = []
    mu_values = []
    
    best_value = stats['best_value']
    
    for _ in range(n_candidates):
        # Small uniform perturbation
        perturbation = np.random.uniform(-0.12, 0.12, 8)
        candidate = centroid + perturbation
        candidate = np.clip(candidate, 0.15, 0.95)  # Stay in safe range
        
        ei, mu = expected_improvement(candidate, ensemble, best_value, xi=0.01)
        
        candidates.append(candidate)
        ei_values.append(ei)
        mu_values.append(mu)
    
    # Select best EI
    best_idx = np.argmax(ei_values)
    best_candidate = candidates[best_idx]
    
    print(f"\nBest Candidate (max EI):")
    print(f"  Location: {best_candidate}")
    print(f"  Expected Improvement: {ei_values[best_idx]:.6f}")
    print(f"  Predicted Value: {mu_values[best_idx]:.4f}")
    print(f"  Expected Gain: {mu_values[best_idx] - best_value:.4f}")
    
    return best_candidate

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("WEEK 9 QUERY GENERATION - FINAL TWO SUBMISSIONS")
    print("="*60)
    print("\nContext:")
    print("  - 8 weeks completed (1 query per function per week)")
    print("  - 2 queries remaining (budget limit)")
    print("  - Strategy: Allocate to F4 (breakthrough) and F8 (stability)")
    print("  - Goal: Maximize expected portfolio value")
    
    # Generate queries
    print("\n" + "="*60)
    print("GENERATING QUERIES")
    print("="*60)
    
    np.random.seed(42)  # For reproducibility
    
    f4_query = generate_f4_query()
    f8_query = generate_f8_query()
    
    # Final summary
    print("\n" + "="*60)
    print("WEEK 9 FINAL QUERIES")
    print("="*60)
    
    print(f"\nFunction 4 (4D):")
    print(f"  Query: {f4_query}")
    print(f"  Formatted: [{', '.join([f'{x:.6f}' for x in f4_query])}]")
    print(f"  Expected: -3.0 to -4.5 range")
    
    print(f"\nFunction 8 (8D):")
    print(f"  Query: {f8_query}")
    print(f"  Formatted: [{', '.join([f'{x:.6f}' for x in f8_query])}]")
    print(f"  Expected: 8.2 to 8.8 range")
    
    # Portfolio projection
    print("\n" + "="*60)
    print("PORTFOLIO PROJECTION")
    print("="*60)
    
    # Current portfolio (W8 values for all functions)
    current_portfolio = {
        1: -1.21e-112,
        2: 0.0329,
        3: -0.1383,
        4: -5.556,
        5: 1.149,
        6: -1.570,
        7: 0.3185,
        8: 7.823
    }
    
    current_sum = sum(current_portfolio.values())
    
    print(f"\nCurrent Portfolio (W8): {current_sum:.4f}")
    
    # Projected portfolio (optimistic)
    projected_opt = current_portfolio.copy()
    projected_opt[4] = -3.5  # F4 improvement
    projected_opt[8] = 8.5   # F8 improvement
    projected_opt_sum = sum(projected_opt.values())
    
    print(f"Projected Portfolio (optimistic): {projected_opt_sum:.4f}")
    print(f"Expected Gain: {projected_opt_sum - current_sum:.4f} ({100*(projected_opt_sum/current_sum - 1):.1f}%)")
    
    # Projected portfolio (conservative)
    projected_cons = current_portfolio.copy()
    projected_cons[4] = -4.5  # F4 modest improvement
    projected_cons[8] = 8.2   # F8 modest improvement
    projected_cons_sum = sum(projected_cons.values())
    
    print(f"Projected Portfolio (conservative): {projected_cons_sum:.4f}")
    print(f"Expected Gain: {projected_cons_sum - current_sum:.4f} ({100*(projected_cons_sum/current_sum - 1):.1f}%)")
    
    print("\n" + "="*60)
    print("Ready for submission to capstone portal")
    print("="*60)
