"""
Week 5 Comprehensive Analysis: W1-W5 Trajectory Analysis

Performs complete trend analysis on all 8 functions across 5 weeks,
identifying performance patterns, tier classification, and strategic insights.
"""

import numpy as np
import sys

# Historical data W1-W5 (all 5 weeks)
historical_data = {
    1: {
        'W1': 2.6065864278618756e-96,
        'W2': 7.570914060942952e-193,
        'W3': -5.384584177282445e-16,
        'W4': -1.560646704467778e-117,
        'W5': 3.4416015849706167e-131,
    },
    2: {
        'W1': 0.3691787538388598,
        'W2': 0.8473573729146894,
        'W3': 0.4074279061230939,
        'W4': -0.05807400895675094,
        'W5': 0.053778481722633775,
    },
    3: {
        'W1': -0.010251690931823796,
        'W2': -0.010450162716101937,
        'W3': -0.07882847061831176,
        'W4': -0.012318067554316293,
        'W5': -0.13592439842996926,
    },
    4: {
        'W1': -13.072131637188551,
        'W2': -13.072131637188551,
        'W3': -28.648038812076084,
        'W4': -12.607647357899442,
        'W5': -27.440890417764923,
    },
    5: {
        'W1': 5.273302329600012,
        'W2': 4.049267429988913,
        'W3': 34.98323399644939,
        'W4': 32.96599170726208,
        'W5': 25.575607090129246,
    },
    6: {
        'W1': -0.6995639652538725,
        'W2': -1.9119879535617619,
        'W3': -1.552441674550123,
        'W4': -1.4792010945616396,
        'W5': -1.293746931550967,
    },
    7: {
        'W1': 0.11959165710190967,
        'W2': 0.14129996220103783,
        'W3': 0.219690205078482,
        'W4': 0.22895976507696808,
        'W5': 0.19344909329957222,
    },
    8: {
        'W1': 8.694471875,
        'W2': 8.73765,
        'W3': 9.4488988470416,
        'W4': 9.4329653859419,
        'W5': 9.3980882498781,
    },
}

def calc_change(old_val, new_val):
    """Calculate percentage change, handling edge cases."""
    if old_val == 0:
        return 100.0 if new_val != 0 else 0.0
    if new_val == 0 and old_val != 0:
        return -100.0
    return ((new_val - old_val) / abs(old_val)) * 100

print("\n" + "="*120)
print("WEEK 5 COMPLETE ANALYSIS: W1-W5 TRAJECTORY FOR ALL 8 FUNCTIONS")
print("="*120 + "\n")

print("COMPREHENSIVE TRAJECTORY TABLE (W1 → W5)\n")
print(f"{'Func':^6} | {'Dim':^4} | {'W1':^16} | {'W2':^16} | {'W3':^16} | {'W4':^16} | {'W5':^16} | {'W4→W5':^10} | {'Status':^20}")
print("-" * 150)

tier_classification = {}

for func_id in range(1, 9):
    data = historical_data[func_id]
    dims = {1:2, 2:2, 3:3, 4:4, 5:4, 6:5, 7:6, 8:8}
    
    w1, w2, w3, w4, w5 = data['W1'], data['W2'], data['W3'], data['W4'], data['W5']
    change_w4_w5 = calc_change(w4, w5)
    change_w1_w5 = calc_change(w1, w5)
    
    # Classify tier
    if func_id in [5, 7]:
        tier = "TIER 1"
        status = "🟢 Elite"
    elif func_id == 2:
        tier = "TIER 2"
        status = "🔄 Recovery"
    elif func_id in [3, 4]:
        tier = "TIER 3"
        status = "🟡 Uncertain"
    elif func_id in [6, 8]:
        tier = "TIER 4"
        status = "🟡 Steady"
    else:
        tier = "TIER 5"
        status = "🔴 Sparse"
    
    tier_classification[func_id] = tier
    
    print(f"F{func_id:1d}    | {dims[func_id]:4d} | {w1:16.6e} | {w2:16.6e} | {w3:16.6e} | {w4:16.6e} | {w5:16.6e} | {change_w4_w5:+9.1f}% | {status:^20}")

print("\n" + "="*120)
print("DETAILED PER-FUNCTION ANALYSIS")
print("="*120 + "\n")

def analyze_function(func_id):
    data = historical_data[func_id]
    w1, w2, w3, w4, w5 = data['W1'], data['W2'], data['W3'], data['W4'], data['W5']
    
    c12 = calc_change(w1, w2)
    c23 = calc_change(w2, w3)
    c34 = calc_change(w3, w4)
    c45 = calc_change(w4, w5)
    c15 = calc_change(w1, w5)
    
    print(f"\n🔹 FUNCTION {func_id} [{tier_classification[func_id]}]")
    print(f"   Trajectory: W1({w1:.6e}) → W2({w2:.6e}) → W3({w3:.6e}) → W4({w4:.6e}) → W5({w5:.6e})")
    print(f"   Change %:   W1→W2({c12:+.1f}%) → W2→W3({c23:+.1f}%) → W3→W4({c34:+.1f}%) → W4→W5({c45:+.1f}%) | Total({c15:+.1f}%)")
    
    # Analysis
    if func_id == 1:
        print(f"   Status: 🔴 CRITICALLY SPARSE")
        print(f"   Issue: Output magnitude ~e-100, no meaningful signal detected")
        print(f"   W6 Action: Final random query; prepare abandonment from W7")
        
    elif func_id == 2:
        print(f"   Status: 🟢 CATASTROPHIC RECOVERY SUCCESS!")
        print(f"   W4→W5: {c45:+.1f}% (crashed -0.058 → recovered +0.0538)")
        print(f"   Analysis: Non-linear landscape confirmed. Distance-based exploration worked.")
        print(f"   W6 Action: Continue exploration via SVM RBF (γ=0.1) around new peak")
        
    elif func_id == 3:
        print(f"   Status: 🔴 UNEXPECTED COLLAPSE!")
        print(f"   W4→W5: {c45:+.1f}% (was recovering -0.012 → crashed -0.136)")
        print(f"   Analysis: W5 exploration backfired. Overfitting to W1-W4 pattern.")
        print(f"   W6 Action: SAFETY RETREAT to W3 query. Use ensemble SVM+Linear.")
        
    elif func_id == 4:
        print(f"   Status: 🔴 CONTINUED DETERIORATION")
        print(f"   W4→W5: {c45:+.1f}% (worsening: -12.6 → -27.4)")
        print(f"   Analysis: Exploration failed. Non-linear surrogates inadequate.")
        print(f"   W6 Action: Reset to W3 query (-28.6, best known). Try ensemble method.")
        
    elif func_id == 5:
        print(f"   Status: 🟢 ELITE PLATEAU (First Regression)")
        print(f"   W4→W5: {c45:+.1f}% (32.97 → 25.58, first decline after +525% growth)")
        print(f"   Analysis: Plateau reached. Micro-exploitation now backfiring.")
        print(f"   W6 Action: Step back to W3/W4 center with ±0.02 perturbations. Use GP.")
        
    elif func_id == 6:
        print(f"   Status: 🟡 STEADY IMPROVEMENT")
        print(f"   W4→W5: {c45:+.1f}% (slight improvement: -1.479 → -1.294)")
        print(f"   Analysis: Consistent positive trend across all weeks. Linear working.")
        print(f"   W6 Action: Continue balanced exploitation (β=1.2 UCB). Maintain linear surrogate.")
        
    elif func_id == 7:
        print(f"   Status: 🟡 ELITE REGRESSION (First Decline)")
        print(f"   W4→W5: {c45:+.1f}% (0.229 → 0.193, growth interrupted)")
        print(f"   Analysis: Micro-exploitation overshot optimum. May have passed peak.")
        print(f"   W6 Action: Grid search near W3/W4 center (0.03 spacing). Use adaptive GP.")
        
    elif func_id == 8:
        print(f"   Status: 🟡 PLATEAU STALLED")
        print(f"   W4→W5: {c45:+.1f}% (minor fluctuations: 9.433 → 9.398)")
        print(f"   Analysis: No meaningful improvement possible; statistical noise floor.")
        print(f"   W6 Action: Final refinement; prepare abandonment. Continue linear baseline.")

# Analyze each function
for func_id in range(1, 9):
    analyze_function(func_id)

print("\n" + "="*120)
print("TIER-BASED SUMMARY & STRATEGIC RECOMMENDATIONS FOR WEEK 6")
print("="*120 + "\n")

print("TIER 1 - Elite Performers (Micro-Exploitation + Plateau Management):")
print("  ├─ F5: Micro-perturbations ±0.02, Gaussian Process surrogate, length_scale=0.05")
print("  └─ F7: Grid search near W3/W4 center (0.03 spacing), adaptive GP kernel\n")

print("TIER 2 - Catastrophic Recovery (Maximum Exploration):")
print("  └─ F2: Continue distance-based exploration, SVM RBF (γ=0.1), margin refinement\n")

print("TIER 3 - Recovering/Uncertain (Safety + Smart Exploration):")
print("  ├─ F3: RESET to W3 query (-0.0788), ensemble SVM+Linear, regularization")
print("  └─ F4: RESET to W3 query (-28.6), ensemble method, voting strategy\n")

print("TIER 4 - Steady Functions (Balanced Refinement):")
print("  ├─ F6: Continue balanced exploitation, linear surrogate, β=1.2")
print("  └─ F8: Final refinement, linear baseline, prepare abandonment\n")

print("TIER 5 - Abandoned Functions (Random Baseline):")
print("  └─ F1: Final random query, abandon from W7 onwards\n")

print("CRITICAL INSIGHTS:")
print("  1. Non-linearity CONFIRMED: F2 recovery proves linear surrogates fail")
print("  2. Micro-exploitation risks: F5/F7 regression suggests plateau overshoot")
print("  3. Strategy backfire: F3/F4 exploration failed; retreat strategy needed")
print("  4. Recovery potential: F2 +193% jump shows high-dimensional search works")
print("  5. Model recommendation: SVM RBF for F2-F4, GP for F5/F7, Linear for F6/F8\n")

print("="*120)
print("END OF WEEK 5 ANALYSIS")
print("="*120 + "\n")
