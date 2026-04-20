"""
Week 4 Results Analysis
Comprehensive analysis of W1→W4 trends and performance evaluation
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import results from each week
exec(open('../submissions/week_01/queries.py').read())
week1_results_data = week1_results

exec(open('../submissions/week_02/queries.py').read())
week2_results_data = week2_results

exec(open('../submissions/week_03/queries.py').read())
week3_results_data = week3_results

exec(open('../submissions/week_04/queries.py').read())
week4_results_data = week4_results

# Rename for use in analysis
week1_results = week1_results_data
week2_results = week2_results_data
week3_results = week3_results_data
week4_results = week4_results_data

def calculate_improvement(w3, w4):
    """Calculate % improvement from W3 to W4"""
    if abs(w3) < 1e-100:  # Handle near-zero values
        return "N/A (sparse)"
    return ((w4 - w3) / abs(w3)) * 100

def categorize_function(func_id, w1_w4_trend):
    """Categorize function performance"""
    if func_id == 1:
        return "🔴 Sparse/Unstable"
    elif func_id in [5, 7]:
        if w1_w4_trend > 100:
            return "🟢 Elite Winner"
        else:
            return "🟡 Winner (declining)"
    elif func_id in [2, 3, 4]:
        if w1_w4_trend < -50:
            return "🔴 Critical Decline"
        elif w1_w4_trend < 0:
            return "🟠 Moderate Decline"
        else:
            return "🟡 Recovery"
    elif func_id in [6, 8]:
        if w1_w4_trend > 0:
            return "🟢 Improving"
        else:
            return "🟡 Stalled"
    return "⚪ Unknown"

# Collect all results
all_results = {
    1: [week1_results[i] for i in range(1, 9)],
    2: [week2_results[i] for i in range(1, 9)],
    3: [week3_results[i] for i in range(1, 9)],
    4: [week4_results[i] for i in range(1, 9)]
}

print("="*100)
print("WEEK 4 RESULTS ANALYSIS: W1 → W2 → W3 → W4 TRENDS")
print("="*100)
print()

# Detailed analysis per function
print(f"{'Func':<6} {'W1':<12} {'W2':<12} {'W3':<12} {'W4':<12} {'W3→W4':<10} {'W1→W4':<10} {'Status':<25}")
print("-"*100)

for func_id in range(1, 9):
    w1 = all_results[1][func_id-1]
    w2 = all_results[2][func_id-1]
    w3 = all_results[3][func_id-1]
    w4 = all_results[4][func_id-1]
    
    w3_w4_change = calculate_improvement(w3, w4)
    w1_w4_change = calculate_improvement(w1, w4)
    status = categorize_function(func_id, 
                                 float(w1_w4_change.split()[0]) if isinstance(w1_w4_change, str) and 'N/A' not in w1_w4_change else 0)
    
    # Format outputs
    w1_str = f"{w1:.4e}" if abs(w1) < 0.01 or abs(w1) > 100 else f"{w1:.6f}"
    w2_str = f"{w2:.4e}" if abs(w2) < 0.01 or abs(w2) > 100 else f"{w2:.6f}"
    w3_str = f"{w3:.4e}" if abs(w3) < 0.01 or abs(w3) > 100 else f"{w3:.6f}"
    w4_str = f"{w4:.4e}" if abs(w4) < 0.01 or abs(w4) > 100 else f"{w4:.6f}"
    
    w3_w4_str = f"{w3_w4_change:+.1f}%" if isinstance(w3_w4_change, float) else w3_w4_change
    w1_w4_str = f"{w1_w4_change:+.1f}%" if isinstance(w1_w4_change, float) else w1_w4_change
    
    print(f"F{func_id:<5} {w1_str:<12} {w2_str:<12} {w3_str:<12} {w4_str:<12} {w3_w4_str:<10} {w1_w4_str:<10} {status:<25}")

print()
print("="*100)
print("KEY INSIGHTS FROM WEEK 4")
print("="*100)

# Function-specific insights
insights = {
    1: {
        "result": week4_results[1],
        "w3_w4": calculate_improvement(week3_results[1], week4_results[1]),
        "insight": "Still critically sparse (~1e-117). Extreme instability persists. Random sampling ineffective."
    },
    2: {
        "result": week4_results[2],
        "w3_w4": calculate_improvement(week3_results[2], week4_results[2]),
        "insight": "CATASTROPHIC: -0.058 (down from 0.407). Exploration strategy failed. Non-linearity confirmed."
    },
    3: {
        "result": week4_results[3],
        "w3_w4": calculate_improvement(week3_results[3], week4_results[3]),
        "insight": "RECOVERY: -0.012 (up from -0.079). Exploration helping. Still needs SVM/NN surrogate."
    },
    4: {
        "result": week4_results[4],
        "w3_w4": calculate_improvement(week3_results[4], week4_results[4]),
        "insight": "IMPROVEMENT: -12.61 (up from -28.65). Exploration working (+56% recovery)."
    },
    5: {
        "result": week4_results[5],
        "w3_w4": calculate_improvement(week3_results[5], week4_results[5]),
        "insight": "SLIGHT DECLINE: 32.97 (down from 34.98). Still elite. Exploitation near optimum plateau."
    },
    6: {
        "result": week4_results[6],
        "w3_w4": calculate_improvement(week3_results[6], week4_results[6]),
        "insight": "MODEST IMPROVEMENT: -1.479 (up from -1.552). Balanced strategy working (+4.7%)."
    },
    7: {
        "result": week4_results[7],
        "w3_w4": calculate_improvement(week3_results[7], week4_results[7]),
        "insight": "SUSTAINED GROWTH: 0.229 (up from 0.220). Exploitation successful (+4.1%)."
    },
    8: {
        "result": week4_results[8],
        "w3_w4": calculate_improvement(week3_results[8], week4_results[8]),
        "insight": "MINOR DECLINE: 9.433 (down from 9.449). Near plateau. May need exploration boost."
    }
}

for func_id, data in insights.items():
    print(f"\nF{func_id}: {data['result']:.6f}")
    print(f"  W3→W4 Change: {data['w3_w4']}")
    print(f"  💡 {data['insight']}")

print()
print("="*100)
print("STRATEGIC RECOMMENDATIONS FOR WEEK 5")
print("="*100)

recommendations = """
🎯 ELITE WINNERS (F5, F7):
   Strategy: Continue exploitation with micro-refinements
   β = 0.3 (tighter than W4's 0.5)
   Action: Small perturbations around W3/W4 best regions
   Rationale: Both near optimal plateau; avoid over-exploration

⚠️  CRITICAL RECOVERY NEEDED (F2):
   Strategy: AGGRESSIVE exploration + SVM surrogate
   β = 5.0 (maximum exploration)
   Action: Sample distant regions, abandon current area
   Rationale: Linear surrogates catastrophically failed; need non-linear model

🔄 RECOVERING (F3, F4):
   Strategy: Continue exploration but with SVM guidance
   β = 2.5 (slightly reduced from 3.0)
   Action: Maintain broad search with smarter candidate selection
   Rationale: Improvement trend confirmed; SVM can accelerate

🟢 STEADY IMPROVERS (F6, F7):
   Strategy: Balanced exploration-exploitation
   β = 1.0 (slightly tighter than W4's 1.5)
   Action: Refine near current best while exploring adjacent regions
   Rationale: Consistent progress; ready for refinement phase

⚪ UNSTABLE (F1):
   Strategy: Abandon optimization; use random baseline
   β = N/A (pure random sampling)
   Action: Accept that this function may be adversarial/chaotic
   Rationale: 4 weeks of attempts show no meaningful signal

🔧 TECHNICAL UPGRADES FOR WEEK 5:
   ✓ Implement SVM surrogates (RBF kernel) for F2, F3, F4
   ✓ Use Gaussian Process for F5, F7 (better uncertainty quantification)
   ✓ Ensemble methods: combine linear + SVM predictions
   ✓ Adaptive β: adjust per function based on surrogate confidence
"""

print(recommendations)

print()
print("="*100)
print("WEEK 5 EXECUTION PLAN")
print("="*100)
print("""
1. Update surrogates:
   - Train SVM (RBF) on all W1-W4 data for F2, F3, F4
   - Train Gaussian Process for F5, F7 (exploit plateau)
   - Keep linear baseline for F6, F8 (still responding well)

2. Generate candidates:
   - F1: Pure random (5,000 samples)
   - F2: LHS exploration (10,000 samples, β=5.0)
   - F3, F4: SVM-guided exploration (5,000 samples, β=2.5)
   - F5, F7: GP-guided exploitation (3,000 samples, β=0.3)
   - F6, F8: Balanced LHS (5,000 samples, β=1.0)

3. Selection:
   - Use Expected Improvement for F5, F7
   - Use Upper Confidence Bound for all others
   - Cross-validate surrogate predictions before selection

4. Quality checks:
   - Ensure queries in [0,1]^n bounds
   - Validate 6 decimal places
   - Verify no duplicates with W1-W4
""")

print("\n✅ Analysis complete. Ready to generate Week 5 queries.")
print("Run: python3 notebooks/generate_week5_queries.py")
