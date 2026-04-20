# WEEK 9 FINAL STRATEGY: THE LAST STAND
## 2 Queries Remaining - Maximum Value Extraction

**Date**: March 9, 2026  
**Context**: Budget exhausted - only 2 submissions left before Week 10 finale  
**Portfolio Status**: 4.49 (down from W6 peak of 69.42)  
**Mission**: Maximize expected value with extreme selectivity

---

## 🎯 STRATEGIC ALLOCATION DECISION

### The Brutal Math
- **10 total submissions allowed** (Week 1-10)
- **8 submissions used** (Week 1-8: one per function)
- **2 submissions remaining** for Weeks 9-10
- **8 functions competing** for 2 slots

### Critical Question: Which 2 Functions to Query?

---

## 📊 FUNCTION-BY-FUNCTION ASSESSMENT (W1-W8 PERFORMANCE)

| Function | Best Value | Worst Value | W8 Value | Trend | Volatility | Confidence | Priority |
|:---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| **F1** | 2.6e-96 | -1.47e-21 | -1.21e-112 | Flat | 0% | 0% | ❌ **SKIP** |
| **F2** | 0.847 | -0.058 | 0.0329 | Chaotic | 78% | 15% | ❌ **SKIP** |
| **F3** | 0.0225 | -0.1383 | -0.1383 | ↓ Declining | 45% | 20% | ❌ **SKIP** |
| **F4** | -5.556 | -28.65 | **-5.556** | ↑ **+69% W8** | 65% | **60%** | ✅ **SELECT** |
| **F5** | 79.327 | 1.149 | 1.149 | ↓ Collapsed | 98% | 5% | ❌ **SKIP** |
| **F6** | -0.700 | -1.912 | -1.570 | Plateau | 22% | 35% | 🟡 Backup |
| **F7** | 0.3705 | 0.120 | 0.3185 | ↓ Declining | 18% | 25% | 🟡 Backup |
| **F8** | 9.449 | 7.416 | 7.823 | Stable | 8% | **70%** | ✅ **SELECT** |

### DECISION: Query F4 and F8

**Rationale:**
1. **F4**: Breakthrough momentum (+69% in W8). First time found exploitable structure in chaotic landscape. Bounded random strategy validated.
2. **F8**: Most reliable function (volatility=8%). Consistently near μ≈8.5. High-dimensional curse less severe than expected.

**Expected Returns:**
- **F4**: -5.556 → **-3.0 target** (+45% improvement) — Continue bounded exploration
- **F8**: 7.823 → **8.5 target** (+9% improvement) — Exploit stability near mean

**Risk Management:**
- Abandon F1 (noise floor), F2/F3/F7 (non-stationary decline), F5 (irreversibly collapsed), F6 (plateau)
- Accept 6-function loss to maximize 2-function gain
- Portfolio optimization: Better to have 2 strong performers than 8 mediocre attempts

---

## 🔬 WEEK 8 KEY INSIGHTS

### What Worked
✅ **F4 Bounded Random Exploration**: +69% improvement  
- Strategy: Uniform sampling within [0.3, 0.7] hypercube  
- Avoided overfitting (no complex models)  
- Lesson: **Simplicity beats sophistication in chaos**

✅ **F8 Steady Optimization**: Minimal decline (-2%)  
- 8D curse less severe than feared  
- GB + RBF SVM ensemble stable  
- Lesson: **High dimensionality manageable with regularization**

### What Failed
❌ **F2 Momentum Collapse**: 0.1429 → 0.0329 (-77%)  
- Recovery was temporary spike, not sustainable trend  
- Lesson: **One good result ≠ pattern discovery**

❌ **F3/F7 Continued Decline**: -31% and -8% respectively  
- No strategy stops non-stationary drift  
- Lesson: **Some landscapes unoptimizable with N<20**

❌ **F5 Still in Ruins**: 1.149 (-98.6% from W6 peak)  
- No recovery path exists  
- Lesson: **Local optima are traps without global view**

---

## 🎲 WEEK 9 QUERY GENERATION METHODOLOGY

### Function 4 Strategy: **Aggressive Bounded Exploitation**
**Approach**: Build on W8 breakthrough
- **W8 query**: [0.42, 0.58, 0.55, 0.50] → **-5.556 (best yet)**
- **W8 result**: Validated bounded region [0.4-0.6] as promising
- **W9 strategy**: Perturb slightly within winning region
- **Uncertainty-driven**: Add small Gaussian noise (σ=0.05) around best location

**Generated Query**:
```python
# Start from W8 location, add controlled perturbation
base = np.array([0.42, 0.58, 0.55, 0.50])
perturbation = np.random.normal(0, 0.05, 4)
query = np.clip(base + perturbation, 0.35, 0.65)  # Stay in validated region
```

**Expected**: -3 to -4 (continued improvement likely)

### Function 8 Strategy: **Cautious Mean Reversion**
**Approach**: Exploit stability near historical mean
- **Historical mean**: 8.82 (W1-W8 average)
- **W8 value**: 7.823 (below mean)
- **W9 strategy**: Sample near centroid of successful queries
- **Safety margin**: Stay in high-density region

**Generated Query**:
```python
# Compute centroid of top 3 queries, add small exploration
top_3_centroid = np.mean([q1, q2, q3], axis=0)
noise = np.random.uniform(-0.10, 0.10, 8)
query = np.clip(top_3_centroid + noise, 0.2, 0.9)
```

**Expected**: 8.2 to 8.8 (slight improvement or stability)

---

## 🎯 WEEK 9 SUBMISSION PLAN

### Queries to Submit (2 only)
1. **Function 4**: [aggressive bounded exploitation]
2. **Function 8**: [mean reversion exploitation]

### Functions to Abandon (6 total)
- **F1**: No signal exists (confirmed noise floor)
- **F2**: Unmanageable chaos (0.8 volatility)
- **F3**: Non-stationary decline
- **F5**: Beyond recovery (-98.6% from peak)
- **F6**: Plateau with no upside
- **F7**: Declining despite all strategies

### Expected Portfolio Impact
- **Conservative estimate**: 4.49 → 5.5 (+22%)
- **Optimistic estimate**: 4.49 → 7.0 (+56%)
- **Worst case**: 4.49 → 4.0 (-11%)

**Note**: Week 10 will receive zero queries (budget exhausted). W9 results are final values for 6 abandoned functions.

---

## 📖 META-LESSONS FROM 8 WEEKS

### Lesson 1: **Sample Efficiency is Impossible with N<20**
- 8 queries per function = coverage of <0.01% in 4D
- <0.0001% in 8D
- Bayesian optimization requires minimum 10×D samples to be effective
- **Conclusion**: This challenge tests decision-making under radical uncertainty, not optimization skill

### Lesson 2: **Simplicity Beats Complexity**
- F4 breakthrough used **bounded random sampling**
- F5 collapse used **sophisticated ensemble + hyperparameter tuning**
- Complex models overfit noise when N<<D
- **Conclusion**: Occam's Razor applies to surrogate model selection

### Lesson 3: **Non-Stationarity Dominates**
- F2, F3, F7 all show drift despite different strategies
- F5 peak was temporary local maximum
- Functions change shape over time (or are multi-modal)
- **Conclusion**: Single-objective optimization assumes stationarity; these functions violate that

### Lesson 4: **Portfolio Risk Management Crucial**
- F5 collapse wiped out 98% of portfolio gain (-70 point loss)
- Diversification would have limited damage
- Equal weighting across functions = naive strategy
- **Conclusion**: In real scenarios, allocate budget based on confidence levels

### Lesson 5: **Exploration > Exploitation with N<20**
- W6 aggressive exploitation (F5) → catastrophic failure
- W8 exploration (F4) → breakthrough
- With minimal data, exploring beats exploiting
- **Conclusion**: Invest in exploration until confidence threshold reached

---

## 🚀 WEEK 10 OUTLOOK

**No queries will be submitted in Week 10** (budget exhausted).

Week 10 serves as:
1. Final results observation for F4 and F8
2. Retrospective analysis of 8-week journey
3. Comprehensive report on lessons learned
4. Strategy recommendations for future practitioners

**Final Portfolio Prediction**: 5-7 range (realistic given constraints)

**Success Criteria**:
- ✅ Identified F4 breakthrough trajectory
- ✅ Validated F8 as most stable function
- ✅ Learned F5 peak-chasing danger
- ✅ Documented non-stationarity evidence
- ✅ Demonstrated defensive decision-making

**Failure Acknowledgment**:
- ❌ Did not achieve F5 peak sustainably
- ❌ Could not solve F2 chaos
- ❌ Misallocated queries to low-confidence functions (F1, F3, F7)

---

## 🎓 PRACTITIONER TAKEAWAYS

If faced with similar challenge:
1. **Budget conservatively**: Start with uniform sampling (Latin Hypercube)
2. **Abandon hopeless cases**: Don't waste queries on noise (F1) or chaos (F2)
3. **Double down on winners**: Allocate budget proportional to confidence
4. **Use simple models**: Ridge regression > Neural networks when N<20
5. **Quantify uncertainty**: Never exploit without confidence intervals
6. **Accept losses**: Perfect optimization impossible with radical constraints
7. **Document rigorously**: Learning > Performance when information is scarce

---

## ✅ NEXT STEPS

1. **Friday, March 9**: Generate and submit W9 queries (F4, F8 only)
2. **Monday, March 12**: Receive W9 results
3. **Friday, March 16**: Observe final W10 standings (no new queries)
4. **March 17-20**: Write comprehensive capstone report
5. **March 21**: Final presentation

**Estimated Time**: 6-8 hours for W9 query generation + analysis

---

**END OF WEEK 9 STRATEGY**

*"In black-box optimization with N<20, knowing when to quit is as important as knowing where to search."*
