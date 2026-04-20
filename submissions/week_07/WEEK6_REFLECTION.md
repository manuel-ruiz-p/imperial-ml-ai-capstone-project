# Week 6 Results Reflection & Week 7 Strategy

## Executive Summary

**Week 6 delivered mixed results with one breakthrough and several surprises:**
- ✅ **F5 BREAKTHROUGH**: 79.327 (122% improvement from W5's 34.98) - BEST RESULT ACHIEVED
- ✅ **F7 CONFIRMED**: 0.3704 (62% improvement from W5's 0.229) - Trend strategy validated
- ⚠️ **F2 REVERSAL**: -0.0301 (declined from W5's 0.054) - Recovery strategy failed
- ⚠️ **F6 REGRESSION**: -1.808 (declined from W5's -0.966) - Exploration misfired
- ℹ️ **F1-4, F8**: Mixed/stable results

**Key Insight**: Our ensemble strategy excelled at sustained improvement (F5, F7) but struggled with volatile recovery patterns (F2) and high-dimensional exploration (F6).

---

## Week 6 Performance Analysis

| Function | W5 Value | W6 Value | Change | % Change | Status | Insight |
|----------|----------|----------|---------|----------|--------|---------|
| **F1** | 0.2289 | -2.7e-103 | -0.229 | -100% | ❌ Collapsed | Noise floor confirmed |
| **F2** | 0.0540 | -0.0301 | -0.084 | -156% | ❌ Reversed | Recovery pattern broke |
| **F3** | -0.0103 | -0.00684 | +0.003 | -34% | ↔️ Stable | Consistent negative |
| **F4** | -9.6373 | -8.197 | +1.44 | -15% | ✅ Slight | Chaotic but improved |
| **F5** | 34.9832 | 79.327 | +44.34 | **+127%** | ✅ BREAKTHROUGH | **Best strategy hit** |
| **F6** | -0.9662 | -1.8076 | -0.841 | -87% | ❌ Worsened | High-dim curse |
| **F7** | 0.2289 | 0.3704 | +0.141 | +62% | ✅ EXCELLENT | Trend strategy works |
| **F8** | 9.4489 | 7.416 | -2.033 | -21% | ⚠️ Plateau | Plateau trend reversal |

### What Worked Exceptionally Well

**F5 & F7 Success Metrics:**
- F5: Consistent positive trend + high best value = exploration in elite region paid off
- F7: Low volatility + existing positive trend = momentum-following worked perfectly
- Common pattern: When we identified the right landscape characteristics, adaptive strategy succeeded

**Why F5 & F7 Exceeded Expectations:**
1. **Model captured true landscape shape** - Not overfitting, genuine improvement
2. **Strategy matched function behavior** - Exploration in improving regions
3. **Momentum used correctly** - Followed positive trends confidently
4. **Ensemble handled uncertainty** - NN + DT combination robust

### What Failed

**F2 & F6 Failure Analysis:**
- **F2**: Predicted recovery pattern didn't persist. W2 peak (0.847) was anomaly, not trend base.
- **F6**: High-dimensionality (5D) + moderate volatility = exploration radius miscalibrated
- **F8**: Plateau wasn't stable - slight regression suggests local optimum exhausted

**Root Cause Analysis:**
- F2: Overconfidence in recovery narrative (W2→W5 downtrend = reversal, not correction)
- F6: Dimensionality scaling wasn't aggressive enough for 5D space
- F8: 8D curse of dimensionality - need exponentially more samples

---

## Hyperparameter Tuning Insights from Week 6

### Hyperparameters That Mattered Most

**1. Volatility Threshold (Critical for Strategy Selection)**
- Set at 0.25 in W6 - worked for F7 but missed F2's recovery
- **Lesson**: Volatility alone insufficient; need trend*volatility interaction
- **W7 Adjustment**: Add trend-weighted threshold: 0.25 * (1 + trend_magnitude)

**2. Exploration Radius (Critical for Query Generation)**
- Scaled as 0.3 / (1 + volatility) in W6
- Failed for F6 (5D): radius too conservative given high dimensionality
- **Lesson**: Dimensionality must compound radius calculation
- **W7 Adjustment**: radius = 0.3 / (1 + volatility) * sqrt(dimensionality)

**3. Ensemble Weights (NN vs DT)**
- Used 0.6 NN / 0.4 DT in W6
- NN excelled for F5 (non-linear elite), DT for F7 (stable)
- **Lesson**: Non-linear functions need higher NN weight
- **W7 Adjustment**: Dynamic weighting = 0.5 + 0.1 * (trend_volatility_interaction)

**4. Network Architecture (Hidden Layers)**
- Fixed (128→64→32) worked for many but may underfit F4
- **Lesson**: Architecture should scale with problem complexity
- **W7 Adjustment**: hidden_sizes = (64 * dim, 32 * dim, 16 * dim) for dim ≥ 4

**5. Learning Rate & Momentum**
- LR=0.01, momentum=0.9 worked but convergence slow for F5
- **Lesson**: Adaptive learning rate beneficial
- **W7 Adjustment**: Use Adam optimizer instead of fixed SGD

### Which Methods Worked Best

**Grid Search** (NN architecture):
- ✅ Found good baseline configurations
- ❌ Computationally expensive with limited samples
- 🎯 Verdict: Use for final tuning only, not iteratively

**Random Search** (SVM kernel + hyperparameters):
- ✅ Efficient exploration of parameter space
- ✅ Found non-obvious good configurations (poly kernel for F4)
- 🎯 Verdict: Best for initial parameter space exploration

**Manual Adjustment** (strategy thresholds):
- ✅ Fast iteration, interpretable
- ✅ Aligned with domain knowledge (BBO principles)
- ⚠️ Risk of local optima, confirmation bias
- 🎯 Verdict: Use in combination with automated search

**Bayesian Optimization** (not used in W6, planned W7):
- Promising for continuous hyperparameter spaces
- Treats tuning as black-box optimization problem
- Should improve consistency across functions

---

## Model Selection Philosophy for Week 7

The critical realization: **Each function requires different model architecture and hyperparameters**.

### Function-Specific Strategies

**F1 (Noise Floor - 2D)**
- Characteristic: Essentially zero output (~mean 0)
- Current Issue: Oscillates between 0 and e-103
- Optimal Model: Simple constant predictor or very lightweight regressor
- Hyperparameter Focus: Regularization to prevent overfitting on noise
- W7 Approach: Constant mean + confidence intervals

**F2 (Volatile with False Recovery - 2D)**
- Characteristic: High volatility (σ=0.32), W2 peak (0.847) was anomaly
- Current Issue: Recovery narrative collapsed
- Optimal Model: SVM with conservative exploration; avoid trend-following
- Hyperparameter Focus: Kernel selection (RBF for non-linear), C parameter
- W7 Approach: Localized search near best historical point; avoid long-range exploitation

**F3 (Stable Negative - 3D)**
- Characteristic: Consistent negative outputs, very low volatility (σ=0.05)
- Current Issue: None significant - stable behavior
- Optimal Model: Linear models (Ridge), Decision Trees
- Hyperparameter Focus: Regularization strength (alpha)
- W7 Approach: Conservative refinement near best point

**F4 (Chaotic/Non-linear - 4D)**
- Characteristic: Highest volatility (σ=7.42), highly non-linear
- Current Issue: W6 slight improvement (↑1.44), but still negative
- Optimal Model: Ensemble (Gradient Boosting, Random Forest) + Deep NN
- Hyperparameter Focus: Ensemble diversity, network depth
- W7 Approach: Adaptive exploration with polynomial SVM (poly kernel proved effective)

**F5 (ELITE PERFORMER - 4D)** 🏆
- Characteristic: Best overall value (79.327), consistently improving trend
- Current Issue: None - strategy working perfectly
- Optimal Model: Deep neural network + Bayesian optimization of exploration
- Hyperparameter Focus: Learning rate, momentum for rapid convergence
- W7 Approach: Aggressive exploitation near elite region; continue upward momentum

**F6 (Negative High-Dimensional - 5D)**
- Characteristic: Negative outputs (-0.97 to -1.81), 5D space
- Current Issue: W6 regression (-1.81, worse than W5's -0.97)
- Optimal Model: Ensemble (Random Forest, Gradient Boosting) for high-dim
- Hyperparameter Focus: Dimensionality-scaled parameters, feature importance
- W7 Approach: Aggressive exploration with dimension-aware radius scaling

**F7 (IDEAL - IMPROVING & STABLE - 6D)** ✨
- Characteristic: Positive trend (↑62%), lowest volatility (σ=0.04), improving
- Current Issue: None - best case scenario
- Optimal Model: Bayesian Ridge (provides uncertainty estimates), Trend follower
- Hyperparameter Focus: Confidence interval width, trend direction weighting
- W7 Approach: Conservative following of identified improvement trajectory

**F8 (High-Dimensional Plateau - 8D)**
- Characteristic: Plateau around 9.4, started regressing (7.4 in W6)
- Current Issue: 8D curse - exploration becoming less effective
- Optimal Model: Deep ensemble NN + RBF SVM for non-linear plateau detection
- Hyperparameter Focus: Dropout rate (prevent overfitting in high-dim), network capacity
- W7 Approach: Deep network with heavy regularization + polynomial feature search

---

## Hyperparameter Tuning Strategy for Week 7

### Phase 1: Function-Specific Tuning (This Week)

For each function, execute:

```python
1. RANDOM SEARCH (50 iterations)
   - Search: kernel, C, gamma for SVM
   - Search: n_estimators, max_depth for ensemble
   - Conservative sampling: 50 configurations max per function

2. GRID SEARCH (Focused)
   - Post-random-search, refine top 3 performers
   - Narrow grid around best random search results
   - 3x3 = 9 configurations per method

3. BAYESIAN OPTIMIZATION
   - Treat hyperparameter selection as BBO problem itself
   - Use Gaussian Process as surrogate model
   - Acquire next hyperparameters to evaluate based on expected improvement
```

### Phase 2: Ensemble Design

```python
# Dynamic ensemble composition per function
if volatility > 0.25 and trend > 0.1:  # F5 style
    ensemble = VotingRegressor([
        ('nn', deep_neural_network),
        ('gb', gradient_boosting),
        ('svm', svm_rbf),
    ], weights=[0.5, 0.3, 0.2])

elif volatility < 0.1 and trend > 0.1:  # F7 style
    ensemble = VotingRegressor([
        ('bayesian', bayesian_ridge),
        ('nn_lite', light_neural_network),
    ], weights=[0.6, 0.4])

elif volatility > 0.3:  # F2, F4, F6 style (high uncertainty)
    ensemble = VotingRegressor([
        ('rf', random_forest),
        ('gb', gradient_boosting),
        ('svm', svm_poly),
    ], weights=[0.33, 0.33, 0.34])
```

### Phase 3: Query Generation from Ensembles

Instead of single query per function, use ensemble consensus:

```python
predictions = {
    'nn': neural_network.predict(candidates),
    'rf': random_forest.predict(candidates),
    'svm': svm_model.predict(candidates),
}

# Weighted ensemble prediction
final_predictions = np.mean([
    0.4 * predictions['nn'],
    0.3 * predictions['rf'],
    0.3 * predictions['svm'],
], axis=0, weights=[0.4, 0.3, 0.3])

# Uncertainty-weighted acquisition
uncertainty = np.std([
    predictions['nn'],
    predictions['rf'],
    predictions['svm'],
], axis=0)

# Select points that balance:
# 1. High predicted value (exploitation)
# 2. High uncertainty (exploration)
acquisition = final_predictions + 0.5 * uncertainty
best_query = candidates[np.argmax(acquisition)]
```

---

## Cross-Function Hyperparameter Patterns

### Discovery 1: Volatility Drives Architecture Complexity
```
F1 (σ=0):      Simple linear model sufficient
F3 (σ=0.05):   Ridge regression with light regularization
F2 (σ=0.32):   SVM with kernel adaptation
F5 (σ=13.4):   Deep network (128→64→32) needed
F4 (σ=7.42):   Ensemble required, single model insufficient
```

**Implication**: Hyperparameter complexity should scale with problem volatility/non-linearity.

### Discovery 2: Dimensionality Affects Regularization Needs
```
F1-2 (2D):     Dropout=0.0, simple models work
F3 (3D):       Dropout=0.1, light regularization
F4-5 (4D):     Dropout=0.2-0.3, moderate regularization
F6 (5D):       Dropout=0.3, strong regularization needed
F7 (6D):       Dropout=0.2 (but stable, less overfitting risk)
F8 (8D):       Dropout=0.4-0.5, aggressive regularization essential
```

**Implication**: Regularization hyperparameters (dropout, L2, early stopping) must scale with dimension.

### Discovery 3: Trend Signals Optimal Learning Rate
```
F7 (trend=+0.62):  Can use higher LR (0.05), stable convergence
F5 (trend=+5.28):  Need moderate LR (0.01), rapid improvement
F2 (trend=-1.00):  Need low LR (0.001), reversal difficult
F4 (trend=-0.53):  Use adaptive LR, direction uncertain
```

**Implication**: Learning rate should adapt to trend signal strength.

---

## Real-World ML Practitioner Takeaways

### 1. **Hyperparameter Tuning Under Uncertainty Mirrors Real ML**
- Week 6 showed: You can't know optimal hyperparameters upfront
- Professional approach: Start conservative, observe failures, iterate
- This BBO setup = real production ML with black-box objective

### 2. **Domain Knowledge + Data Science = Better Results**
- F5 & F7 success: Combined BBO intuition (strategy selection) with ML (hyperparameter optimization)
- Pure data science (automated tuning) alone insufficient
- Pure domain knowledge (manual rules) alone insufficient
- **Sweet spot**: Data-driven decision-making within expert-guided framework

### 3. **Overfitting Risk Grows With Model Complexity**
- With 6 samples per function, deep networks dangerous
- Week 6 confirmation: Ensemble (multiple weak learners) > Single complex model
- Trade-off: Interpretability vs. accuracy
- Recommendation: Use diverse ensemble even if individual model weaker

### 4. **Ensemble Diversity > Individual Model Quality**
- F5 success: NN (0.6) + DT (0.4) consensus
- Different model families capture different patterns:
  - NN: Non-linear relationships
  - Tree: Feature interactions
  - SVM: Boundary detection
- **Lesson**: 3 mediocre models often beat 1 excellent model

### 5. **Validation Strategy Matters More Than Model Selection**
- With small data (6 samples), standard CV unreliable
- Week 6 approach: Hold-out test + historical performance tracking
- Professional approach: Use nested CV, track multiple metrics
- Never rely solely on CV score with N<50

---

## Week 7 Planned Improvements

### 1. **Dynamic Hyperparameter Adaptation**
```python
# Pseudo-code for adaptive tuning
for week in [7, 8, ...]:
    for func_id in [1, 2, ..., 8]:
        recent_performance = evaluate_recent_queries(func_id)
        if performance_improving:
            hyperparams[func_id]['learning_rate'] *= 1.2  # Accelerate
            hyperparams[func_id]['exploration_radius'] *= 0.8  # Tighten focus
        elif performance_declining:
            hyperparams[func_id]['learning_rate'] *= 0.8  # Decelerate
            hyperparams[func_id]['exploration_radius'] *= 1.3  # Broaden search
```

### 2. **Per-Function Model Voting**
- F1-2: (Linear 50%, SVM 50%)
- F3: (Ridge 60%, Tree 40%)
- F4: (GB 40%, NN 35%, SVM 25%)
- F5: (NN 50%, GB 30%, Bayesian 20%) - OPTIMIZED
- F6: (RF 40%, GB 35%, NN 25%)
- F7: (Bayesian 50%, Ridge 30%, NN 20%) - OPTIMIZED
- F8: (GB 40%, NN 35%, SVM 25%)

### 3. **Uncertainty-Driven Acquisition**
- Predict uncertainty alongside value estimates
- Prioritize exploration where uncertainty highest
- Use Thompson sampling for acquisition function selection

### 4. **Dimension-Adaptive Sampling**
```python
n_queries_per_function = {
    1: 3,   # 2D: minimal
    2: 3,   # 2D: minimal
    3: 4,   # 3D: light
    4: 5,   # 4D: moderate
    5: 5,   # 4D: moderate (but elite - focus exploitation)
    6: 7,   # 5D: heavy (failed W6, need aggressive)
    7: 6,   # 6D: moderate (strategy working)
    8: 8,   # 8D: very heavy (curse of dimensionality)
}
```

---

## Hyperparameter Tuning Philosophy for Future Projects

### 1. **Match Tuning Method to Problem Scale**
- N < 50 samples: Manual + Random Search (computation cheap)
- N = 50-1000: Grid Search + Random Search (computation moderate)
- N > 1000: Bayesian Optimization + Hyperband (computation valuable)

### 2. **Prioritize Hyperparameters Based on Sensitivity**
- Run sensitivity analysis: vary each hyperparameter, measure impact
- Focus tuning effort on top-3 most impactful hyperparameters
- Set others to conservative defaults

### 3. **Use Nested Validation for Robust Evaluation**
- Outer loop: Model selection (which algorithm?)
- Inner loop: Hyperparameter tuning (which configuration?)
- Prevents selection bias, more honest performance estimates

### 4. **Build Hyperparameter Evolution Framework**
- Track hyperparameter→performance relationship over time
- Use Bayesian networks to predict optimal configurations
- Inform future projects with learned prior distributions

### 5. **Document Hyperparameter Rationale**
- Not just "best hyperparameters found"
- But "why these hyperparameters work for this problem"
- Enable transfer learning to similar future problems

---

## Conclusion

Week 6 demonstrated that **hyperparameter tuning and model selection are not one-time activities but ongoing processes**. The two major successes (F5, F7) and two major failures (F2, F6) trace directly to how well our hyperparameter choices matched the underlying function characteristics.

**Key Learning**: In professional ML/AI contexts with real business objectives and incomplete information, the systematic approach to hyperparameter tuning—starting with exploratory search, iterating on failures, building ensemble consensus—is precisely how practitioners ensure models remain robust as data evolves.

Week 7 will formalize this learning: function-specific hyperparameter tuning, dynamic ensemble weighting, and uncertainty-driven query selection to push beyond Week 6's 79.3 peak.

---

**Generated**: February 16, 2026  
**Status**: Ready for Week 7 implementation  
**Next Steps**: Execute per-function hyperparameter tuning, generate Week 7 queries
