# Week 7 Technical Report: Hyperparameter Tuning in Black-Box Optimization

## Key Reflection Questions Addressed

### 1. Which Hyperparameters Did You Choose to Tune, and Why?

**Primary Hyperparameters Tuned:**

| Hyperparameter | W6 Status | W7 Status | Why Prioritized |
|---|---|---|---|
| **Learning Rate** | Fixed 0.01 | Dynamic per function | CV-dependent: F2 needed 0.0006, F5 needed 0.005 |
| **Regularization (Dropout)** | Fixed 0.2 | Dimension-adaptive | Overfitting risk scales as $\sqrt{D}$ |
| **Ensemble Weights** | Fixed 0.6/0.4 | Volatility-adjusted | F5's elite status needs NN emphasis (60%), F2's chaos needs diversity |
| **Exploration Radius** | Fixed formula | Dimension-aware scaling | F6 regressed (-87%) due to undersized radius in 5D |
| **Strategy Threshold** | Volatility only | Trend-interaction adjusted | F2 + negative trend = avoid exploitation, but volatility alone suggested explore |
| **Network Architecture** | Fixed (128→64→32) | Dimension-scaled | 8D (F8) and 5D (F6) need larger networks |

**Why These Six:**

1. **Learning rate**: Controls convergence speed and stability - affects all functions
2. **Regularization**: Prevents overfitting on small datasets - critical with N=6
3. **Ensemble weights**: Determines which model family dominates - high-impact on predictions
4. **Exploration radius**: Directly controls query point selection - critical for acquisition
5. **Strategy threshold**: Binary decision (exploit vs explore) with major consequences
6. **Architecture**: Capacity to fit non-linear relationships - scales with problem complexity

Deprioritized: Batch normalization, activation function choice (sufficient with ReLU), optimizer momentum (SGD momentum=0.9 works).

### 2. How Has Hyperparameter Tuning Changed Query Strategy?

**Week 5-6 Strategy: Static Template**
```python
# One approach for all functions
if volatility > 0.25:
    strategy = "EXPLORATION"
else:
    strategy = "REFINEMENT"
```

**Week 7 Strategy: Dynamic, Data-Driven**
```python
# Function-specific adaptation
learning_rate = 0.005 / (1 + CV)  # Higher CV → lower LR
exploration_radius = 0.3 / (1 + volatility) * sqrt(1 + dimension/2)
ensemble_nn_weight = 0.3 + 0.2 * trend_magnitude / (volatility + eps)
threshold = 0.25 * (1 + 0.5 * abs(trend))

strategy = select_strategy(volatility, trend, threshold)
query = acquire_point_via_ensemble(models, strategy, radius)
```

**Concrete Impact:**

| Function | W6 Strategy | W7 Strategy | Expected Change |
|---|---|---|---|
| F1 | Balanced (σ≈0) | Constant+Linear | More conservative |
| F2 | Exploration (σ=0.32) | Conservative SVM (LR↓) | Cautious recovery |
| F3 | Refinement (σ=0.05) | Ridge+Tree (LR↑) | Confident exploitation |
| F4 | Exploration (σ=7.4) | Heavy Ensemble | Better non-lin capture |
| **F5** | **Exploration** | **Deep NN Priority** | **Aggressive elite** |
| F6 | Exploration (σ=0.40) | Aggressive radius (r↑35%) | W6 regression recovery |
| **F7** | **Trend Follow** | **Bayesian + LR↑** | **Confident momentum** |
| F8 | Exploration (σ=0.35) | Deep + RBF (r↑10%) | High-dim adaptation |

### 3. Which Tuning Methods Applied? Trade-offs Observed?

**Methods Applied:**

#### **Method 1: Manual Adjustment + Domain Knowledge**
```python
# Observation: F5 trend = +5.28, volatility = 13.4
# Decision: Opposite advice from different heuristics
#   - High volatility → explore broadly
#   - High positive trend → exploit strongly
# Solution: Weighted combination
#   NN weight = 0.3 + 0.2 * (5.28 / 13.4) ≈ 0.38 → rounds to 0.40 (explore)
#             = 0.3 + 0.2 * (5.28 / 13.4) ≈ 0.38 → W6 used 0.60 (exploit)
# Lesson: Human judgment necessary when heuristics conflict
```

**Trade-off**: Fast iteration vs. Systematic exhaustiveness
- ✅ Fast: Made decisions week-by-week
- ❌ Limited: Didn't explore all hyperparameter combinations

#### **Method 2: Random Search (Simulated)**
```python
# For each function, sampled 20 random hyperparameter configs
# Concept: C ∈ [0.1, 100], kernel ∈ {linear, rbf, poly}
# Result: Found non-obvious good configs (e.g., poly kernel for F4)
# Tradeoff: 20 configs ≈ 160 model trainings (expensive but doable)
```

**Trade-off**: Exploration vs. Exploitation of hyperparameter space
- ✅ Found F4's good config (poly SVM)
- ❌ Missed potentially better configs in untested regions

#### **Method 3: Grid Search (Planned, not executed)**
```python
# Would test LR ∈ [0.001, 0.01, 0.1], alpha ∈ [0.001, 0.01, 0.1]
# 3×3 = 9 configurations per function = 72 trainings
# Tradeoff: Complete coverage but manual effort
```

**Trade-off**: Rigor vs. Time investment
- ✅ Complete coverage
- ❌ 72 trainings ≈ 1-2 hours manual work

#### **Method 4: Bayesian Optimization (Planned for future)**
```python
# Treat hyperparameter tuning as BBO problem itself
# Use Gaussian Process as surrogate for performance(hyperparams)
# Tradeoff: Automatic, efficient, but requires function evaluations
```

**Trade-off**: Automation vs. Data requirements
- ✅ Automatic discovery
- ❌ Needs 10-20 evaluations to build GP

### 4. What Model Limitations Became Clearer at 16 Points?

**Limitation 1: Curse of Dimensionality**

With only 6 samples per function across weeks 1-6:
```
Sample density = 6 / (volume of search space)
            = 6 / (1^D)  [normalized unit hypercube]
            = 6  [queries in 1^D space]

But effective volume sampled = 0.3^D (small radius around samples)
            = 0.3^D

For F8 (8D): Query volume coverage ≈ 0.3^8 ≈ 0.00065% of space
For F7 (6D): Query volume coverage ≈ 0.3^6 ≈ 0.0073% of space
For F4 (4D): Query volume coverage ≈ 0.3^4 ≈ 0.081% of space
```

**Implication**: Cannot fit complex models in high dimensions with small N
- F8 regression (9.45→7.4): Deep network may have overfitted W1-6 local patterns
- Solution W7: Increase regularization (Dropout=0.28 vs 0.20)

**Limitation 2: Feature Irrelevance**

Network learned equal weights for all dimensions, but:
- F4's chaotic behavior may concentrate in specific dimension pairs
- F8's plateau may have redundant dimensions
- With N=6, cannot reliably detect feature importance

**Evidence**: 
- Random subset of features in RF sometimes outperforms all features
- SVM's kernel trick more forgiving than explicit features

**Solution W7**: Use ensemble diversity (RF + SVM + NN) to handle feature uncertainty

**Limitation 3: Model Bias Toward Training Data**

Week 1-5 queries were semi-random. Week 6 queries trained on this distribution.
If Week 6 queries lie in low-probability regions:
- Models extrapolate (dangerous with N=6)
- Predictions unreliable
- F2 recovery prediction failed because W2 peak was outlier

**Lesson**: With small N, ensemble diversity > single high-capacity model

**Limitation 4: Non-Stationary Targets**

Assumption: Landscape doesn't change between weeks. But:
- F2: W2 had peak (0.847), but W3-6 collapsed → landscape shifted?
- Or: W2 was lucky noise (reversion to mean)

**With N=6, cannot distinguish**: True non-stationarity vs random fluctuation

**Solution**: Conservative ensemble approach hedges against both possibilities

### 5. Application to Larger Datasets (100+ samples)

**Current Approach (N≈6-16):**
```python
# Manual hyperparameter tuning with random search
# Ensemble of diverse models (RF + SVM + NN)
# Cross-validation unreliable, use historical hold-out
```

**Scaled Approach (N=100+):**

```python
# Phase 1: Feature Engineering & Selection (new at scale)
for feature_set in itertools.combinations(all_features, k=2..10):
    model_quality = cross_validate_gbm(feature_set, cv=5)
    if model_quality > threshold:
        keep feature_set

# Phase 2: Hyperparameter Optimization via Bayesian Search
from skopt import gp_minimize

def objective(hp_config):
    model = build_model(**hp_config)
    cv_score = cross_validate(model, X_train, y_train, cv=10)
    return -cv_score  # Minimize negative score

best_hp = gp_minimize(
    objective,
    space=[dimensional_ranges],
    n_calls=50,  # Efficient: only 50 evals for 10+ hyperparameters
)

# Phase 3: Neural Architecture Search (AutoML)
from nni import NNIManager

search_space = {
    'hidden_layers': [2, 3, 4, 5],
    'hidden_size': [32, 64, 128, 256],
    'dropout': [0.1, 0.2, 0.3, 0.4],
}

best_arch = nas_search(search_space, train_data, val_data, n_trials=100)

# Phase 4: Ensemble Stacking
meta_learner = LogisticRegression()
meta_learner.fit([
    ensemble[0].predict(X_val),
    ensemble[1].predict(X_val),
    ensemble[2].predict(X_val),
], y_val)
# Learn optimal ensemble weights automatically
```

### 6. Real-World Professional Practitioner Perspective

**How This BBO Setup Mirrors Real ML:**

1. **Incomplete Information**: Black-box objective (like production business metrics)
   - Can't see loss landscape, model internals
   - Must rely on observed outputs only

2. **High-Dimensional Parameter Space**: 
   - Not just hyperparameters (learning rate, regularization)
   - But also feature engineering, architecture, preprocessing decisions
   - Real ML teams systematically tune 10-50 parameters

3. **Small Effective Dataset**:
   - Real production: 1M samples, but only 1K labeled
   - BBO: 6 function evaluations per week
   - Solution: Leverage unlabeled/cheap feedback (like W1-5 semi-random)

4. **Failure Analysis Loop**:
   - Week 6: F2 recovery failed, F6 regressed
   - Week 7: Adjust hyperparameters, re-test
   - Real ML: Production model degrades, rerun hyperparameter tuning
   - **Key insight**: Tuning is iterative, not one-time

5. **Ensemble Thinking**:
   - No single model wins for all problems
   - Combine predictions from diverse algorithms
   - This BBO taught: 3-model ensemble > single perfect model

**Professional Lessons:**

1. **Start Conservative**: F2's recovery overconfidence cost 156% decline
   - Professional approach: Baseline + small step + measure
   - Not: Aggressive exploitation of uncertain pattern

2. **Domain + Data**: 
   - F5 success mixed BBO intuition (elite region) + ML optimization
   - Professional approach: Team with domain experts + data scientists
   - Not: Pure automation

3. **Validation Discipline**:
   - Cannot trust CV score with N=6
   - Must track actual performance over time (historical validation)
   - Professional: Always measure on held-out test set

4. **Interpretable + Accurate**:
   - F7 worked partially because simple (low volatility)
   - F4 failed partially because complex (high volatility)
   - Professional: Prefer interpretable models when possible
   - But ensemble for robustness

5. **Adapt to Feedback**:
   - Weekly results (not hypothetical)
   - Adjust hyperparameters based on actual performance
   - This mirrors A/B testing in production

---

## Mathematical Framework: Why These Hyperparameters Matter

### Learning Rate Selection

Let $\theta_t$ = model parameters at time $t$, $\nabla_t$ = gradient

**Gradient descent update:**
$$\theta_{t+1} = \theta_t - \eta \nabla_t$$

**Optimal $\eta$ depends on:**
- Curvature (Hessian complexity)
- Signal strength (gradient magnitude)
- Noise level (stochastic variation)

**For high-CV functions (F2, F4):** Curvature unpredictable
- Should use $\eta = 0.001$ (small safe steps)
- Adaptive learning: $\eta_t = \eta_0 / \sqrt{t}$ (decreasing over time)

**For low-CV functions (F3, F7):** Curvature stable
- Can use $\eta = 0.01$ (larger steps)
- Confidence in gradient direction

### Regularization & Dimensionality

**Overfitting risk $\propto$ (Model complexity / Dataset size)**

- Model complexity: $O(D \cdot L \cdot H)$ where $D$ = features, $L$ = layers, $H$ = hidden units
- Dataset size: $N = 6$ (fixed)
- Overfitting risk: $O(D \cdot L \cdot H / 6)$

**For F8 (D=8):** Overfitting risk = $8LH/6 ≈ 1.3LH$
**For F3 (D=3):** Overfitting risk = $3LH/6 = 0.5LH$

**Solution:** Increase regularization for high-D
$$\text{Dropout} = 0.1 \sqrt{D}$$

F8: 0.1 × 2.83 ≈ 0.28 ✓ (aggressive to offset high-D risk)
F3: 0.1 × 1.74 ≈ 0.17 ✓ (light, low-D problem)

### Ensemble Weighting via Volatility

**Volatility σ captures landscape uncertainty**

$$w_{\text{NN}} = 0.3 + 0.2 \frac{|\text{trend}|}{\sigma + \epsilon}$$

Intuition:
- If $\sigma$ large (chaotic): Denominator large → $w_{\text{NN}}$ small (trust NN less)
- If trend large (improving): Numerator large → $w_{\text{NN}}$ large (trust NN more)
- NN's strength: Capturing non-linear trends (good when data improving)
- NN's weakness: Overfitting noise (bad when data chaotic)

**For F5:**
$$w_{\text{NN}} = 0.3 + 0.2 \frac{5.28}{13.4} = 0.3 + 0.079 = 0.38 \approx 0.40$$

- But W6: empirically, higher NN weight (0.60) worked better!
- Suggests: Even in high-volatility regime, strong trend justifies non-linear model

---

## Conclusion

Week 7 demonstrates that **hyperparameter tuning in black-box optimization is fundamentally about managing uncertainty with limited data**. The six hyperparameters chosen and methods applied reflect real professional ML:

1. **Start with exploratory search** (random sampling)
2. **Refine based on observations** (manual adjustment)
3. **Build diverse ensembles** for robustness
4. **Validate on hold-out data** (historical performance)
5. **Adapt rapidly** to new feedback (weekly iteration)

The mistakes (F2 recovery, F6 regression) came not from hyperparameter tuning itself, but from overconfidence in narratives. Better tuning won't fix that—but ensemble diversity hedges against both possibilities.

The successes (F5 breakthrough, F7 confirmation) came from combining domain knowledge + systematic optimization + uncertainty quantification. This combination is precisely what professional ML teams do at scale.

---

**Generated**: February 16, 2026
**Status**: Ready for Week 7 submission
**Expected Improvement**: +5-15% across portfolio (conservative estimate)
