# Model Card: BBO Capstone Optimization Approach

**Model Name**: Adaptive Ensemble Bayesian Optimizer (AEBO)  
**Version**: 1.0  
**Date**: March 16, 2026  
**Author**: Imperial ML/AI Capstone Student  
**Framework**: Adapted from Mitchell et al. (2019) "Model Cards for Model Reporting"

---

## Model Overview

### Model Type

**Category**: Bayesian Optimization with Surrogate Models  
**Paradigm**: Sequential Decision-Making Under Uncertainty  
**Architecture**: Ensemble of regression models (Gradient Boosting, Random Forest, SVM RBF, Gaussian Process) combined with acquisition functions (Expected Improvement, Upper Confidence Bound)

### Model Description

AEBO is a meta-optimization strategy that adaptively selects query points for 8 unknown black-box functions under strict budget constraints (10 queries per function). The approach evolved over 9 weeks from random exploration to function-specific Bayesian optimization, incorporating volatility-adaptive strategies, defensive portfolio management, and strategic resource allocation.

**Core Innovation**: Function-specific strategy selection based on observed volatility patterns and performance trajectories, with explicit abandonment of low-confidence functions to concentrate resources on high-potential targets.

### Version History

- **v0.1 (Week 1)**: Random baseline sampling
- **v0.2 (Week 2)**: Boundary vs. interior exploration
- **v0.3 (Weeks 3-5)**: Bayesian optimization with linear surrogates
- **v0.4 (Week 6)**: Ensemble surrogates + exploitation focus
- **v0.5 (Week 7)**: Post-collapse hyperparameter tuning
- **v0.6 (Week 8)**: Simplicity-first bounded random strategies
- **v1.0 (Week 9)**: Strategic allocation + function abandonment

---

## Intended Use

### Primary Use Cases

1. **Black-box function optimization** with expensive evaluations (N<20)
2. **Sequential decision-making** under radical uncertainty
3. **Portfolio management** for multi-objective optimization problems
4. **Exploration-exploitation trade-offs** in data-scarce regimes
5. **Adaptive strategy selection** based on real-time performance feedback

### Target Domains

- Hyperparameter tuning for computationally expensive ML models
- Drug discovery with limited lab budget
- Robotics parameter tuning with real-world experiments
- Hardware design optimization (CAD/CFD simulations)
- A/B testing with constrained user traffic

### User Groups

- **Researchers**: Benchmarking Bayesian optimization approaches
- **Practitioners**: Applying sequential decision-making to real-world problems
- **Students**: Learning exploration-exploitation trade-offs
- **Developers**: Implementing adaptive optimization frameworks

---

## Out-of-Scope Use Cases

### Inappropriate Applications

❌ **High-stakes safety-critical systems**: Insufficient validation for medical, aviation, or autonomous vehicle applications  
❌ **Large-scale optimization**: Designed for N<20 regime; inefficient for big data scenarios  
❌ **Gradient-based optimization**: Assumes no derivative information; use gradient descent when applicable  
❌ **Stationary functions**: Over-engineered for simple convex landscapes  
❌ **Real-time applications**: Weekly cadence inappropriate for low-latency requirements  
❌ **Deterministic guarantees**: Cannot guarantee global optimum in black-box settings  

### Known Limitations

- **Non-stationarity**: Strategy assumes quasi-stationary functions; performs poorly on F2, F3, F7 with temporal drift
- **Sample efficiency**: Requires 3-5 queries minimum for surrogate training; ineffective for single-shot optimization
- **Dimensionality**: Tested up to 8D; performance uncertain beyond 10D
- **Computational cost**: Ensemble training requires 10-60 seconds per query generation (acceptable for expensive evaluations, impractical for cheap functions)

---

## Model Details

### Technical Specifications

**Input**: Function ID (1-8), historical query-evaluation pairs, budget remaining  
**Output**: Single query point in [0,1]^d (d = function dimensionality)

**Core Components**:

1. **Surrogate Models** (ensemble):
   - Gradient Boosting Regressor (n_estimators=100, max_depth=3)
   - Random Forest Regressor (n_estimators=100, max_depth=5)
   - Support Vector Regressor (kernel='rbf', C=1.0, γ='scale')
   - Gaussian Process Regressor (kernel=Matérn ν=2.5)

2. **Acquisition Functions**:
   - Expected Improvement (EI): ξ = 0.01
   - Upper Confidence Bound (UCB): β ∈ [0.5, 5.0] (function-adaptive)

3. **Search Methods**:
   - Latin Hypercube Sampling (100-500 candidates)
   - Random perturbation with Gaussian noise (σ ∈ [0.05, 0.2])
   - Bounded exploration with safety margins

### Training Data

- **Pre-collected samples**: 175 (provided by course)
- **Weekly submissions**: 82 query-evaluation pairs (self-generated)
- **Total training data**: Up to 257 samples (varies by function)

**Data split**: No train/test split used (all historical data used for surrogate training in sequential setting)

### Hyperparameters by Function Group

| Group | Functions | Strategy | β (UCB) | Radius σ | Candidates |
|:---:|:---:|:---|:---:|:---:|:---:|
| **Winners** | F5, F7 | Exploitation | 0.5 | 0.05 | 100 |
| **Improving** | F6, F8 | Balanced | 1.5 | 0.15 | 200 |
| **Declining** | F2, F3, F4 | Exploration | 3.0 | 0.20 | 500 |
| **Sparse** | F1 | Random | 5.0 | 0.30 | 1000 |

### Computational Requirements

- **Hardware**: Standard laptop (M1 MacBook, 16GB RAM)
- **Runtime**: 10-60 seconds per query generation
- **Dependencies**: scikit-learn 1.3+, NumPy 1.24+, SciPy 1.11+
- **Memory**: <500MB peak usage

---

## Strategy Evolution (Weeks 1-9)

### Week 1: Random Baseline
**Approach**: Uniform random sampling in [0,1]^d  
**Rationale**: Establish baseline, explore function variability  
**Outcome**: Extreme variability observed (CV: 23%-450%), one-size-fits-all fails  
**Key Learning**: Functions require heterogeneous strategies

### Week 2: Boundary Exploration
**Approach**: Targeted boundary vs. interior sampling  
**Rationale**: Differential diagnosis of function preferences  
**Outcome**: F5 prefers interior (4.049→34.98, +764%), F2 prefers boundaries  
**Key Learning**: Spatial preferences vary by function

### Week 3: Bayesian Optimization (EI)
**Approach**: Expected Improvement with linear surrogate models  
**Rationale**: Leverage 175 pre-collected samples  
**Outcome**: Mixed (F5 +764%, F7 +56%, but F2-F4 decline)  
**Key Learning**: Linear surrogates insufficient for non-linear functions

### Week 4: Function-Specific Strategies
**Approach**: Group functions by performance trajectory, assign β parameters  
**Rationale**: Winners exploit, declining functions explore  
**Outcome**: Moderated declines, sustained F5/F7 peaks  
**Key Learning**: Adaptive hyperparameters improve robustness

### Week 5: Ensemble Surrogates
**Approach**: Combine GB, RF, SVM, GP predictions (weighted average)  
**Rationale**: Reduce overfitting, capture diverse patterns  
**Outcome**: F5 sustained at 25.58 (−26% from W4 but still strong)  
**Key Learning**: Ensemble reduces volatility but doesn't prevent decline

### Week 6: Exploitation Focus
**Approach**: Aggressive β reduction (0.5 for winners), small exploration radius (σ=0.05)  
**Rationale**: Refine near discovered peaks  
**Outcome**: **F5 BREAKTHROUGH 79.327 (+210%)**, F7 confirmed 0.3705 (+62%)  
**Key Learning**: Deep exploitation works in stable regions, but...

### Week 7: Post-Collapse Analysis
**Approach**: Hyperparameter tuning after F5 collapse (79.327 → 9.247, −88%)  
**Rationale**: Diagnose peak loss, adjust overfitting mitigation  
**Outcome**: **Portfolio crash (69.42 → 5.79, −91.6%)**, F2 recovery +574%, F3/F7 decline  
**Key Learning**: Multi-modal landscapes require extensive sampling, not aggressive exploitation. Peak was LOCAL optimum in deceptive function.

### Week 8: Simplicity-First Recovery
**Approach**: Bounded random walk (F4), conservative mean reversion (F8), recovery attempts (F2, F5)  
**Rationale**: Abandon complex ensembles for chaotic functions, use simplicity over sophistication  
**Outcome**: **F4 BREAKTHROUGH +69%** (−17.894 → −5.556), F2 crash −77%, F5 continued collapse −87%  
**Key Learning**: **"Simplicity beats sophistication when N<20."** F4's bounded random (no ensemble) outperformed F5's complex models.

### Week 9: Strategic Allocation
**Approach**: Allocate final 2 queries to F4 (breakthrough) and F8 (stability), abandon 6 functions  
**Rationale**: Concentrate resources on high-confidence targets under budget constraint (10 total queries)  
**Outcome**: Pending results (expected portfolio ≈4.0)  
**Key Learning**: **Knowing when to quit as valuable as knowing where to search.** Defensive portfolio management essential in black-box environments.

### Week 10: Final Standings (No New Queries)
**Approach**: Observe final rankings, measure strategic success  
**Rationale**: Budget exhausted, evaluate 9-week performance  
**Outcome**: To be determined (March 20, 2026)  
**Retrospective**: Complete final report, document lessons learned

---

## Performance Metrics

### Primary Metric: Portfolio Sum

**Definition**: Sum of absolute best values across all 8 functions at each week

| Week | Portfolio | Change | Key Event |
|:---:|---:|---:|:---|
| W6 | 69.42 | — | F5 peak (79.327) |
| W7 | 5.79 | −91.6% | F5 collapse (9.247) |
| W8 | 4.49 | −22% | F4 breakthrough (−5.556, +69%) |
| W9 | TBD | TBD | Strategic allocation (2 queries) |

### Per-Function Performance

**Success Stories**:
- **F4**: Breakthrough trajectory (−17.894 → −5.556, +69% in W8)
- **F8**: Most stable (8% volatility, mean ≈8.6)
- **F2**: Recovery resilience (+574% W6→W7 despite volatility)

**Failures**:
- **F5**: Catastrophic collapse (79.327 → 1.149, −98.6% from peak)
- **F2**: Chaotic attractor (±77% weekly swings)
- **F3, F7**: Non-stationary decline (−31%, −8% in W8)
- **F1**: Noise floor (~10⁻¹⁰⁰, no exploitable structure)

### Secondary Metrics

1. **Improvement Rate**: +69% (F4, Week 8, best single-week gain)
2. **Stability**: 8% volatility (F8, most reliable)
3. **Risk-Adjusted Return**: F8 mean 8.6 with 8% CV vs. F2 mean 0.27 with 300% CV
4. **Query Efficiency**: 9-10 queries per function (met budget constraint)

---

## Assumptions and Limitations

### Core Assumptions

1. **Quasi-stationarity**: Functions assumed relatively stable over 9-week period
   - **Violated by**: F2, F3, F5, F7 (temporal drift observed)
   - **Impact**: Surrogate predictions inaccurate for non-stationary functions

2. **Smoothness**: Functions assumed locally Lipschitz-continuous
   - **Violated by**: F1 (noise-dominated), F2 (chaotic)
   - **Impact**: Gradient-based acquisition functions unreliable

3. **Sample sufficiency**: N=8-10 assumed adequate for surrogate training
   - **Challenged by**: 8D function (F8) requires >>10 samples for full coverage
   - **Impact**: High-dimensional functions underexplored

4. **Budget adequacy**: 10 queries assumed sufficient to discover global optimum
   - **Result**: Insufficient for all functions; forced prioritization
   - **Impact**: Strategic abandonment necessary (6 functions)

5. **Evaluation determinism**: Functions assumed noise-free
   - **Uncertain for**: F1 (near machine epsilon)
   - **Impact**: Cannot distinguish signal from numerical error

### Known Limitations

**Data Scarcity (N<20)**:
- Dimensionality curse: 8D space with 9 samples = 0.000001% coverage
- Overfitting risk: Complex models memorize noise, fail to generalize
- Surrogate unreliability: Prediction intervals wider than value ranges

**Non-Stationarity**:
- Temporal drift: F3, F7 decline regardless of strategy
- Sudden shifts: F5 collapse not predicted by historical trend
- Adaptation lag: Strategy updates occur weekly, not dynamically

**Black-Box Constraint**:
- No ground truth: Cannot validate surrogate accuracy
- No derivatives: Gradient-free methods only (slower convergence)
- No structure information: Cannot exploit symmetry, separability, convexity

**Computational Constraints**:
- Sequential requirement: Cannot parallelize queries (1 per week per function)
- Real-time infeasible: 10-60 second generation time per query
- Memory limits: High-dimensional Latin Hypercube Sampling (LHS) with 10,000+ candidates infeasible

### Failure Modes

1. **Premature convergence**: Aggressive exploitation (β=0.5) on multi-modal functions (F5 collapse)
2. **Exploration exhaustion**: Budget depletion before sufficient coverage (F8 8D underexplored)
3. **Surrogate misleading**: Model predicts improvement, evaluation yields decline (F2 volatility)
4. **Non-stationarity blindness**: Strategy assumes stability, function drifts (F3, F7)
5. **Noise amplification**: High β in noisy regions leads to random walk (F1)

---

## Ethical Considerations

### Transparency and Reproducibility

**Full disclosure**:
- All queries documented in `submissions/week_XX/queries.py`
- Strategies explained in weekly reflection documents
- Hyperparameters reported in `MODEL_CARD.md` (this document)
- Raw data available in GitHub repository

**Reproducibility guarantees**:
- NumPy random seeds documented
- Library versions specified in `requirements.txt`
- Preprocessing steps detailed in `DATASHEET.md`
- Code available under MIT license

**Why this matters**:
- Enables peer review and validation
- Supports future research building on this work
- Demonstrates academic integrity
- Facilitates real-world adaptation

### Real-World Adaptation Considerations

**When deploying AEBO-like approaches in practice**:

1. **Safety validation**: Ensure queries cannot cause harm (e.g., drug toxicity, robot damage)
2. **Stakeholder consent**: Disclose optimization strategy to affected parties
3. **Bias auditing**: Check for systematic exclusion of regions (e.g., boundary avoidance)
4. **Failure transparency**: Document when optimization converges to poor solutions
5. **Resource equity**: Consider computational cost disparity (ensemble models vs. random sampling)

### Limitations of Transparency

**This model card cannot guarantee**:
- Optimal performance on unseen functions
- Generalization to different budget constraints
- Robustness to adversarial black-box functions
- Fairness across diverse application domains

**Users must**:
- Validate on their specific problem
- Conduct domain-specific risk assessments
- Monitor for unexpected failure modes
- Update strategy based on real-world feedback

---

## Evaluation and Validation

### Evaluation Protocol

**No held-out test set**: Black-box constraint prevents traditional train/test split. Validation occurs through:
1. **Sequential leave-one-out**: Each week uses historical data, evaluates on current week
2. **Surrogate diagnostics**: R² scores, prediction intervals, residual analysis
3. **Portfolio tracking**: Week-over-week performance monitoring
4. **Strategy ablation**: Comparing random, boundary, EI, UCB approaches

### Performance by Evaluation Metric

| Metric | Value | Interpretation |
|:---|---:|:---|
| **Portfolio W9 (projected)** | 4.0 | Conservative estimate |
| **Best single function** | 79.327 (F5 W6) | Discovered elite region, but unsustainable |
| **Most improved** | +69% (F4 W8) | Validated breakthrough trajectory |
| **Most stable** | 8% CV (F8) | Reliable baseline |
| **Query efficiency** | 82/80 (102.5%) | Slightly over budget (F4, F8 got 9th query) |

### Model Uncertainty

**Quantified via**:
- Gaussian Process prediction intervals (σ bounds)
- Ensemble disagreement (standard deviation across 4 models)
- Bootstrap resampling (1000 iterations)

**Example (F4 Week 8)**:
- Ensemble prediction: μ = −6.70, σ = 3.2
- Actual evaluation: −5.556 (within 1σ)
- EI score: 6.59 (high uncertainty, strong acquisition)

---

## Recommendations for Users

### Best Practices

1. **Start simple**: Use random sampling for first 3-5 queries to establish baseline
2. **Function-specific strategies**: Avoid one-size-fits-all hyperparameters
3. **Monitor volatility**: Adapt β based on observed variance (high variance → high β)
4. **Ensemble over single models**: Combine GB, RF, SVM, GP to reduce overfitting
5. **Defensive allocation**: Reserve budget for high-confidence targets, accept losses elsewhere

### When to Use AEBO

✅ **Expensive evaluations**: Cost or time per query >> cost of surrogate training  
✅ **Budget constraints**: N<50 evaluations available  
✅ **Unknown structure**: No gradient or closed-form available  
✅ **Sequential setting**: Can wait for results before next query  
✅ **Multiple functions**: Need portfolio optimization across heterogeneous objectives  

### When NOT to Use AEBO

❌ **Cheap evaluations**: Random search faster when N>1000 feasible  
❌ **Gradient available**: Gradient descent converges faster for smooth functions  
❌ **Convex problems**: Specialized solvers guarantee global optimum  
❌ **Real-time requirements**: 10-60 second query generation too slow  
❌ **High dimensions**: Exponentially inefficient beyond d>10  

---

## Future Improvements

### Planned Enhancements

1. **Dynamic hyperparameters**: Auto-tune β based on historical volatility
2. **Multi-fidelity optimization**: Use cheap approximations before expensive evaluations
3. **Transfer learning**: Leverage patterns across related functions
4. **Adaptive ensemble weighting**: Prioritize models with better track record
5. **Non-stationary detection**: Trigger strategy shift when drift detected

### Research Directions

- **Contextual bandits**: Frame as online learning problem with function contexts
- **Meta-learning**: Learn optimization strategy from multiple BBO problems
- **Safe Bayesian optimization**: Add safety constraints to acquisition functions
- **Batch optimization**: Generate multiple queries simultaneously (if parallelizable)

---

## Model Card Maintenance

### Version Control

- **v1.0 (March 16, 2026)**: Initial release after Week 9 submission
- **v1.1 (March 21, 2026)**: Final update with Week 10 standings
- **v2.0 (Future)**: If used in subsequent research or courses

### Contact Information

- **GitHub Issues**: Preferred for technical questions
- **Repository**: https://github.com/[username]/imperial-ml-ai-capstone-project
- **Course Discussion**: Imperial ML/AI Capstone board (until March 2026)

### Citation

If using this model or approach, please cite:

```
@misc{bbo_capstone_2026,
  author = {Imperial ML/AI Capstone Student},
  title = {Adaptive Ensemble Bayesian Optimizer (AEBO)},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/[username]/imperial-ml-ai-capstone-project}},
  note = {Model Card v1.0}
}
```

---

## Model Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AEBO: Query Generation Pipeline             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Historical Data      │
                    │  - Pre-collected: 175 │
                    │  - Weekly: 8-9 per fn │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Surrogate Training   │
                    │  - Gradient Boosting  │
                    │  - Random Forest      │
                    │  - SVM RBF            │
                    │  - Gaussian Process   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Ensemble Prediction  │
                    │  - Weighted average   │
                    │  - Uncertainty (σ)    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Acquisition Function │
                    │  - EI (exploitation)  │
                    │  - UCB (exploration)  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Candidate Generation │
                    │  - LHS (100-500)      │
                    │  - Random perturbation│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Acquisition Maximizer│
                    │  - Select argmax(EI)  │
                    │  - Apply bounds [0,1] │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Query Point (x*)     │
                    │  - Dimensionality: d  │
                    │  - Format: NumPy array│
                    └───────────────────────┘
```

---

**Document Status**: Complete  
**Last Updated**: March 16, 2026  
**Next Review**: Upon Week 10 final standings (March 20, 2026)  
**Compliance**: Adapted from Mitchell et al. (2019) Model Cards framework
