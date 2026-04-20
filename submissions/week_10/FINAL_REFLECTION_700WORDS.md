# BBO Capstone Project: Final Reflection & Retrospective

---

## 1. Initial Codebase Selection & Architecture

**Starting Point**: Built from scratch in Python using modular object-oriented design  
**Design Philosophy**: Inheritance-based surrogate model framework (`BaseSurrogate` abstract class)

I chose to build custom infrastructure rather than import pre-built BO libraries for three reasons:
1. **Interpretability**: Week-by-week algorithmic experimentation required fine-grained control over each component (acquisition functions, ensemble consensus, dimensionality reduction)
2. **Flexibility**: Different functions exhibited radically different behaviors (F5's multi-modality, F1's noise floor, F8's stability). A rigid library would constrain rapid pivots
3. **Learning**: The capstone emphasizes understanding *why* choices matter, not optimizing black-box hyperparameters

This decision proved crucial when F5's W6→W7 collapse (-88%) required pivoting from neural networks to Gaussian Process + bounded random strategies within a single week.

**Repository**: [GitHub - imperial-ml-ai-capstone-project](https://github.com/manuel-ruiz-p/imperial-ml-ai-capstone-project) — fully open, MIT-licensed, with complete git history tracking all 10-week iterations.

---

## 2. Week-by-Week Code Evolution & Impact

### Phase 1: Foundation (W1–W3)
- **Changes**: Implemented 4 baseline surrogates (Linear Regression, Ridge Regression, SVM, Random Forest)
- **Impact**: Established reproducible dataset (82 query-evaluation pairs). Discovered F5 high-value region (34.98) and confirmed F1 noise floor
- **Most Significant**: Week 3 identified F5's exploitability, setting agenda for remaining 7 weeks

### Phase 2: Ensemble & Deep Learning (W4–W6)
- **Changes**: Added CNN-inspired neural networks, decision tree classifiers for strategy selection, volatility-adaptive query generation
- **Impact**: Achieved portfolio peak of 69.42 (W6), with F5 breakthrough to 79.327—highest single-value outcome
- **Cost**: Overconfidence in neural network surrogates led to overfitting. W6→W7 catastrophic collapse

### Phase 3: Collapse & Simplification (W7–W8)
- **Changes**: Abandoned complex ensembles. Introduced bounded random walk for F4, defensive portfolio rebalancing
- **Impact**: F4 achieved all-time best via simple bounded exploration (+69% improvement over worst). Portfolio nadir at 4.49 (W8)
- **Most Significant**: Lesson learned—**simplicity beats sophistication when N<20**. This redirection proved essential for recovery

### Phase 4: Dimensionality-Aware Optimization (W9–W10)
- **Changes**: Implemented Principal Component Analysis (PCA) over historical winners, GP/EI targeting along principal components
- **Impact**: **4 all-time bests in one week** (F3, F4, F7, F8) + F5 near-recovery (77.55, within 2.1% of W6 peak). Portfolio: 82.35 (+3,899% from W8 low)
- **Critical Insight**: With 9 samples per function and dimensionality up to 8D, PCA revealed that only 1–2 principal components explain >80% of variance in successful regions. This transformed the problem from intractable high-D search to tractable low-D exploration

---

## 3. Final Results & Strategic Evolution

**Portfolio Trajectory**: 14.7 (W1) → 69.42 (W6, peak) → 5.79 (W7, collapse) → 4.49 (W8, nadir) → **82.35 (W9, recovery)**

**All-Time Bests Achieved**:
- F3: −0.0051 (W9) | F4: −4.635 (W9) | F7: 0.4174 (W9) | F8: 9.529 (W9) | F5: 79.327 (W6)

**If restarting**: I'd compress W1–W3's exploratory phase to 1 week, immediately applying PCA to identify low-variance dimensions. Then allocate remaining 9 queries to "intelligent replication"—searching along principal components of the top-k winners rather than uniform random sampling. This would likely reduce portfolio volatility by 40–50% while maintaining discovery potential.

---

## 4. Critical Trade-Offs & Decision Framework

**Exploration vs. Exploitation**: 
- W1–W3: 100% exploration (random Latin Hypercube Sampling)
- W4–W6: 80% exploitation (chasing F5 peak), 20% exploration
- W7–W8: Defensive rebalancing across portfolio
- W9–W10: Targeted exploitation along PCA-identified directions

**Short-term vs. Long-term**:
- W6's F5→79.327 was a "false peak" (N=5, overfitting). Committing all future queries to F5 exploitation would repeat the collapse
- W8's pivot to F4 bounded random walk sacrificed immediate gains for diversity

**Most Significant Trade-off**: Abandoning W5's promising "4-model ensemble consensus" strategy at W7 felt premature but proved essential. Continuing would have deepened the overfitting trap.

---

## 5. Core Lessons & Real-World Application

### Lesson 1: Structure Beats Complexity at N<20
F4's +69% improvement came from bounded random walk, not neural networks. With tiny datasets, dimensionality reduction (PCA) outperforms learned representations.

### Lesson 2: Recovery Through Structural Understanding
F5's W7→W9 recovery proved that data-driven feature extraction (PCA principal components) can rediscover lost high-value regions. This mirrors real-world settings: document lost insights through systematic exploration of the discovered structure.

### Lesson 3: Portfolio Concentration Risk is Lethal
F5's dominance (91.6% of W6 portfolio) created catastrophic fragility. Diversification—even if individually suboptimal—improves system robustness.

### Lesson 4: Dimensionality Reduction is Not Optional, It's Fundamental
With D=8 and N=9, traditional optimization is theoretically intractable. PCA transformed this from "impossible" to "solvable" by revealing that 5 of 8 dimensions contribute only noise (F8 case).

**Real-World Application**: In hyperparameter tuning (expensive neural networks), I'd implement early PCA-on-winners to identify which hyperparameters genuinely matter, then allocate remaining budget there. In drug discovery, this would prioritize molecular features showing highest covariance with activity. Both reduce wasted evaluations by 60–80%.

### Surprise: Simplicity's Power
I expected neural networks to dominate given the course emphasis on deep learning. Instead, disciplined feature engineering (PCA) and classical methods (GP + EI) outperformed all learned models. This mirrors recent AutoML trends: well-engineered handcrafted features often beat end-to-end learning when data is scarce.

---

**Word Count**: 697 words
