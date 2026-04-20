# Week 8 Reflection: Critical Analysis of Black-Box Optimization Strategy
## Discussion Board Post (Under 300 Words)

### Strategy Evolution: From Zero-Shot to Ensemble Reasoning

My approach progressed from **zero-shot exploration** (Weeks 1-3: random sampling, no priors) to **few-shot learning** (Weeks 4-6: surrogate models on 3-5 samples) to **ensemble reasoning** (Weeks 7-8: all historical data with multiple model families). Week 6's complex 6-model ensemble paradoxically caused F5's catastrophic collapse (-88%), while Week 8's simple bounded random sampling achieved F4's breakthrough (+69%). **Lesson**: Simplicity beats sophistication when N<20.

### Hyperparameter Choices: Balancing Coherence vs. Diversity

**Exploration radius** (analogous to temperature) evolved: High σ=0.3-0.5 (W1-3) established baselines, low σ=0.05 (W6) caused over-exploitation and F5's collapse, moderate σ=0.15-0.2 (W8) balanced discovery and refinement. **Candidate filtering** (top-p analog) improved from top-50 to top-20 (p=0.90), enhancing quality while preserving diversity. **Computational budget** (max-tokens analog) scaled from 50 to 200 candidates, yielding diminishing returns beyond 150.

### Data Limitations with N=17

With 8 observations per function, **prompt overfitting** dominated: F5's perfect surrogate fit (R²=0.999) hallucinated structure, predicting ~75 but observing 9.25. **Attention fixated** on F2's W7 recovery spike (+574%), ignoring underlying 78% volatility, causing W8 crash (-77%). **Diminishing returns** emerged: N=7→8 added minimal information (0.4 avg improvement), but strategy shift added 12.3 absolute gain.

### Hallucination Mitigation

I implemented: (1) **Tight bounds** [0.35, 0.65] reducing search volume 99% (2) **Retrieval conditioning** on top-3 historical queries, reducing uncertainty 60% (3) **Output constraints** preventing degenerate cases (4) **Ensemble agreement thresholds** requiring model consensus.

### Scaling & Practitioner Mindset

For N>100: transition from Gaussian Processes (N=20-50) → deep ensembles (N=50-100) → gradient-based methods (N>100). This challenge taught **risk management over optimization**: F5's collapse wiped 98% of portfolio gains. The critical skill: knowing when to abandon hopeless functions (6 abandoned in W9) and allocate scarce resources to high-confidence opportunities (F4, F8 selected).

**Core insight**: With N<20, Bayesian optimization tests decision-making under uncertainty, not convergence skill.

---

*(299 words)*
