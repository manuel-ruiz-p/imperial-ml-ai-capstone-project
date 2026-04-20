# TECHNICAL REPORT — WEEK 10
## Black-Box Function Optimization: Final Submission & Project Retrospective

**Author**: Heber Manuel Ruiz Prado
**Course**: Imperial ML/AI Capstone
**Week**: 10 of 10 — Final Submission
**Date**: March 30, 2026
**Portfolio Status**: W9 = 82.35 (best since W6) | W10 queries submitted

---

## Executive Summary

Week 10 is the final query submission of the capstone project. With 9 observations
per function (W1–W9), the dataset is now large enough to apply PCA for dimensionality
reduction, revealing which input directions genuinely carry exploitable signal versus
which dimensions contribute only noise.

This report documents the W10 query generation strategy, presents the full
10-week retrospective, and connects the trajectory of the project to the course
themes of PCA, clustering, dimensionality reduction, and structured learning from data.

---

## 1. Week 9 Results — The Best Week Since Week 6

| Function | W8 | W9 | Change | Note |
|:---:|---:|---:|---:|:---|
| F1 | ≈0 | ≈0 | — | Noise floor confirmed |
| F2 | 0.033 | **0.481** | +1362% | Strong recovery |
| F3 | −0.138 | **−0.005** | +96% | 🏆 All-time best |
| F4 | −5.556 | **−4.635** | +17% | 🏆 All-time best — 3rd consecutive improvement |
| F5 | 1.149 | **77.553** | +6650% | Near W6 peak (79.327) — GP/EI rediscovered peak region |
| F6 | −1.570 | −0.988 | +37% | Recovered significantly |
| F7 | 0.319 | **0.417** | +31% | 🏆 All-time best |
| F8 | 7.823 | **9.529** | +22% | 🏆 All-time best — surpassed W3 peak of 9.449 |
| **PORTFOLIO** | **4.49** | **82.35** | **+3,899%** | Best single-week gain |

Four all-time bests in one week. The CNN-Inspired + PCA + GP/EI pipeline,
applied to the full W1–W9 history, produced the most broadly successful
result of the entire project.

---

## 2. Week 10 Strategy: PCA-Guided GP/EI

### 2.1 Core Principle

With 9 data points per function, PCA over the top-k historical inputs reveals
the directions in input space that co-vary with high outputs. The principal
components of the winner cluster point toward the most exploitable directions
in the landscape — everything orthogonal to PC1 is predominantly noise.

This mirrors exactly how PCA is applied in supervised dimensionality reduction:
project onto the axes that explain the most variance in the outcome-relevant
subspace, then search there.

### 2.2 PCA Findings

| Function | PC1 Explained Variance | PC1 Direction (dominant dims) | Implication |
|:---:|:---:|:---|:---|
| F4 | 98.8% | Dims 1, 2, 4: [0.50, 0.70, 0.01, −0.51] | Dim 3 nearly irrelevant |
| F5 | 98.3% | Dims 2, 4: [−0.02, −0.69, −0.16, 0.71] | Low dim-2, high dim-4 = peak |
| F8 | 81.0% | Dims 3, 5, 8: [0.06, −0.14, 0.60, 0.20, 0.68, −0.11, −0.13, 0.28] | 5 of 8 dims near-noise |

### 2.3 W10 Queries

```
F1: 0.101613-0.728058
F2: 0.501828-0.474203
F3: 0.483270-0.740795-0.558480
F4: 0.555318-0.550912-0.531991-0.670623
F5: 0.000000-1.000000-0.319539-0.210119
F6: 0.264033-0.460343-0.695113-0.856664-0.528116
F7: 0.228555-0.060712-0.586883-0.448725-1.000000-0.867749
F8: 0.019493-0.553542-0.280020-0.100114-0.630907-0.331912-0.281512-0.000000
```

---

## 3. Full Project Retrospective (W1–W10)

### 3.1 Portfolio Trajectory

| Week | Portfolio | Key Event |
|:---:|---:|:---|
| W1 | ~14.7 | Baseline exploration |
| W2 | ~13.8 | Centroid strategies |
| W3 | ~25.0 | F5 discovery (34.98) |
| W4 | ~29.7 | F5 still climbing |
| W5 | ~17.8 | Mixed results |
| W6 | **69.42** | F5 peak (79.327) — false signal |
| W7 | 5.79 | F5 collapse (−88%) — catastrophic |
| W8 | 4.49 | F4 breakthrough via simplicity |
| W9 | **82.35** | PCA+GP/EI: 4 all-time bests + F5 recovery |
| W10 | TBD | Final submission |

### 3.2 Strategy Evolution

| Phase | Weeks | Approach | Outcome |
|:---|:---:|:---|:---|
| Zero-shot exploration | W1–W3 | LHS, random, centroid | Established baselines, found F5 region |
| Few-shot surrogates | W4–W6 | SVM, Ridge, NN surrogates | F5 peak (W6) but overfit |
| Collapse recovery | W7–W8 | Ensemble consensus, bounded random | F4 breakthrough via simplicity |
| Structure-aware | W9–W10 | PCA + GP/EI on winner clusters | 4 all-time bests, F5 recovery |

### 3.3 All-Time Best Values Achieved

| Function | All-Time Best | Week Achieved |
|:---:|---:|:---:|
| F1 | ≈ 0 (noise) | — |
| F2 | 0.8474 | W2 |
| F3 | **−0.0051** | **W9** |
| F4 | **−4.635** | **W9** |
| F5 | 79.327 | W6 |
| F6 | −0.6996 | W1 |
| F7 | **0.4174** | **W9** |
| F8 | **9.529** | **W9** |

---

## 4. Core Lessons Learned

### Lesson 1: Dimensionality is the Primary Challenge
With N ≤ 9 and D up to 8, we have less than 1 observation per dimension on
average. PCA on the winner cluster makes this tractable: instead of searching
an 8D space, we search along 1–2 principal components that explain >80% of
the variance in successful regions.

### Lesson 2: Simplicity Beats Complexity at N < 20
F4's breakthrough came from bounded uniform sampling (W8), not a 4-model
ensemble. F5's recovery came from a GP trained on 9 points, not a neural
network. The lesson is consistent: model complexity should be proportional
to data availability.

### Lesson 3: Peaks Can Be Rediscovered
The assumption after W7's F5 collapse was that the W6 peak was permanently
inaccessible. W9 disproved this — PCA identified that dim-2 low / dim-4
high was the structural requirement for high F5 values, and GP/EI found the
region again at 77.55. Structural learning enables recovery; pure exploitation
does not.

### Lesson 4: Portfolio Risk Management is Critical
F5's W6→W7 collapse wiped 91.6% of the portfolio. A single function represented
>90% of total portfolio value — classic concentration risk. The correct response
(demonstrated in W8–W9) was to diversify effort across multiple functions via
structural analysis, not to double down on a single peak.

### Lesson 5: The Challenge Tests Decision-Making, Not Convergence
With N < 10 per function, Bayesian optimization cannot converge to the global
optimum. The real skill tested is: given extreme uncertainty and limited budget,
how do you allocate resources, manage risk, and extract the maximum defensible
signal from minimal data? That is closer to real-world ML practice than any
benchmark problem.

---

## 5. Discussion Board Reflection (≤ 350 words)

**Week 10 Reflection — Heber Manuel Ruiz Prado**

**How past patterns shaped latest choices:**
By Week 10, 9 observations per function are enough to apply PCA over the
top-performing inputs. Rather than querying uniformly across all dimensions,
W10 queries move along the principal components of each function's winner
cluster. For F4, 98.8% of variance in top results lies along
[0.50, 0.70, 0.01, −0.51] — meaning dimension 3 contributes almost nothing.
For F8, dims 3, 5, and 8 carry 81% of variance. Every W10 query targets
movement along these axes, not the full D-dimensional space.

**Clusters and recurring regions identified:**
Yes — PCA makes them explicit and quantifiable. The F5 peak cluster requires
low dim-2 and high dim-4; W6 (79.327) and W9 (77.553) both satisfy this, while
W7 and W8 queries violated it and collapsed. The F8 high-value cluster (9.3–9.5)
consistently occupies a narrow band in dims 3, 5, and 8. These are not
assumptions — they are empirically confirmed across 9 weeks.

**Less effective strategies and adjustments:**
Aggressive exploitation with complex ensembles (W6–W7) was the clearest failure.
The surrogate models overfitted the noise when N << D, generating confident but
wrong predictions. The adjustment was progressive: W8 moved to bounded random
(simplicity), W9 added GP/EI as a selection filter, W10 adds PCA as a
dimensionality pre-filter. Each step added structure without adding complexity.

**Parallel to PCA / clustering:**
PCA separates meaningful variance from noise by projecting onto high-eigenvalue
components. My W10 query generation does the same: project the candidate
search space onto PC1 of the winner cluster, discard the rest. K-means would
group inputs by output similarity; PCA reveals what makes that grouping
geometrically coherent. Both tools answer the same question: where is the signal?

**Plotting trends:**
Projected onto PC1 vs output value, each function's scatter would show a
roughly monotonic band — most of the optimizable signal lives in one or two
directions. That is the final, generalizable lesson of this project: in
high-dimensional black-box optimization with N ≤ 21, identifying and querying
along the principal directions of successful observations is the most reliable
path forward.

---

*End of Week 10 Technical Report*
*Project: Imperial ML/AI Capstone — Black-Box Function Optimization*
*Author: Heber Manuel Ruiz Prado*
