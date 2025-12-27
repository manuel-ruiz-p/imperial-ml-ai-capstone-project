# Bayesian Black-Box Optimization Capstone Project

## Section 1: Project Overview

### Purpose and Goal
This capstone project addresses the **Bayesian Black-Box Optimization (BBO)** challenge: maximizing eight unknown, high-dimensional functions under strict evaluation constraints. Rather than seeking perfect optima, the project demonstrates sound optimization strategy, adaptive reasoning, and principled decision-making under uncertainty—core competencies in modern machine learning and operations research.

**Overall Goal**: Submit one strategically chosen query point per function per week for 8 weeks, maximizing cumulative performance across functions with increasing dimensionality (2D → 8D) and complexity.

### Real-World Relevance
Black-box optimization is ubiquitous in practice:
- **Hyperparameter tuning** for neural networks (expensive to evaluate)
- **Drug discovery** and molecular design (limited laboratory budget)
- **A/B testing** and online experimentation (finite user traffic)
- **Robotics control** and reinforcement learning (costly real-world deployments)
- **Hardware design** and expensive simulations (CAD/CFD optimization)

### Career Impact
This project directly strengthens:
- **Design under uncertainty**: Making principled decisions with incomplete information
- **Adaptive problem-solving**: Tailoring strategies to heterogeneous optimization landscapes
- **Empirical reasoning**: Interpreting signals from noisy, sparse data
- **Communication**: Documenting strategy evolution transparently for reproducibility and peer review

---

## Section 2: Inputs and Outputs

### Input Format
**Query Point**: One vector per function per submission
- **Dimensionality**: 2D–8D (varies by function)
- **Domain**: $\mathbf{x} \in [0, 1]^d$ (normalized unit hypercube)
- **Constraints**: Continuous, deterministic input; functions may exhibit noise or multi-modality

**Example (Week 1 Submission)**:
```python
second_submission = [
    np.array([0.05, 0.05]),                                    # F1 (2D)
    np.array([0.5, 0.5]),                                      # F2 (2D)
    np.array([0.35, 0.65, 0.5]),                               # F3 (3D)
    np.array([0.8, 0.2, 0.6, 0.4]),                            # F4 (4D)
    np.array([0.72, 0.28, 0.58, 0.22]),                        # F5 (4D)
    np.array([0.8, 0.6, 0.4, 0.2, 0.5]),                       # F6 (5D)
    np.array([0.25, 0.40, 0.50, 0.70, 0.85, 0.50]),           # F7 (6D)
    np.array([0.15, 0.30, 0.40, 0.48, 0.60, 0.70, 0.85, 0.45]) # F8 (8D)
]
```

### Output Format
**Response**: Scalar function value $f(\mathbf{x}) \in \mathbb{R}$ per query
- **Noise characteristics**: Unknown; may be clean or noisy
- **Scale**: Highly variable across functions (from 2.6e-96 to 8.69)
- **Latency**: One evaluation per function per week (strict constraint)

---

## Section 3: Challenge Objectives

### Optimization Direction
**Maximize** all eight functions: goal is to find $\mathbf{x}^* = \arg\max_{\mathbf{x} \in [0,1]^d} f(\mathbf{x})$

### Critical Constraints
1. **Limited Budget**: 1 query per function per week (8 weeks total = 8 evaluations per function max)
2. **Unknown Structure**: No gradient, analytical form, or derivative information
3. **Function Heterogeneity**: Each function has distinct characteristics:
   - **F1–F2** (2D): Low-dimensional, foundational
   - **F3–F5** (3D–4D): Mid-dimensional; F3 is transformed minimization (inverse/negation)
   - **F6–F8** (5D–8D): High-dimensional curse; F7 is hyperparameter tuning context
4. **Output Variability**: Outputs range from sparse (F1: 2.6e-96) to moderate (F8: 8.69)

### Success Metrics
- **Primary**: Sound strategy and clear reasoning (not perfection)
- **Secondary**: Improvement trajectory across weeks
- **Tertiary**: Adaptive refinement based on observed data

---

## Section 4: Technical Approach

### Week 1–2 Strategy: Differential Diagnosis

Rather than applying a uniform algorithm, we adopt a **function-specific** strategy informed by initial outputs:

#### Data-Driven Classification
| Function | Output | Dimensionality | Challenge | Strategy | Exploration % |
|----------|--------|-----------------|-----------|----------|---------------|
| F1 | 2.6e-96 | 2D | Extreme sparsity | Exploration | 95% |
| F2 | 0.369 | 2D | Multi-modal | Hybrid | 40% |
| F3 | -0.0103 | 3D | Near-zero (minimization) | Exploitation | 20% |
| F4 | -13.07 | 4D | Large negative | Exploration | 80% |
| F5 | 5.27 | 4D | High positive (unimodal) | Exploitation | 25% |
| F6 | -0.70 | 5D | High-dim negative | Exploration-Hybrid | 65% |
| F7 | 0.120 | 6D | Hyperparameter tuning | Balanced Hybrid | 50% |
| F8 | 8.69 | 8D | High output, complex | Exploitation-Hybrid | 45% |

### Reasoning and Heuristics

**Exploration (F1, F4, F6):**
- **F1**: Sparse output indicates localized peaks or Gaussian sources. Sample boundary corners (e.g., [0.05, 0.05]) to detect non-zero regions.
- **F4**: Large negative output suggests poor initialization. Invert Week 1 point systematically to escape local minima.
- **F6**: High-dimensional negative output implies curse of dimensionality. Mirror Week 1 to test opposite region and reduce epistemic uncertainty.

**Exploitation (F3, F5):**
- **F3**: Output already near zero (target). Small perturbations (±0.02) around existing point optimize locally.
- **F5**: Strong positive output (5.27) suggests unimodal or well-conditioned landscape. Tight perturbations maximize signal/noise ratio.

**Balanced Hybrid (F2, F7, F8):**
- **F2**: Moderate output and suspected multi-modality warrant UCB-style balance: refine promising region while testing alternatives.
- **F7**: Hyperparameter tuning context implies gradual refinement toward optimal configuration. Shift toward center [0.5, 0.5, ...].
- **F8**: High output tempts exploitation, but 8D complexity demands uncertainty hedging. Combine small perturbations with strategic boundary sampling.

### Week 3+ Adaptive Refinement
- **If sparse functions improve**: Expand exploration radius; apply Latin hypercube or stratified sampling.
- **If F2 confirms multimodality**: Implement restart/local optima hunting.
- **If high-dimensional functions plateau**: Incorporate dimensionality reduction (PCA, random projections).
- **Future**: Consider Bayesian approaches (Gaussian process surrogate + expected improvement) if budget permits.

### Why This Approach Works
1. **Evidence-based**: Decisions justified by Week 1 outputs, not assumptions.
2. **Heterogeneous**: Respects function diversity; avoids one-size-fits-all.
3. **Efficient**: Allocates evaluations where uncertainty is highest.
4. **Transparent**: Each decision is documented and adaptable.

---

**Status**: Week 2 submission complete. Awaiting Week 2 outputs to refine Week 3 strategy.
