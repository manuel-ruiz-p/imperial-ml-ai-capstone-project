# Imperial ML/AI Capstone: Black-Box Function Optimization

**Course**: Machine Learning & Artificial Intelligence Capstone  
**Challenge**: 8 Black-Box Functions Optimization via Bayesian Methods  
**Period**: Weeks 1–10 (Complete)  
**Status**: ✅ All 10 Weeks Complete | 🏁 Project Concluded | Best Portfolio: W9 = 82.35

---

## 📋 Project Overview (COMPREHENSIVE GUIDE - All Details Consolidated Here)

Maximize **8 unknown (black-box) functions** via Bayesian optimization under strict constraints:

- **Limited Budget**: 1 query per function per week (8 weeks maximum)
- **Unknown Structure**: No gradients, closed forms, or derivative information  
- **High Dimensionality**: Functions range from 2D to 8D
- **Realistic Scenario**: Mirrors hyperparameter tuning, drug discovery, robotics

### Real-World Applications
- **Hyperparameter Optimization**: Expensive neural network tuning
- **Drug Discovery**: Limited lab budget for molecular screening
- **Robotics**: Costly real-world robot deployments
- **Hardware Design**: Expensive CAD/CFD simulations
- **A/B Testing**: Constrained online experimentation

---

## 📋 Documentation & Transparency

**Complete transparency documentation available**:

- **[DATASHEET.md](DATASHEET.md)**: Comprehensive dataset documentation following Gebru et al. (2018) framework
  - Motivation, composition, collection process
  - 82 query-evaluation pairs across 8 functions
  - Preprocessing, uses, distribution, and maintenance
  
- **[MODEL_CARD.md](MODEL_CARD.md)**: Full optimization approach documentation following Mitchell et al. (2019) framework
  - AEBO (Adaptive Ensemble Bayesian Optimizer) specifications
  - Strategy evolution across 9 weeks
  - Performance metrics, assumptions, and limitations
  - Ethical considerations and reproducibility guidelines

These documents support research transparency, reproducibility, and real-world adaptation of Bayesian optimization methods under data-scarce constraints (N<20 regime).

---

### 🏆 Key Achievements & Critical Lessons
- **F5 BREAKTHROUGH W6: 79.327** → **COLLAPSE W7: 9.247 (-88%)** → **RECOVERY W9: 77.553** — Peak region rediscovered via PCA-guided GP/EI. Lesson: Multi-modal landscapes require systematic exploration, not one-shot exploitation.
- **F4 THREE-WEEK STREAK: W7: -17.894 → W8: -5.556 → W9: -4.635 (all-time best)** — Bounded exploration + PCA guidance produced consecutive improvements.
- **F8 ALL-TIME BEST W9: 9.529** — Surpassed W3 peak (9.449). Centroid-of-winners strategy targeting the high-value cluster proved decisive.
- **F7 ALL-TIME BEST W9: 0.4174** — GP/EI along principal component recovered declining trend.
- **F3 ALL-TIME BEST W9: -0.0051** — Near-zero achieved; best result in 9 weeks.
- **W9 Portfolio: 4.49 → 82.35 (+3,899%)** — Best single-week gain of the project, driven by F5 recovery (77.55) and 4 simultaneous all-time bests.
- **Final Lesson**: PCA over top-k historical inputs reveals which dimensions carry exploitable signal. With N≤21, dimensionality reduction is not optional — it is the strategy.

---

## 📊 Week-by-Week Results Summary (W1→W9 Complete — W10 Submitted)

| **Function** | **Dim** | **W1** | **W2** | **W3** | **W4** | **W5** | **W6** | **W7** | **W8** | **W9** | **Status** |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **F1** | 2 | 2.6e-96 | 7.6e-193 | -5.4e-16 | -1.6e-117 | 3.4e-131 | -2.7e-103 | -1.5e-21 | -1.2e-112 | -2.3e-192 | 🔴 Noise Floor |
| **F2** | 2 | 0.369 | 0.847 | 0.407 | -0.058 | 0.054 | -0.030 | **+0.143** | 0.033 | **0.481** | 🟡 Volatile Recovery |
| **F3** | 3 | -0.010 | -0.011 | -0.079 | -0.012 | -0.136 | -0.080 | -0.106 | -0.138 | **-0.005** | 🟢 **All-time best** |
| **F4** | 4 | -13.07 | -13.07 | -28.65 | -12.61 | -27.44 | -14.20 | -17.89 | -5.556 | **-4.635** | 🟢 **All-time best** |
| **F5** | 4 | 5.273 | 4.049 | 34.98 | 32.97 | 25.58 | **79.327** | 9.247 | 1.149 | **77.553** | 🟢 Near peak |
| **F6** | 5 | -0.700 | -1.912 | -1.552 | -1.479 | -1.294 | -1.808 | -1.594 | -1.570 | -0.988 | 🟡 Improved |
| **F7** | 6 | 0.120 | 0.141 | 0.220 | 0.229 | 0.193 | 0.371 | 0.345 | 0.319 | **0.417** | 🟢 **All-time best** |
| **F8** | 8 | 8.694 | 8.738 | 9.449 | 9.433 | 9.398 | 7.416 | 8.001 | 7.823 | **9.529** | 🟢 **All-time best** |
| **PORTFOLIO** | — | — | — | — | — | — | **69.42** | **5.79** | **4.49** | **82.35** | 🚀 Best since W6 |

**Critical Week 9 Analysis**:
- 🟢 **4 ALL-TIME BESTS IN ONE WEEK**: F3 (−0.005), F4 (−4.635), F7 (0.417), F8 (9.529) — PCA-guided GP/EI targeting winner clusters paid off across all queried functions.
- 🟢 **F5 NEAR-RESURRECTION: 1.149 → 77.553** — GP/EI rediscovered the peak region (dim-2 low, dim-4 high). W6 peak (79.327) nearly matched.
- 🟢 **F8 NEW RECORD: 9.529** — Surpasses W3 historical best of 9.449. Centroid-of-winners + PCA guidance confirmed the high-value cluster.
- 🟡 **F2 STRONG RECOVERY: 0.033 → 0.481** — Not yet at W2 peak (0.847) but significant bounce.
- 🔴 **F1 CONFIRMED NOISE FLOOR** — No exploitable signal exists; all outputs machine-precision noise.
- **PORTFOLIO: 4.49 → 82.35 (+3,899%)** — Best single-week gain of the project. F5 (77.55) dominates but F8/F4/F7/F3 simultaneous improvements validate the PCA strategy broadly.
- **META-LESSON W9**: "PCA over top-k historical inputs reveals which dimensions carry real signal. With N≤21, reducing the effective search dimensionality is not optional — it IS the strategy." Query along principal components of the winner cluster, not uniformly across all dimensions.

---

## 🗂️ Repository Structure

```
imperial-ml-ai-capstone-project/
├── README.md                          # This file (navigation hub)
├── STRUCTURE.md                       # Detailed directory guide
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── data/
│   ├── raw/                           # Pre-collected samples (175 total)
│   │   └── function_1-8/              # 10-40 per function
│   └── processed/                     # [Reserved for future]
│
├── initial_data/                      # Course-provided original data
│   ├── function_1-8/
│   ├── README.md
│   └── QUICK_START.md
│
├── src/                               # Core production-quality code
│   ├── utils/
│   │   ├── data_loading.py           # Load data from all sources
│   │   ├── formatting.py             # Query validation & formatting
│   │   └── visualization.py          # Plotting utilities
│   ├── models/
│   │   ├── base_surrogate.py         # Abstract surrogate interface
│   │   ├── linear_models.py          # Linear/Logistic regression (W2-W3)
│   │   ├── svm_models.py             # SVM RBF kernel (W4+)
│   │   └── nn_models.py              # Neural networks (W5+)
│   └── optimisation/
│       └── bayesian_helpers.py        # EI, UCB, LHS, search methods
│
├── submissions/
│   ├── week_01/queries.py             # W1 queries + results
│   ├── week_02/queries.py             # W2 queries + results
│   ├── week_03/
│   │   ├── queries.py                 # W3 queries + results
│   │   ├── WEEK3_SUBMISSION.md        # W3 strategy rationale
│   │   ├── WEEK3_QUICK_REFERENCE.txt
│   │   └── INITIAL_DATA_VALIDATION.md
│   └── week_04/                       # W4 in progress
│
├── results/                           # Visualizations & analysis
│   ├── [PNG figures: progress, scope, exploration]
│   ├── summary_table.txt
│   └── scope_summary.txt
│
├── notebooks/                         # Analysis scripts
│   ├── visualize_progress.py          # Generate W1-W3 plots
│   ├── visualize_function_scope.py    # Generate scope plots
│   ├── week4_input_recommendations.py # Strategy analysis
│   └── generate_week4_queries.py      # Query generation
│
├── reflections/
│   ├── week_01.md
│   ├── week_02.md
│   ├── week_03.md
│   └── week4_module16_reflection.py   # Module 16 concepts
│
└── docs/
    ├── methodology.md                 # Optimization strategy
    ├── modelling_evolution.md         # Surrogate progression
    └── lessons_learned.md             # Week-by-week insights
```

---

## 📂 Complete Directory Reference & File Guide

### `/data/` — Data Management

**Purpose**: Centralized data access point. Pre-collected function samples organized here.

- **175 total pre-collected samples** across 8 functions
- Loaded by `src/utils/data_loading.py`
- Used as training data for surrogates starting Week 2
- **Never modify**—treat as immutable reference

**Structure**:
```
data/raw/
├── function_1/ → initial_inputs.npy (40, 2), initial_outputs.npy (40,)
├── function_2/ → initial_inputs.npy (20, 2), initial_outputs.npy (20,)
└── ... (functions 3-8 with 10-40 samples each)
```

### `/initial_data/` — Course-Provided Data

**Purpose**: Original course data for reference. Duplicate of pre-collected samples.

```
initial_data/
├── function_1-8/              # Original course data
├── README.md                  # Original course README
└── QUICK_START.md             # Course quick-start guide
```

### `/src/` — Production Code

#### `/src/utils/` — Utility Modules

**data_loading.py** (132 lines)
- `load_function_data(function_id)` → (X, y)
- `load_all_functions()` → dict of all 8 functions
- `load_weekly_results(week)` → dict of results

**formatting.py** (140 lines)
- `array_to_submission_string(query)` → "0.123456-0.654321-..."
- `validate_submission(queries)` → (bool, [errors])
- **Checks**: Dimensionality, range [0,1], 6 decimal places

**visualization.py** (140 lines)
- Plotting utilities for consistent styling

#### `/src/models/` — Surrogate Models

**base_surrogate.py**
- Abstract `BaseSurrogate` class
- Interface: `fit(X, y)`, `predict(X)`, `predict_with_uncertainty(X)`

**linear_models.py** (280 lines)
- `LinearRegressionSurrogate` (Week 2-3, baseline, tested on all functions)
- `LogisticRegressionSurrogate` (classification)
- **Limitation**: Cannot capture non-linearity (F2, F3, F4 decline)

**svm_models.py** (Scaffolded)
- `SVMSurrogate` with RBF kernel (Week 4+)
- Expected to handle non-linear landscapes

**nn_models.py** (Placeholder)
- Neural network surrogates (Week 5+)

#### `/src/optimisation/` — Bayesian Optimization

**bayesian_helpers.py** (320 lines)
- `expected_improvement(mu, sigma, best_y)` → acquisition scores
- `upper_confidence_bound(mu, sigma, beta)` → UCB scores
- `latin_hypercube_search(bounds, n_candidates)` → candidate points
- `grid_search()`, `random_search()` → search methods

### `/submissions/` — Weekly Query Management

**Week 1: Exploratory Baseline**
- **Strategy**: Uniform random sampling
- **File**: `week_01/queries.py`
- **Key Insight**: One-size-fails-all—functions require heterogeneous strategies

**Week 2: Differential Diagnosis**
- **Strategy**: Boundary exploration + interior refinement
- **File**: `week_02/queries.py`
- **Key Insight**: F5 prefers interior, F2 prefers boundaries

**Week 3: Expected Improvement via Bayesian Optimization**
- **Strategy**: EI acquisition function + linear regression surrogates
- **File**: `week_03/queries.py`
- **Data**: 175 pre-collected samples + W1-W2 submissions
- **Results**:
  - F5 breakthrough: +764% (4.049 → 34.98)
  - F7 success: +56% (0.141 → 0.220)
  - F2-F4 decline: Linear surrogates insufficient for non-linear functions
- **Documentation**: 
  - `WEEK3_SUBMISSION.md` — Detailed strategy rationale
  - `WEEK3_QUICK_REFERENCE.txt` — Quick lookup table
  - `INITIAL_DATA_VALIDATION.md` — Data verification

**Week 4+: Adaptive Per-Function Strategies**
- **Winners (F5, F7)**: Deep exploitation (β=0.5 in UCB)
- **Improving (F6, F8)**: Balanced exploration-exploitation (β=1.5)
- **Declining (F2, F3, F4)**: Shallow broad exploration (β=3.0)
- **Sparse (F1)**: Random high-uncertainty sampling (β=5.0)

### `/results/` — Visualizations & Analysis Output

**Progress Plots** (W1-W3 Trends):
- `output_progression.png` — Bar chart per function
- `improvement_trajectory.png` — Weekly % changes
- `2d_exploration.png` — Input trajectories
- `high_dim_projection.png` — PCA projection
- `output_boundaries.png` — Min/max/mean landscape

**Scope Analysis** (W3 Coverage):
- `input_space_coverage_2d.png` — Density visualization
- `dimension_wise_coverage.png` — Per-dimension ranges
- `output_range_spectrum.png` — Output distribution
- `query_density_heatmap.png` — Regional density

**Summary**:
- `summary_table.txt` — Numeric W1-W3 results
- `scope_summary.txt` — Metrics per function

### `/notebooks/` — Analysis & Generation Scripts

**visualize_progress.py** (285 lines)
- Generates W1-W3 progress visualizations
- Outputs: 6 PNG figures + summary_table.txt
- Run: `python3 notebooks/visualize_progress.py`

**visualize_function_scope.py** (340 lines)
- Generates scope/coverage analysis
- Outputs: 4 PNG figures + scope_summary.txt
- Run: `python3 notebooks/visualize_function_scope.py`

**week4_input_recommendations.py**
- Analyzes W1-W3 performance by group
- Categorizes: Winners, Improving, Declining, Sparse
- Run: `python3 notebooks/week4_input_recommendations.py`

**generate_week4_queries.py**
- Generates W4 submission queries
- Applies group-specific strategies
- Run: `python3 notebooks/generate_week4_queries.py`

### `/reflections/` — Weekly Reflection Documents

**week_01.md** — Exploratory phase reflection  
**week_02.md** — Differential diagnosis reflection  
**week_03.md** — Targeted optimization reflection  
**week4_module16_reflection.py** — Module 16 concepts
- Connects capstone to hierarchical learning, AlexNet, exploration-exploitation
- Responds to 6 course reflection prompts
- ~380 words (ready for discussion board)

### `/docs/` — Detailed Technical Documentation

**methodology.md**
- Optimization strategy overview
- Differential diagnosis principle
- Per-function strategy justification
- Surrogate model roadmap

**modelling_evolution.md**
- Week-by-week surrogate progression
- Linear regression limitations (W2-W3)
- SVM upgrade (W4)
- Neural networks (W5+) planning

**lessons_learned.md**
- Week 1: Extreme variability insight
- Week 2: Boundary vs. interior trade-off
- Week 3: Surrogate effectiveness varies per function
- Overall: Function-specific strategies essential

---

## 🚀 Quick Start

### Installation

```bash
# Clone & set up
git clone https://github.com/username/imperial-ml-ai-capstone-project.git
cd imperial-ml-ai-capstone-project

# Create environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Load & Use

```python
# Load pre-collected samples
from src.utils.data_loading import load_all_functions, load_weekly_results
all_data = load_all_functions()
X_f5, y_f5 = all_data[5]  # 40 samples for F5

# Load submission results
from submissions.week_03.queries import week3_queries, week3_results
w3_query = week3_queries[5]    # [0.014688, 0.641578, 0.349456, 0.493352]
w3_result = week3_results[5]   # 34.98 (breakthrough!)

# Generate W4 queries
exec(open('notebooks/generate_week4_queries.py').read())
```

---

## 📚 Documentation & Quick Links by Purpose

| **Goal** | **Resource** |
|----------|-------------|
| **Dataset documentation** | **[DATASHEET.md](DATASHEET.md)** |
| **Model documentation** | **[MODEL_CARD.md](MODEL_CARD.md)** |
| See week-by-week strategy | `docs/methodology.md` |
| Track model evolution | `docs/modelling_evolution.md` |
| Learn from results | `docs/lessons_learned.md` |
| Review W1 strategy | `reflections/week_01.md` |
| Review W2 strategy | `reflections/week_02.md` |
| Review W3 strategy | `reflections/week_03.md` |
| Understand W4 approach | `submissions/week_03/WEEK3_SUBMISSION.md` |
| Module 16 reflection | `reflections/week4_module16_reflection.py` |

---

## 🔬 Technical Details

### Bayesian Optimization Stack
- **Acquisition Functions**: Expected Improvement (EI), Upper Confidence Bound (UCB)
- **Surrogates** (Progressive):
  - W1-W3: Linear regression (baseline, fast)
  - W4+: Support Vector Machines (RBF kernel, non-linear)
  - W5+: Neural networks (complex landscapes)
- **Search**: Latin Hypercube Sampling, grid search, random search
- **Validation**: 175 pre-collected samples per function

### Key Hyperparameters by Function Group

| Group | Functions | Strategy | β (UCB) | Action |
|:---:|:---:|:---|:---:|:---|
| **Winners** | F5, F7 | Exploitation | 0.5 | Refine near optimum |
| **Improving** | F6, F8 | Balanced | 1.5 | Continue trajectory |
| **Declining** | F2, F3, F4 | Exploration | 3.0 | Escape local optima |
| **Sparse** | F1 | Random | 5.0 | High-uncertainty sampling |

---

## 📈 Performance Highlights

- **F5 W6 Breakthrough**: 79.327 (+127% from W5, +1405% from W1) — Best result achieved, elite region exploitation
- **F7 W6 Confirmed**: 0.3704 (+62% from W5, +208% from W1) — Trend-following validation, momentum strategy works
- **Data Leverage**: 175 pre-collected samples + 6 weeks queries enabled reliable ensemble training
- **Ensemble Success**: 2 breakdowns (F5, F7) vs 3 failures (F2, F6, F8) — hyperparameter tuning needed for adaptation
- **W6 Insight**: Volatility-adaptive strategies and function-specific model selection critical for Week 7+

---

## 🔄 Future Work (Weeks 8+)

- [x] Week 7: Hyperparameter tuning (learning rate, regularization, ensemble weights, radius scaling, thresholds, architecture)
- [ ] Implement SVM surrogates for non-linear functions (W4 baseline, improve in W8)
- [ ] Neural network models for complex landscapes (initial W5, enhance in W8)
- [ ] Deep ensemble methods (combining 3+ surrogates with voting)
- [ ] Bayesian optimization for hyperparameter tuning itself
- [ ] Advanced acquisition functions (EI variants, Thompson sampling)
- [ ] Uncertainty quantification via confidence intervals

### Why This Approach Works
1. **Evidence-based**: Decisions justified by Week 1 outputs, not assumptions.
2. **Heterogeneous**: Respects function diversity; avoids one-size-fits-all.
3. **Efficient**: Allocates evaluations where uncertainty is highest.
4. **Transparent**: Each decision is documented and adaptable.

---

**Status**: Week 2 submission complete. Awaiting Week 2 outputs to refine Week 3 strategy.
