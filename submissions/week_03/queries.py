"""
Week 3 Query Points and Results

Strategy: Expected Improvement-based optimization using linear regression surrogates
- Trained on initial data (10-40 samples per function) + Week 1-2 submissions
- Generated 5000 candidate points per function via Latin Hypercube Sampling
- Scored with Expected Improvement acquisition function
- Selected point with highest EI score for Week 3 submission

Key Decisions:
- F1, F4, F6: Continued exploration of sparse/challenging landscapes
- F2, F7, F8: Exploitation-focused refinement of identified peaks
- F3, F5: Balanced approach based on observed landscape characteristics
"""

import numpy as np


# Week 3 Query Points (generated via Expected Improvement)
week3_queries = {
    1: np.array([0.754891, 0.704403]),  # F1 (2D)
    2: np.array([0.686831, 0.530211]),  # F2 (2D)
    3: np.array([0.039713, 0.302029, 0.315311]),  # F3 (3D)
    4: np.array([0.728602, 0.982928, 0.708406, 0.027707]),  # F4 (4D)
    5: np.array([0.014688, 0.641578, 0.349456, 0.493352]),  # F5 (4D)
    6: np.array([0.575333, 0.108777, 0.034359, 0.840559, 0.517247]),  # F6 (5D)
    7: np.array([0.102635, 0.201553, 0.788679, 0.155646, 0.990262, 0.833759]),  # F7 (6D)
    8: np.array([0.018659, 0.622726, 0.428889, 0.224671, 0.701438, 0.385308, 0.247735, 0.172798]),  # F8 (8D)
}

# Week 3 Results (received from platform)
week3_results = {
    1: -5.384584177282445e-16,       # F1
    2: 0.4074279061230939,            # F2
    3: -0.07882847061831176,          # F3
    4: -28.648038812076084,           # F4
    5: 34.98323399644939,             # F5
    6: -1.552441674550123,            # F6
    7: 0.219690205078482,             # F7
    8: 9.4488988470416,               # F8
}

