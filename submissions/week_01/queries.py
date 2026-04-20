"""
Week 1 Query Points and Results

Strategy: Exploratory baseline queries across all functions
- Sample from different regions of input space (corners, center, edges)
- Gather initial output characteristics to guide Week 2+ strategy

Dimensionality: 2D, 2D, 3D, 4D, 4D, 5D, 6D, 8D
"""

import numpy as np


# Week 1 Query Points (submitted)
week1_queries = {
    1: np.array([0.250000, 0.750000]),
    2: np.array([0.750000, 0.250000]),
    3: np.array([0.333333, 0.666667, 0.500000]),
    4: np.array([0.200000, 0.800000, 0.400000, 0.600000]),
    5: np.array([0.700000, 0.300000, 0.600000, 0.200000]),
    6: np.array([0.200000, 0.400000, 0.600000, 0.800000, 0.500000]),
    7: np.array([0.150000, 0.350000, 0.550000, 0.750000, 0.950000, 0.450000]),
    8: np.array([0.125000, 0.250000, 0.375000, 0.500000, 0.625000, 0.750000, 0.875000, 0.437500])
}

# Week 1 Results (received)
week1_results = {
    1: 2.6065864278618756e-96,
    2: 0.3691787538388598,
    3: -0.010251690931823796,
    4: -13.072131637188551,
    5: 5.273302329600012,
    6: -0.6995639652538725,
    7: 0.11959165710190967,
    8: 8.694471875
}
