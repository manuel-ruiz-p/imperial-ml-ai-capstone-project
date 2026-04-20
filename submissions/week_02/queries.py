"""
Week 2 Query Points and Results

Strategy: Differentiated approach based on Week 1 analysis
- F1, F4, F6: Exploration-heavy (95%, 80%, 65%)
- F2, F7, F8: Balanced hybrid (40%, 50%, 45%)
- F3, F5: Exploitation-heavy (20%, 25%)

Key Insight: Output magnitude and sparsity indicate landscape accessibility.
Sparse/negative outputs → exploration; High positive → exploitation.
"""

import numpy as np


# Week 2 Query Points (submitted)
week2_queries = {
    1: np.array([0.050000, 0.050000]),                                    # F1 (2D)
    2: np.array([0.500000, 0.500000]),                                    # F2 (2D)
    3: np.array([0.350000, 0.650000, 0.500000]),                          # F3 (3D)
    4: np.array([0.800000, 0.200000, 0.600000, 0.400000]),                # F4 (4D)
    5: np.array([0.720000, 0.280000, 0.580000, 0.220000]),                # F5 (4D)
    6: np.array([0.800000, 0.600000, 0.400000, 0.200000, 0.500000]),     # F6 (5D)
    7: np.array([0.250000, 0.400000, 0.500000, 0.700000, 0.850000, 0.500000]),  # F7 (6D)
    8: np.array([0.150000, 0.300000, 0.400000, 0.480000, 0.600000, 0.700000, 0.850000, 0.450000])  # F8 (8D)
}

# Week 2 Results (received)
week2_results = {
    1: 7.570914060942952e-193,
    2: 0.8473573729146894,
    3: -0.010450162716101937,
    4: -13.072131637188551,
    5: 4.049267429988913,
    6: -1.9119879535617619,
    7: 0.14129996220103783,
    8: 8.73765
}
