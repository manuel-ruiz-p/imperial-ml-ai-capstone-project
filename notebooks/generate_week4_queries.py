#!/usr/bin/env python3
"""Week 4 Query Generation - Submission Format"""

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Week 3 queries (reference for refinement)
week3_queries = {
    1: np.array([0.754891, 0.704403]),
    2: np.array([0.686831, 0.530211]),
    3: np.array([0.039713, 0.302029, 0.315311]),
    4: np.array([0.728602, 0.982928, 0.708406, 0.027707]),
    5: np.array([0.014688, 0.641578, 0.349456, 0.493352]),
    6: np.array([0.575333, 0.108777, 0.034359, 0.840559, 0.517247]),
    7: np.array([0.102635, 0.201553, 0.788679, 0.155646, 0.990262, 0.833759]),
    8: np.array([0.018659, 0.622726, 0.428889, 0.224671, 0.701438, 0.385308, 0.247735, 0.172798]),
}

# Generate Week 4 queries based on strategy
week4_queries = {}

# F1 (2D) - SPARSE: Random exploration
dim = 2
w4 = np.random.uniform(0, 1, dim)
week4_queries[1] = w4

# F2 (2D) - DECLINING: Explore edge region
dim = 2
# Try lower-left corner region
w4 = np.array([0.15, 0.15]) + np.random.uniform(-0.05, 0.05, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[2] = w4

# F3 (3D) - DECLINING: Jump to opposite region
dim = 3
# W2 was [1,1,1], W3 was [0.04, 0.30, 0.32] → Try middle-upper region
w4 = np.array([0.65, 0.70, 0.60]) + np.random.uniform(-0.08, 0.08, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[3] = w4

# F4 (4D) - DECLINING: Explore different corner
dim = 4
# Try lower corner region
w4 = np.array([0.15, 0.20, 0.50, 0.85]) + np.random.uniform(-0.08, 0.08, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[4] = w4

# F5 (4D) - EXCELLENT: Local refinement around W3
dim = 4
w4 = week3_queries[5] + np.random.uniform(-0.02, 0.02, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[5] = w4

# F6 (5D) - IMPROVING: Small step + exploration
dim = 5
# Continue trend from W2→W3, small adjustment
w4 = week3_queries[6] + np.random.uniform(-0.05, 0.05, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[6] = w4

# F7 (6D) - EXCELLENT: Local refinement around W3
dim = 6
w4 = week3_queries[7] + np.random.uniform(-0.03, 0.03, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[7] = w4

# F8 (8D) - IMPROVING: Small step + exploration
dim = 8
w4 = week3_queries[8] + np.random.uniform(-0.04, 0.04, dim)
w4 = np.clip(w4, 0, 1)
week4_queries[8] = w4

# Format output for submission
print("\n" + "="*90)
print("WEEK 4 QUERIES - SUBMISSION FORMAT".center(90))
print("="*90 + "\n")

print("Format: x1-x2-x3-...-xn (each value to 6 decimal places)\n")

for func_id in range(1, 9):
    w4 = week4_queries[func_id]
    
    # Format as submission string
    submission_str = "-".join([f"0.{str(int(round((v % 1) * 1e6))).zfill(6)}" if v < 1 else "1.000000" for v in w4])
    
    # Better formatting
    submission_str = "-".join([f"{v:.6f}" for v in w4])
    
    print(f"F{func_id}: {submission_str}")

print("\n" + "="*90)
print("WEEK 4 SUBMISSION TEMPLATE\n")
print("Copy-paste the following into your submission:\n")

# Create submission dict format
submission_data = {}
for func_id in range(1, 9):
    w4 = week4_queries[func_id]
    submission_str = "-".join([f"{v:.6f}" for v in w4])
    submission_data[func_id] = submission_str
    print(f"Function {func_id}: {submission_str}")

print("\n" + "="*90)
print("\nStrategyGuide:\n")
print("✅ F1 (SPARSE):  Random sampling - function near zero")
print("⚠️  F2 (DECLINING): Escape to edge region [0.15, 0.15]")
print("⚠️  F3 (DECLINING): Jump to opposite region [0.65, 0.70, 0.60]")
print("⚠️  F4 (DECLINING): Explore corner [0.15, 0.20, 0.50, 0.85]")
print("📈 F5 (EXCELLENT): Refine near W3 [0.015, 0.642, 0.349, 0.493]")
print("✅ F6 (IMPROVING): Continue trend from W3 [0.575, 0.109, 0.034, 0.841, 0.517]")
print("📈 F7 (EXCELLENT): Refine near W3 [0.103, 0.202, 0.789, 0.156, 0.990, 0.834]")
print("✅ F8 (IMPROVING): Continue trend from W3 [0.019, 0.623, 0.429, 0.225, 0.701, 0.385, 0.248, 0.173]")
print("\n" + "="*90)
