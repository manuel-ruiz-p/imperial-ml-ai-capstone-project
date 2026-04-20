"""
Capstone Project: Black-Box Optimization

This package provides utilities and models for iterative optimization of
unknown (black-box) functions using Bayesian optimization principles and
adaptive surrogate models.

Modules:
- utils: Data loading, formatting, visualization
- models: Surrogate models (linear, SVM, neural networks)
- optimisation: Search strategies and infill criteria
"""

from . import utils, models, optimisation

__version__ = "0.1.0"
__author__ = "Student Researcher"

__all__ = ["utils", "models", "optimisation"]
