"""
Models module initialization.

Provides access to all surrogate models used across different weeks.
"""

from .base_surrogate import BaseSurrogate
from .linear_models import LinearRegressionSurrogate, LogisticRegressionSurrogate

# TODO: Week 3 - Add SVM-based surrogates
# from .svm_models import SVMSurrogate, RBFSurrogate

# TODO: Week 4+ - Add neural network surrogates
# from .nn_models import NeuralNetSurrogate, GaussianProcessSurrogate

__all__ = [
    "BaseSurrogate",
    "LinearRegressionSurrogate",
    "LogisticRegressionSurrogate",
    # "SVMSurrogate",
    # "NeuralNetSurrogate",
    # "GaussianProcessSurrogate",
]
