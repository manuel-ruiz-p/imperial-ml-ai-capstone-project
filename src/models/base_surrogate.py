"""
Base surrogate model class and interface for capstone project.

All surrogate models (linear, SVM, neural networks) inherit from BaseSurrogate.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseSurrogate(ABC):
    """
    Abstract base class for surrogate models.
    
    A surrogate is a fast approximation of an expensive black-box function,
    trained on observed input-output pairs. Used for:
    - Guiding query point selection (infill criterion)
    - Approximating high-dimensional function behavior
    - Uncertainty quantification (for some models)
    """
    
    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Human-readable name for the surrogate (e.g., "LinearRegression")
        """
        self.name = name
        self.is_fitted = False
        self.training_data = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSurrogate":
        """
        Fit the surrogate model to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training input points, shape (n_samples, n_features)
        y : np.ndarray
            Training output values, shape (n_samples,)
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new input points.
        
        Parameters
        ----------
        X : np.ndarray
            Input points to predict, shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted output values, shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Not all surrogates support uncertainty (e.g., plain linear regression).
        Override this method if uncertainty is available.
        
        Parameters
        ----------
        X : np.ndarray
            Input points, shape (n_samples, n_features)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, uncertainties) or (predictions, zeros) if not supported
        """
        pass
    
    def get_metadata(self) -> dict:
        """Return metadata about the fitted model."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'n_training_samples': len(self.training_data[1]) if self.training_data else 0,
        }
