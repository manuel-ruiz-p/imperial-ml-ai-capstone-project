"""
Linear and logistic regression surrogate models.

Week 2 strategy: Use simple linear/logistic regression to build initial
understanding of function landscapes before moving to more complex surrogates.

Classes:
- LinearRegressionSurrogate: Standard linear regression (Week 2)
- LogisticRegressionSurrogate: For binary or bounded outputs
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from .base_surrogate import BaseSurrogate


class LinearRegressionSurrogate(BaseSurrogate):
    """
    Linear regression surrogate model.
    
    Assumes: output = w^T * x + b
    
    Pros:
    - Interpretable: weights indicate feature importance
    - Fast to train and predict
    - Provides baseline for more complex models
    
    Cons:
    - Assumes linear relationship (likely too simple for Week 3+)
    - No uncertainty estimates
    
    Use Case:
    - Week 2: Initial understanding of function landscapes
    - High-dimensional functions (curse of dimensionality)
    """
    
    def __init__(self, normalize: bool = True):
        """
        Parameters
        ----------
        normalize : bool, default=True
            Whether to standardize inputs before fitting
        """
        super().__init__("LinearRegression")
        self.model = LinearRegression()
        self.scaler = StandardScaler() if normalize else None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionSurrogate":
        """
        Fit linear regression model.
        
        Parameters
        ----------
        X : np.ndarray
            Training inputs, shape (n_samples, n_features)
        y : np.ndarray
            Training outputs, shape (n_samples,)
        
        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y)
        self.X_train = X
        self.y_train = y
        self.training_data = (X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new inputs.
        
        Parameters
        ----------
        X : np.ndarray
            Input points, shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted outputs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outputs (no uncertainty for linear regression).
        
        Returns zeros for uncertainty as linear regression doesn't
        provide principled confidence intervals.
        
        Parameters
        ----------
        X : np.ndarray
            Input points
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, zeros)
        """
        predictions = self.predict(X)
        uncertainties = np.zeros_like(predictions)
        return predictions, uncertainties
    
    def get_feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature importance (coefficients) from linear model.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (feature_indices, coefficients)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        coefficients = self.model.coef_
        return np.arange(len(coefficients)), coefficients


class LogisticRegressionSurrogate(BaseSurrogate):
    """
    Logistic regression for binary classification or bounded outputs.
    
    Useful for functions with known bounds or binary responses.
    Not commonly used in this capstone but included for completeness.
    
    Use Case:
    - Functions known to be binary (e.g., constraint satisfaction)
    - Bounded outputs (via sigmoid transformation)
    """
    
    def __init__(self, normalize: bool = True, threshold: float = 0.0):
        """
        Parameters
        ----------
        normalize : bool, default=True
            Whether to standardize inputs
        threshold : float, default=0.0
            Decision threshold (default: classify as positive if output > 0)
        """
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler() if normalize else None
        self.threshold = threshold
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionSurrogate":
        """
        Fit logistic regression (convert continuous y to binary classification).
        
        Parameters
        ----------
        X : np.ndarray
            Training inputs
        y : np.ndarray
            Training outputs (converted to binary: y > threshold)
        
        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        # Convert to binary: 1 if y > threshold, else 0
        y_binary = (y > self.threshold).astype(int)
        
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y_binary)
        self.X_train = X
        self.y_train = y
        self.training_data = (X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class (0 or 1).
        
        Parameters
        ----------
        X : np.ndarray
            Input points
        
        Returns
        -------
        np.ndarray
            Predicted binary labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with probability estimates (used as uncertainty proxy).
        
        Parameters
        ----------
        X : np.ndarray
            Input points
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, probabilities)
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X).max(axis=1)
        return predictions, probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability estimates for both classes."""
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
