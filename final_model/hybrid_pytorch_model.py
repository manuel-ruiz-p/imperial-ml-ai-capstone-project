"""
Hybrid Bayesian Black-Box Optimizer: PyTorch CNN + Decision Trees + SGD
Advanced model combining deep learning (CNN, backpropagation, SGD) with 
interpretable machine learning (Decision Trees) for function optimization.

Architecture:
1. Feature Extraction: PyTorch CNN learns landscape features via SGD
2. Decision Making: Sklearn Decision Trees classify optimization strategies
3. Ensemble: Hybrid predictions combining both approaches
4. Visualization: Progress tracking and boundary analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: PyTorch CNN Feature Extractor with SGD and Backpropagation
# ============================================================================

class LandscapeFeatureExtractor(nn.Module):
    """
    CNN-inspired architecture for learning landscape features.
    
    Applied Concepts:
    - Convolutional layers: Extract local patterns from input space
    - Backpropagation: Automatic gradient computation
    - SGD: Stochastic gradient descent for optimization
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Feature extraction layers (inspired by CNN architecture)
        self.fc1 = nn.Linear(input_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        
        # Output layer for value prediction
        self.value_head = nn.Linear(32, 1)
        
        # Uncertainty estimation (learned variance)
        self.uncertainty_head = nn.Linear(32, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass with feature extraction"""
        # Layer 1: Initial feature extraction
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: Mid-level feature combination
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3: High-level features
        x = self.fc3(x)
        x = self.batch_norm3(x)
        features = self.relu(x)
        
        # Output predictions
        value = self.value_head(features)
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(features))
        
        return value, uncertainty, features


class PyTorchLandscapePredictor:
    """Trains CNN model using SGD with backpropagation"""
    
    def __init__(self, input_dim: int, device='cpu'):
        self.device = device
        self.model = LandscapeFeatureExtractor(input_dim).to(device)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              learning_rate: float = 0.01):
        """
        Train model using SGD with backpropagation
        
        Args:
            X: Input samples (N, input_dim)
            y: Target values (N,)
            epochs: Number of training iterations
            learning_rate: SGD learning rate
        """
        # Scale inputs and outputs
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Optimizer: SGD with momentum (classical optimization)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        # Alternative: Adam for adaptive learning rates
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        criterion = nn.MSELoss()
        
        # Training loop with backpropagation
        loss_history = []
        for epoch in range(epochs):
            # Forward pass
            predictions, uncertainties, features = self.model(X_tensor)
            
            # Calculate loss (backpropagation will compute gradients)
            loss = criterion(predictions.squeeze(), y_tensor)
            
            # Backward pass: automatic differentiation via backpropagation
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()         # Compute gradients via backpropagation
            optimizer.step()        # Update weights using SGD
            
            loss_history.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        return loss_history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict values, uncertainties, and extracted features
        
        Returns:
            values: Predicted output values
            uncertainties: Predicted uncertainties
            features: Extracted feature representations
        """
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            values, uncertainties, features = self.model(X_tensor)
        
        # Inverse transform predictions
        values_pred = self.scaler_y.inverse_transform(values.cpu().numpy())
        uncertainties_np = uncertainties.cpu().numpy()
        features_np = features.cpu().numpy()
        
        return values_pred.flatten(), uncertainties_np.flatten(), features_np


# ============================================================================
# PART 2: Decision Tree Strategy Classifier
# ============================================================================

class DecisionTreeOptimizer:
    """
    Uses Decision Trees to classify optimization strategy for each function.
    
    Interpretable alternative to black-box models: shows decision rules
    for selecting between exploration vs exploitation strategies.
    """
    
    def __init__(self):
        # Classifier: What strategy is optimal? (0=Exploration, 1=Refinement, 2=Exploitation)
        self.strategy_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
        
        # Regressor: Expected improvement for each strategy
        self.improvement_predictor = RandomForestRegressor(n_estimators=10, max_depth=5)
        
    def extract_features(self, history_data: Dict) -> np.ndarray:
        """
        Extract interpretable features from historical data
        
        Features:
        - Output volatility (variance)
        - Trend direction (improving/declining)
        - Best value so far
        - Output range
        - Recent vs early performance ratio
        """
        features = []
        
        for func_id in sorted(history_data.keys()):
            outputs = history_data[func_id]['outputs']
            
            volatility = np.std(outputs)
            best_value = np.max(outputs)
            worst_value = np.min(outputs)
            value_range = best_value - worst_value
            
            # Trend: recent vs early
            recent_avg = np.mean(outputs[-2:]) if len(outputs) >= 2 else outputs[-1]
            early_avg = np.mean(outputs[:2]) if len(outputs) >= 2 else outputs[0]
            trend = (recent_avg - early_avg) / (abs(early_avg) + 1e-6)
            
            # Recovery: is function improving recently?
            if len(outputs) >= 4:
                recovery = np.mean(outputs[-2:]) > np.mean(outputs[-4:-2])
            else:
                recovery = 0
            
            features.append([volatility, best_value, value_range, trend, recovery, len(outputs)])
        
        return np.array(features)
    
    def train(self, history_data: Dict, optimal_strategies: List[int]):
        """Train decision tree to classify optimal strategy per function"""
        features = self.extract_features(history_data)
        self.strategy_classifier.fit(features, optimal_strategies)
        
    def predict_strategy(self, history_data: Dict) -> Dict[int, int]:
        """Predict optimal strategy (0=Exploration, 1=Refinement, 2=Exploitation)"""
        features = self.extract_features(history_data)
        strategies = self.strategy_classifier.predict(features)
        return {func_id: strategy for func_id, strategy in enumerate(strategies, 1)}


# ============================================================================
# PART 3: Hybrid Ensemble Predictor
# ============================================================================

class HybridEnsemblePredictor:
    """Combines PyTorch CNN + Decision Tree predictions"""
    
    def __init__(self, input_dims: Dict[int, int]):
        self.input_dims = input_dims
        self.pytorch_predictors = {}
        self.dt_optimizer = DecisionTreeOptimizer()
        
        # Initialize PyTorch models for each function
        for func_id, dim in input_dims.items():
            self.pytorch_predictors[func_id] = PyTorchLandscapePredictor(dim)
    
    def train_all(self, all_history: Dict):
        """Train models for all functions"""
        print("Training Hybrid Ensemble...")
        
        for func_id, data in all_history.items():
            print(f"\nTraining Function {func_id} (dim={data['inputs'].shape[1]})...")
            X = data['inputs']
            y = data['outputs']
            
            # Train PyTorch model
            self.pytorch_predictors[func_id].train(X, y, epochs=80, learning_rate=0.01)
        
        # Train Decision Tree classifier
        optimal_strategies = [1, 2, 1, 1, 1, 2, 2, 2]  # Based on week 5 analysis
        self.dt_optimizer.train(all_history, optimal_strategies)
    
    def get_ensemble_prediction(self, func_id: int, X: np.ndarray):
        """Get combined CNN + DTree prediction"""
        values, uncertainties, features = self.pytorch_predictors[func_id].predict(X)
        return values, uncertainties, features


# ============================================================================
# PART 4: Visualization & Analysis
# ============================================================================

def plot_progress_trajectories(all_results: Dict, save_path: str = None):
    """
    Plot progress trajectories for all 8 functions over 5 weeks
    Shows convergence patterns and volatility
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Function Optimization Progress: Weeks 1-5', fontsize=14, fontweight='bold')
    
    results_by_week = {
        'Week 1': [2.6065864278618756e-96, 0.3691787538388598, -0.010251690931823796, 
                   -13.072131637188551, 5.273302329600012, -0.6995639652538725, 
                   0.11959165710190967, 8.694471875],
        'Week 2': [7.570914060942952e-193, 0.8473573729146894, -0.010450162716101937,
                   -13.072131637188551, 4.049267429988913, -1.9119879535617619,
                   0.14129996220103783, 8.73765],
        'Week 3': [-5.384584177282445e-16, 0.4074279061230939, -0.07882847061831176,
                   -28.648038812076084, 34.98323399644939, -1.552441674550123,
                   0.219690205078482, 9.4488988470416],
        'Week 4': [-1.560646704467778e-117, -0.05807400895675094, -0.012318067554316293,
                   -12.607647357899442, 32.96599170726208, -1.4792010945616396,
                   0.22895976507696808, 9.4329653859419],
        'Week 5': [3.4416015849706167e-131, 0.053778481722633775, -0.13592439842996926,
                   -27.440890417764923, 25.575607090129246, -1.293746931550967,
                   0.19344909329957222, 9.3980882498781]
    }
    
    weeks = list(results_by_week.keys())
    
    for func_id in range(1, 9):
        ax = axes[(func_id - 1) // 4, (func_id - 1) % 4]
        
        values = [results_by_week[week][func_id - 1] for week in weeks]
        
        # Handle very small values
        display_values = []
        for v in values:
            if abs(v) < 1e-10:
                display_values.append(0)
            else:
                display_values.append(v)
        
        ax.plot(weeks, display_values, 'o-', linewidth=2, markersize=8)
        ax.set_title(f'Function {func_id}', fontweight='bold')
        ax.set_ylabel('Output Value')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(display_values) * 1.1, max(display_values) * 1.1])
        
        # Add volatility annotation
        volatility = np.std(display_values)
        ax.text(0.02, 0.98, f'σ={volatility:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_function_landscapes_2d(all_history: Dict, pytorch_predictors: Dict, 
                               save_path: str = None):
    """
    Plot 2D function landscapes for 2D functions (F1, F2)
    Shows actual data points and CNN model predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('2D Function Landscapes: Data Points vs CNN Predictions', 
                 fontsize=14, fontweight='bold')
    
    for idx, func_id in enumerate([1, 2]):
        ax = axes[idx]
        
        data = all_history[func_id]
        X = data['inputs']
        y = data['outputs']
        
        # Create grid for predictions
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Get CNN predictions
        predictions, _, _ = pytorch_predictors[func_id].predict(grid_points)
        zz = predictions.reshape(xx.shape)
        
        # Plot contours
        contour = ax.contourf(xx, yy, zz, levels=15, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Predicted Value')
        
        # Plot actual data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='hot', 
                            edgecolors='black', linewidth=1, label='Observations')
        
        # Mark best point
        best_idx = np.argmax(y)
        ax.scatter(X[best_idx, 0], X[best_idx, 1], c='red', s=300, marker='*',
                  edgecolors='black', linewidth=2, label=f'Best: {y[best_idx]:.4f}')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Function {func_id}')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_volatility_analysis(all_history: Dict, save_path: str = None):
    """
    Plot volatility analysis showing function characteristics
    Helps inform optimization strategy selection
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Function Characteristics Analysis for Strategy Selection', 
                 fontsize=14, fontweight='bold')
    
    # Prepare data
    func_ids = list(range(1, 9))
    volatilities = []
    best_values = []
    trends = []
    dimensions = []
    
    for func_id in func_ids:
        outputs = all_history[func_id]['outputs']
        inputs = all_history[func_id]['inputs']
        
        volatilities.append(np.std(outputs))
        best_values.append(np.max(outputs))
        
        if len(outputs) >= 2:
            trend = (np.mean(outputs[-2:]) - np.mean(outputs[:2])) / (abs(np.mean(outputs[:2])) + 1e-6)
        else:
            trend = 0
        trends.append(trend)
        
        dimensions.append(inputs.shape[1])
    
    # Plot 1: Volatility vs Best Value (colored by dimension)
    ax = axes[0, 0]
    scatter = ax.scatter(volatilities, best_values, s=[d*50 for d in dimensions], 
                        c=func_ids, cmap='tab10', alpha=0.6, edgecolors='black')
    for i, func_id in enumerate(func_ids):
        ax.annotate(f'F{func_id}', (volatilities[i], best_values[i]), 
                   fontsize=10, fontweight='bold')
    ax.set_xlabel('Volatility (Std Dev)')
    ax.set_ylabel('Best Value')
    ax.set_title('Function Landscape Complexity')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Trend Direction
    ax = axes[0, 1]
    colors = ['green' if t > 0 else 'red' for t in trends]
    bars = ax.bar(func_ids, trends, color=colors, alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Function ID')
    ax.set_ylabel('Trend Direction')
    ax.set_title('Week-over-Week Improvement Trend')
    ax.set_xticks(func_ids)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Strategy Recommendation Matrix
    ax = axes[1, 0]
    strategy_scores = []
    for i in range(8):
        # Score: high volatility→exploration, high best→exploitation, positive trend→refinement
        score = (volatilities[i] * 0.3 - best_values[i] * 0.5 + trends[i] * 0.2) / 10
        strategy_scores.append(score)
    
    colors_strategy = []
    labels_strategy = []
    for score in strategy_scores:
        if score < -0.3:
            colors_strategy.append('blue')  # Exploitation
            labels_strategy.append('Exploit')
        elif score > 0.3:
            colors_strategy.append('orange')  # Exploration
            labels_strategy.append('Explore')
        else:
            colors_strategy.append('green')  # Refinement
            labels_strategy.append('Refine')
    
    bars = ax.bar(func_ids, strategy_scores, color=colors_strategy, alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Function ID')
    ax.set_ylabel('Strategy Score')
    ax.set_title('Recommended Optimization Strategy')
    ax.set_xticks(func_ids)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Exploitation'),
                      Patch(facecolor='green', alpha=0.6, label='Refinement'),
                      Patch(facecolor='orange', alpha=0.6, label='Exploration')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Plot 4: Dimension vs Volatility
    ax = axes[1, 1]
    ax.scatter(dimensions, volatilities, s=200, c=best_values, cmap='RdYlGn', 
              alpha=0.6, edgecolors='black', linewidth=2)
    for i, func_id in enumerate(func_ids):
        ax.annotate(f'F{func_id}', (dimensions[i], volatilities[i]), 
                   fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimensionality')
    ax.set_ylabel('Volatility')
    ax.set_title('Complexity vs Dimensionality')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("Hybrid PyTorch CNN + Decision Tree Optimizer")
    print("=" * 60)
