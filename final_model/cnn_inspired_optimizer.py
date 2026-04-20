"""
CNN-Inspired Bayesian Black-Box Optimizer
Final Model: Applies deep learning principles to function optimization

Core Architecture:
- Feature Extraction Layer: Analyze historical data patterns
- Hierarchical Refinement: Progressive zoom from broad to fine-grained
- Regularization: Prevent overfitting (aggressive exploitation)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean


class CNNInspiredOptimizer:
    """Hierarchical optimizer inspired by CNN feature extraction and pooling"""
    
    def __init__(self, initial_data, dims):
        """
        Initialize with week 1-5 data
        
        Args:
            initial_data: dict with 'inputs' and 'outputs' arrays
            dims: dimensionality of each function
        """
        self.dims = dims
        self.history_inputs = initial_data['inputs']
        self.history_outputs = initial_data['outputs']
        self.best_idx = np.argmax(self.history_outputs)
        self.best_point = self.history_inputs[self.best_idx]
        self.best_value = self.history_outputs[self.best_idx]
        
        # Layer metrics for decision-making
        self.volatility = self._compute_volatility()
        self.trend = self._compute_trend()
        self.recovery = self._compute_recovery()
    
    def _compute_volatility(self):
        """Measure output variance (like CNN sensitivity maps)"""
        return np.std(self.history_outputs)
    
    def _compute_trend(self):
        """Measure performance trend over weeks"""
        week_size = len(self.history_outputs) // 5
        recent_avg = np.mean(self.history_outputs[-week_size:])
        early_avg = np.mean(self.history_outputs[:week_size])
        return (recent_avg - early_avg) / (early_avg + 1e-6)
    
    def _compute_recovery(self):
        """Check if function recovering after dip"""
        if len(self.history_outputs) < 10:
            return False
        recent_5 = self.history_outputs[-5:]
        prev_5 = self.history_outputs[-10:-5]
        return np.mean(recent_5) > np.mean(prev_5)
    
    def layer_1_exploration(self, num_points=4):
        """
        Initial Convolution Layer: Broad exploration
        For highly volatile or unknown landscapes
        """
        points = []
        
        # Strategy 1: Random exploration in full space
        points.append(np.random.uniform(0, 1, self.dims))
        
        # Strategy 2: Exploration around best point (±0.3)
        neighbor = self.best_point + np.random.normal(0, 0.15, self.dims)
        points.append(np.clip(neighbor, 0, 1))
        
        # Strategy 3: Opposite corner exploration
        opposite = 1 - self.best_point + np.random.normal(0, 0.1, self.dims)
        points.append(np.clip(opposite, 0, 1))
        
        # Strategy 4: Extreme exploitation (if pool suggests)
        if num_points >= 4:
            extreme = self._find_local_extremes(top_k=2)[0]
            points.append(extreme)
        
        return np.array(points[:num_points])
    
    def layer_2_refinement(self, num_points=4):
        """
        Pooling Layer: Intermediate refinement
        For functions with identifiable trends
        """
        points = []
        
        # Strategy 1: Conservative neighborhood search (±0.2)
        radius = 0.2 * (1 - self.volatility / 100)  # Scale by volatility
        neighbor = self.best_point + np.random.uniform(-radius, radius, self.dims)
        points.append(np.clip(neighbor, 0, 1))
        
        # Strategy 2: Trend-following direction
        if self.trend > 0:  # Improving trend
            direction = self._estimate_gradient_direction()
            points.append(np.clip(self.best_point + 0.1 * direction, 0, 1))
        
        # Strategy 3: Cluster analysis refinement
        cluster_center = self._find_cluster_center(top_k=5)
        points.append(cluster_center)
        
        # Strategy 4: Recovery-aware point
        if self.recovery and num_points >= 4:
            recent_best = self._find_local_extremes(k_weeks=1)[0]
            points.append(recent_best)
        
        return np.array(points[:num_points])
    
    def layer_3_exploitation(self, num_points=4):
        """
        Fully Connected Layer: Fine-grained exploitation
        Maximum conservative refinement near identified optimum
        """
        points = []
        
        # Strategy 1: Micro-perturbation (±0.01)
        micro = self.best_point + np.random.normal(0, 0.005, self.dims)
        points.append(np.clip(micro, 0, 1))
        
        # Strategy 2: Orthogonal perturbations
        for _ in range(min(2, num_points - 1)):
            orth = self.best_point.copy()
            idx = np.random.randint(0, self.dims)
            orth[idx] += np.random.normal(0, 0.01)
            points.append(np.clip(orth, 0, 1))
        
        # Strategy 3: Ridge exploration (±0.005 perpendicular to best)
        if num_points >= 4:
            ridge = self.best_point + np.random.normal(0, 0.003, self.dims)
            points.append(np.clip(ridge, 0, 1))
        
        return np.array(points[:num_points])
    
    def select_query_layer(self):
        """Decide which layer based on function characteristics"""
        # High volatility → need Layer 1 (exploration)
        if self.volatility > 0.3:
            return 1
        
        # Plateau or recovering → Layer 2 (refinement)
        if abs(self.trend) < 0.05 and self.recovery:
            return 2
        
        # Stable high performance → Layer 3 (exploitation)
        if self.best_value > 0.7 and self.volatility < 0.15:
            return 3
        
        # Default: Layer 2 (balanced)
        return 2
    
    def generate_queries(self, num_points):
        """Generate next batch of queries"""
        layer = self.select_query_layer()
        
        if layer == 1:
            return self.layer_1_exploration(num_points)
        elif layer == 2:
            return self.layer_2_refinement(num_points)
        else:
            return self.layer_3_exploitation(num_points)
    
    def _find_local_extremes(self, top_k=1, k_weeks=None):
        """Find local extremes in search space"""
        indices = np.argsort(self.history_outputs)[-top_k:]
        return self.history_inputs[indices]
    
    def _estimate_gradient_direction(self):
        """Estimate improvement direction from recent history"""
        recent = self.history_inputs[-5:] if len(self.history_inputs) >= 5 else self.history_inputs
        weights = np.arange(1, len(recent) + 1)
        direction = np.average(recent, axis=0, weights=weights) - self.best_point
        norm = np.linalg.norm(direction)
        return direction / (norm + 1e-6)
    
    def _find_cluster_center(self, top_k=5):
        """Find center of top-k performers"""
        indices = np.argsort(self.history_outputs)[-top_k:]
        return np.mean(self.history_inputs[indices], axis=0)
