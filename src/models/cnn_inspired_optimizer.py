"""
Week 6 Deep Learning-Inspired Model: CNN-Based Hierarchical Optimization

Applies CNN principles to BBO:
- Progressive feature extraction: coarse-to-fine refinement (like CNN pooling layers)
- Multi-scale processing: examine landscape at multiple resolutions
- Depth-efficiency trade-off: balance between exploration depth and overfitting risk
- Regularization: avoid aggressive exploitation on limited data (215 samples)
- Hierarchical learning: build understanding progressively from broad to specific

Theory:
CNN convolutions extract features at multiple abstraction levels. Similarly, we can
think of our optimization as progressively refining our understanding:
  - Layer 1 (W1-W2): Broad exploration, identify general landscape shape
  - Layer 2 (W3-W4): Intermediate exploration, locate promising regions
  - Layer 3 (W5): First exploitation attempt, test plateau boundaries
  - Layer 4 (W6): Refined exploitation, adaptive refinement based on learned features

This mirrors LeNet's breakthrough: building complex understanding from simple building blocks.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats.qmc import LatinHypercube

# Historical data: W1-W5 (15 data points total per function)
historical_data = {
    1: {
        'queries': [
            [0.250000, 0.750000],
            [0.050000, 0.050000],
            [0.754891, 0.704403],
            [0.374540, 0.950714],
            [0.929616, 0.316376],
        ],
        'results': [2.6065864278618756e-96, 7.570914060942952e-193, -5.384584177282445e-16, 
                   -1.560646704467778e-117, 3.4416015849706167e-131],
    },
    2: {
        'queries': [
            [0.750000, 0.250000],
            [0.500000, 0.500000],
            [0.686831, 0.530211],
            [0.173199, 0.159866],
            [0.984082, 0.997991],
        ],
        'results': [0.3691787538388598, 0.8473573729146894, 0.4074279061230939,
                   -0.05807400895675094, 0.053778481722633775],
    },
    3: {
        'queries': [
            [0.333333, 0.666667, 0.500000],
            [0.350000, 0.650000, 0.500000],
            [0.039713, 0.302029, 0.315311],
            [0.594963, 0.644959, 0.529293],
            [0.094455, 0.311399, 0.225967],
        ],
        'results': [-0.010251690931823796, -0.010450162716101937, -0.07882847061831176,
                   -0.012318067554316293, -0.13592439842996926],
    },
    4: {
        'queries': [
            [0.200000, 0.800000, 0.400000, 0.600000],
            [0.800000, 0.200000, 0.600000, 0.400000],
            [0.728602, 0.982928, 0.708406, 0.027707],
            [0.208588, 0.216178, 0.533292, 0.773294],
            [0.674055, 0.965114, 0.741781, 0.048580],
        ],
        'results': [-13.072131637188551, -13.072131637188551, -28.648038812076084,
                   -12.607647357899442, -27.440890417764923],
    },
    5: {
        'queries': [
            [0.700000, 0.300000, 0.600000, 0.200000],
            [0.720000, 0.280000, 0.580000, 0.220000],
            [0.014688, 0.641578, 0.349456, 0.493352],
            [0.033484, 0.654876, 0.337950, 0.480625],
            [0.000000, 0.653906, 0.374032, 0.519541],
        ],
        'results': [5.273302329600012, 4.049267429988913, 34.98323399644939,
                   32.96599170726208, 25.575607090129246],
    },
    6: {
        'queries': [
            [0.200000, 0.400000, 0.600000, 0.800000, 0.500000],
            [0.800000, 0.600000, 0.400000, 0.200000, 0.500000],
            [0.575333, 0.108777, 0.034359, 0.840559, 0.517247],
            [0.543673, 0.089201, 0.036835, 0.833754, 0.496370],
            [0.447812, 0.116655, 0.108676, 0.805596, 0.481036],
        ],
        'results': [-0.6995639652538725, -1.9119879535617619, -1.552441674550123,
                   -1.4792010945616396, -1.293746931550967],
    },
    7: {
        'queries': [
            [0.150000, 0.350000, 0.550000, 0.750000, 0.950000, 0.450000],
            [0.250000, 0.400000, 0.500000, 0.700000, 0.850000, 0.500000],
            [0.102635, 0.201553, 0.788679, 0.155646, 0.990262, 0.833759],
            [0.109346, 0.179923, 0.776208, 0.147628, 0.987626, 0.850870],
            [0.070161, 0.171326, 0.805916, 0.183311, 0.953336, 0.821749],
        ],
        'results': [0.11959165710190967, 0.14129996220103783, 0.219690205078482,
                   0.22895976507696808, 0.19344909329957222],
    },
    8: {
        'queries': [
            [0.125000, 0.250000, 0.375000, 0.500000, 0.625000, 0.750000, 0.875000, 0.437500],
            [0.150000, 0.300000, 0.400000, 0.480000, 0.600000, 0.700000, 0.850000, 0.450000],
            [0.018659, 0.622726, 0.428889, 0.224671, 0.701438, 0.385308, 0.247735, 0.172798],
            [0.000000, 0.623865, 0.436282, 0.188387, 0.710042, 0.358950, 0.212939, 0.208709],
            [0.235697, 0.815314, 0.215750, 0.128421, 0.651928, 0.386742, 0.366773, 0.147227],
        ],
        'results': [8.694471875, 8.73765, 9.4488988470416, 9.4329653859419, 9.3980882498781],
    },
}

dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

class HierarchicalOptimizer:
    """
    CNN-Inspired Hierarchical Optimizer
    
    Applies deep learning concepts to BBO:
    1. Feature Extraction: Identify landscape characteristics from historical data
    2. Progressive Refinement: Coarse-to-fine search (like pooling layers)
    3. Regularization: Balance exploration/exploitation to avoid overfitting
    4. Depth Management: Use progressively deeper analysis without losing efficiency
    """
    
    def __init__(self, func_id):
        self.func_id = func_id
        self.dim = dims[func_id]
        self.queries = np.array(historical_data[func_id]['queries'])
        self.results = np.array(historical_data[func_id]['results'])
        
    def extract_landscape_features(self):
        """
        Extract features from landscape (analogous to CNN feature maps):
        - Trend: directional preference
        - Volatility: sensitivity to small changes
        - Locality: how clustered are good solutions
        """
        # Recent trend: are we improving or declining?
        trend = self.results[-1] - self.results[-2]
        
        # Volatility: range of results
        volatility = np.std(self.results)
        
        # Locality: distance between best and current queries
        best_idx = np.argmax(self.results)
        current_dist = np.linalg.norm(self.queries[-1] - self.queries[best_idx])
        
        return {
            'trend': trend,
            'volatility': volatility,
            'locality': current_dist,
            'best_idx': best_idx,
            'best_value': self.results[best_idx],
        }
    
    def generate_layer1_candidates(self, n=100):
        """
        Layer 1: Broad exploration (like conv1 in LeNet)
        Generate diverse candidates across input space
        """
        sampler = LatinHypercube(d=self.dim, seed=100 + self.func_id)
        candidates = sampler.random(n)
        return candidates
    
    def generate_layer2_candidates(self, best_idx, n=50):
        """
        Layer 2: Intermediate refinement (like conv2 in LeNet)
        Focus around best-known point with adaptive radius
        """
        features = self.extract_landscape_features()
        best_point = self.queries[best_idx]
        
        # Adaptive radius: smaller if volatile (reduces overfitting risk)
        base_radius = 0.3
        radius = base_radius / (1 + features['volatility'])
        
        # Generate candidates in neighborhood
        candidates = []
        for _ in range(n):
            point = best_point + np.random.normal(0, radius, self.dim)
            point = np.clip(point, 0, 1)
            candidates.append(point)
        
        return np.array(candidates)
    
    def generate_layer3_candidates(self, best_idx, n=30):
        """
        Layer 3: Fine-grained refinement (like pooling + FC layers)
        Ultra-conservative micro-exploitation around best point
        """
        best_point = self.queries[best_idx]
        
        # Micro-refinement with ultra-conservative perturbations
        candidates = []
        for _ in range(n):
            point = best_point + np.random.uniform(-0.01, 0.01, self.dim)
            point = np.clip(point, 0, 1)
            candidates.append(point)
        
        return np.array(candidates)
    
    def score_candidates(self, candidates, layer):
        """
        Score candidates based on layer (analogous to activation functions):
        - Layer 1: Prefer diverse candidates far from history
        - Layer 2: Balance diversity with proximity to best
        - Layer 3: Prefer candidates very close to best
        """
        scores = np.zeros(len(candidates))
        
        for i, candidate in enumerate(candidates):
            if layer == 1:
                # Layer 1: Maximize distance from all historical queries
                distances = np.linalg.norm(self.queries - candidate, axis=1)
                scores[i] = np.min(distances)  # Prefer isolation
            
            elif layer == 2:
                # Layer 2: Balance distance from worst + proximity to best
                best_idx = np.argmax(self.results)
                dist_best = np.linalg.norm(self.queries[best_idx] - candidate)
                scores[i] = -dist_best  # Prefer proximity to best
            
            elif layer == 3:
                # Layer 3: Strong preference for best-point neighborhood
                best_idx = np.argmax(self.results)
                dist_best = np.linalg.norm(self.queries[best_idx] - candidate)
                scores[i] = -dist_best ** 2  # Quadratic penalty for distance
        
        return scores
    
    def generate_week6_query(self):
        """
        Generate Week 6 query using hierarchical CNN-inspired approach
        """
        features = self.extract_landscape_features()
        
        # Adaptive layer selection based on function characteristics
        if features['trend'] > 0:  # Improving
            # Continue Layer 2 refinement
            candidates = self.generate_layer2_candidates(features['best_idx'], n=50)
            layer = 2
        elif features['volatility'] > 1.0:  # High volatility (non-linear)
            # Return to Layer 1 broad exploration
            candidates = self.generate_layer1_candidates(n=100)
            layer = 1
        else:  # Steady or declining
            # Fine-grained Layer 3 exploitation
            candidates = self.generate_layer3_candidates(features['best_idx'], n=50)
            layer = 3
        
        # Score candidates
        scores = self.score_candidates(candidates, layer)
        best_idx = np.argmax(scores)
        
        return np.round(candidates[best_idx], 6), {
            'layer': layer,
            'features': features,
            'score': scores[best_idx],
        }


def generate_week6_queries():
    """Generate Week 6 queries using hierarchical CNN-inspired optimizer"""
    week6_queries = {}
    strategy_notes = {}
    
    print("\n" + "="*100)
    print("WEEK 6 QUERY GENERATION: CNN-INSPIRED HIERARCHICAL OPTIMIZATION")
    print("="*100 + "\n")
    
    for func_id in range(1, 9):
        optimizer = HierarchicalOptimizer(func_id)
        query, metadata = optimizer.generate_week6_query()
        
        week6_queries[func_id] = query
        strategy_notes[func_id] = metadata
        
        layer_name = {1: "Layer 1 (Broad Exploration)", 
                     2: "Layer 2 (Intermediate Refinement)", 
                     3: "Layer 3 (Fine-Grained Exploitation)"}[metadata['layer']]
        
        print(f"F{func_id}: {layer_name}")
        print(f"  Query: {query}")
        print(f"  Score: {metadata['score']:.4f}")
        print(f"  Best value so far: {metadata['features']['best_value']:.6e}")
        print(f"  Recent trend: {metadata['features']['trend']:+.6e}\n")
    
    return week6_queries, strategy_notes


if __name__ == "__main__":
    queries, notes = generate_week6_queries()
    
    print("="*100)
    print("WEEK 6 QUERIES (CNN-INSPIRED HIERARCHICAL)")
    print("="*100)
    for func_id in range(1, 9):
        print(f"F{func_id}: {'-'.join(f'{x:.6f}' for x in queries[func_id])}")
