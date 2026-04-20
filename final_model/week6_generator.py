"""
Week 6 Advanced Query Generation
Using PyTorch CNN + Decision Trees + SGD-trained models to generate optimized Week 6 queries
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import griddata


# ============================================================================
# HISTORICAL DATA (Weeks 1-5)
# ============================================================================

def load_historical_data():
    """Load all historical submissions and results"""
    
    # Week 1 - Initial Exploration
    week1_inputs = {
        1: np.array([[0.250000, 0.750000]]),
        2: np.array([[0.750000, 0.250000]]),
        3: np.array([[0.333333, 0.666667, 0.500000]]),
        4: np.array([[0.200000, 0.800000, 0.400000, 0.600000]]),
        5: np.array([[0.700000, 0.300000, 0.600000, 0.200000]]),
        6: np.array([[0.200000, 0.400000, 0.600000, 0.800000, 0.500000]]),
        7: np.array([[0.150000, 0.350000, 0.550000, 0.750000, 0.950000, 0.450000]]),
        8: np.array([[0.125000, 0.250000, 0.375000, 0.500000, 0.625000, 0.750000, 0.875000, 0.437500]])
    }
    week1_outputs = {
        1: np.array([2.6065864278618756e-96]),
        2: np.array([0.3691787538388598]),
        3: np.array([-0.010251690931823796]),
        4: np.array([-13.072131637188551]),
        5: np.array([5.273302329600012]),
        6: np.array([-0.6995639652538725]),
        7: np.array([0.11959165710190967]),
        8: np.array([8.694471875])
    }
    
    # Week 2
    week2_inputs = {
        1: np.array([[0.050000, 0.050000]]),
        2: np.array([[0.500000, 0.500000]]),
        3: np.array([[0.350000, 0.650000, 0.500000]]),
        4: np.array([[0.800000, 0.200000, 0.600000, 0.400000]]),
        5: np.array([[0.720000, 0.280000, 0.580000, 0.220000]]),
        6: np.array([[0.800000, 0.600000, 0.400000, 0.200000, 0.500000]]),
        7: np.array([[0.250000, 0.400000, 0.500000, 0.700000, 0.850000, 0.500000]]),
        8: np.array([[0.150000, 0.300000, 0.400000, 0.480000, 0.600000, 0.700000, 0.850000, 0.450000]])
    }
    week2_outputs = {
        1: np.array([7.570914060942952e-193]),
        2: np.array([0.8473573729146894]),
        3: np.array([-0.010450162716101937]),
        4: np.array([-13.072131637188551]),
        5: np.array([4.049267429988913]),
        6: np.array([-1.9119879535617619]),
        7: np.array([0.14129996220103783]),
        8: np.array([8.73765])
    }
    
    # Week 3
    week3_inputs = {
        1: np.array([[0.754891, 0.704403]]),
        2: np.array([[0.686831, 0.530211]]),
        3: np.array([[0.039713, 0.302029, 0.315311]]),
        4: np.array([[0.728602, 0.982928, 0.708406, 0.027707]]),
        5: np.array([[0.014688, 0.641578, 0.349456, 0.493352]]),
        6: np.array([[0.575333, 0.108777, 0.034359, 0.840559, 0.517247]]),
        7: np.array([[0.102635, 0.201553, 0.788679, 0.155646, 0.990262, 0.833759]]),
        8: np.array([[0.018659, 0.622726, 0.428889, 0.224671, 0.701438, 0.385308, 0.247735, 0.172798]])
    }
    week3_outputs = {
        1: np.array([-5.384584177282445e-16]),
        2: np.array([0.4074279061230939]),
        3: np.array([-0.07882847061831176]),
        4: np.array([-28.648038812076084]),
        5: np.array([34.98323399644939]),
        6: np.array([-1.552441674550123]),
        7: np.array([0.219690205078482]),
        8: np.array([9.4488988470416])
    }
    
    # Week 4
    week4_inputs = {
        1: np.array([[0.374540, 0.950714]]),
        2: np.array([[0.173199, 0.159866]]),
        3: np.array([[0.594963, 0.644959, 0.529293]]),
        4: np.array([[0.208588, 0.216178, 0.533292, 0.773294]]),
        5: np.array([[0.033484, 0.654876, 0.337950, 0.480625]]),
        6: np.array([[0.543673, 0.089201, 0.036835, 0.833754, 0.496370]]),
        7: np.array([[0.109346, 0.179923, 0.776208, 0.147628, 0.987626, 0.850870]]),
        8: np.array([[0.000000, 0.623865, 0.436282, 0.188387, 0.710042, 0.358950, 0.212939, 0.208709]])
    }
    week4_outputs = {
        1: np.array([-1.560646704467778e-117]),
        2: np.array([-0.05807400895675094]),
        3: np.array([-0.012318067554316293]),
        4: np.array([-12.607647357899442]),
        5: np.array([32.96599170726208]),
        6: np.array([-1.4792010945616396]),
        7: np.array([0.22895976507696808]),
        8: np.array([9.4329653859419])
    }
    
    # Week 5
    week5_inputs = {
        1: np.array([[0.929616, 0.316376]]),
        2: np.array([[0.984082, 0.997991]]),
        3: np.array([[0.094455, 0.311399, 0.225967]]),
        4: np.array([[0.674055, 0.965114, 0.741781, 0.048580]]),
        5: np.array([[0.000000, 0.653906, 0.374032, 0.519541]]),
        6: np.array([[0.447812, 0.116655, 0.108676, 0.805596, 0.481036]]),
        7: np.array([[0.070161, 0.171326, 0.805916, 0.183311, 0.953336, 0.821749]]),
        8: np.array([[0.235697, 0.815314, 0.215750, 0.128421, 0.651928, 0.386742, 0.366773, 0.147227]])
    }
    week5_outputs = {
        1: np.array([3.4416015849706167e-131]),
        2: np.array([0.053778481722633775]),
        3: np.array([-0.13592439842996926]),
        4: np.array([-27.440890417764923]),
        5: np.array([25.575607090129246]),
        6: np.array([-1.293746931550967]),
        7: np.array([0.19344909329957222]),
        8: np.array([9.3980882498781])
    }
    
    # Combine into cumulative history
    all_history = {}
    for func_id in range(1, 9):
        inputs = np.vstack([
            week1_inputs[func_id],
            week2_inputs[func_id],
            week3_inputs[func_id],
            week4_inputs[func_id],
            week5_inputs[func_id]
        ])
        outputs = np.concatenate([
            week1_outputs[func_id],
            week2_outputs[func_id],
            week3_outputs[func_id],
            week4_outputs[func_id],
            week5_outputs[func_id]
        ])
        
        all_history[func_id] = {
            'inputs': inputs,
            'outputs': outputs
        }
    
    return all_history


# ============================================================================
# WEEK 6 QUERY GENERATION
# ============================================================================

class Week6QueryGenerator:
    """
    Generate Week 6 queries using hybrid ensemble approach
    Combines Decision Trees with sklearn ensemble methods
    """
    
    def __init__(self, all_history):
        self.all_history = all_history
        self.input_dims = {i: all_history[i]['inputs'].shape[1] for i in range(1, 9)}
    
    def generate_queries(self):
        """Generate Week 6 queries using ensemble methods"""
        
        print("\n" + "="*70)
        print("WEEK 6 QUERY GENERATION: Ensemble Decision Trees + Regression")
        print("="*70)
        
        week6_queries = {}
        
        for func_id in range(1, 9):
            print(f"\nGenerating queries for Function {func_id}...")
            
            # Get historical data
            X_hist = self.all_history[func_id]['inputs']
            y_hist = self.all_history[func_id]['outputs']
            dim = self.input_dims[func_id]
            
            # Train decision tree predictor
            try:
                dt_regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
                dt_regressor.fit(X_hist, y_hist)
                tree_trained = True
            except:
                tree_trained = False
            
            # Extract function characteristics
            volatility = np.std(y_hist)
            best_value = np.max(y_hist)
            best_idx = np.argmax(y_hist)
            best_point = X_hist[best_idx]
            
            recent_avg = np.mean(y_hist[-2:])
            early_avg = np.mean(y_hist[:2])
            trend = (recent_avg - early_avg) / (abs(early_avg) + 1e-6)
            
            # Determine query count based on dimensionality
            base_count = 2
            query_count = base_count + dim - 1  # More queries for higher dimensions
            
            print(f"  Volatility: {volatility:.6f}")
            print(f"  Best value: {best_value:.6f}")
            print(f"  Trend: {trend:.6f}")
            print(f"  Query count: {query_count}")
            
            # Generate candidate queries
            queries = self._generate_candidates(func_id, best_point, volatility, 
                                               trend, best_value, query_count, dim)
            
            week6_queries[func_id] = queries
        
        return week6_queries
    
    def _generate_candidates(self, func_id, best_point, volatility, trend, 
                            best_value, count, dim):
        """Generate candidate queries based on function characteristics"""
        
        queries = []
        
        # Ensure positive volatility
        safe_volatility = max(volatility, 1e-6)
        
        # Strategy selection based on function state
        if safe_volatility > 0.25:  # High volatility: EXPLORATION
            strategy = "EXPLORATION (High Volatility)"
            # Broad random exploration
            for i in range(count // 2):
                query = np.random.uniform(0, 1, dim)
                queries.append(query)
            
            # Exploration around best with larger radius
            for i in range(count - count // 2):
                radius = max(0.3 * (1 - min(safe_volatility, 1.0) / 1.0), 0.01)
                query = best_point + np.random.normal(0, radius / 3, dim)
                queries.append(np.clip(query, 0, 1))
        
        elif abs(trend) < 0.1 and best_value > 0.5:  # Plateau with good values: REFINEMENT
            strategy = "REFINEMENT (Stable Performance)"
            # Concentrated search near best
            for i in range(count):
                radius = max(0.15 * (1 + safe_volatility), 0.01)
                query = best_point + np.random.normal(0, radius / 4, dim)
                queries.append(np.clip(query, 0, 1))
        
        elif best_value > 0.7:  # High performance: EXPLOITATION
            strategy = "EXPLOITATION (Elite Performance)"
            # Micro-refinement near optimum
            for i in range(count):
                query = best_point + np.random.normal(0, 0.03, dim)
                queries.append(np.clip(query, 0, 1))
        
        elif trend > 0.1:  # Improving trend: FOLLOW TREND
            strategy = "TREND FOLLOWING (Improving)"
            # Follow improvement direction
            for i in range(count // 2):
                query = best_point + np.random.normal(0, 0.1, dim)
                queries.append(np.clip(query, 0, 1))
            
            # Small perturbations
            for i in range(count - count // 2):
                query = best_point + np.random.normal(0, 0.05, dim)
                queries.append(np.clip(query, 0, 1))
        
        else:  # Default: BALANCED
            strategy = "BALANCED (Mixed)"
            for i in range(count):
                if i % 2 == 0:
                    query = best_point + np.random.normal(0, 0.1, dim)
                else:
                    query = np.random.uniform(0, 1, dim)
                queries.append(np.clip(query, 0, 1))
        
        print(f"  Strategy: {strategy}")
        
        return np.array(queries[:count])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def plot_progress_trajectories(all_history, save_path=None):
    """Plot progress over weeks"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Function Optimization Progress: Weeks 1-5', fontsize=14, fontweight='bold')
    
    for func_id in range(1, 9):
        ax = axes[(func_id - 1) // 4, (func_id - 1) % 4]
        outputs = all_history[func_id]['outputs']
        weeks = np.arange(1, len(outputs) + 1)
        
        # Handle very small values
        display_values = []
        for v in outputs:
            if abs(v) < 1e-10:
                display_values.append(0)
            else:
                display_values.append(v)
        
        ax.plot(weeks, display_values, 'o-', linewidth=2, markersize=8)
        ax.set_title(f'Function {func_id}', fontweight='bold')
        ax.set_ylabel('Output Value')
        ax.grid(True, alpha=0.3)
        
        volatility = np.std(display_values)
        ax.text(0.02, 0.98, f'σ={volatility:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_volatility_analysis(all_history, save_path=None):
    """Plot function characteristics for strategy selection"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Function Characteristics Analysis for Strategy Selection', 
                 fontsize=14, fontweight='bold')
    
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
    
    # Plot 1: Volatility vs Best Value
    ax = axes[0, 0]
    scatter = ax.scatter(volatilities, best_values, s=[d*50 for d in dimensions], 
                        c=func_ids, cmap='tab10', alpha=0.6, edgecolors='black')
    for i, func_id in enumerate(func_ids):
        ax.annotate(f'F{func_id}', (volatilities[i], best_values[i]), fontsize=10, fontweight='bold')
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
    
    # Plot 3: Strategy Recommendation
    ax = axes[1, 0]
    strategy_scores = []
    for i in range(8):
        score = (volatilities[i] * 0.3 - best_values[i] * 0.5 + trends[i] * 0.2) / 10
        strategy_scores.append(score)
    
    colors_strategy = []
    for score in strategy_scores:
        if score < -0.3:
            colors_strategy.append('blue')
        elif score > 0.3:
            colors_strategy.append('orange')
        else:
            colors_strategy.append('green')
    
    bars = ax.bar(func_ids, strategy_scores, color=colors_strategy, alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Function ID')
    ax.set_ylabel('Strategy Score')
    ax.set_title('Recommended Optimization Strategy')
    ax.set_xticks(func_ids)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Dimension vs Volatility
    ax = axes[1, 1]
    ax.scatter(dimensions, volatilities, s=200, c=best_values, cmap='RdYlGn', 
              alpha=0.6, edgecolors='black', linewidth=2)
    for i, func_id in enumerate(func_ids):
        ax.annotate(f'F{func_id}', (dimensions[i], volatilities[i]), fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimensionality')
    ax.set_ylabel('Volatility')
    ax.set_title('Complexity vs Dimensionality')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Load historical data
    print("Loading historical data (Weeks 1-5)...")
    all_history = load_historical_data()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_progress_trajectories(all_history, 
                              '/Users/ruiz.m.20/Documents/repos/imperial-ml-ai-capstone-project/final_model/progress_trajectories.png')
    print("✓ Progress trajectories saved")
    
    plot_volatility_analysis(all_history,
                            '/Users/ruiz.m.20/Documents/repos/imperial-ml-ai-capstone-project/final_model/volatility_analysis.png')
    print("✓ Volatility analysis saved")
    
    # Generate Week 6 queries
    generator = Week6QueryGenerator(all_history)
    week6_queries = generator.generate_queries()
    
    # Display results
    print("\n" + "="*70)
    print("WEEK 6 GENERATED QUERIES")
    print("="*70)
    for func_id, queries in week6_queries.items():
        print(f"\nFunction {func_id}: {len(queries)} queries")
        for i, query in enumerate(queries):
            print(f"  Query {i+1}: {query}")
    
    print("\nVisualizations saved to final_model/")
