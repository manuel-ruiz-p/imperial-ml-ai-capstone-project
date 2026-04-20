"""
Week 6 Results Analysis & Week 7 Strategy
Adaptive Model Selection with Hyperparameter Tuning

Results Summary:
F1: -2.736e-103 (essentially 0)
F2: -0.0301 (declined from 0.054)
F3: -0.00684 (stable negative)
F4: -8.197 (volatile negative)
F5: 79.327 (BREAKTHROUGH! Best overall)
F6: -1.808 (worse than expected)
F7: 0.3704 (confirmed trend improvement)
F8: 7.416 (slight plateau decline)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

# ============================================================================
# WEEK 6 RESULTS & HISTORICAL DATA
# ============================================================================

class Week6DataCollector:
    """Compile all historical data through Week 6"""
    
    def __init__(self):
        # Initialize data dictionary for all weeks
        self.data = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        self.function_dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load data from previous weeks"""
        # Week 1-5 data (from previous submissions)
        week1_outputs = {
            1: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            2: np.array([0.0547, 0.8474, 0.1453, 0.5381, 0.0705]),
            3: np.array([-0.0226, -0.0031, 0.0225, -0.0104, 0.0049]),
            4: np.array([-0.3869, -6.7214, -14.1254, -20.5294, -9.6373]),
            5: np.array([11.4566, 22.5647, 47.6899, 24.983, 34.9832]),
            6: np.array([-0.9662, -0.6831, -0.7506, -0.6996, -0.8526]),
            7: np.array([0.1286, 0.1697, 0.2290, 0.2111, 0.2289]),
            8: np.array([9.4489, 9.5219, 9.4324, 9.4271, 9.4489]),
        }
        
        # Week 6 results
        week6_outputs = {
            1: np.array([-2.736e-103]),
            2: np.array([-0.0301]),
            3: np.array([-0.00684]),
            4: np.array([-8.197]),
            5: np.array([79.327]),
            6: np.array([-1.808]),
            7: np.array([0.3704]),
            8: np.array([7.416]),
        }
        
        # Combine weeks 1-5 and add week 6
        for func_id in range(1, 9):
            combined = np.concatenate([week1_outputs[func_id], week6_outputs[func_id]])
            self.data[func_id] = combined
    
    def get_function_data(self, func_id: int) -> np.ndarray:
        """Get all historical outputs for a function"""
        return self.data[func_id]
    
    def get_statistics(self, func_id: int) -> Dict[str, float]:
        """Compute comprehensive statistics"""
        outputs = self.data[func_id]
        return {
            'mean': np.mean(outputs),
            'std': np.std(outputs),
            'min': np.min(outputs),
            'max': np.max(outputs),
            'range': np.max(outputs) - np.min(outputs),
            'variance': np.var(outputs),
            'median': np.median(outputs),
            'skewness': (np.mean(outputs) - np.median(outputs)) / (np.std(outputs) + 1e-8),
            'cv': np.std(outputs) / (np.abs(np.mean(outputs)) + 1e-8),  # Coefficient of variation
        }
    
    def get_trend_analysis(self, func_id: int) -> Dict[str, Any]:
        """Analyze trends across weeks"""
        outputs = self.data[func_id]
        
        # Week-by-week averages
        weekly_means = [np.mean(outputs[i*5:(i+1)*5]) for i in range(6)]
        
        # Trend direction
        recent_trend = weekly_means[-1] - weekly_means[-2] if len(weekly_means) > 1 else 0
        long_term_trend = weekly_means[-1] - weekly_means[0] if len(weekly_means) > 1 else 0
        
        return {
            'weekly_means': weekly_means,
            'recent_trend': recent_trend,
            'long_term_trend': long_term_trend,
            'trend_direction': 'improving' if recent_trend > 0 else 'declining',
            'stability': 1.0 - (np.std(np.diff(weekly_means)) / (np.std(weekly_means) + 1e-8)),
        }

# ============================================================================
# WEEK 6 PERFORMANCE EVALUATION
# ============================================================================

class Week6Evaluation:
    """Evaluate Week 6 performance vs expectations"""
    
    def __init__(self):
        self.collector = Week6DataCollector()
        self.results = {}
    
    def evaluate_all(self) -> Dict[int, Dict]:
        """Comprehensive evaluation for all functions"""
        for func_id in range(1, 9):
            outputs = self.collector.get_function_data(func_id)
            week6_value = outputs[-1]  # Last entry is W6
            week5_values = outputs[-2]  # W5 value
            
            stats = self.collector.get_statistics(func_id)
            trend = self.collector.get_trend_analysis(func_id)
            
            # Evaluate success
            improvement = week6_value - week5_values
            improvement_pct = (improvement / (np.abs(week5_values) + 1e-8)) * 100
            
            self.results[func_id] = {
                'week6_value': week6_value,
                'week5_value': week5_values,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'is_improvement': improvement > 0,
                'statistics': stats,
                'trend': trend,
                'all_values': outputs,
            }
        
        return self.results

# ============================================================================
# ADAPTIVE MODEL SELECTION STRATEGY
# ============================================================================

class AdaptiveModelSelector:
    """Select best model(s) for each function based on characteristics"""
    
    def __init__(self):
        self.evaluator = Week6Evaluation()
        self.results = self.evaluator.evaluate_all()
    
    def select_models_for_function(self, func_id: int) -> Dict[str, Any]:
        """
        Choose optimal models based on function characteristics
        
        Strategy:
        - Low volatility, improving trend → Trend-following (Linear, Ridge)
        - High volatility, improving → Ensemble (Random Forest, Gradient Boosting)
        - Chaotic/non-linear → Deep Learning (Neural Networks)
        - Known structure → SVM with kernel selection
        - Need interpretability → Decision Trees
        """
        
        func_results = self.results[func_id]
        stats = func_results['statistics']
        trend = func_results['trend']
        
        cv = stats['cv']  # Coefficient of variation
        std = stats['std']
        trend_status = trend['trend_direction']
        stability = trend['stability']
        
        models = {}
        rationale = {}
        
        # Function-specific model selection
        
        if func_id == 1:  # Flat/Noise floor
            models = {
                'constant': ('Constant', {'value': np.mean(func_results['all_values'])}),
                'linear': ('LinearRegression', {}),
            }
            rationale[1] = "F1 is essentially noise (~0). Model: Constant baseline + Linear for robustness."
        
        elif func_id == 2:  # High volatility, recovery pattern
            models = {
                'ensemble': ('RandomForest', {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 2}),
                'svm': ('SVR', {'kernel': 'rbf', 'gamma': 'auto', 'C': 10}),
                'nn': ('MLP', {'hidden_layer_sizes': (64, 32), 'alpha': 0.1, 'learning_rate': 'adaptive'}),
            }
            rationale[2] = f"F2 shows volatile recovery. CV={cv:.2f}. Model: Ensemble (RF) for non-linearity + SVM + NN."
        
        elif func_id == 3:  # Stable negative
            models = {
                'linear': ('Ridge', {'alpha': 1.0}),
                'tree': ('DecisionTree', {'max_depth': 4, 'min_samples_leaf': 2}),
                'svr': ('SVR', {'kernel': 'linear', 'C': 1.0}),
            }
            rationale[3] = f"F3 is stable but negative. Std={std:.4f}. Model: Linear (Ridge) + DT for interpretability."
        
        elif func_id == 4:  # Highly chaotic
            models = {
                'gb': ('GradientBoosting', {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3}),
                'nn_large': ('MLP', {'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.01, 'early_stopping': True}),
                'svm_poly': ('SVR', {'kernel': 'poly', 'degree': 3, 'C': 100}),
            }
            rationale[4] = f"F4 is chaotic. Std={std:.2f}. Model: Gradient Boosting + Deep NN + Polynomial SVM."
        
        elif func_id == 5:  # Elite performer, improving rapidly
            models = {
                'nn_optimized': ('MLP', {'hidden_layer_sizes': (256, 128, 64), 'beta_1': 0.9, 'beta_2': 0.999, 'alpha': 0.001}),
                'ensemble_boosted': ('XGBRegressor', {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6}),
                'bayesian': ('BayesianRidge', {'n_iter': 1000, 'alpha_1': 1e-6}),
            }
            rationale[5] = f"F5 BREAKTHROUGH (79.3)! Improving rapidly. Model: Deep NN + Boosting + Bayesian for uncertainty."
        
        elif func_id == 6:  # Negative, moderate volatility
            models = {
                'ensemble': ('RandomForest', {'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 1}),
                'nn': ('MLP', {'hidden_layer_sizes': (64, 32), 'alpha': 0.05, 'learning_rate_init': 0.001}),
                'ridge': ('Ridge', {'alpha': 0.1}),
            }
            rationale[6] = f"F6 negative with CV={cv:.2f}. Model: RF + NN + Ridge (ensemble diversity)."
        
        elif func_id == 7:  # Improving, stable - BEST CASE
            models = {
                'linear': ('Ridge', {'alpha': 0.1}),
                'bayesian': ('BayesianRidge', {'n_iter': 500}),
                'nn_lite': ('MLP', {'hidden_layer_sizes': (32,), 'alpha': 0.1, 'learning_rate': 'optimal'}),
            }
            rationale[7] = f"F7 IDEAL: Improving (trend up), stable. Model: Linear (Ridge) + Bayesian for confidence intervals."
        
        elif func_id == 8:  # Plateau region, high-dimensional
            models = {
                'ensemble': ('GradientBoosting', {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4}),
                'svm_rbf': ('SVR', {'kernel': 'rbf', 'gamma': 'scale', 'C': 50}),
                'nn_deep': ('MLP', {'hidden_layer_sizes': (128, 64, 32, 16), 'alpha': 0.01}),
            }
            rationale[8] = f"F8 plateau in 8D. Model: GB + SVM RBF + Deep NN for high-dimensional non-linearity."
        
        return {
            'func_id': func_id,
            'models': models,
            'rationale': rationale.get(func_id, ""),
            'characteristics': {
                'cv': cv,
                'std': std,
                'trend': trend_status,
                'stability': stability,
            }
        }
    
    def get_all_model_selections(self) -> Dict:
        """Get model selections for all functions"""
        return {func_id: self.select_models_for_function(func_id) for func_id in range(1, 9)}

# ============================================================================
# HYPERPARAMETER TUNING ENGINE
# ============================================================================

class HyperparameterTuner:
    """Systematic hyperparameter tuning with multiple strategies"""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, func_id: int):
        self.X_train = X_train
        self.y_train = y_train
        self.func_id = func_id
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X_train)
        self.best_models = {}
        self.tuning_results = {}
    
    def tune_neural_network(self) -> Tuple[MLPRegressor, Dict]:
        """Grid search for neural network hyperparameters"""
        param_grid = {
            'hidden_layer_sizes': [(32,), (64, 32), (128, 64), (128, 64, 32)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh'],
        }
        
        mlp = MLPRegressor(max_iter=500, early_stopping=True, validation_fraction=0.1)
        grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_scaled, self.y_train)
        
        return grid_search.best_estimator_, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
        }
    
    def tune_svm(self) -> Tuple[SVR, Dict]:
        """Random search for SVM hyperparameters"""
        param_dist = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': np.logspace(-2, 3, 10),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 10)),
            'degree': [2, 3, 4, 5],
        }
        
        svm = SVR()
        random_search = RandomizedSearchCV(svm, param_dist, n_iter=20, cv=3, n_jobs=-1, 
                                          scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(self.X_scaled, self.y_train)
        
        return random_search.best_estimator_, {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
        }
    
    def tune_random_forest(self) -> Tuple[RandomForestRegressor, Dict]:
        """Grid search for Random Forest hyperparameters"""
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_scaled, self.y_train)
        
        return grid_search.best_estimator_, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }
    
    def tune_gradient_boosting(self) -> Tuple[GradientBoostingRegressor, Dict]:
        """Grid search for Gradient Boosting"""
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.8, 0.9, 1.0],
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_scaled, self.y_train)
        
        return grid_search.best_estimator_, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }
    
    def tune_bayesian_regression(self) -> Tuple[BayesianRidge, Dict]:
        """Bayesian optimization for Ridge regression"""
        param_dist = {
            'alpha_1': np.logspace(-8, -4, 10),
            'alpha_2': np.logspace(-8, -4, 10),
            'lambda_1': np.logspace(-8, -4, 10),
            'lambda_2': np.logspace(-8, -4, 10),
            'n_iter': [100, 300, 500, 1000],
        }
        
        br = BayesianRidge()
        random_search = RandomizedSearchCV(br, param_dist, n_iter=15, cv=3, 
                                          scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(self.X_scaled, self.y_train)
        
        return random_search.best_estimator_, {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
        }
    
    def run_all_tuning(self) -> Dict:
        """Execute all tuning methods"""
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING FOR FUNCTION {self.func_id}")
        print(f"{'='*70}")
        print(f"Training samples: {len(self.y_train)}, Features: {self.X_train.shape[1]}\n")
        
        results = {}
        
        try:
            print("Tuning Neural Network...")
            results['neural_network'] = self.tune_neural_network()
        except Exception as e:
            print(f"  ⚠ NN tuning failed: {e}")
        
        try:
            print("Tuning SVM...")
            results['svm'] = self.tune_svm()
        except Exception as e:
            print(f"  ⚠ SVM tuning failed: {e}")
        
        try:
            print("Tuning Random Forest...")
            results['random_forest'] = self.tune_random_forest()
        except Exception as e:
            print(f"  ⚠ RF tuning failed: {e}")
        
        try:
            print("Tuning Gradient Boosting...")
            results['gradient_boosting'] = self.tune_gradient_boosting()
        except Exception as e:
            print(f"  ⚠ GB tuning failed: {e}")
        
        try:
            print("Tuning Bayesian Ridge...")
            results['bayesian_ridge'] = self.tune_bayesian_regression()
        except Exception as e:
            print(f"  ⚠ Bayesian tuning failed: {e}")
        
        self.tuning_results = results
        return results

# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

def main():
    """Execute comprehensive Week 6-7 analysis"""
    
    # 1. Week 6 Evaluation
    print("\n" + "="*70)
    print("WEEK 6 RESULTS EVALUATION")
    print("="*70)
    
    evaluator = Week6Evaluation()
    results = evaluator.evaluate_all()
    
    for func_id in range(1, 9):
        r = results[func_id]
        trend_arrow = "↑" if r['is_improvement'] else "↓"
        print(f"\nF{func_id}: {r['week6_value']:12.6f} {trend_arrow} "
              f"(vs W5: {r['week5_value']:12.6f}, Δ={r['improvement']:+10.6f})")
        print(f"  Volatility: {r['statistics']['std']:8.4f}, Mean: {r['statistics']['mean']:10.4f}, "
              f"CV: {r['statistics']['cv']:6.2f}")
        print(f"  Trend: {r['trend']['trend_direction']}, Stability: {r['trend']['stability']:.3f}")
    
    # 2. Model Selection
    print("\n" + "="*70)
    print("ADAPTIVE MODEL SELECTION STRATEGY")
    print("="*70)
    
    selector = AdaptiveModelSelector()
    selections = selector.get_all_model_selections()
    
    for func_id, selection in selections.items():
        print(f"\nF{func_id}: {selection['rationale']}")
        print(f"  Selected Models: {', '.join([name for name, _ in selection['models'].values()])}")
    
    return {
        'evaluation': results,
        'model_selections': selections,
    }

if __name__ == "__main__":
    analysis = main()
    print("\n" + "="*70)
    print("Week 6-7 Analysis Complete")
    print("="*70)
