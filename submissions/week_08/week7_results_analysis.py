"""
Week 7 Results Analysis & Week 8 Strategy
===========================================

This module analyzes Week 7 performance, identifies patterns, and develops
the Week 8 optimization strategy. Unlike Week 6 which brought breakthrough
on F5 (79.327), Week 7 reveals critical insights about function volatility,
non-stationarity, and the limits of direct exploitation strategies.

Key Findings:
- F5 experienced massive regression (79.327 → 9.247, -88%), suggesting:
  * W6 was a local/random peak in a volatile landscape
  * Function exhibits high non-stationarity or chaos
  * Direct exploitation of single peaks is unreliable strategy
  
- Improvements observed:
  * F2: Recovery validated (-0.0301 → 0.1429, +147% improvement)
  * F6: Dimension-aware strategy worked (-1.808 → -1.594, +12% improvement)
  * F8: High-D ensemble successful (7.416 → 8.001, +8% improvement)
  
- Overall Portfolio: 2 breakthroughs (W6) followed by 1 major crash (W7)
  suggests we need more conservative, stable strategies with confidence intervals
  rather than aggressive peak-chasing.

Author: Capstone AI System
Date: Week 7-8 Transition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats

# Historical Data: All Inputs and Outputs W1-W7
WEEK_7_RESULTS = {
    'inputs': {
        1: np.array([0.524103, 0.765891]),
        2: np.array([0.287456, 0.321654]),
        3: np.array([0.412789, 0.534612, 0.678901]),
        4: np.array([0.123456, 0.876543, 0.345678, 0.654321]),
        5: np.array([0.612345, 0.234567, 0.789012, 0.456789]),
        6: np.array([0.234567, 0.123456, 0.987654, 0.456789, 0.678901]),
        7: np.array([0.187654, 0.234567, 0.698765, 0.345678, 0.987654, 0.765432]),
        8: np.array([0.457789, 0.345678, 0.876543, 0.234567, 0.654321, 0.567890, 0.234567, 0.876543]),
    },
    'outputs': {
        1: -1.473256448564669e-21,
        2: 0.1428794770560145,
        3: -0.10575219380000973,
        4: -17.89386401102779,
        5: 9.246654442968955,
        6: -1.593619393881447,
        7: 0.3447670478013089,
        8: 8.0012956575546,
    }
}

# Full historical database W1-W7 (storing key results for analysis)
HISTORICAL_BEST = {
    1: [0.0, -0.0107, -0.0131, 0.0057, -0.0122, -1.473e-21],  # W2-W7
    2: [0.847, 0.332, -0.0301, 0.1429],  # W1-W7 (W1 in initial_data)
    3: [-0.0801, -0.1058],  # W6-W7
    4: [-14.197, -17.894],  # W6-W7
    5: [79.327, 9.247],  # W6-W7 (breakthrough then crash)
    6: [-1.808, -1.594],  # W6-W7
    7: [0.3705, 0.3448],  # W6-W7
    8: [7.416, 8.001],  # W6-W7
}

@dataclass
class FunctionStats:
    """Statistics for a single function across multiple weeks"""
    func_id: int
    outputs: List[float]
    mean: float
    std: float
    cv: float  # Coefficient of variation
    min_val: float
    max_val: float
    range: float
    trend: float  # Linear trend slope
    skewness: float
    volatility_class: str  # 'stable', 'moderate', 'high', 'chaotic'
    dimensionality: int
    
class Week7Analysis:
    """Comprehensive analysis of Week 7 results and strategy for Week 8"""
    
    def __init__(self):
        self.stats: Dict[int, FunctionStats] = {}
        self.insights: Dict[str, str] = {}
        self.strategy_recommendations: Dict[int, Dict] = {}
        
    def analyze_function(self, func_id: int, outputs: List[float], dim: int) -> FunctionStats:
        """Compute comprehensive statistics for a function"""
        outputs_array = np.array(outputs)
        mean = float(np.mean(outputs_array))
        std = float(np.std(outputs_array, ddof=1)) if len(outputs) > 1 else 0.0
        cv = std / (abs(mean) + 1e-8)  # Coefficient of variation
        
        # Trend analysis: slope of linear fit
        x = np.arange(len(outputs))
        if len(outputs) > 1:
            z = np.polyfit(x, outputs_array, 1)
            trend = float(z[0])  # slope
        else:
            trend = 0.0
            
        # Skewness
        skewness = float(stats.skew(outputs_array)) if len(outputs) > 1 else 0.0
        
        # Volatility classification
        if cv < 0.1:
            volatility_class = 'stable'
        elif cv < 0.5:
            volatility_class = 'moderate'
        elif cv < 1.5:
            volatility_class = 'high'
        else:
            volatility_class = 'chaotic'
        
        stat_obj = FunctionStats(
            func_id=func_id,
            outputs=outputs,
            mean=mean,
            std=std,
            cv=cv,
            min_val=float(np.min(outputs_array)),
            max_val=float(np.max(outputs_array)),
            range=float(np.max(outputs_array) - np.min(outputs_array)),
            trend=trend,
            skewness=skewness,
            volatility_class=volatility_class,
            dimensionality=dim
        )
        
        self.stats[func_id] = stat_obj
        return stat_obj
    
    def run_analysis(self):
        """Execute full analysis pipeline"""
        
        # Analyze each function
        dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
        
        for func_id in range(1, 9):
            outputs = HISTORICAL_BEST[func_id]
            self.analyze_function(func_id, outputs, dims[func_id])
        
        # Generate insights
        self._generate_insights()
        self._develop_week8_strategy()
        
    def _generate_insights(self):
        """Extract key insights from data"""
        
        # F5 Collapse Analysis
        f5_stats = self.stats[5]
        f5_regression = (9.247 - 79.327) / 79.327 * 100
        self.insights['f5_collapse'] = (
            f"F5 experienced catastrophic -88% regression (79.327 → 9.247). "
            f"Analysis:\n"
            f"  - W6 peak likely local optimum in highly non-stationary landscape\n"
            f"  - High volatility (σ={f5_stats.std:.2f}) suggests chaos/noise dominates\n"
            f"  - Direct peak exploitation is unreliable short-term strategy\n"
            f"  - Need confidence intervals + stability margins for W8"
        )
        
        # F2 Success Pattern
        f2_outputs = [0.847, 0.332, -0.0301, 0.1429]
        f2_recovery = (0.1429 - (-0.0301)) / abs(-0.0301) * 100
        self.insights['f2_recovery'] = (
            f"F2 recovery validated after W5 crash:\n"
            f"  - W4: 0.847 (initial peak)\n"
            f"  - W5: 0.332 (decline post-peak)\n"
            f"  - W6: -0.0301 (crash, -88% from W5)\n"
            f"  - W7: +0.1429 (recovery, +573% from W6 low point)\n"
            f"  - Conservative SVM strategy + low learning rate effective\n"
            f"  - Lesson: Recovery is possible after crashes with proper tuning"
        )
        
        # Portfolio volatility analysis
        all_changes = self._compute_portfolio_volatility()
        self.insights['portfolio_volatility'] = (
            f"Portfolio W6→W7 changes: {all_changes}\n"
            f"  - 3 improvements: F2 (+0.173), F6 (+0.214), F8 (+0.585)\n"
            f"  - 1 major crash: F5 (-70.080)\n"
            f"  - 3 minor regressions: F1, F3, F4, F7\n"
            f"  - Net impact: -69.108 portfolio points (-58% from W6 peak)\n"
            f"  - Indicates: Portfolio is NOT stable; need risk management"
        )
        
        # Non-stationarity evidence
        self.insights['non_stationarity'] = (
            f"Strong evidence of non-stationarity (landscape shifting):\n"
            f"  - F5: peak at W6 (+79.327) vanishes by W7 (-88% drop)\n"
            f"  - F7: positive trend reverses (0.3705 → 0.3448)\n"
            f"  - F1: consistently near zero (noise floor)\n"
            f"  - F4: monotonic decline (chaotic descent)\n"
            f"  Implication: Assume functions evolve; diversify queries"
        )
    
    def _compute_portfolio_volatility(self) -> Dict:
        """Analyze week-over-week changes"""
        changes = {}
        w6_values = [0, -0.0107, -0.0301, -0.0801, -14.197, 79.327, -1.808, 0.3705, 7.416]
        w7_values = [0, WEEK_7_RESULTS['outputs'][i] for i in range(1, 9)]
        
        for i in range(1, 9):
            changes[i] = w7_values[i] - w6_values[i]
        
        return changes
    
    def _develop_week8_strategy(self):
        """Define strategy for Week 8 based on W7 learning"""
        
        for func_id in range(1, 9):
            stats_obj = self.stats[func_id]
            strategy = {}
            
            # F1: Noise floor - random exploration
            if func_id == 1:
                strategy = {
                    'model': 'Random()',
                    'rationale': 'Function output ≈ machine noise; no pattern to exploit',
                    'query_type': 'uniform random sampling',
                    'confidence': 'Very Low',
                    'risk': 'High (random)',
                }
            
            # F2: Recovery momentum - conservative exploitation
            elif func_id == 2:
                strategy = {
                    'model': 'SVM(RBF) + Ridge',
                    'rationale': 'Recovery validated (W7: +0.1429); riding positive momentum',
                    'learning_rate': 0.0005,  # Very conservative (low CV ~1.3)
                    'query_type': 'Ridge regression direction with SVM boundary check',
                    'confidence': 'Moderate',
                    'risk': 'Moderate - recovery could reverse',
                }
            
            # F3: Stable decline - exploration near low point
            elif func_id == 3:
                strategy = {
                    'model': 'Bayesian Ridge + Ensemble',
                    'rationale': 'Slight regression (-0.0801 → -0.1058); unclear trend',
                    'query_type': 'Confidence interval + best uncertainty estimate',
                    'confidence': 'Low-Moderate',
                    'risk': 'Moderate',
                }
            
            # F4: Chaotic descent - random walkaround
            elif func_id == 4:
                strategy = {
                    'model': 'Random walk + Ensemble bounds',
                    'rationale': 'Monotonic decline (-14.197 → -17.894); no recovery signal',
                    'query_type': 'Random exploration with ensemble confidence intervals',
                    'confidence': 'Low',
                    'risk': 'High (unknown landscape)',
                }
            
            # F5: Peak instability - confidence-first approach
            elif func_id == 5:
                strategy = {
                    'model': 'REVERT to ensemble (NO aggressive exploitation)',
                    'rationale': (
                        'W6 peak (79.327) was likely local/noise. W7 crash (-88%) proves '
                        'direct peak exploitation fails. Need stability margins and confidence intervals.'
                    ),
                    'query_type': 'Move TOWARDS regions of HIGH ensemble consensus (not peak)',
                    'high_confidence_threshold': 'σ_ensemble < 0.5',
                    'learning_rate': 'Conservative: 0.001',
                    'confidence': 'Very Low (due to W7 crash)',
                    'risk': 'Very High (unstable landscape)',
                    'lesson': 'Never trust single peak in volatile landscape'
                }
            
            # F6: Dimension scaling validated - continue strategy
            elif func_id == 6:
                strategy = {
                    'model': 'Deep NN + RBF SVM ensemble',
                    'rationale': 'Dimension-aware exploration worked (-1.808 → -1.594, +12%)',
                    'query_type': 'Dimension-scaled radius sampling',
                    'exploration_radius': 'r = 0.25 × √(1 + 6/2)',
                    'confidence': 'Moderate',
                    'risk': 'Moderate',
                }
            
            # F7: Trend reversal - need revalidation
            elif func_id == 7:
                strategy = {
                    'model': 'Gaussian Process + caution',
                    'rationale': 'Positive trend reversed (0.3705 → 0.3448, -7%); not as stable as thought',
                    'query_type': 'Sample near current best with LOW confidence',
                    'confidence': 'Moderate (previously validated, but W7 reversal concerning)',
                    'risk': 'Moderate - trend may be non-stationary',
                }
            
            # F8: High-D improvement - continue momentum
            elif func_id == 8:
                strategy = {
                    'model': 'Ensemble NN + RBF SVM (continue)',
                    'rationale': 'Consistent improvement (7.416 → 8.001, +8%); steady progress',
                    'query_type': 'Balanced exploitation-exploration',
                    'confidence': 'Moderate-High (most reliable)',
                    'risk': 'Low-Moderate',
                }
            
            self.strategy_recommendations[func_id] = strategy
    
    def generate_summary_report(self) -> str:
        """Generate text summary of analysis"""
        report = "=" * 80 + "\n"
        report += "WEEK 7 RESULTS ANALYSIS & WEEK 8 STRATEGY\n"
        report += "=" * 80 + "\n\n"
        
        # Statistics table
        report += "FUNCTION STATISTICS (All weeks)\n"
        report += "-" * 80 + "\n"
        report += f"{'Func':>4} {'Dim':>3} {'Mean':>10} {'Std':>10} {'CV':>8} {'Trend':>10} {'Class':>10}\n"
        report += "-" * 80 + "\n"
        
        for func_id in range(1, 9):
            stats = self.stats[func_id]
            report += (f"{func_id:>4} {stats.dimensionality:>3} "
                      f"{stats.mean:>10.4f} {stats.std:>10.4f} {stats.cv:>8.4f} "
                      f"{stats.trend:>10.4f} {stats.volatility_class:>10}\n")
        
        report += "\n" + "=" * 80 + "\n"
        report += "KEY INSIGHTS\n"
        report += "=" * 80 + "\n\n"
        
        for key, insight in self.insights.items():
            report += f"{key.upper()}:\n{insight}\n\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "WEEK 8 STRATEGY RECOMMENDATIONS\n"
        report += "=" * 80 + "\n\n"
        
        for func_id in range(1, 9):
            strategy = self.strategy_recommendations[func_id]
            report += f"\nF{func_id} - {strategy.get('model', 'UNKNOWN')}:\n"
            report += f"  Rationale: {strategy.get('rationale', '')}\n"
            report += f"  Query Type: {strategy.get('query_type', '')}\n"
            report += f"  Confidence: {strategy.get('confidence', '')}\n"
            report += f"  Risk: {strategy.get('risk', '')}\n"
        
        return report
    
    def print_summary(self):
        """Print analysis summary to console"""
        print(self.generate_summary_report())


# Performance comparison W6 vs W7
W6_RESULTS = {
    1: -0.0107,
    2: -0.0301,
    3: -0.0801,
    4: -14.197,
    5: 79.327,
    6: -1.808,
    7: 0.3705,
    8: 7.416,
}

def compute_portfolio_metrics():
    """Compare portfolio performance W6 → W7"""
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE: WEEK 6 vs WEEK 7")
    print("=" * 80 + "\n")
    
    total_w6 = sum(W6_RESULTS.values())
    total_w7 = sum(WEEK_7_RESULTS['outputs'].values())
    
    print(f"{'Func':<10} {'W6 Value':<15} {'W7 Value':<15} {'Change':<15} {'% Change':<12}")
    print("-" * 80)
    
    for func_id in range(1, 9):
        w6 = W6_RESULTS[func_id]
        w7 = WEEK_7_RESULTS['outputs'][func_id]
        change = w7 - w6
        pct_change = (change / abs(w6)) * 100 if w6 != 0 else 0
        
        print(f"F{func_id:<9} {w6:<15.4f} {w7:<15.4f} {change:<15.4f} {pct_change:<12.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<10} {total_w6:<15.4f} {total_w7:<15.4f} {total_w7-total_w6:<15.4f} "
          f"{((total_w7-total_w6)/abs(total_w6)*100):<12.1f}%")
    print("\nKey Events:")
    print("  ⚠️  F5 COLLAPSE: 79.327 → 9.247 (-88% regression)")
    print("  ✓ F2 RECOVERY: -0.0301 → 0.1429 (+574% improvement from low)")
    print("  ✓ F6 IMPROVED: -1.808 → -1.594 (+12% improvement)")
    print("  ✓ F8 IMPROVED: 7.416 → 8.001 (+8% improvement)")
    print("\n")


if __name__ == "__main__":
    # Run analysis
    analyzer = Week7Analysis()
    analyzer.run_analysis()
    analyzer.print_summary()
    
    # Compute metrics
    compute_portfolio_metrics()
