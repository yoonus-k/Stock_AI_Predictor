"""
Backtesting.py Integration Module

Professional backtesting framework using the modern backtesting.py library.
Provides a clean, efficient replacement for Backtrader with enhanced RL/ML integration.

Key Features:
- 10x simpler API than Backtrader
- Built-in comprehensive metrics (50+)
- Interactive visualizations with Bokeh
- Vectorized execution (faster performance)
- Native RL/ML integration
- Professional MLflow integration
- Advanced optimization capabilities

Modules:
    data_provider: Data loading and preparation from database
    strategies: RL trading strategies for backtesting.py
    evaluator: Main portfolio evaluation orchestrator
    metrics: Comprehensive metrics extraction and analysis
    mlflow_integration: Seamless MLflow callback integration
    optimization: Strategy parameter optimization
    risk_management: Risk metrics and portfolio analysis
    visualization: Interactive plots and reporting
"""

from .evaluator import BacktestingPyEvaluator
from .strategies import RLTradingStrategy
from .data_provider import DataProvider
from .metrics import MetricsExtractor



__version__ = "1.0.0"
__author__ = "Stock AI Predictor Team"

# Main exports
__all__ = [
    'BacktestingPyEvaluator',
    'RLTradingStrategy', 
    'DataProvider',
    'MetricsExtractor',
]
