"""
Backtrader Integration Module for RL Model Performance Analysis

This module provides comprehensive backtesting capabilities using the Backtrader library
to replace the simplified portfolio evaluation in the MLflow callback. It supports:

- Professional-grade backtesting with 50+ metrics
- Clean separation: SimplePriceFeed for OHLCV, ObservationManager for RL features
- Long and short positions with realistic trade execution
- Multi-position trading and hedging strategies
- Integration with existing RL models and MLflow infrastructure
- Real-time performance analysis during training

Components:
- rl_strategy.py: Backtrader strategy using RL model predictions
- data_feeds.py: Simplified price feeds + separate observation management
- analyzers.py: Custom analyzers for RL-specific metrics
- portfolio_evaluator.py: Main evaluation orchestrator
- metrics_extractor.py: Extract and format comprehensive metrics

Usage:
    from Backtrader.portfolio_evaluator import BacktraderPortfolioEvaluator
    
    evaluator = BacktraderPortfolioEvaluator(initial_cash=100000)
    metrics = evaluator.evaluate_portfolio(model, environment_data)
"""

__version__ = "1.0.0"
__author__ = "Stock AI Predictor Team"

# Import main components for easy access
try:
    from .portfolio_evaluator import BacktraderPortfolioEvaluator
    from .rl_strategy import RLTradingStrategy
    from .data_feeds import (
        SimplePriceFeed, 
        ObservationManager,
        create_price_feed,
        create_observation_manager,
        create_complete_backtest_data,
        # Legacy compatibility
        EnvironmentDataFeed,
        PandasEnvironmentFeed
    )
    from .analyzers import RLPerformanceAnalyzer, RLActionAnalyzer
    from .metrics_extractor import MetricsExtractor
    
    __all__ = [
        'BacktraderPortfolioEvaluator',
        'RLTradingStrategy', 
        'SimplePriceFeed',
        'ObservationManager',
        'create_price_feed',
        'create_observation_manager', 
        'create_complete_backtest_data',
        'RLPerformanceAnalyzer',
        'RLActionAnalyzer',
        'MetricsExtractor',
        # Legacy
        'EnvironmentDataFeed',
        'PandasEnvironmentFeed'
    ]
    
except ImportError as e:
    # Allow module to load even if some components are missing during development
    import logging
    logging.getLogger(__name__).warning(f"Some backtrader components not available: {e}")
    __all__ = []
