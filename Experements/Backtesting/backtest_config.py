"""
Backtesting Configuration Module

This module defines configuration parameters and utilities for backtesting
strategies based on pattern recognition. It provides:

1. Standard configuration templates for different recognition techniques
2. Configuration validation and suggestion utilities
3. Parameter ranges for optimization

Usage:
    Import this module to create standardized backtesting configurations
    or to validate user-defined parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from enum import Enum


class RecognitionTechnique(Enum):
    """Enumeration of available pattern recognition techniques."""
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    COMBINED = "combined"
    DISTANCE_BASED = "distance_based"
    
    @classmethod
    def from_string(cls, technique: str) -> 'RecognitionTechnique':
        """Convert string to enum value, case-insensitive."""
        technique = technique.lower().replace(" ", "_")
        for t in cls:
            if t.value == technique:
                return t
        raise ValueError(f"Unknown recognition technique: {technique}")


class ExitStrategy(Enum):
    """Enumeration of available trade exit strategies."""
    PATTERN_BASED = "pattern_based"  # Use pattern's historical max gain/drawdown
    FIXED = "fixed"                  # Use fixed take profit/stop loss
    TRAILING = "trailing"            # Use trailing stop loss
    TIME_BASED = "time_based"        # Exit after specific time period
    DUAL = "dual"                    # Combination of time and price-based exits
    
    @classmethod
    def from_string(cls, strategy: str) -> 'ExitStrategy':
        """Convert string to enum value, case-insensitive."""
        strategy = strategy.lower().replace(" ", "_")
        for s in cls:
            if s.value == strategy:
                return s
        raise ValueError(f"Unknown exit strategy: {strategy}")


@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    # Basic parameters
    stock_id: int
    timeframe_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Pattern recognition parameters
    recognition_technique: RecognitionTechnique
    n_pips: int = 5
    lookback: int = 24
    hold_period: int = 6
    returns_hold_period: int = 12
    distance_measure: int = 3
    mse_threshold: float = 0.03
    
    # Model specific parameters
    model_params: Dict[str, Any] = None
    
    # Trade execution parameters
    exit_strategy: ExitStrategy = ExitStrategy.PATTERN_BASED
    fixed_tp_pct: Optional[float] = None
    fixed_sl_pct: Optional[float] = None
    trailing_sl_pct: Optional[float] = None
    time_exit_periods: Optional[int] = None
    reward_risk_min: float = 1.0
    
    # Additional configuration
    config_name: str = "default"
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure model_params is a dictionary
        if self.model_params is None:
            self.model_params = self._get_default_model_params()
        
        # Validate exit strategy parameters
        self._validate_exit_strategy()
        
        # Convert recognition_technique to enum if it's a string
        if isinstance(self.recognition_technique, str):
            self.recognition_technique = RecognitionTechnique.from_string(self.recognition_technique)
            
        # Convert exit_strategy to enum if it's a string
        if isinstance(self.exit_strategy, str):
            self.exit_strategy = ExitStrategy.from_string(self.exit_strategy)
    
    def _get_default_model_params(self) -> Dict[str, Any]:
        """Get default model parameters based on recognition technique."""
        if self.recognition_technique == RecognitionTechnique.SVM:
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True
            }
        elif self.recognition_technique == RecognitionTechnique.RANDOM_FOREST:
            return {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
        elif self.recognition_technique == RecognitionTechnique.COMBINED:
            return {
                'svm': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'probability': True
                },
                'rf': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'random_state': 42
                },
                'voting_weights': [0.5, 0.5]  # Weights for SVM and RF
            }
        elif self.recognition_technique == RecognitionTechnique.DISTANCE_BASED:
            return {
                'distance_metric': 'euclidean',
                'max_distance': 0.2,
                'min_similarity': 0.7
            }
        return {}
    
    def _validate_exit_strategy(self):
        """Validate that the exit strategy has required parameters."""
        if self.exit_strategy == ExitStrategy.FIXED:
            if self.fixed_tp_pct is None or self.fixed_sl_pct is None:
                raise ValueError("Fixed exit strategy requires fixed_tp_pct and fixed_sl_pct")
        elif self.exit_strategy == ExitStrategy.TRAILING:
            if self.trailing_sl_pct is None:
                raise ValueError("Trailing exit strategy requires trailing_sl_pct")
        elif self.exit_strategy == ExitStrategy.TIME_BASED:
            if self.time_exit_periods is None:
                raise ValueError("Time-based exit strategy requires time_exit_periods")
        elif self.exit_strategy == ExitStrategy.DUAL:
            if self.time_exit_periods is None or (self.fixed_tp_pct is None and self.trailing_sl_pct is None):
                raise ValueError("Dual exit strategy requires time_exit_periods and either fixed_tp_pct or trailing_sl_pct")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for database storage."""
        config_dict = {
            'stock_id': self.stock_id,
            'timeframe_id': self.timeframe_id,
            'n_pips': self.n_pips,
            'lookback': self.lookback,
            'hold_period': self.hold_period,
            'returns_hold_period': self.returns_hold_period,
            'distance_measure': self.distance_measure,
            'name': self.config_name,
            'description': self.description,
            'recognition_technique': self.recognition_technique.value,
            'model_params': str(self.model_params),
            'exit_strategy': self.exit_strategy.value,
        }
        
        # Add optional parameters if they exist
        for attr in ['fixed_tp_pct', 'fixed_sl_pct', 'trailing_sl_pct', 
                    'time_exit_periods', 'reward_risk_min', 'mse_threshold']:
            value = getattr(self, attr, None)
            if value is not None:
                config_dict[attr] = value
                
        return config_dict


# Predefined configurations for different recognition techniques
DEFAULT_CONFIGS = {
    'svm_standard': BacktestConfig(
        stock_id=1,
        timeframe_id=2,
        train_start=pd.Timestamp("2019-01-01"),
        train_end=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-01-02"),
        test_end=pd.Timestamp("2023-12-31"),
        recognition_technique=RecognitionTechnique.SVM,
        n_pips=5,
        lookback=24,
        hold_period=6,
        returns_hold_period=12,
        distance_measure=3,
        mse_threshold=0.03,
        config_name="SVM Standard",
        description="Standard configuration using SVM for pattern recognition"
    ),
    'rf_standard': BacktestConfig(
        stock_id=1,
        timeframe_id=2,
        train_start=pd.Timestamp("2019-01-01"),
        train_end=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-01-02"),
        test_end=pd.Timestamp("2023-12-31"),
        recognition_technique=RecognitionTechnique.RANDOM_FOREST,
        n_pips=5,
        lookback=24,
        hold_period=6,
        returns_hold_period=12,
        distance_measure=3,
        mse_threshold=0.05,  # RF typically needs a higher threshold
        config_name="Random Forest Standard",
        description="Standard configuration using Random Forest for pattern recognition"
    ),
    'combined_standard': BacktestConfig(
        stock_id=1,
        timeframe_id=2,
        train_start=pd.Timestamp("2019-01-01"),
        train_end=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-01-02"),
        test_end=pd.Timestamp("2023-12-31"),
        recognition_technique=RecognitionTechnique.COMBINED,
        n_pips=5,
        lookback=24,
        hold_period=6,
        returns_hold_period=12,
        distance_measure=3,
        mse_threshold=0.04,
        config_name="Combined Models",
        description="Using both SVM and Random Forest with voting"
    ),
}


def create_optimization_configs(
    base_config: BacktestConfig,
    param_ranges: Dict[str, List[Any]]
) -> List[BacktestConfig]:
    """
    Create a list of configurations for parameter optimization.
    
    Args:
        base_config: Base configuration to start from
        param_ranges: Dict of parameter names and their possible values
        
    Returns:
        List of BacktestConfig objects with different parameter combinations
    """
    import itertools
    
    # Create parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Create configs for each combination
    configs = []
    for combo in param_combinations:
        # Create a copy of the base config
        config_dict = base_config.to_dict()
        
        # Update with the current parameter combination
        for i, param_name in enumerate(param_names):
            config_dict[param_name] = combo[i]
        
        # Create a new config with the updated parameters
        config = BacktestConfig(**config_dict)
        configs.append(config)
    
    return configs


def get_recommended_config(stock_id: int, timeframe_id: int) -> BacktestConfig:
    """
    Get a recommended configuration based on historical performance.
    
    Args:
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        
    Returns:
        Recommended BacktestConfig
    """
    from Data.Database.db import Database
    
    db = Database()
    
    # Query the database for the best performing config
    query = """
    SELECT c.*, p.win_rate, p.profit_factor
    FROM experiment_configs c
    JOIN performance_metrics p ON c.config_id = p.config_id
    WHERE p.stock_id = ? AND p.timeframe_id = ?
    ORDER BY p.win_rate * p.profit_factor DESC
    LIMIT 1
    """
    
    cursor = db.connection.cursor()
    cursor.execute(query, (stock_id, timeframe_id))
    result = cursor.fetchone()
    
    if result:
        # Create a config from the database result
        config = BacktestConfig(
            stock_id=stock_id,
            timeframe_id=timeframe_id,
            train_start=pd.Timestamp("2019-01-01"),  # Default, to be overridden by the user
            train_end=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-02"),
            test_end=pd.Timestamp("2023-12-31"),
            recognition_technique=RecognitionTechnique.from_string(result['recognition_technique']),
            n_pips=result['n_pips'],
            lookback=result['lookback'],
            hold_period=result['hold_period'],
            returns_hold_period=result['returns_hold_period'],
            distance_measure=result['distance_measure'],
            config_name=result['name'],
            description=result['description']
        )
        
        # Add any optional parameters
        for attr in ['mse_threshold', 'fixed_tp_pct', 'fixed_sl_pct', 
                    'trailing_sl_pct', 'time_exit_periods', 'reward_risk_min']:
            if attr in result and result[attr] is not None:
                setattr(config, attr, result[attr])
        
        return config
    
    # If no config found, return a default one
    return DEFAULT_CONFIGS['svm_standard']


if __name__ == "__main__":
    # Example of creating optimization configs
    base_config = DEFAULT_CONFIGS['svm_standard']
    param_ranges = {
        'n_pips': [3, 5, 7],
        'lookback': [12, 24, 36],
        'hold_period': [4, 6, 8]
    }
    
    optimization_configs = create_optimization_configs(base_config, param_ranges)
    print(f"Created {len(optimization_configs)} optimization configurations")
    
    # Print details of first few configs
    for i, config in enumerate(optimization_configs[:3]):
        print(f"\nConfig {i+1}:")
        print(f"  n_pips: {config.n_pips}")
        print(f"  lookback: {config.lookback}")
        print(f"  hold_period: {config.hold_period}")
