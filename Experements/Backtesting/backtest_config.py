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


# Recognition technique constants
class RecognitionTechnique:
    """Simple class with constants for available pattern recognition techniques."""
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    COMBINED = "combined"
    DISTANCE_BASED = "distance_based"
    
    @classmethod
    def valid_techniques(cls):
        """Return a list of valid recognition techniques."""
        return [cls.SVM, cls.RANDOM_FOREST, cls.COMBINED, cls.DISTANCE_BASED]
    
    @classmethod
    def from_string(cls, technique: str) -> str:
        """Convert string to standard technique string, case-insensitive."""
        if technique is None:
            return cls.SVM  # Default
            
        technique = str(technique).lower().replace(" ", "_")
        if technique in cls.valid_techniques():
            return technique
        raise ValueError(f"Unknown recognition technique: {technique}")
        
    @classmethod
    def to_display_string(cls, technique: str) -> str:
        """Convert technique string to display format."""
        if technique is None:
            return "SVM"
        return str(technique).replace("_", " ").title()


# Exit strategy constants
class ExitStrategy:
    """Simple class with constants for available trade exit strategies."""
    PATTERN_BASED = "pattern_based"  # Use pattern's historical max gain/drawdown
    FIXED = "fixed"                  # Use fixed take profit/stop loss
    TRAILING = "trailing"            # Use trailing stop loss
    TIME_BASED = "time_based"        # Exit after specific time period
    DUAL = "dual"                    # Combination of time and price-based exits
    
    @classmethod
    def valid_strategies(cls):
        """Return a list of valid exit strategies."""
        return [cls.PATTERN_BASED, cls.FIXED, cls.TRAILING, cls.TIME_BASED, cls.DUAL]
    
    @classmethod
    def from_string(cls, strategy: str) -> str:
        """Convert string to standard strategy string, case-insensitive."""
        if strategy is None:
            return cls.PATTERN_BASED  # Default
            
        strategy = str(strategy).lower().replace(" ", "_")
        if strategy in cls.valid_strategies():
            return strategy
        raise ValueError(f"Unknown exit strategy: {strategy}")
        
    @classmethod
    def to_display_string(cls, strategy: str) -> str:
        """Convert strategy string to display format."""
        if strategy is None:
            return "Pattern Based"
        return str(strategy).replace("_", " ").title()


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
    recognition_technique: str
    n_pips: int = 5
    lookback: int = 24
    hold_period: int = 6
    returns_hold_period: int = 12
    distance_measure: int = 2
    mse_threshold: float = 0.03
    
    # Model specific parameters
    model_params: Dict[str, Any] = None
    
    # Trade execution parameters
    exit_strategy: str = ExitStrategy.PATTERN_BASED
    fixed_tp_pct: Optional[float] = None
    fixed_sl_pct: Optional[float] = None
    trailing_sl_pct: Optional[float] = None
    time_exit_periods: Optional[int] = None
    reward_risk_min: float = 1.0
    
    # Additional configuration
    config_id: Optional[int] = None
    config_name: str = "default"
    description: str = ""    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure model_params is a dictionary
        if self.model_params is None:
            self.model_params = self._get_default_model_params()
        
        # Handle recognition_technique conversion
        if hasattr(self.recognition_technique, '__class__') and self.recognition_technique.__class__.__name__ == 'Series':
            # Extract the first value if it's a Series
            recognition_str = self.recognition_technique.iloc[0] if len(self.recognition_technique) > 0 else None
            self.recognition_technique = RecognitionTechnique.from_string(recognition_str)
        elif isinstance(self.recognition_technique, str) or self.recognition_technique is None:
            self.recognition_technique = RecognitionTechnique.from_string(self.recognition_technique)
        
        # Handle exit_strategy conversion
        if hasattr(self.exit_strategy, '__class__') and self.exit_strategy.__class__.__name__ == 'Series':
            # Extract the first value if it's a Series
            exit_str = self.exit_strategy.iloc[0] if len(self.exit_strategy) > 0 else None
            self.exit_strategy = ExitStrategy.from_string(exit_str)
        elif isinstance(self.exit_strategy, str) or self.exit_strategy is None:
            self.exit_strategy = ExitStrategy.from_string(self.exit_strategy)
            
        # Validate exit strategy parameters
        self._validate_exit_strategy()    
    def _get_default_model_params(self) -> Dict[str, Any]:
        """Get default model parameters based on recognition technique."""
        # Handle if recognition_technique is a pandas Series
        if hasattr(self.recognition_technique, '__class__') and self.recognition_technique.__class__.__name__ == 'Series':
            # Extract the first value if it's a Series
            recognition_technique = self.recognition_technique.iloc[0] if len(self.recognition_technique) > 0 else None
            # Convert to standard string
            if recognition_technique is not None:
                recognition_technique = RecognitionTechnique.from_string(recognition_technique)
            else:
                # Default to SVM if we can't determine the type
                recognition_technique = RecognitionTechnique.SVM
        else:
            recognition_technique = self.recognition_technique
        
        if recognition_technique == RecognitionTechnique.SVM:
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True
            }
        elif recognition_technique == RecognitionTechnique.RANDOM_FOREST:
            return {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
        elif recognition_technique == RecognitionTechnique.COMBINED:
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
        elif recognition_technique == RecognitionTechnique.DISTANCE_BASED:
            return {
                'distance_metric': 'euclidean',
                'max_distance': 0.2,
                'min_similarity': 0.7
            }
        return {}    
    def _validate_exit_strategy(self):
        """Validate that the exit strategy has required parameters."""
        # Handle if exit_strategy is a pandas Series
        if hasattr(self.exit_strategy, '__class__') and self.exit_strategy.__class__.__name__ == 'Series':
            # Extract the first value if it's a Series
            exit_strategy = self.exit_strategy.iloc[0] if len(self.exit_strategy) > 0 else None
            # Convert to standard string
            if exit_strategy is not None:
                exit_strategy = ExitStrategy.from_string(exit_strategy)
            else:
                # Default to PATTERN_BASED if we can't determine the type
                return
        else:
            exit_strategy = self.exit_strategy

        if exit_strategy == ExitStrategy.FIXED:
            if self.fixed_tp_pct is None or self.fixed_sl_pct is None:
                raise ValueError("Fixed exit strategy requires fixed_tp_pct and fixed_sl_pct")
        elif exit_strategy == ExitStrategy.TRAILING:
            if self.trailing_sl_pct is None:
                raise ValueError("Trailing exit strategy requires trailing_sl_pct")
        elif exit_strategy == ExitStrategy.TIME_BASED:
            if self.time_exit_periods is None:
                raise ValueError("Time-based exit strategy requires time_exit_periods")
        elif exit_strategy == ExitStrategy.DUAL:
            if self.time_exit_periods is None or (self.fixed_tp_pct is None and self.trailing_sl_pct is None):
                raise ValueError("Dual exit strategy requires time_exit_periods and either fixed_tp_pct or trailing_sl_pct")    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for database storage."""
        # Helper function to extract value from pandas Series if needed
        def extract_value(val):
            if hasattr(val, '__class__') and val.__class__.__name__ == 'Series':
                return val.iloc[0] if len(val) > 0 else None
            return val
            
        # Get simple string values for recognition_technique and exit_strategy
        recognition_technique = extract_value(self.recognition_technique)
        exit_strategy = extract_value(self.exit_strategy)
            
        config_dict = {
            'stock_id': extract_value(self.stock_id),
            'timeframe_id': extract_value(self.timeframe_id),
            'n_pips': int(extract_value(self.n_pips)),
            'lookback': int(extract_value(self.lookback)),
            'hold_period': int(extract_value(self.hold_period)),
            'returns_hold_period': int(extract_value(self.returns_hold_period)),
            'distance_measure': int(extract_value(self.distance_measure)),
            'name': extract_value(self.config_name),
            'description': extract_value(self.description),
            'recognition_technique': recognition_technique,
            'model_params': str(self.model_params),
            'exit_strategy': exit_strategy,
        }
        
        # Add config_id if it exists
        if self.config_id is not None:
            config_dict['config_id'] = extract_value(self.config_id)
        
        # Add optional parameters if they exist
        for attr in ['fixed_tp_pct', 'fixed_sl_pct', 'trailing_sl_pct', 
                    'time_exit_periods', 'reward_risk_min', 'mse_threshold']:
            value = getattr(self, attr, None)
            if value is not None:
                config_dict[attr] = extract_value(value)
                # Convert numeric values that might be in Series
                if attr in ['fixed_tp_pct', 'fixed_sl_pct', 'trailing_sl_pct', 'reward_risk_min', 'mse_threshold']:
                    try:
                        config_dict[attr] = float(config_dict[attr])
                    except (TypeError, ValueError):
                        pass
                elif attr == 'time_exit_periods':
                    try:
                        config_dict[attr] = int(config_dict[attr])
                    except (TypeError, ValueError):
                        pass
                
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
