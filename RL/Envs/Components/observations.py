import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import gymnasium as gym
from gymnasium import spaces

from .observation_normalizer import ObservationNormalizer


class ObservationHandler:
    """
    A class to handle observation creation and processing in the trading environment.
    Extracts features from data and creates observation vectors.
    """
    
    def __init__(self, normalize_observations: bool = True , normalization_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize the observation handler
        
        Args:
            normalize_observations: Whether to normalize observations
        """
        self.normalize_observations = normalize_observations
        
        # Initialize portfolio metrics for observation
        self.initial_balance = 0
        self.balance = 0
        self.position = 0
        self.position_value = 0
        self.max_drawdown = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.steps_without_action = 0
        
        if normalize_observations:
           
            self.normalizer = ObservationNormalizer(
                enable_adaptive_scaling=False,
                output_range= normalization_range,
            )
            self.observation_space = self.normalizer.get_normalized_observation_space()
        else:
            # Sample observation size (should match feature extraction logic)
            sample_observation = 31
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sample_observation,),
                dtype=np.float32
            )
    
    def reset(self, initial_balance: float):
        """
        Reset observation handler for a new episode
        
        Args:
            initial_balance: Starting balance for the episode
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.position_value = 0
        self.max_drawdown = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.steps_without_action = 0
        
        if self.normalize_observations:
            self.normalizer.reset()
    
    def update_portfolio_metrics(self, balance: float, position: float, position_value: float,
                                max_drawdown: float, winning_trades: int, trade_count: int,
                                steps_without_action: int):
        """
        Update portfolio metrics for observation creation
        
        Args:
            balance: Current account balance
            position: Current position size
            position_value: Current position value
            max_drawdown: Maximum drawdown experienced
            winning_trades: Number of winning trades
            trade_count: Total number of trades
            steps_without_action: Number of consecutive HOLD actions
        """
        self.balance = balance
        self.position = position
        self.position_value = position_value
        self.max_drawdown = max_drawdown
        self.winning_trades = winning_trades
        self.trade_count = trade_count
        self.steps_without_action = steps_without_action
    
    def get_observation(self, data_point: pd.Series) -> np.ndarray:
        """
        Create observation vector from data point and portfolio metrics
        
        Args:
            data_point: Current data point with market features
            
        Returns:
            np.ndarray: Observation vector
        """
        features = []
        
        # Add base pattern features (directly from DataFrame)
        base_features = [
            data_point['probability'],
            data_point['action'],
            data_point['reward_risk_ratio'],
            data_point['max_gain'],
            data_point['max_drawdown'],
            data_point['mse'],
            data_point['expected_value']
        ]
        features.extend(base_features)
        
        # Technical indicators - based on actual column names in the dataset
        technical_indicators = ['rsi', 'atr', 'atr_ratio']
        for indicator in technical_indicators:
            if indicator in data_point:
                features.append(data_point[indicator])
            
        # Sentiment features - based on actual column names
        sentiment_features = ['unified_sentiment', 'sentiment_count']
        for feature in sentiment_features:
            if feature in data_point:
                features.append(data_point[feature])
            
        # COT data - based on actual column names in your dataset
        cot_features = [
            'net_noncommercial', 'net_nonreportable',
            'change_nonrept_long', 'change_nonrept_short',
            'change_noncommercial_long', 'change_noncommercial_short'
        ]
        for feature in cot_features:
            if feature in data_point:
                features.append(data_point[feature])
            
        # Time-based features - based on actual column names
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                        'asian_session', 'london_session', 'ny_session']
        for feature in time_features:
            if feature in data_point:
                features.append(data_point[feature])
        
        # Make sure we have at least the base features
        while len(features) < 5:
            features.append(0.0)
            
        # Convert to numpy array
        market_features = np.array(features, dtype=np.float32)
        
        # Portfolio features
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position_value / (self.balance + self.position_value + 1e-6),  # Position ratio
            self.position,  # Absolute position size
            self.max_drawdown,  # Add drawdown as a feature
            self.winning_trades / (self.trade_count + 1e-6),  # Add win rate as a feature
            min(self.steps_without_action / 10, 1.0),  # Normalized consecutive hold count
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Normalize if required
        if self.normalize_observations:
            observation = self.normalizer.normalize_observation(observation)
            
        return observation
