import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import gymnasium as gym
from gymnasium import spaces

from .observation_normalizer import ObservationNormalizer
from .trading_state import TradingState


class ObservationHandler:
    """
    A class to handle observation creation and processing in the trading environment.
    Extracts features from data and creates observation vectors.
    """
                
    # Market features from database (24 features)
    MARKET_FEATURE_NAMES = [
        # Pattern features (7)
        "probability", "action", "reward_risk_ratio", "max_gain", "max_drawdown", "mse", "expected_value",
        # Technical indicators (3)
        "rsi", "atr", "atr_ratio",
        # Sentiment features (1)
        "unified_sentiment",
        # COT data (6)
        "change_nonrept_long", "change_nonrept_short",
        "change_noncommercial_long", "change_noncommercial_short",
        "change_noncommercial_delta", "change_nonreportable_delta",
        # Time features (7)
        "hour_sin", "hour_cos", "day_sin", "day_cos", "asian_session", "london_session", "ny_session"
    ]
    
    # Portfolio features calculated at runtime (6 features)
    PORTFOLIO_FEATURE_NAMES = [
        "balance_ratio", "portfolio_max_drawdown", "win_rate", "avg_pnl_per_hour", "decisive_exits", "recovery_factor"
    ]
    
    # Complete feature list (24 + 6 = 30 features)
    ALL_FEATURE_NAMES = MARKET_FEATURE_NAMES + PORTFOLIO_FEATURE_NAMES
    
    def __init__(self, normalize_observations: bool = True, normalization_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize the observation handler
        
        Args:
            normalize_observations: Whether to normalize observations
            normalization_range: Range for normalized values
        """
        self.normalize_observations = normalize_observations
        self.trading_state = None  # Will be set later

        if normalize_observations:
            self.normalizer = ObservationNormalizer(
                output_range=normalization_range,
            )
            self.observation_space = self.normalizer.get_normalized_observation_space()
        else:
            # Sample observation size (should match feature extraction logic)
            # Updated to match actual feature count: 7+3+1+6+7+6 = 30 features
            sample_observation = 30  # 7 pattern + 3 technical + 1 sentiment + 6 COT + 7 time + 6 portfolio
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sample_observation,),
                dtype=np.float32
            )
    
    def set_trading_state(self, trading_state: TradingState):
        """
        Set the shared trading state reference
        
        Args:
            trading_state: Shared trading state object
        """
        self.trading_state = trading_state
    
    def reset(self, initial_balance: float):
        """
        Reset observation handler for a new episode
        
        Args:
            initial_balance: Starting balance for the episode
        """
        if self.normalize_observations:
            self.normalizer.reset()    
 
    def get_observation(self, market_features: pd.Series) -> np.ndarray:
        """
        Create observation vector from data point and portfolio metrics
        
        Args:
            data_point: Current data point with market features
            
        Returns:
            np.ndarray: Observation vector
        """
        # Portfolio features from trading state
        if self.trading_state:
            portfolio_features = np.array([
                self.trading_state.equity / self.trading_state.initial_balance,  # Normalized balance
                self.trading_state.drawdown,  # Add portfolio current drawdown
                self.trading_state.get_win_rate(),  # Add win rate as a feature
                self.trading_state.get_avg_pnl_per_hour(),  # P&L efficiency metric
                self.trading_state.get_decisive_exits_ratio(),  # Exit strategy effectiveness
                self.trading_state.get_recovery_factor(),  # Risk-adjusted performance
            ], dtype=np.float32)
        else:
            # Fallback if trading_state not set (shouldn't happen)
            portfolio_features = np.zeros(6, dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Normalize if required
        if self.normalize_observations:
            observation = self.normalizer.normalize_observation(observation)
            
        #print(f"Observation shape: {np.array(observation).shape}")
            
        return observation
   