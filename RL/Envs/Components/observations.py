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
        self.drawdown = 0
        self.max_drawdown = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.steps_without_action = 0
          # New tracking variables for enhanced metrics
        self.total_pnl = 0.0
        self.total_holding_hours = 1.0  # Avoid division by zero
        self.tp_hits = 0
        self.sl_hits = 0
        self.time_exits = 0
        self.max_balance = 0.0      # New performance metrics
        self.total_exits = 0  # Total number of exits (tp + sl + timeout)
        self.total_gains = 0.0  # Sum of all positive trades
        self.peak_balance = 0.0  # Track peak balance for recovery factor
        
        if normalize_observations:
            self.normalizer = ObservationNormalizer(
                output_range= normalization_range,
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

    
    def reset(self, initial_balance: float):
        """
        Reset observation handler for a new episode
        
        Args:
            initial_balance: Starting balance for the episode
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.drawdown = 0
        self.max_drawdown = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.steps_without_action = 0
          # Reset enhanced metrics
        self.total_pnl = 0.0
        self.total_holding_hours = 1.0  # Avoid division by zero
        self.tp_hits = 0
        self.sl_hits = 0
        self.time_exits = 0
        self.max_balance = 0.0
        
        # Reset new performance metrics
        self.total_exits = 0  # Total number of exits (tp + sl + timeout)
        self.total_gains = 0.0  # Sum of all positive trades
        self.peak_balance = initial_balance  # Track peak balance for recovery factor
        
        if self.normalize_observations:
            self.normalizer.reset()
    
    def update_portfolio_metrics(self, balance: float,drawdown:float, max_drawdown: float, winning_trades: int, trade_count: int,
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
        self.drawdown = drawdown
        self.max_drawdown = max_drawdown
        self.winning_trades = winning_trades
        self.trade_count = trade_count
        self.steps_without_action = steps_without_action
    
    def get_observation(self, market_features: pd.Series) -> np.ndarray:
        """
        Create observation vector from data point and portfolio metrics
        
        Args:
            data_point: Current data point with market features
            
        Returns:
            np.ndarray: Observation vector
        """
        
        # Portfolio features
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.drawdown,  # Add portfolio current drawdown
            self.winning_trades / (self.trade_count ) if self.trade_count else 0.5,  # Add win rate as a feature
            self.calculate_avg_pnl_per_hour(),  # P&L efficiency metric
            self.calculate_decisive_exits(),  # Exit strategy effectiveness
            self.calculate_recovery_factor(),  # Risk-adjusted performance
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Normalize if required
        if self.normalize_observations:
            observation = self.normalizer.normalize_observation(observation)
            
        return observation
    
    def update_trade_metrics(self, trade_pnl_pct: float, exit_reason: str, holding_hours: float, balance: float):
        """
        Update trading metrics when a trade is completed
        
        Args:
            trade_pnl_pct: Trade P&L as percentage
            exit_reason: Exit reason ('tp', 'sl', 'time')
            holding_hours: Actual holding time in hours
            balance: Current balance after trade
        """
        # Update total PnL and holding hours
        self.total_pnl += trade_pnl_pct
        self.total_holding_hours += holding_hours
        
        # Update exit counters based on exit reason
        if exit_reason == 'tp':
            self.tp_hits += 1
        elif exit_reason == 'sl':
            self.sl_hits += 1
        elif exit_reason == 'time':
            self.time_exits += 1
            
        # Track total exits
        self.total_exits += 1
        
        # Update gains if positive trade
        if trade_pnl_pct > 0:
            self.total_gains += trade_pnl_pct
            
        # Update peak balance for recovery factor calculation
        self.peak_balance = max(self.peak_balance, balance)
    
    def calculate_avg_pnl_per_hour(self) -> float:
        """
        Calculate average P&L per hour of holding time
        
        Returns:
            float: Average P&L per hour (higher is better)
        """
        if self.total_holding_hours <= 0:
            return 0.0
        return self.total_pnl / self.total_holding_hours
    
    def calculate_decisive_exits(self) -> float:
        """
        Calculate ratio of decisive exits (TP/SL) vs timeouts
        
        Returns:
            float: Ratio of decisive exits [0.0, 1.0] (higher is better)
        """
        if self.total_exits <= 0:
            return 0.0
        decisive_exits = self.tp_hits + self.sl_hits
        return decisive_exits / self.total_exits
    
    def calculate_recovery_factor(self) -> float:
        """
        Calculate recovery factor: gains relative to maximum drawdown
        
        Returns:
            float: Recovery factor (higher is better, >1.0 is good)
        """
        if abs(self.max_drawdown) < 1e-6:  # No meaningful drawdown
            return 1.0 if self.total_gains > 0 else 0.0
        return self.total_gains / abs(self.max_drawdown)
