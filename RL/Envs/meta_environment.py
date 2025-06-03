"""
Meta-environment for training meta-models that aggregate timeframe-specific predictions
Uses predictions from daily, weekly, and monthly models as part of the state
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Utils.meta_model import TimeframeModelEnsemble
from RL.Envs.trading_env import TradingEnv

class MetaTradingEnv(gym.Env):
    """
    Environment for training meta-models that combine predictions from multiple timeframe models
    Extends the base trading environment with multi-timeframe awareness
    """
    
    def __init__(
        self,
        data,
        daily_model_path: Optional[str] = None,
        weekly_model_path: Optional[str] = None,
        monthly_model_path: Optional[str] = None,
        include_model_predictions: bool = True,
        normalize_observations: bool = True,
        reward_type: str = "combined",
        window_size: int = 20,
    ):
        """
        Initialize the meta-trading environment
        
        Args:
            data: DataFrame with market data
            daily_model_path: Path to trained daily model
            weekly_model_path: Path to trained weekly model
            monthly_model_path: Path to trained monthly model
            include_model_predictions: Whether to include model predictions in state
            normalize_observations: Whether to normalize observations
            reward_type: Type of reward function to use
            window_size: Size of observation window
        """
        super(MetaTradingEnv, self).__init__()
        
        # Store configuration
        self.data = data
        self.normalize = normalize_observations
        self.reward_type = reward_type
        self.window_size = window_size
        self.include_model_predictions = include_model_predictions
        
        # Create base trading environments for each timeframe
        self.envs = {}
        
        # Create timeframe model ensemble
        self.model_ensemble = TimeframeModelEnsemble(
            daily_model_path=daily_model_path,
            weekly_model_path=weekly_model_path,
            monthly_model_path=monthly_model_path
        )
        self.timeframes = self.model_ensemble.timeframes
        
        # Calculate base observation dimensions
        self.base_obs_dim = 0
        if len(self.data.columns) > 0:
            self.base_obs_dim = len(self.data.columns) - 1  # Exclude target column
        
        # Determine action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Calculate total observation dimension
        # Base features + model predictions (if included)
        total_obs_dim = self.base_obs_dim
        if include_model_predictions:
            # Add space for each timeframe model's prediction and confidence
            total_obs_dim += len(self.timeframes) * 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to starting state"""
        self.current_step = 0
        self.balance = 10000.0  # Starting balance
        self.position = 0.0
        self.trades = []
        self.trade_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action):
        """
        Execute action and advance environment by one step
        
        Args:
            action: Action to take (continuous value between -1 and 1)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Process the action value (between -1 and 1)
        action_value = float(action[0])
        
        # Current price and next price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Store current position for calculating reward
        prev_position = self.position
        
        # Update position based on action
        # Action between -1 and 1: Negative for sell, positive for buy
        self.position = action_value
        
        # Track the trade if position changed significantly
        if abs(self.position - prev_position) > 0.1:
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'action': 'buy' if self.position > prev_position else 'sell',
                'position_delta': self.position - prev_position
            })
        
        # Calculate reward
        reward = self._calculate_reward(prev_position)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades)
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """
        Get current environment observation
        Combines market data with timeframe model predictions
        """
        if self.current_step >= len(self.data):
            # If we're at the end of the data, return the last observation
            return np.zeros(self.observation_space.shape)
        
        # Get market data features
        market_features = self.data.iloc[self.current_step].values[:-1]  # Exclude target
        
        if self.normalize:
            # Simple normalization based on recent history
            if self.current_step >= self.window_size:
                window = self.data.iloc[self.current_step - self.window_size:self.current_step]
                means = window.mean()
                stds = window.std()
                stds = np.where(stds == 0, 1, stds)  # Avoid division by zero
                market_features = (market_features - means[:-1]) / stds[:-1]
        
        # Start with market features
        obs_components = [market_features]
        
        # Add model predictions if configured
        if self.include_model_predictions:
            # Prepare observations for each timeframe model
            timeframe_observations = {}
            
            for timeframe in self.timeframes:
                # Use the same market features for each timeframe
                # In a real implementation, you might use different features per timeframe
                timeframe_observations[timeframe] = market_features
            
            # Get predictions from ensemble
            if timeframe_observations:
                action, metadata = self.model_ensemble.predict_action(timeframe_observations)
                
                # Add each model's prediction and confidence to observation
                for timeframe in self.timeframes:
                    if timeframe in metadata["timeframe_predictions"]:
                        # Add prediction value
                        obs_components.append(np.array([metadata["timeframe_predictions"][timeframe][0]]))
                        # Add confidence score
                        obs_components.append(np.array([metadata["timeframe_confidences"][timeframe]]))
        
        # Combine all observation components
        obs = np.concatenate(obs_components)
        
        # Ensure observation has the right shape
        if obs.shape != self.observation_space.shape:
            # Pad or truncate to expected size
            if len(obs) < self.observation_space.shape[0]:
                padding = np.zeros(self.observation_space.shape[0] - len(obs))
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:self.observation_space.shape[0]]
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, prev_position):
        """
        Calculate reward based on position and price changes
        
        Args:
            prev_position: Previous position value
            
        Returns:
            Calculated reward
        """
        if self.current_step < 1:
            return 0
        
        # Get price data
        current_price = self.data.iloc[self.current_step]['close']
        prev_price = self.data.iloc[self.current_step - 1]['close']
        
        # Calculate price change percentage
        price_change_pct = (current_price - prev_price) / prev_price
        
        # Base reward is position * price_change
        # If long (positive position) and price goes up, positive reward
        # If short (negative position) and price goes down, positive reward
        position_reward = self.position * price_change_pct * 100
        
        # Penalty for large position changes (to encourage smoother trading)
        position_change = abs(self.position - prev_position)
        transaction_penalty = position_change * 0.01
        
        # Combine different reward components based on reward type
        if self.reward_type == "simple":
            return position_reward
        elif self.reward_type == "with_penalty":
            return position_reward - transaction_penalty
        else:  # combined or default
            # Add more reward components here if needed
            return position_reward - transaction_penalty
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.current_step > 0:
            current_price = self.data.iloc[self.current_step]['close']
            prev_price = self.data.iloc[self.current_step - 1]['close']
            price_change = current_price - prev_price
            
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f} ({price_change:.2f})")
            print(f"Position: {self.position:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Trades: {len(self.trades)}")
            print("-------------------")
    
    def close(self):
        """Clean up environment resources"""
        pass
