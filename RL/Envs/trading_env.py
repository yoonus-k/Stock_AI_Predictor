import numpy as np
import gymnasium as gym
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional, Union

# Import components
from .Components.rewards import RewardCalculator
from .Components.actions import ActionHandler
from .Components.observations import ObservationHandler


class TradingEnv(gym.Env):
    def __init__(
        self,
        data,
        initial_balance=100000,
        reward_type="combined",
        normalize_observations=True,
        normalization_range=(-1.0, 1.0),
        timeframe_id=5,
    ):
        """
        Trading environment for reinforcement learning with modular components

        Args:
            data: DataFrame with market data
            initial_balance: Starting account balance
            reward_type: Type of reward function to use
            normalize_observations: Whether to normalize observations
            normalization_range: Range for normalized observations
            enable_adaptive_scaling: Whether to adapt scaling during training
            timeframe_id: Identifier for the timeframe (e.g., 5 for 1H , 6 for 4H, etc.)
        """
        super(TradingEnv, self).__init__()        
        self.data = data
        self.index = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Current position size (shares/contracts)
        self.position_value = 0  # Dollar value of current position
        self.current_price = 0
        self.timeframe = timeframe_id  # Store timeframe information

        # Initialize trade tracking variables
        self.entry_price = 0
        self.tp_price = 0
        self.sl_price = 0
        self.trade_direction = 0  # 1=long, -1=short, 0=neutral
        self.peak_balance = initial_balance
        self.last_trade_idx = -10  # For tracking trade frequency

        # Initialize modular components
        self.reward_calculator = RewardCalculator(reward_type=reward_type)
        self.action_handler = ActionHandler()
        self.observation_handler = ObservationHandler(
            normalize_observations=normalize_observations,
            normalization_range=normalization_range,
        )

        # Set up action and observation spaces
        self.action_space = self.action_handler.action_space
        self.observation_space = self.observation_handler.observation_space

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple: (observation, info)
        """
        self.index = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.tp_price = 0
        self.sl_price = 0
        self.trade_direction = 0
        self.peak_balance = self.initial_balance
        self.last_trade_idx = -10

        # Reset components
        self.reward_calculator.reset()
        self.action_handler.reset()
        self.observation_handler.reset(self.initial_balance)

        super().reset(seed=seed)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Action from the agent [action_type, position_size, risk_reward_multiplier]

        Returns:
            Tuple: (observation, reward, done, truncated, info)
        """
        # Get current data point
        current_data = self.data.iloc[self.index]

        # Process action using action handler
        action_type, position_size, risk_reward_multiplier = (
            self.action_handler.process_action(action)
        )

        # Default price (in real markets, use actual price data)
        self.current_price = 1.0

        # Track peak balance for drawdown calculation
        self.peak_balance = max(self.peak_balance, self.balance)

        reward = 0
        base_reward = 0
        done = False
        truncated = False
        trade_open = False
        trade_closed = False
        trade_pnl = 0

        # Get pattern information from data
        max_gain = current_data.get("max_gain", 0.01)
        max_drawdown = current_data.get("max_drawdown", -0.01)
        pattern_action = current_data.get("action", 0)

        # Handle HOLD action
        if action_type != 0:  # If not HOLD
            reward += 0.001  # Small bonus for taking action

        if action_type == 0:  # HOLD
            # Calculate hold penalty
            reward = self.reward_calculator.calculate_hold_penalty(
                self.action_handler.steps_without_action
            )

            # # In the step() method, adjust the termination condition
            # if self.action_handler.steps_without_action > 100:  # Increase from 50 to 100
            #     done = True
            #     truncated = True
            #     reward = -0.005  # Reduce penalty from -0.01 to -0.005
        # Handle BUY action
        elif action_type == 1:  # BUY
            # Apply position sizing with performance adjustment
            adjusted_position_size = self.action_handler.adaptive_position_sizing(
                position_size,
                self.balance,
                self.peak_balance,
                self.reward_calculator.returns_history,
                current_data.get("atr_ratio", None),
            )

            # Calculate position value and size
            position_value = self.balance * adjusted_position_size
            self.position = position_value / self.current_price
            self.entry_price = self.current_price

            # Calculate take profit and stop loss levels
            self.tp_price, self.sl_price = self.action_handler.calculate_trade_targets(
                action_type,
                self.entry_price,
                max_gain,
                max_drawdown,
                risk_reward_multiplier,
                pattern_action,
            )

            self.trade_direction = 1  # Long
            trade_open = True
            self.last_trade_idx = self.index

        # Handle SELL action
        elif action_type == 2:  # SELL
            # Apply position sizing with performance adjustment
            adjusted_position_size = self.action_handler.adaptive_position_sizing(
                position_size,
                self.balance,
                self.peak_balance,
                self.reward_calculator.returns_history,
                current_data.get("atr_ratio", None),
            )

            # Calculate position value and size
            position_value = self.balance * adjusted_position_size
            self.position = position_value / self.current_price
            self.entry_price = self.current_price

            # Calculate take profit and stop loss levels
            self.tp_price, self.sl_price = self.action_handler.calculate_trade_targets(
                action_type,
                self.entry_price,
                max_gain,
                max_drawdown,
                risk_reward_multiplier,
                pattern_action,
            )

            self.trade_direction = -1  # Short
            trade_open = True
            self.last_trade_idx = self.index

        # Process trade results (check if TP/SL hit)
        if trade_open:
            # Get price movement data
            mfe = current_data.get("mfe", 0.01)  # Maximum favorable excursion
            mae = current_data.get("mae", -0.01)  # Maximum adverse excursion
            actual_return = current_data.get("actual_return", 0.0)

            # Calculate price extremes
            highest_price = self.entry_price * (1 + mfe)
            lowest_price = self.entry_price * (1 + mae)

            # For long positions
            if self.trade_direction == 1:
                # check for stop loss and take profit conditions
                if lowest_price <= self.sl_price:
                    # Hit stop loss
                    base_reward = (self.sl_price - self.entry_price) / self.entry_price
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                elif highest_price >= self.tp_price:
                    # Hit take profit
                    base_reward = (self.tp_price - self.entry_price) / self.entry_price
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True

                else:
                    # Close at actual return
                    base_reward = actual_return
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True

            # For short positions
            elif self.trade_direction == -1:
                if highest_price >= self.sl_price:
                    # Hit stop loss
                    base_reward = -(self.sl_price - self.entry_price) / self.entry_price
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                elif lowest_price <= self.tp_price:
                    # Hit take profit
                    base_reward = -(self.tp_price - self.entry_price) / self.entry_price
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True

                else:
                    # Close at actual return
                    base_reward = -actual_return
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True

        # Calculate final reward
        if trade_closed:
            # Update reward calculator metrics
            self.reward_calculator.update_metrics(self.balance + trade_pnl, trade_pnl)

            # Calculate reward based on configured type
            reward = self.reward_calculator.calculate_reward(base_reward, trade_pnl)

        # Update portfolio value
        self.position_value = self.position * self.current_price
        portfolio_balance = self.balance + (trade_pnl if trade_closed else 0)

        # Build info dictionary
        info = {
            "portfolio_balance": portfolio_balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "trade_direction": self.trade_direction,
            "trade_pnl": trade_pnl if trade_closed else 0,
            "max_drawdown": self.reward_calculator.max_drawdown,
        }

        # Add action statistics
        info.update(self.action_handler.get_action_statistics())

        # Close position if trade completed
        if trade_closed:
            self.balance += trade_pnl
            self.position = 0
            self.entry_price = 0
            self.trade_direction = 0

        # Move to next timestep
        self.index += 1

        # Check for episode termination conditions
        if self.balance < self.initial_balance * 0.9:  # 10% drawdown
            done = True
            truncated = True
            reward -= 0.1  # Additional penalty

        if self.index >= len(self.data):
            done = True
            truncated = True
            observation = None

        else:
            # Update observation handler with current metrics
            self.observation_handler.update_portfolio_metrics(
                self.balance,
                self.position,
                self.position_value,
                self.reward_calculator.max_drawdown,
                self.reward_calculator.winning_trades,
                self.reward_calculator.trade_count,
                self.action_handler.steps_without_action,
            )

            observation = self._get_observation()

        return observation, reward, done, truncated, info

    def _get_observation(self):
        """Get current observation from observation handler"""
        data_point = self.data.iloc[self.index]
        return self.observation_handler.get_observation(data_point)
