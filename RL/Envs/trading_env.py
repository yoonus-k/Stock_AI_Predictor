import numpy as np
import gymnasium as gym
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional, Union
from datetime import datetime, timedelta

# Import components
from .Components.rewards import RewardCalculator
from .Components.actions import ActionHandler
from .Components.observations import ObservationHandler

from Data.Database.db import Database


class TradingEnv(gym.Env):
    def __init__(
        self,
        feutures=None,
        initial_balance=100000,
        reward_type="combined",
        normalize_observations=True,
        normalization_range=(-1.0, 1.0),
        timeframe_id=5,
        stock_id=1,
        start_date="2024-01-01",
        end_date="2025-01-01",
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
            stock_id: Identifier for the stock (used with database connection)
            db_connection: Database connection for dynamic data retrieval
        """
        super(TradingEnv, self).__init__()
        self.feutures = feutures
        self.index = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_price = 1
        self.timeframe = timeframe_id  # Store timeframe information
        self.stock_id = stock_id  # Store stock identifier

        # Initialize database connection and price fetcher if provided
        self.db_connection = Database()
        self.stock_data = self.db_connection.get_stock_data_range(
            stock_id=stock_id,
            timeframe_id=timeframe_id,
            start_date=start_date,
            end_date=end_date,
        )

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
        self.entry_price = 1
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
            action: Action from the agent [action_type, position_size, risk_reward_multiplier, hold_time_hours]

        Returns:
            Tuple: (observation, reward, done, truncated, info)
        """
        # Get current data point
        current_data = self.feutures.iloc[self.index]

        # Process action using action handler
        action_type, position_size, risk_reward_multiplier, hold_time_hours = (
            self.action_handler.process_action(action)
        )

        # Track peak balance for drawdown calculation
        self.peak_balance = max(self.peak_balance, self.balance)

        reward = 0
        base_reward = 0
        done = False
        truncated = False
        trade_pnl_pct = 0
        trade_closed = False
        trade_info = {}

        # Get pattern information from data
        pattern_action = current_data.get(
            "action", 0
        )  # Execute trade based on action parameters
        if action_type != 0:  # If not HOLD
            # Execute trade and get results
            trade_pnl_pct, trade_closed, trade_info = self.execute_trade(
                action_type,
                position_size,
                risk_reward_multiplier,
                hold_time_hours,
                pattern_action,
            )

            # Update balance
            self.balance *= trade_pnl_pct + 1

            # Update trade metrics in observation handler when trade is completed
            if trade_closed and trade_info:
                self.observation_handler.update_trade_metrics(
                    trade_pnl_pct=trade_info.get("trade_pnl_pct", 0.0),
                    exit_reason=trade_info.get("exit_reason", "time"),
                    holding_hours=trade_info.get("hold_time_hours", 0),
                    balance=self.balance,
                )

            # Small bonus for taking action (adjusted based on pattern matching)
            pattern_match_bonus = 0.001
            if pattern_action == action_type:  # Agent agrees with pattern
                pattern_match_bonus = 0.002  # Double bonus for correct action
            reward += pattern_match_bonus

            self.last_trade_idx = self.index
        else:  # HOLD action
            # Calculate hold penalty
            reward = self.reward_calculator.calculate_hold_penalty(
                self.action_handler.steps_without_action
            )

            # Enhanced holding penalty adjustment based on market conditions
            missed_opportunity_penalty = 0
            if pattern_action != 0 and self.action_handler.steps_without_action > 5:
                # Penalize more for ignoring strong signals
                pattern_probability = current_data.get("probability", 0.5)
                if pattern_probability > 0.7:
                    missed_opportunity_penalty = -0.002

            reward += missed_opportunity_penalty

            # Extended holding penalty adjustment
            if self.action_handler.steps_without_action > 100:
                if self.action_handler.steps_without_action > 200:
                    reward -= 0.001 * (
                        self.action_handler.steps_without_action - 200
                    )  # Progressive penalty

        # Calculate reward based on trade outcome
        if trade_closed:
            # Enhanced reward calculation incorporating multiple factors
            # Base reward considers both raw PnL and position sizing
            base_reward = trade_pnl_pct

            # Adjust base reward based on position size (risk taken)
            # More risk (higher position size) should require higher returns for same reward
            risk_adjustment = 1.0 - (
                0.5 * position_size
            )  # Lower position sizes get bonus
            base_reward *= risk_adjustment

            # Adjust reward based on holding time efficiency
            # Shorter successful trades should be rewarded more
            time_efficiency = min(1.0, 48 / max(6, hold_time_hours))  # 48h is baseline
            base_reward *= 0.8 + (
                0.4 * time_efficiency
            )  # Up to 40% bonus for efficient trades

            # Pattern matching quality bonus
            position_size_modifier = trade_info.get("position_size_modifier", 1.0)
            pattern_quality_bonus = 1.0
            if position_size_modifier == 1.0:  # Perfect match
                pattern_quality_bonus = 1.2  # 20% bonus
            elif position_size_modifier == 0.5:  # Contrary to pattern
                pattern_quality_bonus = 0.8  # 20% penalty
            base_reward *= pattern_quality_bonus

            # Bonus for hitting take profit vs stop loss
            exit_reason = trade_info.get("exit_reason", "")
            if exit_reason == "tp":
                base_reward *= 1.2  # 20% bonus for taking profit
            elif exit_reason == "sl":
                base_reward *= 0.9  # 10% penalty for stop loss

            # Risk-reward achievement bonus
            rrr_achieved = trade_info.get("rrr_achieved", 0)
            rrr_target = trade_info.get("rrr_target", 1.0)
            if rrr_achieved > 0 and rrr_target > 0:
                rrr_ratio = min(rrr_achieved / rrr_target, 2.0)  # Cap at 200%
                rrr_bonus = 1.0 + (
                    0.1 * (rrr_ratio - 1.0)
                )  # +/- 10% based on RRR achievement
                base_reward *= max(0.8, min(1.2, rrr_bonus))  # Limit impact to +/- 20%

            # Update reward calculator metrics with actual PnL
            self.reward_calculator.update_metrics(self.balance, trade_pnl_pct)

            # Calculate final reward using risk-adjusted metrics
            reward = self.reward_calculator.calculate_reward(base_reward, trade_pnl_pct)

        # Build info dictionary
        info = {
            "portfolio_balance": self.balance,
            "position_size": position_size,
            "risk_reward_ratio": trade_info.get("rrr_achieved", risk_reward_multiplier),
            "hold_time_hours": hold_time_hours,
            "entry_price": self.entry_price,
            "trade_direction": self.trade_direction,
            "trade_pnl_pct": trade_pnl_pct,
            "base_reward": base_reward,
            "pattern_action": pattern_action,
            "max_drawdown": self.reward_calculator.max_drawdown,
        }

        # Add trade info and action statistics
        if trade_info:
            info.update(trade_info)
        info.update(self.action_handler.get_action_statistics())

        # Move to next timestep and reset the trade parameters
        self.index += 1
        self.entry_price = 1
        self.tp_price = 0
        self.sl_price = 0
        self.trade_direction = 0

        # Check for episode termination conditions
        if self.balance < self.initial_balance * 0.8:  # 10% drawdown
            done = True
            truncated = True
            reward -= 0.1  # Additional penalty
        #print(self.index, len(self.feutures))
        if self.index  >= len(self.feutures):
            done = True
            truncated = True
            observation = None
        else:
            # Update observation handler with current metrics
            self.observation_handler.update_portfolio_metrics(
                self.balance,
                self.reward_calculator.max_drawdown,
                self.reward_calculator.winning_trades,
                self.reward_calculator.trade_count,
                self.action_handler.steps_without_action,
            )
            observation = self._get_observation()

        return observation, reward, done, truncated, info

    def _get_observation(self):
        """Get current observation from observation handler"""
        data_point = self.feutures.iloc[self.index]
        return self.observation_handler.get_observation(data_point)

    def execute_trade(
        self,
        action_type: int,
        position_size: float,
        risk_reward_multiplier: float,
        hold_time_hours: int,
        pattern_action: int,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Execute a trade based on the action parameters and stock data

        Args:
            action_type: Action type (1=BUY, 2=SELL)
            position_size: Fraction of portfolio to risk (0.05-1.0)
            risk_reward_multiplier: Multiplier for risk-reward ratio (0.5-3.0)
            hold_time_hours: Position holding time in hours (6-168)
            pattern_action: Action suggested by pattern (1=BUY, 2=SELL)

        Returns:
            Tuple[float, bool, Dict]: (trade_pnl, trade_closed, trade_info)
        """
        # Exit early if action is HOLD
        if action_type == 0:
            return 0.0, False, {"trade_status": "hold"}  # Get current data point
        current_data = self.feutures.iloc[self.index]

        # Get current timestamp from data , the date is the index of the DataFrame
        current_timestamp = self.feutures.index[self.index]

        self.entry_price = self.stock_data.loc[current_timestamp]["close_price"]

        # Calculate TP, SL and position size modifier based on pattern match
        self.tp_price, self.sl_price, position_size_modifier = (
            self.action_handler.calculate_trade_targets(
                action_type,
                self.entry_price,
                current_data.get("max_gain", 0.01),
                current_data.get("max_drawdown", -0.01),
                risk_reward_multiplier,
                pattern_action,
            )
        )

        # Apply position size modifier based on pattern match quality
        adjusted_position_size = position_size * position_size_modifier

        # Set trade direction
        self.trade_direction = 1 if action_type == 1 else -1  # 1=Long, -1=Short

        # Simulate trade outcome
        trade_closed = True
        exit_reason = "normal"
        exit_price = self.entry_price
        trade_pnl_pct = 0.0
        actual_hold_time = hold_time_hours
        gain_risk_reward = 0.0  # in RR

        if current_timestamp is not None:
            # Simulate the trade
            exit_price, exit_reason, actual_hold_time, gain_risk_reward = (
                self.simulate_trade_outcome(
                    current_timestamp=current_timestamp,
                    entry_price=self.entry_price,
                    tp_price=self.tp_price,
                    sl_price=self.sl_price,
                    hold_time_hours=hold_time_hours,
                    trade_direction=self.trade_direction,
                )
            )
        # Calculate trade metrics
        trade_pnl_pct = adjusted_position_size * gain_risk_reward

        # Build trade info dictionary with enhanced metrics
        trade_info = {
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "trade_direction": self.trade_direction,
            "position_size": adjusted_position_size,
            "position_size_modifier": position_size_modifier,
            "risk_reward_multiplier": risk_reward_multiplier,
            "trade_pnl_pct": trade_pnl_pct,
            "exit_reason": exit_reason,
            "intended_hold_time_hours": hold_time_hours,
            "actual_hold_time_hours": actual_hold_time,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "rrr_achieved": gain_risk_reward,
            "rrr_target": risk_reward_multiplier,
            "pattern_match_quality": (
                "full"
                if position_size_modifier == 1.0
                else (
                    "partial"
                    if position_size_modifier == 0.75
                    else "contrary" if position_size_modifier == 0.5 else "none"
                )
            ),
        }

        return trade_pnl_pct, trade_closed, trade_info

    def simulate_trade_outcome(
        self,
        current_timestamp: datetime,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        hold_time_hours: int,
        trade_direction: int,
    ) -> Tuple[float, str, float, float]:
        """
        Simulate a trade outcome based on price data

        Args:
            entry_price: Entry price
            tp_price: Take profit price
            sl_price: Stop loss price
            hold_time_hours: Maximum holding time in hours
            trade_direction: 1=Long, -1=Short
            timeframe_id: Timeframe identifier
            price_data: Historical price data

        Returns:
            Tuple[float, str, float]: (exit_price, exit_reason, actual_hold_time,gain)
        """
        price_data = self.stock_data
        if price_data.empty:
            return entry_price, "no_data", 0

        # Calculate end timestamp
        end_timestamp = current_timestamp + timedelta(hours=float(hold_time_hours))
        # filter the desired price data range
        price_data = price_data[
            (price_data.index >= current_timestamp)
            & (price_data.index <= end_timestamp)
        ]

        # Default values if no conditions met
        exit_price = entry_price
        exit_reason = "time"
        actual_hold_time = hold_time_hours
        gain_risk_reward = 0.0  # in RR

        # Simulate trade outcome , whether it hits TP, SL or just times out
        # first get the maximum and minimum prices during the trade duration

        # get the close prices for the trade duration
        close_prices = price_data["close_price"].values
        max_price = np.max(close_prices)
        min_price = np.min(close_prices)

        # if the trade is long
        if trade_direction == 1:
            # Check if SL is hit
            if min_price <= sl_price:
                exit_price = sl_price
                exit_reason = "sl"

                sl_hits = price_data[
                    price_data["close_price"] <= sl_price
                ]  # Use <= instead of ==
                if len(sl_hits) > 0:
                    actual_hold_time = (
                        sl_hits.index[0] - current_timestamp
                    ).total_seconds() / 3600.0

                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)
            # Check if TP is hit
            elif max_price >= tp_price:
                exit_price = tp_price
                exit_reason = "tp"

                tp_hits = price_data[price_data["close_price"] >= tp_price]
                if len(tp_hits) > 0:
                    # Use the first occurrence of TP hit
                    actual_hold_time = (
                        tp_hits.index[0] - current_timestamp
                    ).total_seconds() / 3600.0

                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)
            else:
                # Trade times out
                exit_price = close_prices[-1]
                actual_hold_time = hold_time_hours
                exit_reason = "time"
                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)
        else:  # Short trade
            # Check if SL is hit
            if max_price >= sl_price:
                exit_price = sl_price
                exit_reason = "sl"
                
                sl_hits = price_data[
                    price_data["close_price"] >= sl_price
                ]
                if len(sl_hits) > 0:
                    # Use the first occurrence of SL hit
                    actual_hold_time = (
                        sl_hits.index[0] - current_timestamp
                    ).total_seconds() / 3600.0
    
                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)
            # Check if TP is hit
            elif min_price <= tp_price:
                exit_price = tp_price
                exit_reason = "tp"
                
                tp_hits = price_data[price_data["close_price"] <= tp_price]
                if len(tp_hits) > 0:
                    # Use the first occurrence of TP hit
                    actual_hold_time = (
                        tp_hits.index[0] - current_timestamp
                    ).total_seconds() / 3600.0
                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)
            else:
                # Trade times out
                exit_price = close_prices[-1]
                actual_hold_time = hold_time_hours
                exit_reason = "time"
                gain_risk_reward = (exit_price - entry_price) / (entry_price - sl_price)

        return exit_price, exit_reason, actual_hold_time, gain_risk_reward
