"""
Enhanced Trading Environment (V2) for Reinforcement Learning

Features:
- Multiple concurrent positions (up to 10 active trades)
- Persistent positions across multiple timesteps
- Dynamic tracking of equity and unrealized PnL
- Vectorized operations for performance
- Realistic reward function based on portfolio performance
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import gymnasium as gym
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional, Union
from datetime import datetime, timedelta
from collections import OrderedDict

# Import components
from .Components.rewards import RewardCalculator
from .Components.actions import ActionHandler
from .Components.observations import ObservationHandler
from .Components.trading_state import TradingState


from Data.Database.db import Database


class TradingEnv(gym.Env):
    """
    Enhanced Trading Environment (V2) for Reinforcement Learning
    
    This environment simulates realistic trading with persistent positions
    across multiple timesteps. Key features include:
    - Multiple concurrent positions (up to 10 active trades)
    - Position management with TP/SL/time exits
    - Dynamic equity and unrealized PnL tracking
    - Proper margin trading with leverage limits
    - Vectorized operations for performance
    - Portfolio-based reward function
    """
    def __init__(
        self,
        features=None,
        initial_balance=100000,
        reward_type="combined",
        normalize_observations=True,
        normalization_range=(-1.0, 1.0),
        timeframe_id=5,
        stock_id=1,
        start_date="2024-01-01",
        end_date="2025-01-01",
        max_positions=10,
        commission_rate=0.001,
        enable_short=True,
        margin_requirement=0.01,  # 1% margin requirement by default
        max_leverage=100,  # 100x max leverage by default
        verbose=False
    ):
        """
        Initialize the trading environment with data and parameters.

        Args:
            features: DataFrame with market features
            initial_balance: Starting account balance
            reward_type: Type of reward function to use
            normalize_observations: Whether to normalize observations
            normalization_range: Range for normalized observations
            timeframe_id: Identifier for the timeframe
            stock_id: Identifier for the stock
            start_date: Start date for data range
            end_date: End date for data range
            max_positions: Maximum number of concurrent positions
            commission_rate: Transaction cost as a percentage
            enable_short: Whether to allow short selling
            margin_requirement: Percentage of position value required as margin (e.g., 0.1 = 10%)
            max_leverage: Maximum allowed leverage (e.g., 10 = 10x)
        """
        super(TradingEnv, self).__init__()
        self.verbose = verbose
        
        # Store original features for reference
        self.original_features = features 
        
        # Process the features dataframe
        if features is not None:
            # Drop config_id and timeframe_id columns if they exist
            if "config_id" in features.columns:
                features = features.drop(columns=["config_id"])
            if "timeframe_id" in features.columns:
                features = features.drop(columns=["timeframe_id"])
                
        self.features = features
        
        # Initialize basic parameters
        self.index = 0  # Index for time progression through stock data
        self.timeframe = timeframe_id
        self.stock_id = stock_id
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.enable_short = enable_short
        self.margin_requirement = margin_requirement
        self.max_leverage = max_leverage
        
        # Initialize centralized trading state
        self.trading_state = TradingState(initial_balance=initial_balance)
        
        # Initialize database connection and fetch price data
        self.db_connection = Database()
        
        if features is not None:
            features_start_date = self.features.index[0].strftime('%Y-%m-%d %H:%M:%S')
            # Get the end date from the features index
            features_end_date = self.features.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            self.stock_data = self.db_connection.get_stock_data_range(
                stock_id=stock_id,
                timeframe_id=timeframe_id,
                start_date=features_start_date,
                end_date=features_end_date,
            )
            
            # Pre-calculate numpy arrays for faster access
            self.close_price_array = np.array(self.stock_data["close_price"])
            self.high_price_array = np.array(self.stock_data["high_price"])
            self.low_price_array = np.array(self.stock_data["low_price"])
            self.timestamp_array = np.array(self.stock_data.index, dtype='datetime64[ns]')
            
            # setup fast access to features data
            # Convert features DataFrame to numpy array for faster access
            self._features_array = self.features.values
            numpy_timestamps = np.array(self.features.index).astype('datetime64[ns]')
            self._timestamp_to_index = {np.datetime64(ts, 'ns'): idx for idx, ts in enumerate(numpy_timestamps)}
            self._feature_columns = self.features.columns
            self._feature_name_to_idx = {name: idx for idx, name in enumerate(self.features.columns)}
            
            self.latest_features_timestamp = self.features.iloc[0].name
            self.latest_features_data = self._features_array[0]
            
            # Detect timeframe from data
            if len(self.timestamp_array) > 1:
                # Calculate time difference between first two timestamps
                time_diff = self.timestamp_array[1] - self.timestamp_array[0]
                time_diff_seconds = time_diff.astype('timedelta64[s]').astype(np.int64)
                self.time_step_hours = time_diff_seconds / 3600
            else:
                # Default if can't determine
                self.time_step_hours = 1.0
            
        # Initialize modular components
        self.reward_calculator = RewardCalculator(reward_type=reward_type)
        self.action_handler = ActionHandler()
        self.observation_handler = ObservationHandler(
            normalize_observations=normalize_observations,
            normalization_range=normalization_range,
        )
        
        # Set trading state in all components
        self.reward_calculator.set_trading_state(self.trading_state)
        self.action_handler.set_trading_state(self.trading_state)
        self.observation_handler.set_trading_state(self.trading_state)

        # Set up action and observation spaces
        self.action_space = self.action_handler.action_space
        self.observation_space = self.observation_handler.observation_space

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple: (observation, info)
        """
        # Reset indices and centralized trading state
        self.index = 0
        self.trading_state.reset()
        
        # Reset features data tracking
        if self.features is not None:
            self.latest_features_timestamp = self.features.iloc[0].name
            self.latest_features_data = self._features_array[0]

        # Reset components with shared trading state
        self.reward_calculator.reset()
        self.action_handler.reset()
        self.observation_handler.reset(self.trading_state.initial_balance)

        super().reset(seed=seed)

        observation = self.observation_handler.get_observation(self.latest_features_data)
        info = {}

        return observation, info      
    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Action from the agent [action_type, position_size, risk_reward_multiplier, hold_time_hours]

        Returns:
            Tuple: (observation, reward, done, truncated, info)
            
        Process:
            1- Process existing trades first
                This updates available margin when trades close
                Calculates unrealized PnL for remaining positions
                Updates trade statistics (win/loss counts, etc.)
                
            2- Update portfolio metrics (equity, drawdown)
                Calculate equity based on balance + unrealized PnL
                Update drawdown metrics based on new equity
                This needs to happen after trade processing but before new trade execution
            
            3- Execute new trades last
                Uses most up-to-date available margin after closes
                Uses current equity for position sizing
                Won't affect calculations for existing positions
                
            4- Final portfolio update

                Update observation handler with final portfolio state
                Calculate reward based on complete portfolio changes
        """
        # Get current price data from stock data index
        current_price = self.close_price_array[self.index]
        current_timestamp = self.timestamp_array[self.index]
        # Convert timestamp to pandas Timestamp for consistent datetime handling
        current_timestamp = pd.Timestamp(current_timestamp)
        
        current_data = self.latest_features_data
        
        # Store initial state for reward calculation
        initial_equity = self.trading_state.equity
        
        # Process action using action handler
        action_type, position_size, risk_reward_multiplier, hold_time_hours = (
            self.action_handler.process_action(action)
        )
        #######################################
        ####### Process existing trades first
        #######################################
        # Process existing trades first - check for TP/SL/time exits and calculate unrealized PnL
        closed_positions = self._process_trades(current_price, current_timestamp)
        
        #######################################
        ####### Update portfolio metrics (equity, drawdown)
        #######################################
        # Update equity (actual balance + unrealized PnL)
        self.trading_state.update_equity(self.trading_state.balance + self.trading_state.unrealized_pnl)
        
        #######################################
        ####### Execute new trades last
        #######################################
        # Execute new trade if not a HOLD action, we have capacity, and features data is available
        new_position = None
        if action_type != 0 and len(self.trading_state.active_positions) < self.max_positions:
            new_position = self._execute_new_trade(
                current_data,
                action_type,
                position_size,
                risk_reward_multiplier,
                hold_time_hours,
                current_price,
                current_timestamp
            )
            
            # Track action frequency
            self.trading_state.steps_without_action = 0
        else:
            # No action taken, increment counter
            self.trading_state.steps_without_action += 1          
        
        # Calculate reward based on change in equity and other factors
        reward = self._calculate_reward(
            initial_equity=initial_equity,
            closed_positions=closed_positions,
            new_position=new_position,
            action_type=action_type,
            current_data=current_data
        )
         

        # Check if episode is finished
        done = False
        truncated = False
        
        # Terminate if significant drawdown (80%)
        if self.trading_state.max_drawdown <= -0.8:
            done = True
            truncated = True
            reward -= 0.1  # Additional penalty
            
        # Terminate if we've reached the end of the dataset
        if self.index >= len(self.stock_data) - 1:
            done = True
            truncated = True
            self.index -= 1  # Stay on last index to avoid out of bounds
        
        # Build info dictionary with detailed metrics
        if self.verbose:
            info = self._build_info_dict(action_type, position_size, risk_reward_multiplier, 
                                        closed_positions, new_position)
        else:
            info = {
                'action_type': action_type,
                'position_size': position_size,
                'risk_reward_multiplier': risk_reward_multiplier,
                'closed_positions_count': len(closed_positions),
                'new_position': new_position is not None,
                'equity': self.trading_state.equity,
                'unrealized_pnl': self.trading_state.unrealized_pnl,
                'drawdown': self.trading_state.drawdown,
                'max_drawdown': self.trading_state.max_drawdown,
            }
        
        # Move to next timestep
        self.index += 1
        
        #######################################
        ####### Get next Obs
        #######################################   
        # get the next timestamp
       
        next_timestamp = self.timestamp_array[self.index]
        # Check if features data exists for current timestamp
        # O(1) dictionary lookup instead of O(n) DataFrame index scan
        if next_timestamp in self._timestamp_to_index:
            # Direct array indexing - O(1) operation
            idx = self._timestamp_to_index[next_timestamp]
            features_vector = self._features_array[idx]
            
            # Only create Series if absolutely needed by downstream code
            #current_data = pd.Series(features_vector, index=self._feature_columns)
            
            self.latest_features_data = features_vector
            self.latest_features_timestamp = next_timestamp
        
        # Prepare the observationself.timestamp_array
        observation = self.observation_handler.get_observation(self.latest_features_data)
        
        
        return observation, reward, done, truncated, info

    def _process_trades(self, current_price: float, current_timestamp):
        """
        Process all active trades to check for exits (TP, SL, time) and calculate unrealized PnL
        
        Args:
            current_price: Current price
            current_timestamp: Current timestamp
            
        Returns:
            list: List of positions that were closed in this step
        """
        closed_positions = []
        positions_to_keep = []
        total_unrealized_pnl = 0
        
        # Get high and low prices for this candle for more accurate TP/SL checking
        current_high = self.high_price_array[self.index]
        current_low = self.low_price_array[self.index]
        
        for position in self.trading_state.active_positions:
            # Check if the position should be closed
            exit_reason = None
            exit_price = current_price
            
            # For long positions
            if position['direction'] == 1:
                # Check for SL hit first using low price (considering wicks)
                if current_low <= position['sl_price']:  
                    exit_reason = "sl"
                    exit_price = position['sl_price']  # Use SL price as exit price
                # Then check for TP hit using high price (considering wicks)
                elif current_high >= position['tp_price']:  
                    exit_reason = "tp"
                    exit_price = position['tp_price']  # Use TP price as exit price
            # For short positions
            elif position['direction'] == -1:
                # Check for SL hit first using high price (considering wicks)
                if current_high >= position['sl_price']:  
                    exit_reason = "sl"
                    exit_price = position['sl_price']
                # Then check for TP hit using low price (considering wicks)
                elif current_low <= position['tp_price']:  
                    exit_reason = "tp"
                    exit_price = position['tp_price']
                    
            # Check for time-based exit
            if self.index >= position['exit_index']:
                # Only apply time exit if no other exit condition was met
                if exit_reason is None:
                    exit_reason = "time"

            # Process position exit if needed
            if exit_reason is not None:
                # Release margin reservation
                self.trading_state.margin_used -= position['margin_required']
                self.trading_state.available_margin += position['margin_required']
                
                # Calculate PnL (exit_price - entry_price) * direction * size
                price_diff = (exit_price - position['entry_price']) * position['direction']
                pnl = price_diff * position['size']
                
                # Calculate PnL as percentage of equity at entry
               # Approximate equity at entry
                pnl_pct = pnl / self.trading_state.balance
                
                # Update balance with PnL
                self.trading_state.balance += pnl
      
                # Calculate actual hold time in hours
                hold_time = (current_timestamp - position['entry_timestamp']).total_seconds() / 3600
                
                # Prepare closed position data
                closed_position = position.copy()
                closed_position.update({
                    'exit_price': exit_price,
                    'exit_timestamp': current_timestamp,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'actual_hold_time': hold_time,
                })
                
                # Add to closed positions list and trade history
                closed_positions.append(closed_position)
                self.trading_state.trade_history.append(closed_position)
                
                # Update trading state with trade metrics
                self.trading_state.update_from_trade(
                    trade_pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    holding_hours=hold_time
                )

            else:
                # Calculate unrealized PnL for this position (no commission)
                price_diff = (current_price - position['entry_price']) * position['direction']
                position_pnl = price_diff * position['size']
                
                # Add to total unrealized PnL
                total_unrealized_pnl += position_pnl
                
                # Keep the position active
                positions_to_keep.append(position)
                
        # Update active positions list
        self.trading_state.active_positions = positions_to_keep
        
        # Store total unrealized PnL
        self.trading_state.unrealized_pnl = total_unrealized_pnl
        
        return closed_positions

    def _execute_new_trade(self, current_data, action_type, position_size, risk_reward_multiplier, 
                         hold_time_hours, current_price, current_timestamp):
        """
        Execute a new trade based on the action
        
        Args:
            current_data: Current data point
            action_type: Action type (1=BUY, 2=SELL)
            position_size: Position size as fraction of portfolio
            risk_reward_multiplier: Risk-reward ratio multiplier
            hold_time_hours: Intended hold time in hours
            current_price: Current price
            current_timestamp: Current timestamp
            
        Returns:
            dict: The new position or None if no trade executed
        """
        # Exit early if action is HOLD
        if action_type == 0:
            return None
        
        # Check if we can open a short position
        if action_type == 2 and not self.enable_short:
            return None
        
        # Get pattern information from data
        pattern_action = current_data[self._feature_name_to_idx.get("action", 0)]
        
        # Calculate TP, SL and position size modifier based on pattern match
        tp_price, sl_price, position_size_modifier = (
            self.calculate_trade_targets(
                action_type,
                current_price,
                current_data[self._feature_name_to_idx.get("max_gain", 0)],
                current_data[self._feature_name_to_idx.get("max_drawdown", 0)],
                risk_reward_multiplier,
                pattern_action,
            )
        )
        
        # Apply position size modifier based on pattern match quality
        adjusted_position_size = position_size * position_size_modifier
        
        # Set trade direction
        direction = 1 if action_type == 1 else -1  # 1=Long, -1=Short        
        
        # Calculate position size in units
        risk_value = self.trading_state.equity * adjusted_position_size
        position_units = (direction*risk_value) / (current_price - sl_price)

        # Calculate position value and required margin
        position_value = position_units * current_price
        margin_required = position_value * self.margin_requirement
        
        # Check for sufficient available margin
        if margin_required > self.trading_state.available_margin:
            # Not enough available margin - reject the trade
            return None
        
        # Generate unique position ID
        position_id = self.trading_state.generate_position_id()
        
        # Calculate exit index (how many steps to hold)
        steps_to_hold = int(round(hold_time_hours / self.time_step_hours))
        
        new_position = {
            'id': position_id,
            'direction': direction,
            'size': position_units,
            'entry_price': current_price,
            'entry_timestamp': current_timestamp,
            'exit_index': self.index+steps_to_hold,
            'entry_index': self.index,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'hold_time': hold_time_hours,
            'risk_reward_multiplier': risk_reward_multiplier,
            'position_size': adjusted_position_size,
            'pattern_match': pattern_action == action_type,
            'position_size_modifier': position_size_modifier,
            'position_value': position_value,
            'margin_required': margin_required
        }
        
        # Update available margin (subtract margin required)
        self.trading_state.update_margin(self.trading_state.margin_used + margin_required)
        
        # Add position to active positions list
        self.trading_state.active_positions.append(new_position)
        
        return new_position

    def _calculate_reward(self, initial_equity, closed_positions, new_position, action_type, current_data):
        """
        Calculate reward based on portfolio performance and actions
        
        Args:
            initial_equity: Equity at the beginning of the step
            closed_positions: List of positions closed in this step
            new_position: New position opened in this step, if any
            action_type: Action type (0=HOLD, 1=BUY, 2=SELL)
            current_data: Current data point
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        pattern_action = current_data[self._feature_name_to_idx.get("action", 0)]
        pattern_prob = current_data[self._feature_name_to_idx.get("probability", 0.5)]
        
        # 1. Portfolio change component - reward based on equity growth
        equity_change_pct = (self.trading_state.equity - initial_equity) / initial_equity
        reward += equity_change_pct * 10  # Scale up for meaningful rewards
        
        pnl_pct = 0.0
        pnl_pct_total = 0.0
        # 2. Process closed positions rewards
        for position in closed_positions:
            # Base reward on P&L
            pnl_pct = position['pnl_pct']
            pnl_pct_total += pnl_pct
            pnl_reward = pnl_pct * 5  # Scale factor for closed trades
            
            # Adjust reward based on exit reason
            if position['exit_reason'] == 'tp':
                pnl_reward *= 1.2  # Bonus for hitting take profit
            elif position['exit_reason'] == 'sl':
                pnl_reward *= 0.8  # Slight penalty for hitting stop loss
                
            # Adjust reward based on hold time efficiency
            hold_time_efficiency = min(1.0, position['hold_time'] / position['actual_hold_time']) if position['actual_hold_time'] > 0 else 0.5
            pnl_reward *= (0.8 + 0.4 * hold_time_efficiency)  # Reward efficiency
            
            # Add to total reward
            reward += pnl_reward
            
        # 3. Trade execution quality rewards
        if action_type != 0:  # If an action was taken
    
            # Reward/penalty based on pattern match
            if pattern_action == action_type:  # Action aligns with pattern
                pattern_match_reward = 0.002 * pattern_prob  # Scale by pattern confidence
                reward += pattern_match_reward
            elif pattern_action != 0 and pattern_action != action_type:  # Action contradicts pattern
                pattern_mismatch_penalty = -0.001 * pattern_prob  # Small penalty
                reward += pattern_mismatch_penalty
        
        # 4. Holding penalty for extended periods of inaction
        if action_type == 0:  # HOLD action
            # Basic hold penalty from reward calculator
            hold_penalty = self.reward_calculator.calculate_hold_penalty(
                self.trading_state.steps_without_action
            )
            
            # Enhanced holding penalty for ignoring strong signals
            if pattern_action != 0 and self.trading_state.steps_without_action > 5:
                if pattern_prob > 0.7:
                    hold_penalty -= 0.002  # Stronger penalty for ignoring clear signals
                    
            reward += hold_penalty
            
        # 5. Risk management rewards/penalties
        # Penalize excessive drawdowns
        if self.trading_state.drawdown < -0.1:  # More than 10% drawdown
            drawdown_penalty = self.trading_state.drawdown * 0.5  # Scale factor
            reward += drawdown_penalty
            
        # Reward efficient use of capital (having active positions)
        capital_usage = len(self.trading_state.active_positions) / self.max_positions
        if capital_usage > 0:
            capital_usage_reward = 0.0005 * capital_usage  # Small reward for using capital
            reward += capital_usage_reward
            
        # 6. Apply risk-adjusted reward via reward calculator
        final_reward = self.reward_calculator.calculate_reward(reward, pnl_pct_total)
            
        return final_reward
        
    def _build_info_dict(self, action_type, position_size, risk_reward_multiplier, 
                        closed_positions, new_position):
        """
        Build information dictionary for step return
        
        Args:
            action_type: Action type (0=HOLD, 1=BUY, 2=SELL)
            position_size: Position size as fraction of portfolio
            risk_reward_multiplier: Risk-reward ratio multiplier
            closed_positions: List of positions closed in this step
            new_position: New position opened in this step, if any
            
        Returns:
            dict: Information dictionary
        """
        # Calculate win rate
        win_rate = self.trading_state.get_win_rate()
            
        # Calculate profit factor
        profit_factor = 1.0
        if self.trading_state.loss_amount > 0:
            profit_factor = self.trading_state.win_amount / self.trading_state.loss_amount
  
        # Basic info
        info = {
            'cash_balance': self.trading_state.balance,
            'available_margin': self.trading_state.available_margin,
            'equity': self.trading_state.equity,
            'unrealized_pnl': self.trading_state.unrealized_pnl,
            'margin_used': self.trading_state.margin_used,
            'max_leverage': self.max_leverage,
            'drawdown': self.trading_state.drawdown,
            'max_drawdown': self.trading_state.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': self.trading_state.trade_count,
            'active_positions': len(self.trading_state.active_positions),
            'tp_exits': self.trading_state.tp_exits,
            'sl_exits': self.trading_state.sl_exits,
            'time_exits': self.trading_state.time_exits,
        }
        
        # Add action details
        info.update({
            'action_type': action_type,
            'position_size': position_size,
            'risk_reward': risk_reward_multiplier,
        })
        
        # Add new position details if any
        if new_position:
            info['new_position'] = {
                'id': new_position['id'],
                'direction': 'long' if new_position['direction'] == 1 else 'short',
                'size': new_position['size'],
                'entry_price': new_position['entry_price'],
                'tp_price': new_position['tp_price'],
                'sl_price': new_position['sl_price'],
                'hold_time': new_position['hold_time'],
            }
            
        # Add closed position details if any
        if closed_positions:
            info['closed_positions'] = []
            for pos in closed_positions:
                info['closed_positions'].append({
                    'id': pos['id'],
                    'direction': 'long' if pos['direction'] == 1 else 'short',
                    'pnl': pos['pnl'],
                    'pnl_pct': pos['pnl_pct'],
                    'exit_reason': pos['exit_reason'],
                    'hold_time': pos['actual_hold_time'],
                })
        
        return info
    
    def calculate_trade_targets(self, action_type: int, entry_price: float, max_gain: float, 
                              max_drawdown: float, risk_reward_multiplier: float,
                              pattern_action: int = 0) -> Tuple[float, float, float]:
        """
        Calculate trade targets (TP, SL) based on action and market context
        
        Args:
            action_type: Action type (1=BUY, 2=SELL)
            entry_price: Current/entry price
            max_gain: Maximum expected gain from pattern
            max_drawdown: Maximum expected drawdown from pattern
            risk_reward_multiplier: Risk-reward ratio multiplier
            pattern_action: Pattern's suggested action (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            Tuple[float, float, float]: (tp_price, sl_price, position_size_modifier)
            - tp_price: Take profit price level
            - sl_price: Stop loss price level
            - position_size_modifier: Adjustment factor for position size based on conviction
        """
     
        # Calculate price movements adjusting for direction
        if action_type == 1:  # BUY (LONG)
            # Long position: TP above entry, SL below entry
            tp_move = abs(max_gain) * risk_reward_multiplier  # Scaled by RR multiplier
            sl_move = abs(max_drawdown)*1.5  # add 50% buffer to drawdown
            
            tp_price = entry_price * (1 + tp_move)
            sl_price = entry_price * (1 - sl_move)
        else:  # SELL (SHORT)
            # Short position: TP below entry, SL above entry
            tp_move = abs(max_gain) * risk_reward_multiplier  # Scaled by RR multiplier
            sl_move = abs(max_drawdown) *1.5  # add 50% buffer to drawdown
            
            tp_price = entry_price * (1 - tp_move)
            sl_price = entry_price * (1 + sl_move)
        
        # Calculate position size modifier based on pattern match
        position_size_modifier = 1.0  # Default, no adjustment
        
        # If pattern suggests opposite action, reduce size
        if pattern_action != 0 and pattern_action != action_type:
            position_size_modifier = 0.5  # Half size when going against pattern
        
        # If pattern confirms action, increase size
        elif pattern_action == action_type:
            position_size_modifier = 1.0  # 50% larger size when confirmed
        
        return tp_price, sl_price, position_size_modifier

    def render(self):
        """
        Render the environment (not implemented)
        """
        pass

    def close(self):
        """
        Close the environment
        """
        pass

