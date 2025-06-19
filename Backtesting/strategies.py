"""
RL Trading Strategies for Backtesting.py

Professional trading strategies using backtesting.py framework.
Maintains compatibility with existing RL models while leveraging
backtesting.py's built-in features for clean, efficient backtesting.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import backtesting.py
try:
    from Lib.Backtesting import Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    Strategy = object  # Fallback
    BACKTESTING_AVAILABLE = False
    print("⚠️ backtesting.py not installed. Install with: pip install backtesting")

# Import RL observation normalizer if available
try:
    from RL.Envs.Components.observation_normalizer import ObservationNormalizer
except ImportError:
    ObservationNormalizer = None

logger = logging.getLogger(__name__)


class RLTradingStrategy(Strategy):
    """
    Professional RL Trading Strategy for backtesting.py
    
    This strategy replicates the essential functionality of the Backtrader RLTradingStrategy
    but uses backtesting.py's clean, efficient framework. It handles:
    
    - MultiDiscrete action space [3, 20, 10, 10] (same as Backtrader version)
    - Multiple position management with hedging
    - Advanced risk management (stop loss, take profit, position sizing)
    - Seamless integration with RL models and observation managers
    """
    # Define required parameters as class variables for backtesting.py
    model = None  # RL model
    observation_manager = None  # Observation manager
    # Risk management parameters (same as Backtrader version)
    max_positions = 10  # Maximum number of concurrent positions
    enable_hedging = True  # Allow hedging positions
    enable_short = True  # Allow short positions
    position_sizing = 'fixed'  # Position sizing method
    
    # Trading parameters
    min_hold_periods = 1  # Minimum hold periods
    max_hold_periods = 48  # Maximum hold periods
    commission = 0.001  # Commission rate
    
    def __init__(self, broker=None, data=None, params=None):
        """Initialize strategy with backtesting.py parameters."""
        # Pass parameters to parent Strategy class
        super().__init__(broker, data, params)

    def init(self):
        """Initialize strategy with RL model and parameters."""
        # Trading parameters (using class variables with defaults)
        self.min_hold_periods = getattr(self, 'min_hold_periods', 1)
        self.max_hold_periods = getattr(self, 'max_hold_periods', 24)
        self.commission_rate = getattr(self, 'commission', 0.001)
        
        # Initialize observation normalizer if available
        self.normalizer = None
        if ObservationNormalizer is not None:
            self.normalizer = ObservationNormalizer(
                output_range=(-1.0, 1.0),
                clip_outliers=True
            )
            
        self.current_observation = None  # Current observation for RL model
            
        self.position_size_devider =1   # Position size divisor for fixed sizing
        
        # Minimal position tracking (just to handle custom logic not in backtesting.py)
        # DEPRECATED: Using self.trades instead - keeping for backward compatibility during migration
        self.position_counter = -1    # Unique position IDs
        
        # Action space mapping
        self.action_mappings = self._initialize_action_mappings()
        self.model_action_history =[]
        
        # Risk management tracking
        self.initial_equity = self._broker._cash
        self.peak_equity = self._broker._cash
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Portfolio metrics for RL observation
        self.trade_wins = 0
        self.trade_losses = 0
        self.trade_total_pnl = 0.0
        self.trade_hours = 0
        self.start_time = self.data.index[0] if len(self.data) > 0 else None
        
        logger.info(f"🚀 RL Strategy initialized with ${self._broker._cash:,.2f}")
        logger.info(f"   Max positions: {self.max_positions}")
        logger.info(f"   Hedging enabled: {self.enable_hedging}")
        logger.info(f"   Short selling enabled: {self.enable_short}")
    
    def _initialize_action_mappings(self) -> Dict[str, Any]:
        """Initialize action space mappings (exact same as Backtrader version)."""
        return {
            'action_type': [0, 1, 2],  # HOLD, BUY, SELL
            'position_sizes': [0.001 + i * 0.001 for i in range(20)],  # 0.001 to 0.02 (20 steps)
            'risk_rewards': [0.5 + i * 0.25 for i in range(10)],  # 0.5x to 3.0x
            'hold_times': [3 + i * 3 for i in range(10)]  # 3h to 30h
        }

    def next(self):
        """
        Execute one step of the trading strategy.
        
        Order of operations:
        1. Update equity tracking for risk metrics
        2. Check for automatic closed positions
        3. Check time-based exits
        4. Get RL action and execute new trades
        5. Apply portfolio-level risk management
        """
        try:
            current_bar = len(self.data) - 1
            current_price = self.data.Close[-1]
            current_time = self.data.index[-1]
            
            # 1. Update equity tracking for risk management
            current_equity = self.equity

            self._update_drawdown_tracking(current_equity)
            
            # 5. Final risk management check
            self._manage_risk_limits(current_equity)
            
            # 4. Get RL action and execute new trades (only if not at risk limits)
            if self._can_open_new_position(current_equity):
                actions = self._get_rl_action(current_bar)
                # Execute new action
                self._execute_action(actions, current_price, current_time)
            
        except Exception as e:
            logger.error(f"Error in strategy.next(): {e}")
            
            
    def _get_rl_action(self, current_bar: int) -> List[int]:
        """Get action from RL model."""
        if not self.model or not self.observation_manager:
            # Fallback raise an error if model or observation manager not set
            logger.error("RL model or observation manager not set, cannot get action")
        
        try:
            # get the current time from the data
            current_time = self.data.index[current_bar]
            
            # Get market observation from observation manager
            self.current_observation = self.observation_manager.get_market_observation(current_time)


            # Calculate portfolio features
            portfolio_features = self._calculate_portfolio_features()
            
            # Combine market and portfolio features
            observation_array =np.concatenate([self.current_observation , portfolio_features]) 
    
    
            if self.normalizer:
                observation_array = self.normalizer.normalize_observation(observation_array)
                
            # Get RL model prediction with the complete observation
            raw_action, _ = self.model.predict(observation_array, deterministic=True)
            
            action_type, position_size_idx, risk_reward_idx, hold_time_idx = raw_action
            
            # Map action indices to actual values
            position_size = self.action_mappings['position_sizes'][min(position_size_idx, 19)]
            position_size/=self.position_size_devider  # Apply position size divisor
            risk_reward = self.action_mappings['risk_rewards'][min(risk_reward_idx, 9)]
            hold_time = self.action_mappings['hold_times'][min(hold_time_idx, 9)]
            
            # record the model's action history
            self.model_action_history.append({
                'action_type': action_type,
                'position_size': position_size,
                'risk_reward': risk_reward,
                'hold_time': hold_time,
                'timestamp': current_time
            })
            
            
            # Log the action
            # convert back to list
            actions = [action_type, position_size, risk_reward, hold_time]
                
            return actions
            
        except Exception as e:
            logger.warning(f"RL prediction failed: {e}, using fallback")
            return [0, 0, 0, 0]  # Hold on error
    
            
    def _execute_action(self, actions: List[int], current_price: float, current_time):
        """Execute trading action based on RL model output."""
        action_type, position_size, risk_reward, hold_time = actions
        
        #if action is 0 , then return
        if action_type == 0:
            logger.info("🤚 HOLD action received, no trade executed")
            return
        
        # Get cash available
        equity = self.equity
        risk_value = equity * position_size
        
        # get the current observations
        obs = self.current_observation
        
        max_drawdown = abs(obs[4]) # Maximum drawdown in index 4 refer to the observation structure
        max_gain = abs(obs[3])  # Maximum gain in index 5
        
        # Action type: 0=Hold, 1=Buy, 2=Sell
        if action_type == 1:  # BUY
            # Calculate entry/stops
            stop_pct =   max_drawdown* 1.5  # Dynamic stop based on pattern drawdown
            target_pct = max_gain * risk_reward  # TP based on RR ratio
        
            # Create position
            self._open_long_position(
                current_price=current_price,
                entry_time=current_time,
                risk_value=risk_value,
                stop_loss_pct=stop_pct,
                take_profit_pct=target_pct,
                hold_time=hold_time,
                entry_bar=len(self.data)-1
            )
                
        elif action_type == 2:  # SELL
            # Check if we can take shorts (hedging rules)
            can_short = self.enable_hedging 
            can_short = can_short and self.enable_short  # Short permission
            
            if can_short:
               
                stop_pct =   max_drawdown * 1.5  # Dynamic stop based on pattern drawdown
                target_pct = max_gain * risk_reward  # TP based on RR ratio
            
                # Create position
                self._open_short_position(
                    current_price=current_price,
                    entry_time=current_time,
                    risk_value=risk_value,
                    stop_loss_pct=stop_pct,
                    take_profit_pct=target_pct,
                    hold_time=hold_time,
                    entry_bar=len(self.data)-1
                )    
                
    def _can_open_new_position(self, current_equity: float) -> bool:
        """
        Check if we can open new positions based on risk limits.
        """
        # Check position count limit
        if len(self.trades) >= self.max_positions:
            return False
        
        # Check drawdown limit
        if abs(self.max_drawdown) > 0.80:  # 80% max drawdown
            return False
        
        # Check available cash
        if self._broker._cash < current_equity * 0.05:  # Need at least 5% cash
            return False
        
        return True

    def _calculate_portfolio_features(self) -> Dict[str, float]:
        """
        Calculate portfolio features needed for the RL observation.
        These features are used by the RL model for decision making.
        """
        current_equity = self._broker._cash
        total_returns =(self._broker._cash/ self.initial_equity) - 1
        #######################################
        # Calculate win rate from closed trades
        #######################################
        total_trades = len(self.closed_trades)
        win_rate = self.win_rate 
        
        ##########################################
        # Calculate average pnl per hour
        ##########################################
        avg_pnl_per_hour = 0.0
        if total_trades > 0:
            total_returns = (self._broker._cash/ self.initial_equity) - 1  # Total return
            total_holding_time = self._broker.total_hold_time
            avg_pnl_per_hour = total_returns / total_holding_time
        
        ###########################################
        # Recovery factor - Return / Max Drawdown
        ###########################################
        recovery_factor = 0.0
        total_returns = (self._broker._cash/ self.initial_equity) - 1
        if abs(self.max_drawdown) < 1e-6:  # No meaningful drawdown
            recovery_factor = 1.0 if total_returns  > 0 else 0.0
        else:
            # Calculate recovery factor astotal_returns relative to max drawdown
            #total_returns are calculated as (current equity / initial equity) - 1
            # Recovery factor =total_returns / abs(max_drawdown)
            # If max_drawdown is zero, set recovery factor to zero
            recovery_factor =total_returns / abs(self.max_drawdown)
    
        ###########################################
        # Balance ratio - Current equity / Initial equity
        ############################################
        balance_ratio = current_equity / self.initial_equity if self.initial_equity > 0 else 1.0
        
        ###########################################
        # Calculate decisive exits ratio
        ###########################################
        decisive_exits = self.calculate_decisive_exits()
        
        portfolio_features = [balance_ratio,
                              self.drawdown,
                              win_rate,
                              avg_pnl_per_hour,
                              decisive_exits,
                              recovery_factor]
        
        # convert to np.array for compatibility with RL models
        portfolio_features = np.array(portfolio_features, dtype=np.float32)
        
        return portfolio_features
        # return {
        #     "balance_ratio": balance_ratio,
        #     "portfolio_max_drawdown": self.drawdown,
        #     "win_rate": win_rate,
        #     "avg_pnl_per_hour": avg_pnl_per_hour,
        #     "decisive_exits": decisive_exits,
        #     "recovery_factor": recovery_factor
        # }
        
    def calculate_decisive_exits(self):
        """
        Calculate the decisive exits ratio - how often trades are closed by hitting targets vs. timing out.
        
        Returns:
            float: A ratio between 0-1 representing the "decisiveness" of trade exits
        """
        exits = self.exit_reasons_counts
        total_exits = exits['sl'] + exits['tp'] + exits['time']
                    
        if total_exits == 0:
            return 0  # Default when no trades
        
        # Decisive exits are those that hit stops or targets (not time-based)
        decisive_exits =  exits['sl'] + exits['tp']
        
        # Calculate the ratio
        decisive_ratio = decisive_exits / total_exits
        
        return decisive_ratio

    def _open_long_position(
        self, 
        current_price: float, 
        entry_time, 
        risk_value: float,
        stop_loss_pct: float, 
        take_profit_pct: float,
        hold_time: int,
        entry_bar: int
    ):
        """
        Open a long position with proper risk management.
        Uses backtesting.py's built-in SL/TP functionality.
        """
        try:
            # Calculate position parameters
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            size = int(risk_value // (current_price - stop_loss))  # Position size based on risk
            
            # Create unique position ID
            position_id = self._create_position_id()
            
            # Execute trade using backtesting.py's built-in SL/TP functionality
            result = self.buy(size=size, sl=stop_loss, tp=take_profit, tag=position_id,hold_time=hold_time)
            
            logger.info(f"🔵 Opened LONG #{position_id}: "
                       f"{size:.2f} size @ ${current_price:.2f}, "
                       f"SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
            return None
    
    def _open_short_position(
        self, 
        current_price: float, 
        entry_time, 
        risk_value: float,
        stop_loss_pct: float, 
        take_profit_pct: float,
        hold_time: int,
        entry_bar: int
    ):
        """
        Open a short position with proper risk management.
        Uses backtesting.py's built-in SL/TP functionality.
        """
        try:
            # Calculate position parameters
            stop_loss = current_price * (1 + stop_loss_pct)  # For shorts, stop is above entry
            take_profit = current_price * (1 - take_profit_pct)  # For shorts, target is below entry
            size = (-risk_value // (current_price - stop_loss))
            
            # Create unique position ID
            position_id = self._create_position_id()
            
            # Execute trade using backtesting.py's built-in SL/TP functionality
            result=self.sell(size=size, sl=stop_loss, tp=take_profit, tag=position_id, hold_time=hold_time)
            
            
            logger.info(f"🔴 Opened SHORT #{position_id}: "
                       f"{size:.2f} size @ ${current_price:.2f}, "
                       f"SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
        
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
            return None
            
     
    def _manage_risk_limits(self, current_equity: float):
        """Manage overall portfolio risk limits."""
        # Check overall drawdown limit
        if self.max_drawdown > 0.95:  # 20% max drawdown
            logger.warning(f"⚠️ Maximum drawdown exceeded: {self.max_drawdown:.2%}")
            self._close_all_positions("RISK_LIMIT")
        
      
        # One-liner risk calculation with list comprehension
        total_risk = sum(
            abs(trade.size) * abs(trade.entry_price - trade.sl)
            for trade in self.trades
        )
     
        risk_pct = total_risk / current_equity if current_equity > 0 else 0
        
        if risk_pct > 0.10:  # 10% max portfolio risk
            #logger.warning(f"⚠️ Maximum portfolio risk exceeded: {risk_pct:.2%}")
            self._close_all_positions("RISK_LIMIT")
            
    def _close_all_positions(self, reason: str):
        """Close all open positions (emergency risk management)."""
        # Get all active trades with their tags
        positions_to_close = [trade.tag for trade in self.trades if trade.exit_price is None]
        
        for pos_id in positions_to_close:
            self._close_position(pos_id, reason)
    
    def _close_position(self, position_id: str, reason: str):
        """Close position using backtesting.py's Trade.close() method."""
        try:
            # Find the corresponding backtesting.py Trade object by tag (position_id)
            for trade in self.trades:
                if trade.tag == position_id and trade.exit_price is None:
                    # Close the trade using proper API
                    trade.close()
                    logger.info(f"🔄 Closed  #{position_id}: {reason}")
                    
                    return
            
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def _create_position_id(self) -> str:
        """Create unique position ID."""
        self.position_counter += 1
        return int(self.position_counter)
    
    
    def _update_drawdown_tracking(self, current_equity: float):
        """Update equity tracking for drawdown calculation."""
        self.peak_equity = max(self.peak_equity, current_equity)
        
        self.drawdown = (current_equity / self.peak_equity) -1
        self.max_drawdown = min(self.max_drawdown, self.drawdown )

