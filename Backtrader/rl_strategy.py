"""
RL Trading Strategy for Backtrader

This module implements a Backtrader strategy that uses RL model predictions
to make trading decisions. It interfaces with the model's MultiDiscrete action space
and handles position management, risk control, and performance tracking.
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path for normalizer import
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from RL.Envs.Components.observation_normalizer import ObservationNormalizer
except ImportError:
    ObservationNormalizer = None
    print("Warning: ObservationNormalizer not available - observations will not be normalized")

logger = logging.getLogger(__name__)


class RLTradingStrategy(bt.Strategy):
    """
    Backtrader strategy that uses RL model predictions for trading decisions.
    
    This strategy:
    - Takes observations from the data feed
    - Uses the RL model to predict actions
    - Executes trades based on the MultiDiscrete action space
    - Manages multiple positions and hedging
    - Tracks comprehensive performance metrics    """
    
    params = (
        ('model', None),  # RL model for predictions
        ('observation_manager', None),  # ObservationManager for RL features
        ('initial_cash', 100000),  # Starting cash
        ('position_sizing', 'fixed'),  # Position sizing method
        ('max_positions', 5),  # Maximum concurrent positions
        ('risk_per_trade', 0.01),  # Risk per trade (1% - more conservative)
        ('enable_hedging', True),  # Allow hedging (long + short)
        ('enable_short', True),  # Allow short positions
        ('commission', 0.001),  # Commission rate
        ('slippage', 0.0005),  # Slippage rate
        ('verbose', False),  # Verbose logging
    )
    
    def __init__(self):
        """Initialize the strategy."""
        super().__init__()
          # Model interface
        self.model = self.p.model
        self.observation_manager = self.p.observation_manager
        
        # Initialize normalizer for observations
        self.normalizer = None
        if ObservationNormalizer is not None:
            self.normalizer = ObservationNormalizer(
                enable_adaptive_scaling=False,
                output_range=(-1.0, 1.0),
                clip_outliers=True
            )
            logger.info("ObservationNormalizer initialized successfully")
        else:
            logger.warning("ObservationNormalizer not available - using raw observations")
        
        # Position tracking
        self.positions_log = []  # Track all position details
        self.current_positions = {}  # Active positions
        self.position_counter = 0  # Unique position IDs
        self.pending_orders = {}  # Track pending orders by ID
        
        # Performance tracking
        self.trade_log = []
        self.daily_returns = []
        self.equity_curve = []
        
        # Action space mapping (MultiDiscrete [3, 20, 10, 10])
        self.action_mappings = self._initialize_action_mappings()
          # Risk management
        self.max_drawdown = 0.0
        self.peak_equity = self.p.initial_cash
        
        logger.info(f"Initialized RLTradingStrategy with {self.p.initial_cash} initial cash")
        if self.observation_manager:
            logger.info("Using ObservationManager for RL features")
            # Initialize portfolio calculator with correct starting balance
            if hasattr(self.observation_manager, 'portfolio_calculator'):
                self.observation_manager.portfolio_calculator.reset(self.p.initial_cash)
                logger.info(f"Portfolio calculator initialized with {self.p.initial_cash} balance")
        else:
            logger.warning("No ObservationManager provided - using basic price features")
    
    def _initialize_action_mappings(self) -> Dict[str, List]:
        """Initialize action space mappings."""
        return {
            'action_type': [0, 1, 2],  # HOLD, BUY, SELL
            'position_sizes': [0.005 + i * 0.005 for i in range(20)],  # 0.5% to 10%
            'risk_rewards': [0.5 + i * 0.25 for i in range(10)],  # 0.5x to 3.0x
            'hold_times': [3 + i * 3 for i in range(10)]  # 3h to 30h
        }
    
    def next(self):
        """Execute strategy logic for each bar."""
        try:
            # Get current market data
            current_price = self.data.close[0]
            current_time = bt.num2date(self.data.datetime[0])
            
            # Update equity tracking
            current_equity = self.broker.get_value()
            self.equity_curve.append({
                'datetime': current_time,
                'equity': current_equity,
                'cash': self.broker.get_cash(),
                'positions_value': current_equity - self.broker.get_cash()
            })
              # Update drawdown tracking
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Update portfolio calculator with current balance
            if self.observation_manager and hasattr(self.observation_manager, 'portfolio_calculator'):
                self.observation_manager.portfolio_calculator.update_balance(current_equity)
            
            # Get observation vector from data feed
            observation = self._get_observation()
            
            if observation is not None and self.model is not None:
                # Get model prediction
                action = self._get_model_prediction(observation)
                
                # Execute trading logic
                self._execute_action(action, current_price, current_time)
            
            # Manage existing positions
            self._manage_positions(current_price, current_time)
              # Log performance periodically
            if len(self.equity_curve) % 100 == 0 and self.p.verbose:
                self._log_performance()
                
        except Exception as e:
            logger.error(f"Error in strategy.next(): {e}")
    def notify_order(self, order):
        """Handle order status notifications from Backtrader."""
        try:
            # Track order status changes with detailed logging
            order_id = getattr(order, 'ref', None)
            status_names = {
                0: 'Created', 1: 'Submitted', 2: 'Accepted', 3: 'Partial',
                4: 'Completed', 5: 'Canceled', 6: 'Margin', 7: 'Rejected'
            }
            status_name = status_names.get(order.status, f'Unknown({order.status})')
            
            # Always log order status for debugging
            if self.p.verbose:
                order_type = 'BUY' if order.isbuy() else 'SELL'
                logger.info(f"Order {order_id}: {status_name} - {order_type} {order.size} @ {order.price or 'Market'}")
                
                # Log additional details for rejected orders
                if order.status == 7:  # Rejected
                    logger.error(f"Order REJECTED - Size: {order.size}, Price: {order.price}, "
                               f"Cash: ${self.broker.get_cash():.2f}, Value: ${self.broker.get_value():.2f}")
            
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                # Remove from pending orders
                if order_id in self.pending_orders:
                    position_info = self.pending_orders.pop(order_id)
                    
                    if order.status == order.Completed:
                        if self.p.verbose:
                            logger.info(f"Order {order_id} EXECUTED: {order.executed.size} @ {order.executed.price:.4f}")
                    elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                        if self.p.verbose:
                            logger.warning(f"Order {order_id} FAILED: {status_name}")
                        # Remove position from tracking if main order failed
                        if 'position_id' in position_info:
                            self.current_positions.pop(position_info['position_id'], None)
                            
        except Exception as e:
            logger.error(f"Error in notify_order: {e}")
    
    def _get_observation(self) -> Optional[np.ndarray]:
        """Get observation vector using ObservationManager."""
        try:
            if self.observation_manager:
                # Use ObservationManager to get current observation
                current_time = bt.num2date(self.data.datetime[0])
                raw_observation = self.observation_manager.get_observation_array(current_time)
                
                # CRITICAL FIX: Normalize observation using our normalizer
                if raw_observation is not None and self.normalizer is not None:
                    normalized_observation = self.normalizer.normalize_observation(raw_observation)
                    return normalized_observation
                else:
                    if self.normalizer is None:
                        logger.warning("No normalizer available - using raw observations (NOT RECOMMENDED)")
                    return raw_observation
            else:
                return self._create_basic_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            return self._create_basic_observation()

    def _create_basic_observation(self) -> np.ndarray:
        """Create basic observation from price data when ObservationManager not available."""
        try:
            # Simple price-based features as fallback
            return np.array([
                self.data.close[0] / self.data.close[-20] if len(self.data) > 20 else 1.0,  # 20-period price ratio
                (self.data.high[0] - self.data.low[0]) / self.data.close[0],  # Price range
                self.broker.get_cash() / self.p.initial_cash,  # Cash ratio
            ])
        except:
            return np.zeros(3)  # Fallback array
    
  
    
    def _get_model_prediction(self, observation: np.ndarray) -> List[int]:
        """Get action prediction from RL model."""
        try:
            if hasattr(self.model, 'predict'):
                # Standard stable-baselines3 interface
                action, _ = self.model.predict(observation, deterministic=True)
                return action.tolist() if isinstance(action, np.ndarray) else action
            else:
                # Fallback: random action                
                 return [
                    np.random.choice([0, 1, 2]),  # action_type
                    np.random.randint(0, 20),     # position_size_idx
                    np.random.randint(0, 10),     # risk_reward_idx
                    np.random.randint(0, 10)      # hold_time_idx
                ]
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            # Return HOLD action as safe fallback
            return [0, 0, 0, 0]
    
    def _execute_action(self, action: List[int], price: float, timestamp: datetime):
        """Execute trading action using proper Backtrader bracket orders."""
        try:
            action_type, pos_size_idx, rr_idx, hold_time_idx = action
            
            # Map indices to actual values
            position_size = self.action_mappings['position_sizes'][pos_size_idx]
            risk_reward = self.action_mappings['risk_rewards'][rr_idx]
            hold_time = self.action_mappings['hold_times'][hold_time_idx]
            
            if action_type == 0:  # HOLD
                return
              # CRITICAL FIX: Proper position sizing and validation
            cash_available = self.broker.get_cash()
            margin_available = cash_available * 100  # Use 100x leverage for margin trading
            
            # Validate minimum cash available
            if cash_available < 1000:  # Need at least $1000 to trade
                if self.p.verbose:
                    logger.warning(f"Insufficient cash: ${cash_available:.2f}")
                return
                
            # Calculate risk amount (percentage of cash)
            risk_value = cash_available * position_size
            if risk_value < 100:  # Minimum trade value
                return
            
            # Get volatility-based stop loss/take profit levels
            try:
                current_observation = self.observation_manager.get_market_observation(timestamp)
                volatility = current_observation.get('volatility', 0.02)
                # Use volatility for more realistic stop loss levels
                stop_loss_pct = min(0.05, max(0.01, volatility * 2))  # 1-5% stop loss
                take_profit_pct = min(0.10, max(0.02, volatility * risk_reward))  # 2-10% take profit
            except:
                stop_loss_pct = 0.02  # 2% stop loss
                take_profit_pct = 0.04  # 4% take profit
            
            # FIXED: Calculate position size properly
            if action_type == 1:  # BUY
                stop_loss_price = price * (1 - stop_loss_pct)
                take_profit_price = price * (1 + take_profit_pct)
                
                # Calculate max shares we can afford
                max_shares = int(cash_available * 0.95 / price)  # Use 95% of cash for margin
                
                # Calculate shares based on risk (proper formula)
                risk_per_share = price - stop_loss_price
                if risk_per_share > 0:
                    shares_by_risk = int(risk_value / risk_per_share)
                    size = min(max_shares, shares_by_risk)
                else:
                    size = max_shares
                    
            else:  # SELL (short)
                if not self.p.enable_short:
                    return
                stop_loss_price = price * (1 + stop_loss_pct)
                take_profit_price = price * (1 - take_profit_pct)
                
                # For shorts, calculate based on margin requirements
                max_shares = int(cash_available * 0.5 / price)  # Conservative 50% for shorts
                
                # Calculate shares based on risk
                risk_per_share = stop_loss_price - price
                if risk_per_share > 0:
                    shares_by_risk = int(risk_value / risk_per_share)
                    size = min(max_shares, shares_by_risk)
                else:
                    size = max_shares
            
            # CRITICAL: Validate position size
            if size <= 0:
                if self.p.verbose:
                    logger.warning(f"Invalid position size: {size}")
                return
                
            if size > 10000:  # Sanity check for max position size
                if self.p.verbose:
                    logger.warning(f"Position size too large: {size}, capping at 1000")
                size = 1000
            
            # Validate we have enough cash for the trade
            required_cash = size * price * 1.1  # Add 10% buffer for fees/slippage
            if required_cash > margin_available:
                if self.p.verbose:
                    logger.warning(f"Insufficient cash for trade: need ${required_cash:.2f}, have ${cash_available:.2f}")
                return
            
            # Create bracket orders with validated parameters
            if action_type == 1:  # BUY
                main_order = self.buy(size=size)
                if main_order:
                    # Create stop loss order
                    stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_loss_price, parent=main_order)
                    # Create take profit order  
                    limit_order = self.sell(size=size, exectype=bt.Order.Limit, price=take_profit_price, parent=main_order)
                    
            else:  # SELL (short)
                main_order = self.sell(size=size)
                if main_order:
                    # Create stop loss order (buy to cover)
                    stop_order = self.buy(size=size, exectype=bt.Order.Stop, price=stop_loss_price, parent=main_order)
                    # Create take profit order (buy to cover)
                    limit_order = self.buy(size=size, exectype=bt.Order.Limit, price=take_profit_price, parent=main_order)
            
            # Track position with timeout
            if main_order:
                position_id = self.position_counter
                self.position_counter += 1
                
                # Schedule timeout closure
                timeout_date = timestamp + timedelta(hours=hold_time)
                
                self.current_positions[position_id] = {
                    'main_order': main_order,
                    'stop_order': stop_order if 'stop_order' in locals() else None,
                    'limit_order': limit_order if 'limit_order' in locals() else None,
                    'action_type': action_type,
                    'entry_price': price,
                    'size': size,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'entry_time': timestamp,
                    'timeout_date': timeout_date,
                    'position_size_pct': position_size,
                    'risk_reward': risk_reward,
                    'status': 'pending'
                }
                  # Track order for notifications
                self.pending_orders[main_order.ref] = {
                    'position_id': position_id,
                    'order_type': 'main',
                    'action_type': action_type
                }
                
                if self.p.verbose:
                    action_name = 'BUY' if action_type == 1 else 'SELL'
                    logger.info(f"{timestamp}: {action_name} {size:.4f} @ {price:.4f}, "
                              f"SL: {stop_loss_price:.4f}, TP: {take_profit_price:.4f}")
                    
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    def _manage_positions(self, current_price: float, current_time: datetime):
        """Manage existing positions - simplified since we're using bracket orders."""
        positions_to_close = []
        
        for pos_id, position in self.current_positions.items():
            try:
                # Check if position should be closed due to timeout
                timeout_date = position['timeout_date']
                
                if current_time >= timeout_date:
                    # Close position due to timeout
                    current_position = self.getposition()
                    if current_position.size != 0:
                        # Cancel any pending stop/limit orders
                        if 'stop_order' in position and position['stop_order']:
                            self.cancel(position['stop_order'])
                        if 'limit_order' in position and position['limit_order']:
                            self.cancel(position['limit_order'])
                        
                        # Close position manually
                        if current_position.size > 0:  # Long position
                            self.sell(size=abs(current_position.size))
                        else:  # Short position
                            self.buy(size=abs(current_position.size))
                        
                        # Log trade
                        self._log_trade(position, current_price, current_time, 'time_exit')
                        positions_to_close.append(pos_id)
                        
                        if self.p.verbose:
                            logger.info(f"Position {pos_id} closed due to timeout")
                    else:
                        # Position already closed by stop/limit orders
                        positions_to_close.append(pos_id)
                        
            except Exception as e:
                logger.error(f"Error managing position {pos_id}: {e}")
                positions_to_close.append(pos_id)
        
        # Remove closed positions
        for pos_id in positions_to_close:
            self.current_positions.pop(pos_id, None)
    
    def _log_trade(self, position: Dict, exit_price: float, exit_time: datetime, exit_reason: str):
        """Log completed trade."""
        try:
            entry_price = position['entry_price']
            size = position['size']
            action_type = position['action_type']
            
            # Calculate P&L
            if action_type == 1:  # Long
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - exit_price) / entry_price
            
            pnl_absolute = pnl_pct * abs(size) * entry_price
            
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'action_type': 'BUY' if action_type == 1 else 'SELL',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl_pct': pnl_pct,
                'pnl_absolute': pnl_absolute,
                'exit_reason': exit_reason,
                'hold_time_hours': (exit_time - position['entry_time']).total_seconds() / 3600,
                'position_size_pct': position['position_size_pct'],
                'risk_reward': position['risk_reward']
            }
            self.trade_log.append(trade_record)
            
            # Update portfolio calculator with completed trade
            if self.observation_manager and hasattr(self.observation_manager, 'portfolio_calculator'):
                # Map exit reasons to match portfolio calculator expectations
                portfolio_exit_reason = exit_reason
                if exit_reason == 'take_profit':
                    portfolio_exit_reason = 'tp'
                elif exit_reason == 'stop_loss':
                    portfolio_exit_reason = 'sl'
                elif exit_reason == 'time_exit':
                    portfolio_exit_reason = 'timeout'
                
                # Record trade with portfolio calculator
                self.observation_manager.portfolio_calculator.record_trade(
                    pnl=pnl_absolute,
                    exit_reason=portfolio_exit_reason,
                    holding_hours=trade_record['hold_time_hours']
                )
            
            if self.p.verbose:
                logger.info(f"Trade closed: {trade_record['action_type']} "
                          f"P&L: {pnl_pct:.2%} ({pnl_absolute:.2f}) "
                          f"Reason: {exit_reason}")
                          
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def _log_performance(self):
        """Log current performance metrics."""
        try:
            if not self.equity_curve:
                return
            
            current_equity = self.equity_curve[-1]['equity']
            total_return = (current_equity - self.p.initial_cash) / self.p.initial_cash
            
            logger.info(f"Performance Update:")
            logger.info(f"  Current Equity: {current_equity:.2f}")
            logger.info(f"  Total Return: {total_return:.2%}")
            logger.info(f"  Max Drawdown: {self.max_drawdown:.2%}")
            logger.info(f"  Active Positions: {len(self.current_positions)}")
            logger.info(f"  Completed Trades: {len(self.trade_log)}")
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def stop(self):
        """Called when strategy stops - final cleanup."""
        try:
            # Close any remaining positions
            if self.getposition():
                self.close()
            
            # Final performance log
            if self.p.verbose:
                logger.info("Strategy stopped. Final performance:")
                self._log_performance()
                
        except Exception as e:
            logger.error(f"Error in strategy.stop(): {e}")
    
    def get_trade_log(self) -> List[Dict]:
        """Get complete trade log."""
        return self.trade_log.copy()
    
    def get_equity_curve(self) -> List[Dict]:
        """Get equity curve data."""
        return self.equity_curve.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            if not self.trade_log or not self.equity_curve:
                return {}
            
            # Trade-based metrics
            trades_df = pd.DataFrame(self.trade_log)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
            
            # Equity-based metrics
            equity_df = pd.DataFrame(self.equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - self.p.initial_cash) / self.p.initial_cash
            
            # Returns for Sharpe calculation
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std() if equity_df['returns'].std() > 0 else 0
            
            return {
                'total_return': total_return,
                'final_equity': final_equity,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_winning_trade': avg_win,
                'avg_losing_trade': avg_loss,
                'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
