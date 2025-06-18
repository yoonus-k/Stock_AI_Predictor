"""
Unified Trading State for RL Environment

This module provides a centralized state management system for trading environments
to eliminate redundant code and keep the trading state synchronized across
various environment components.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

class TradingState:
    """
    A centralized class to manage the trading state information shared across
    environment components like reward calculation, action handling, and observations.
    """
    
    def __init__(self, initial_balance: float = 100000):
        """
        Initialize the trading state with default values
        
        Args:
            initial_balance: Starting account balance
        """
        # Core financial tracking
        self.initial_balance = initial_balance
        self.balance = initial_balance       # Cash balance
        self.equity = initial_balance        # Total portfolio value (cash + positions)
        self.peak_equity = initial_balance   # Highest equity value reached
        self.drawdown = 0.0                  # Current drawdown as percentage
        self.max_drawdown = 0.0              # Maximum drawdown experienced
        
        # Margin tracking
        self.margin_used = 0.0               # Amount of margin currently in use
        self.available_margin = initial_balance  # Available margin for new positions
        
        # Trade statistics
        self.trade_count = 0                 # Total number of completed trades
        self.win_amount = 0.0                # Total profit from winning trades
        self.loss_amount = 0.0               # Total loss from losing trades
        
        # Position tracking
        self.active_positions = []           # List of active trade positions
        self.unrealized_pnl = 0.0            # Current unrealized profit/loss
        
        # Trade exit statistics
        self.tp_exits = 0                    # Number of take profit exits
        self.sl_exits = 0                    # Number of stop loss exits
        self.time_exits = 0                  # Number of time-based exits
        self.total_exits = 0                 # Total number of all exits
        
        # for consistency and performance tracking
        self.most_profitable_return = 0.0  # Highest return from a single trade
        self.most_losing_return = 0.0   # Lowest return from a single trade
        
        # some stats eg. std, mean, etc.
        self.mean_return = 0.0                # Mean return from trades
        self.var_return = 0.0     # Variance of returns
        self.std_return = 0.0                 # Standard deviation of returns
        
        # Advanced metrics
        self.total_pnl_pct = 0.0             # Total P&L percentage
        self.total_holding_hours = 0.0       # Total position holding time (hours)
        self.steps_without_action = 0        # Consecutive steps without trading action
        
        # History tracking
        self.equity_curve = []               # Historical equity values
        self.trade_history = []              # History of completed trades
        self.returns_history = []            # Historical returns
        self.position_counter = 0            # Counter for generating unique position IDs
    
    def reset(self, initial_balance: Optional[float] = None):
        """
        Reset the trading state for a new episode
        
        Args:
            initial_balance: Optional new initial balance
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
            
        # Reset core financial tracking
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Reset margin tracking
        self.margin_used = 0.0
        self.available_margin = self.initial_balance
        
        # Reset trade statistics
        self.trade_count = 0
        self.win_amount = 0.0
        self.loss_amount = 0.0
        
        # Reset position tracking
        self.active_positions = []
        self.unrealized_pnl = 0.0
        
        # Reset exit statistics
        self.tp_exits = 0
        self.sl_exits = 0
        self.time_exits = 0
        self.total_exits = 0
        
        # Reset performance tracking
        self.most_profitable_return = 0.0
        self.most_losing_return = 0.0
        
        # some stats eg. std, mean, etc.
        self.mean_return = 0.0    # Mean return from trades
        self.var_return = 0.0     # Variance of returns
        self.std_return = 0.0     # Standard deviation of returns
        
        # Reset advanced metrics
        self.total_pnl_pct = 0.0
        self.total_holding_hours = 0.0
        self.steps_without_action = 0
        
        # Reset history tracking
        self.equity_curve = []
        self.trade_history = []
        self.returns_history = []    
        self.position_counter = 0
    
    def update_from_trade(self, trade_pnl_pct: float, exit_reason: str, holding_hours: float):
        """
        Update state metrics when a trade is completed
        
        Args:
            trade_pnl_pct: Trade P&L as percentage
            exit_reason: Exit reason ('tp', 'sl', 'time')
            holding_hours: Actual holding time in hours
        """
        # Update trade count and P&L metrics
        self.trade_count += 1
        self.total_pnl_pct += trade_pnl_pct
        self.total_holding_hours += holding_hours
        self.returns_history.append(trade_pnl_pct)
        
        # update statistics
        self.calculate_std_return(trade_pnl_pct)

        # Update exit counters based on exit reason
        self.total_exits += 1
        if exit_reason == 'tp':
            self.win_amount += trade_pnl_pct
            self.tp_exits += 1
            self.most_profitable_return = max(self.most_profitable_return, trade_pnl_pct)
        elif exit_reason == 'sl':
            self.sl_exits += 1
            self.loss_amount += trade_pnl_pct
            self.most_losing_return = min(self.most_losing_return, trade_pnl_pct)
        elif exit_reason == 'time':
            self.time_exits += 1
            if trade_pnl_pct > 0:
                self.win_amount += trade_pnl_pct
                self.most_profitable_return = max(self.most_profitable_return, trade_pnl_pct)
            else:
                self.loss_amount += trade_pnl_pct
                self.most_losing_return = min(self.most_losing_return, trade_pnl_pct)
    
    def update_equity(self, new_equity: float):
        """
        Update equity and related metrics
        
        Args:
            new_equity: New equity value
        """
        self.equity = new_equity
        self.equity_curve.append(new_equity)
        
        # Update peak equity and drawdown metrics
        self.peak_equity = max(self.peak_equity, new_equity)
        self.drawdown = (new_equity / self.peak_equity) - 1.0
        self.max_drawdown = min(self.max_drawdown, self.drawdown)
    
    def update_margin(self, margin_used: float):
        """
        Update margin usage metrics
        
        Args:
            margin_used: New margin used value
        """
        self.margin_used = margin_used
        self.available_margin = self.equity - margin_used
        
    def calculate_std_return(self , trade_pnl_pct: float):
        """
        Calculate the standard deviation of returns based on the returns history ( incremental update)
        Args:
            trade_pnl_pct: Profit/loss percentage from the trade
        Returns:
            float: Standard deviation of returns
            
        See Docs for more details on the calculation
        """
        # if it's the first variation calculation, initialize
        n = self.trade_count
        if self.var_return == 0.0 and n >= 2:
            return_previous = self.returns_history[0]
            return_current = trade_pnl_pct
            self.mean_return = (return_current+return_previous) / n
            self.var_return = ((return_previous - self.mean_return) ** 2 + (return_current - self.mean_return) ** 2)/ (n - 1)
            self.std_return = np.sqrt(self.var_return)
        elif n > 2:
            # Incremental update of variance and standard deviation
            self.var_return = (((n-2)/(n-1))*self.var_return)+((1/n)*(trade_pnl_pct - self.mean_return) ** 2)
            self.std_return = np.sqrt(self.var_return)
            self.mean_return = self.mean_return + (trade_pnl_pct - self.mean_return) / n
        else:
            # Not enough data to calculate variance
            self.var_return = 0.0
            self.std_return = 0.0
            self.mean_return = 0.0
    
    def get_win_rate(self) -> float:
        """
        Get the current win rate
        
        Returns:
            float: Win rate as a percentage (0-1)
        """
        if self.trade_count == 0:
            return 0.5  # Default win rate when no trades
        return self.tp_exits / self.trade_count
    
    
    def get_consistency_score(self) -> float:
        """
        Calculate a consistency score based on win rate and average holding time
        Formula: (1- (abs(max(return)))/ abs(all returns)) 
        
        Returns:
            float: Consistency score (0-1)
        """
        absolute_max_return = max(abs(self.most_profitable_return), abs(self.most_losing_return))
        absolute_return_all = abs(self.loss_amount) + abs(self.win_amount)
        
        consistency_score = 1 - (absolute_max_return / absolute_return_all) if absolute_return_all > 0 else 1.0
        
        return consistency_score
    
    def get_avg_pnl_per_hour(self) -> float:
        """
        Calculate average P&L per hour of holding time
        
        Returns:
            float: Average P&L per hour
        """
        if self.total_holding_hours <= 0:
            return 0.0
        return self.total_pnl_pct / self.total_holding_hours
    
    def get_decisive_exits_ratio(self) -> float:
        """
        Calculate ratio of decisive exits (TP/SL) vs timeouts
        
        Returns:
            float: Ratio of decisive exits (0-1)
        """
        if self.total_exits <= 0:
            return 0.0
        decisive_exits = self.tp_exits + self.sl_exits
        return decisive_exits / self.total_exits
    
    def get_recovery_factor(self) -> float:
        """
        Calculate recovery factor: gains relative to maximum drawdown
        
        Returns:
            float: Recovery factor
        """
        if abs(self.max_drawdown) < 1e-2:  # No meaningful drawdown < 1% (0.01)
            return 1.0 if self.total_pnl_pct > 0 else 0.0
        current_pnl_pct = (self.equity / self.initial_balance) - 1.0
        recovery_factor = current_pnl_pct / abs(self.max_drawdown)
        #print(f"Recovery Factor: {recovery_factor:.2f} (Current P&L: {current_pnl_pct:.2%}, Max Drawdown: {abs(self.max_drawdown):.2%})")
        return recovery_factor
    
    def generate_position_id(self) -> int:
        """
        Generate a unique position ID
        
        Returns:
            int: Unique position ID
        """
        self.position_counter += 1
        return self.position_counter
