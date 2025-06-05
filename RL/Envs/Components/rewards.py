import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple

class RewardCalculator:
    """
    A class to handle all reward calculation in the trading environment.
    Implements various reward types including standard, Sharpe, Sortino, drawdown-focused,
    Calmar, win-rate, consistency, and combined rewards.
    """
    
    def __init__(self, reward_type: str = 'combined', max_history_length: int = 20):
        """
        Initialize reward calculator with specified reward type
        
        Args:
            reward_type: Type of reward function to use ('sharpe', 'sortino', 'drawdown_focus',
                          'calmar', 'win_rate', 'consistency', 'combined')
            max_history_length: Maximum length of returns history for rolling calculations
        """
        self.reward_type = reward_type
        self.max_history_length = max_history_length
        
        # Initialize tracking variables
        self.reset()
    
    def reset(self):
        """Reset all tracking variables for a new episode."""
        self.returns_history = []
        self.equity_curve = []
        self.peak_balance = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_amount = 0
        self.loss_amount = 0
        self.max_drawdown = 0
    
    def update_metrics(self, balance: float, trade_pnl_pct: float):
        """
        Update tracking metrics when a trade is completed
        
        Args:
            balance: Current account balance
            trade_pnl: Profit/loss from the trade
        """
        # Update equity curve
        self.equity_curve.append(balance)
        
        # Update peak balance for drawdown calculation
        self.peak_balance = max(self.peak_balance, balance)
        
        # Update trade metrics
        if trade_pnl_pct > 0:
            self.winning_trades += 1
            self.win_amount += trade_pnl_pct
        elif trade_pnl_pct < 0:
            self.losing_trades += 1
            self.loss_amount += abs(trade_pnl_pct)
        
        self.trade_count = self.winning_trades + self.losing_trades
    
    def calculate_reward(self, base_reward: float, trade_pnl_pct: float = 0) -> float:
        """
        Calculate reward based on the selected reward type
        
        Args:
            base_reward: Base reward from P&L
            trade_pnl: Profit/loss from the trade
            
        Returns:
            float: The calculated reward value
        """
        if trade_pnl_pct != 0:  # Only update metrics if we had a trade
            # Update returns history
            self.returns_history.append(trade_pnl_pct)
            
            # Keep history at max length
            if len(self.returns_history) > self.max_history_length:
                self.returns_history.pop(0)
        
        # Select reward function based on type
        if self.reward_type == 'sharpe':
            return self.calculate_sharpe_reward(base_reward)
        elif self.reward_type == 'sortino':
            return self.calculate_sortino_reward(base_reward)
        elif self.reward_type == 'drawdown_focus':
            return base_reward + self.calculate_drawdown_penalty()
        elif self.reward_type == 'calmar':
            return self.calculate_calmar_reward(base_reward)
        elif self.reward_type == 'win_rate':
            return base_reward + self.calculate_win_rate_bonus()
        elif self.reward_type == 'consistency':
            return base_reward + self.calculate_consistency_bonus()
        elif self.reward_type == 'combined':
            return self.calculate_combined_reward(base_reward)
        else:
            return base_reward  # Default to base reward
    
    def calculate_sharpe_reward(self, base_reward: float) -> float:
        """Calculate reward using Sharpe ratio component"""
        # Calculate rolling Sharpe ratio (annualized)
        if len(self.returns_history) > 1:
            returns_mean = np.mean(self.returns_history)
            returns_std = np.std(self.returns_history) + 1e-6  # Avoid division by zero
            sharpe = returns_mean / returns_std * np.sqrt(252)  # Annualized
            
            # Sharpe ratio bonus (higher is better)
            sharpe_bonus = np.clip(0.1 * sharpe, -0.5, 0.5)  # Limit impact
            return base_reward + sharpe_bonus
        return base_reward

    def calculate_sortino_reward(self, base_reward: float) -> float:
        """Calculate reward using Sortino ratio (focusing on downside risk)"""
        # Calculate Sortino ratio (only considering downside deviation)
        if len(self.returns_history) > 1:
            returns_mean = np.mean(self.returns_history)
            
            # Calculate downside deviation (only negative returns)
            downside_returns = [r for r in self.returns_history if r < 0]
            if downside_returns:
                downside_dev = np.std(downside_returns) + 1e-6
                sortino = returns_mean / downside_dev * np.sqrt(252)
                
                # Sortino ratio bonus (penalizes negative returns more heavily)
                sortino_bonus = np.clip(0.15 * sortino, -0.6, 0.6)
                return base_reward + sortino_bonus
        return base_reward

    def calculate_drawdown_penalty(self) -> float:
        """Calculate penalty based on drawdown"""
        # Calculate current drawdown
        if len(self.equity_curve) > 0:
            peak_equity = max(self.equity_curve)
            current_equity = self.equity_curve[-1]
            current_drawdown = (current_equity / peak_equity) - 1
            
            # Update max drawdown
            self.max_drawdown = min(self.max_drawdown, current_drawdown)
            
            # Drawdown penalty (more severe as drawdown increases)
            if current_drawdown < -0.03:  # 3% drawdown
                return -0.05
            elif current_drawdown < -0.05:  # 5% drawdown
                return -0.1
            elif current_drawdown < -0.08:  # 8% drawdown
                return -0.5
            elif current_drawdown < -0.1:  # 10% drawdown (critical for prop firms)
                return -1.0
        return 0

    def calculate_win_rate_bonus(self) -> float:
        """Calculate bonus based on win rate and profit factor"""
        if self.trade_count > 5:  # Need enough trades for meaningful calculation
            win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
            
            # Win rate bonus (scaled to be small but meaningful)
            win_bonus = 0.05 * (win_rate - 0.5) * 2  # -0.05 to 0.05 range
            
            # Profit factor bonus (win amount / loss amount)
            profit_factor = self.win_amount / (self.loss_amount + 1e-6)
            profit_factor_bonus = min(0.05, 0.01 * (profit_factor - 1))  # Cap at 0.05
            
            return win_bonus + profit_factor_bonus
        return 0

    def calculate_consistency_bonus(self, base_reward: float) -> float:
        """Reward consistency of returns"""
        if len(self.returns_history) > 5:
            # Calculate rolling average
            avg_return = np.mean(self.returns_history)
            
            # Calculate how close current return is to average (consistency)
            deviation = abs(base_reward - avg_return)
            normalized_deviation = deviation / (abs(avg_return) + 1e-6)
            
            # Reward low deviation (consistency)
            if normalized_deviation < 0.5:
                return 0.05  # Small bonus for consistent returns
        return 0

    def calculate_calmar_reward(self, base_reward: float) -> float:
        """Calculate reward based on Calmar ratio (return / max drawdown)"""
        if abs(self.max_drawdown) > 0.001 and len(self.returns_history) > 2:
            returns_mean = np.mean(self.returns_history)
            # Calculate Calmar ratio (return / max drawdown)
            calmar = returns_mean / abs(self.max_drawdown + 1e-6) * 252  # Annualized
            
            # Calmar ratio bonus (heavily rewards good return/drawdown ratio)
            calmar_bonus = np.clip(0.2 * calmar, -0.4, 0.4)
            return base_reward + calmar_bonus
        return base_reward

    def calculate_combined_reward(self, base_reward: float) -> float:
        """Calculate combined risk-adjusted reward"""
        # Base reward from P&L
        adjusted_reward = base_reward
        
        # Add Sharpe ratio component
        sharpe_component = self.calculate_sharpe_reward(base_reward) - base_reward
        adjusted_reward += (0.2 * sharpe_component)  # 20% weight
        
        # Add sortino ratio component (better handling of downside risk)
        sortino_component = self.calculate_sortino_reward(base_reward) - base_reward
        adjusted_reward += (0.15 * sortino_component)  # 15% weight
        
        # Add drawdown penalty
        drawdown_penalty = self.calculate_drawdown_penalty()
        adjusted_reward += drawdown_penalty  # Full impact of drawdown penalty
        
        # Add calmar ratio component
        calmar_component = self.calculate_calmar_reward(base_reward) - base_reward
        adjusted_reward += (0.15 * calmar_component)  # 15% weight
        
        # Add win rate bonus
        win_bonus = self.calculate_win_rate_bonus()
        adjusted_reward += (0.3 * win_bonus)  # 30% weight
        
        # Add consistency bonus
        consistency_bonus = self.calculate_consistency_bonus(base_reward)
        adjusted_reward += (0.1 * consistency_bonus)  # 10% weight
        
        return adjusted_reward
        
    def calculate_hold_penalty(self, steps_without_action: int) -> float:
        """
        Calculate penalty for consecutive HOLD actions
        
        Args:
            steps_without_action: Number of consecutive HOLD actions
            
        Returns:
            float: Penalty value (negative)
        """
            # Instead of harsh penalty, use gradual increase
        if steps_without_action < 10:
            return -0.0001  # Very small penalty
        elif steps_without_action < 30:
            return -0.001   # Moderate penalty
        else:
            return -0.01    # Larger penalty only after extended holding