import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple

from .trading_state import TradingState

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
        self.trading_state = None  # Will be set later
        
        # Initialize tracking variables
        self.reset()
    
    def set_trading_state(self, trading_state: TradingState):
        """
        Set the shared trading state reference
        
        Args:
            trading_state: Shared trading state object
        """
        self.trading_state = trading_state
    
    def reset(self):
        """Reset all tracking variables for a new episode."""
        pass
        
    
    def calculate_reward(self, base_reward: float, trade_pnl_pct: float = 0) -> float:
        """
        Calculate reward based on the selected reward type
        
        Args:
            base_reward: Base reward from P&L
            trade_pnl_pct: Profit/loss from the trade
            
        Returns:
            float: The calculated reward value
        """

        # Select reward function based on type
        if self.reward_type == 'sharpe':
            return self.calculate_sharpe_reward(base_reward)
        elif self.reward_type == 'drawdown_focus':
            return base_reward + self.calculate_drawdown_penalty()
        elif self.reward_type == 'win_rate':
            return base_reward + self.calculate_win_rate_bonus()
        elif self.reward_type == 'consistency':
            return base_reward + self.calculate_consistency_bonus()
        elif self.reward_type == 'combined':
            return self.calculate_combined_reward(base_reward)
        else:
            return base_reward  # Default to base reward
    
    def calculate_sharpe_reward(self, base_reward: float) -> float:
        """
        Calculate reward using Sharpe ratio to adjust base reward
        
        Args:
            base_reward: Base reward value
            
        Returns:
            float: Adjusted reward value
        """
        if len(self.trading_state.returns_history) < 2:
            return base_reward
            
        mean_return = self.trading_state.mean_return
        std_return = self.trading_state.std_return
        
        # Avoid division by zero
        if std_return < 1e-6:
            sharpe_ratio = 0 if mean_return < 0 else 1
        else:
            sharpe_ratio = mean_return / std_return
            
        # Adjust the base reward by the Sharpe ratio (scaled)
        return base_reward * (1 + min(1, max(-0.5, sharpe_ratio * 0.2)))
    
    
    def calculate_drawdown_penalty(self) -> float:
        """
        Calculate a penalty based on drawdown magnitude
        
        Returns:
            float: Drawdown penalty
        """
        # Penalty scales with drawdown size
        penalty = self.trading_state.drawdown * 2  # More impact than base reward
        return min(0, penalty)  # Ensure it's a penalty (negative or zero)
    
    def calculate_win_rate_bonus(self) -> float:
        """
        Calculate a bonus based on win rate
        
        Returns:
            float: Win rate bonus
        """
        if  self.trading_state.trade_count == 0:
            return 0.0
            
        win_rate = self.trading_state.get_win_rate()
        # Sigmoid-like scaling to reward win rates > 0.5 and penalize < 0.5
        bonus = (win_rate - 0.5) * 0.1  # Scale factor
        return bonus
    
    def calculate_consistency_bonus(self) -> float:
        """
        Calculate a bonus for consistent returns
        
        Returns:
            float: Consistency bonus
        """
        if len(self.trading_state.returns_history) < 3:
            return 0.0
            
        consistency_score = self.trading_state.get_consistency_score()
        
        bonus= (consistency_score - 0.5) * 0.1  # Small bonus for consistency
        return bonus
       
    
    def calculate_combined_reward(self, base_reward: float) -> float:
        """
        Calculate combined reward using multiple metrics
        
        Args:
            base_reward: Base reward value
            
        Returns:
            float: Combined reward value
        """
        # Apply Sharpe ratio adjustment
        sharpe_adjustment = self.calculate_sharpe_reward(1.0) - 1.0
        
        # Apply drawdown penalty
        drawdown_penalty = self.calculate_drawdown_penalty()
        
        # Apply win rate adjustment
        win_rate_bonus = self.calculate_win_rate_bonus()
        
        # Apply consistency bonus
        consistency_bonus = self.calculate_consistency_bonus()
        
        # Combine all adjustments
        combined_reward = base_reward * (1 + sharpe_adjustment * 0.5)
        combined_reward += drawdown_penalty * 0.1
        combined_reward += win_rate_bonus
        combined_reward += consistency_bonus
        # if combined_reward >=2:
        #     print(f"⚠️ High combined reward: {combined_reward:.4f} (base: {base_reward:.4f}, sharpe: {sharpe_adjustment:.4f}, drawdown: {drawdown_penalty:.4f}, win rate: {win_rate_bonus:.4f}, consistency: {consistency_bonus:.4f})")
        return combined_reward
    
    def calculate_hold_penalty(self, steps_without_action: int) -> float:
        """
        Calculate a penalty for consecutive HOLD actions
        
        Args:
            steps_without_action: Number of consecutive HOLD actions
            
        Returns:
            float: Hold penalty
        """
        # Only penalize after several consecutive HOLDs
        if steps_without_action < 10:
            return 0.0
            
        # Increasing penalty for extended HOLDs
        penalty = -0.0001 * (steps_without_action - 9)
        return max(-0.005, penalty)  # Cap the penalty