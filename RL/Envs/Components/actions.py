import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

class ActionHandler:
    """
    A class to handle action space definition and action processing
    in the trading environment.
    Uses MultiDiscrete action space for better compatibility with PPO.
    """
    def __init__(self):
        """
        Initialize the action handler with MultiDiscrete action space
        
        Action space components:
        - Action type: 0=HOLD, 1=BUY, 2=SELL
        - Position size: discretized (0=0.005, 1=0.01, ..., 19=0.1) representing account risk percentage
        - Risk reward: discretized (0=0.5, 1=0.75, ..., 9=3.0)
        - Hold time: discretized holding periods in hours (0=3h, 1=6h, 2=9h, ..., 9=30h)
        """# Define the action space as MultiDiscrete
        self.action_space = MultiDiscrete([
            3,              # Action type: 0=HOLD, 1=BUY, 2=SELL
            20,             # Position size: discretized (0=0.005, 1=0.01, ..., 19=0.1) (% from the account to risk)
            10,             # Risk reward: discretized (0=0.5, 1=0.75, ..., 9=3.0)
            10              # Hold time: discretized (0=3h, 1=6h, 2=9h, 3=12h, 4=15h, 5=18h, 6=21h, 7=24h, 8=27h, 9=30h)
        ])
        
        # Action tracking for monitoring
        self.action_counts = [0, 0, 0]  # [HOLD, BUY, SELL] counts
        self.steps_without_action = 0  # Track consecutive HOLD actions
    
    def reset(self):
        """Reset action tracking variables"""
        self.action_counts = [0, 0, 0]
        self.steps_without_action = 0
    def process_action(self, action: np.ndarray) -> Tuple[int, float, float, int]:
        """
        Process the raw action from the agent into actionable components
        
        Args:
            action: Raw action array from agent [action_type, position_size_idx, risk_reward_idx, hold_time_idx]
            
        Returns:
            Tuple[int, float, float, int]: (action_type, position_size, risk_reward_multiplier, hold_time_hours)
              - action_type: 0=HOLD, 1=BUY, 2=SELL
              - position_size: Fraction of portfolio to risk (0.005-0.1)
              - risk_reward_multiplier: Multiplier for risk-reward ratio (0.5-3.0)
              - hold_time_hours: Position holding time in hours (3-30)
        """
        # Extract discrete action components
        action_type = int(action[0])  # Already 0, 1, or 2
        
        # Map position size from discrete index to continuous value
        # 0->0.005, 1->0.01, ..., 19->0.1
        position_size = 0.005 + (action[1] * 0.005)
        
        # Map risk reward from discrete index to continuous value
        # 0->0.5, 1->0.75, ..., 9->3.0
        risk_reward_multiplier = 0.5 + (action[2] * 0.25)
        
        # Map hold time from discrete index to hours
        # Using 3h increments: 3h, 6h, 9h, 12h, 15h, 18h, 21h, 24h, 27h, 30h
        hold_time_hours = 3 + (action[3] * 3)
        
        # Update action tracking
        if action_type == 0:  # HOLD
            self.steps_without_action += 1
        else:  # BUY or SELL
            self.steps_without_action = 0
            
        # Update action counts for monitoring
        self.action_counts[action_type] += 1
        
        return action_type, position_size, risk_reward_multiplier, hold_time_hours
    
    def adaptive_position_sizing(self, base_position_size: float, balance: float, 
                               peak_balance: float, returns_history: List[float],
                               atr_ratio: Optional[float] = None) -> float:
        """
        Adjust position size based on performance and drawdown
        
        Args:
            base_position_size: Base position size to adjust
            balance: Current account balance
            peak_balance: Highest balance achieved
            returns_history: List of recent returns
            atr_ratio: Current ATR ratio as volatility measure (optional)
            
        Returns:
            float: Adjusted position size (0.1-1.0)
        """
        # Start with base position size
        adjusted_size = base_position_size
        
        # 1. Drawdown-based adjustment
        drawdown = (balance / peak_balance) - 1
        
        # Reduce position size in drawdown
        if drawdown < -0.02:  # 2% drawdown
            adjusted_size *= 0.8  # 20% reduction
        if drawdown < -0.05:  # 5% drawdown
            adjusted_size *= 0.6  # Additional reduction (total: 52% reduction)
        if drawdown < -0.08:  # 8% drawdown
            adjusted_size *= 0.5  # Additional reduction (total: 76% reduction)
            
        # 2. Win/loss streak adjustment
        recent_trades = returns_history[-5:] if len(returns_history) >= 5 else returns_history
        if recent_trades:
            recent_wins = sum(1 for r in recent_trades if r > 0)
            recent_losses = sum(1 for r in recent_trades if r < 0)
            
            # Increase size on winning streak (with cap)
            if recent_wins >= 3 and recent_wins > recent_losses:
                win_ratio = recent_wins / len(recent_trades)
                adjusted_size *= min(1.2, 1 + (win_ratio * 0.2))  # Max 20% increase
                
            # Decrease size on losing streak
            elif recent_losses >= 3 and recent_losses > recent_wins:
                loss_ratio = recent_losses / len(recent_trades)
                adjusted_size *= max(0.5, 1 - (loss_ratio * 0.5))  # Max 50% decrease
        
        # 3. Volatility-based adjustment (if atr_ratio is available)
        if atr_ratio is not None:
            # Baseline ATR ratio (consider this "normal" volatility)
            baseline_atr = 0.002  # Adjust based on your asset
            
            # Adjust position size inversely to volatility
            vol_adjustment = baseline_atr / (atr_ratio + 1e-6)
            vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)  # Limit adjustment range
            
            adjusted_size *= vol_adjustment
        
        # Ensure position size stays within allowed limits (0.1 to 1.0)
        return np.clip(adjusted_size, 0.1, 1.0)
        
    def get_action_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about actions taken during the episode
        
        Returns:
            Dict: Statistics about actions
        """        
        total_actions = sum(self.action_counts)
        if total_actions == 0:
            return {
                "hold_pct": 0.0,
                "buy_pct": 0.0,
                "sell_pct": 0.0,
                "consecutive_holds": self.steps_without_action
            }
        return {
            "hold_pct": self.action_counts[0] / total_actions * 100,
            "buy_pct": self.action_counts[1] / total_actions * 100,
            "sell_pct": self.action_counts[2] / total_actions * 100,
            "consecutive_holds": self.steps_without_action
        }
        
    def calculate_trade_targets(self, action_type: int, entry_price: float, max_gain: float, 
                              max_drawdown: float, risk_reward_multiplier: float, 
                              pattern_action: int) -> Tuple[float, float, float]:
        """
        Calculate take profit, stop loss prices, and position size modifier for a trade
        
        Args:
            action_type: Action type (1=BUY, 2=SELL)
            entry_price: Entry price for the trade
            max_gain: Maximum expected gain from pattern
            max_drawdown: Maximum expected drawdown from pattern
            risk_reward_multiplier: Risk-reward ratio multiplier
            pattern_action: Action suggested by pattern (1=BUY, 2=SELL)
            
        Returns:
            Tuple[float, float, float]: (take_profit_price, stop_loss_price, position_size_modifier)
        """
        # Default position size modifier (will be adjusted based on pattern match)
        position_size_modifier = 1.0
        
        # For long position (BUY)
        if action_type == 1:
            # If agent agrees with pattern - 100% position size
            if pattern_action == 1:
                tp_price = entry_price * (1 + (abs(max_gain) * risk_reward_multiplier))
                sl_price = entry_price * (1 - abs(max_drawdown) )
                position_size_modifier = 1.0  # Full position size for exact pattern match
            # If agent contradicts pattern - 50% position size
            elif pattern_action == 2:
                tp_price = entry_price * (1 + (abs(max_drawdown) * risk_reward_multiplier))
                sl_price = entry_price * (1 - abs(max_gain) )
                position_size_modifier = 0.5  # Half position size for contradicting pattern
            else:
                # Default if pattern_action is neutral - 75% position size
                tp_price = entry_price * (1 + (0.01 * risk_reward_multiplier))
                sl_price = entry_price * (1 - (0.01 ))
                position_size_modifier = 0.75  # 75% position size for neutral pattern
        
        # For short position (SELL)
        elif action_type == 2:
            # If agent agrees with pattern - 100% position size
            if pattern_action == 2:
                tp_price = entry_price * (1 - (abs(max_gain) * risk_reward_multiplier))
                sl_price = entry_price * (1 + abs(max_drawdown) )
                position_size_modifier = 1.0  # Full position size for exact pattern match
            # If agent contradicts pattern - 50% position size
            elif pattern_action == 1:
                tp_price = entry_price * (1 - (abs(max_drawdown) * risk_reward_multiplier))
                sl_price = entry_price * (1 + abs(max_gain))
                position_size_modifier = 0.5  # Half position size for contradicting pattern
            else:
                # Default if pattern_action is neutral - 75% position size
                tp_price = entry_price * (1 - (0.01 * risk_reward_multiplier))
                sl_price = entry_price * (1 + (0.01 ))
                position_size_modifier = 0.75  # 75% position size for neutral pattern
        else:
            # No TP/SL for HOLD actions
            tp_price = 0.0
            sl_price = 0.0
            position_size_modifier = 0.0
            
        return tp_price, sl_price, position_size_modifier