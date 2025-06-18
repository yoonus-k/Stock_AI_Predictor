import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

from .trading_state import TradingState

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
        """
        # Define the action space as MultiDiscrete
        self.action_space = MultiDiscrete([
            3,              # Action type: 0=HOLD, 1=BUY, 2=SELL
            20,             # Position size: discretized (0=0.001, 1=0.002, ..., 19=0.02) (% from the account to risk)
            10,             # Risk reward: discretized (0=0.5, 1=0.75, ..., 9=3.0)
            10              # Hold time: discretized (0=3h, 1=6h, 2=9h, 3=12h, 4=15h, 5=18h, 6=21h, 7=24h, 8=27h, 9=30h)
        ])
        
        self.trading_state = None  # Will be set later
    
    def set_trading_state(self, trading_state: TradingState):
        """
        Set the shared trading state reference
        
        Args:
            trading_state: Shared trading state object
        """
        self.trading_state = trading_state
    
    def reset(self):
        """Reset action tracking variables"""
        pass
    
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
        # 0->0.001, 1->0.002, ..., 19->0.02
        position_size = round(0.001 + (action[1] * 0.001), 4)
        
        # Map risk reward from discrete index to continuous value
        # 0->0.5, 1->0.75, ..., 9->3.0
        risk_reward_multiplier = 0.5 + (action[2] * 0.25)
        
        # Map hold time from discrete index to hours
        # Using 3h increments: 3h, 6h, 9h, 12h, 15h, 18h, 21h, 24h, 27h, 30h
        hold_time_hours = 3 + (action[3] * 3)
        
        return action_type, position_size, risk_reward_multiplier, hold_time_hours
    