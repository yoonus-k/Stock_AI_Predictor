import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TupleActionWrapper(gym.Wrapper):
    """
    Wrapper that converts a Tuple action space to a Box action space for compatibility with SB3.
    This allows us to use algorithms like PPO with our custom trading environment.
    
    The wrapper converts:
    - Tuple(Discrete(3), Box([0.1, 0.5], [1.0, 3.0], (2,), float32))
    - to Box([0, 0.1, 0.5], [2, 1.0, 3.0], (3,), float32)
    
    Where the first dimension represents the action type (0=hold, 1=buy, 2=sell),
    and the other two dimensions represent position_size and risk_reward_ratio.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Extract original action space components
        discrete_space = self.env.action_space[0]
        box_space = self.env.action_space[1]
        
        # Get information from the spaces
        self.n_discrete = discrete_space.n  # Number of discrete actions (3)
        self.box_low = box_space.low        # Lower bounds for continuous actions [0.1, 0.5]
        self.box_high = box_space.high      # Upper bounds for continuous actions [1.0, 3.0]
        
        # Create a new Box action space with an extra dimension for the discrete action
        self.action_space = spaces.Box(
            low=np.array([0.0, self.box_low[0], self.box_low[1]]),  # [0, 0.1, 0.5]
            high=np.array([float(self.n_discrete - 1), self.box_high[0], self.box_high[1]]),  # [2, 1.0, 3.0]
            dtype=np.float32
        )
    
    def step(self, action):
        """
        Convert the Box action back to a Tuple action before passing to the environment.
        
        Parameters:
            action: Box action from the agent [action_type, position_size, risk_reward_ratio]
            
        Returns:
            Standard step return (observation, reward, done, truncated, info)
        """
        # Convert continuous action_type to discrete by rounding
        discrete_action = int(round(min(max(action[0], 0.0), self.n_discrete - 1)))
        
        # Clip continuous values to be within bounds
        position_size = np.clip(action[1], self.box_low[0], self.box_high[0])
        risk_reward = np.clip(action[2], self.box_low[1], self.box_high[1])
        
        # Create tuple action for the environment
        tuple_action = (
            discrete_action,
            np.array([position_size, risk_reward], dtype=np.float32)
        )
        
        # Pass the converted action to the environment
        return self.env.step(tuple_action)
