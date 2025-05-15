import numpy as np
from stable_baselines3 import PPO

class RLPolicy:
    def __init__(self, model_path=None):
        # Load or define RL model
        if model_path:
            self.model = PPO.load("models/pattern_sentiment_rl_model.zip")
        else:
            self.model = None  # Placeholder until trained

        # Define action mapping
        self.action_types = ["HOLD", "BUY", "SELL"]
        
        # Default position size (can be overridden by model)
        self.default_position_size = 0.1  # 10% of portfolio

    def build_state(self, pattern_data, sentiment_data, portfolio_state=None):
        """
        Construct state from features.
        Now includes portfolio information to match the new observation space.
        """
        # Market features (8 dimensions)
        market_features = np.array([
            pattern_data.get('probability', 0),
            pattern_data.get('action', 0),
            pattern_data.get('reward_risk_ratio', 0),
            pattern_data.get('max_gain', 0),
            pattern_data.get('max_drawdown', 0),
            sentiment_data.get('impact_score', 0),
            sentiment_data.get('news_score', 0),
            sentiment_data.get('twitter_score', 0),
        ], dtype=np.float32)
        
        # Portfolio features (3 dimensions)
        if portfolio_state is None:
            portfolio_state = {
                'normalized_balance': 1.0,  # Full balance available
                'position_ratio': 0.0,      # No current position
                'position_size': 0.0        # No shares held
            }
            
        portfolio_features = np.array([
            portfolio_state['normalized_balance'],
            portfolio_state['position_ratio'],
            portfolio_state['position_size']
        ], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_features])

    def select_action(self, state):
        """
        Returns both action type and position size.
        """
        if self.model is None:
            raise RuntimeError("RL model not loaded or trained.")

        # Get action from model (now returns dict with action_type and position_size)
        action_dict = self.model.predict(state, deterministic=True)[0]
        
        return {
            'action_type': self.action_types[action_dict['action_type']],
            'position_size': float(action_dict['position_size'][0])  # Convert to scalar
        }

    def estimate_confidence(self, state):
        """
        Optional: Confidence can be used to modulate position size.
        Now returns both action confidence and position size confidence.
        """
        return {
            'action_confidence': 0.6,  # Placeholder
            'size_confidence': 0.5      # Placeholder
        }