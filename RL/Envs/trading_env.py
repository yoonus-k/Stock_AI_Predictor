import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PatternSentimentEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(PatternSentimentEnv, self).__init__()
        
        self.data = data
        self.index = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Current position size (shares/contracts)
        self.position_value = 0  # Dollar value of current position
        self.current_price = 0
        self.trade_size = 0.1  # Fraction of balance to trade (10%)
        
        self.entry_price = 0
        self.tp_price = 0
        self.sl_price = 0 
        self.trade_direction = 0  # 1=long, -1=short, 0=neutral

        # Observation space: pattern features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,),  # Added portfolio metrics
            dtype=np.float32
        )

        # New action space:
        # [action_type, position_size]
        # action_type: 0=hold, 1=buy, 2=sell
        # position_size: fraction of portfolio (0.1-1.0)
        self.action_space = spaces.Box(
            low=np.array([0, 0.1]),  # [action_type, position_size]
            high=np.array([2, 1.0]),  # 0=hold, 1=buy, 2=sell
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.index = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        
        super().reset(seed=seed)
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_data = self.data.iloc[self.index]
        self.current_price = current_data['price']
        reward = 0
        done = False
        truncated = False
        trade_open = False
        trade_closed = False  # Track if we closed a position this step
        
        action_dict = {
        'action_type': int(np.round(action[0])),  # Convert to discrete
        'position_size': np.array([action[1]], dtype=np.float32)
        }
        # Execute trade
        action_type = action_dict['action_type']
        position_size = action_dict['position_size'][0]  # Convert to scalar
        # Close previous position if exists
        # if self.position != 0:
        #     # Calculate PnL
        #     price_change = (self.current_price - self.entry_price) / self.entry_price
        #     if self.position > 0:  # Long position
        #         reward = price_change
        #     else:  # Short position
        #         reward = -price_change
                
        #     self.balance += self.position * self.current_price
        #     self.position = 0
        #     self.entry_price = 0
        #     self.trade_direction = 0
        
        # Open new position
        pattern_action = current_data['action']
        if action_type == 1:  # Go Long
            if pattern_action == 1:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                self.tp_price = self.entry_price * (1 + current_data['max_gain'])
                self.sl_price = self.entry_price * (1 + current_data['max_drawdown'])
                self.trade_direction = 1
                trade_open = True
            elif pattern_action == 2:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                self.tp_price = self.entry_price * (1 + current_data['max_drawdown'])
                self.sl_price = self.entry_price * (1 + current_data['max_gain'])
                self.trade_direction = 1
                trade_open = True
            
        elif action_type == 2:  # Go Short
            if pattern_action == 2:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price  # Negative for short
                self.entry_price = self.current_price
                self.tp_price = self.entry_price * (1 + current_data['max_gain'])
                self.sl_price = self.entry_price * (1 + current_data['max_drawdown'])
                self.trade_direction = -1
                trade_open = True
                
            elif pattern_action == 1:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price  # Negative for short
                self.entry_price = self.current_price
                self.tp_price = self.entry_price * (1 + current_data['max_drawdown'])
                self.sl_price = self.entry_price * (1 + current_data['max_gain'])
                self.trade_direction = -1
                trade_open = True
                
        # Check if trade hit TP/SL
        if trade_open :
            # Calculate TP and SL prices based on entry price and current data
            high, low = current_data['high'], current_data['low']
            highest_price = self.entry_price * (1 + high)
            lowest_price = self.entry_price * (1 + low)
            
            if self.trade_direction == 1:  # Long
                if highest_price >= self.tp_price:
                    reward = (self.tp_price - self.entry_price)/self.entry_price  # Hit TP
                    trade_closed = True
                elif lowest_price <= self.sl_price:
                    reward = (self.sl_price - self.entry_price)/self.entry_price  # Hit SL
                    trade_closed = True
                else: # close the position and the acual return
                    reward = current_data['actual_return']  # Close position at current price
                    trade_closed = True
                    
            elif self.trade_direction == -1:  # Short
                if lowest_price <= self.tp_price:
                    reward = -(self.tp_price - self.entry_price)/self.entry_price  # Hit TP
                    trade_closed = True
                elif high >= self.sl_price:
                    reward = -(self.sl_price - self.entry_price)/self.entry_price  # Hit SL
                    trade_closed = True
                else: # close the position and the acual return
                    reward = -current_data['actual_return']
                    trade_closed = True
                    
    
        # Calculate portfolio value
        self.position_value = self.position * self.current_price
        portfolio_balance = self.balance + (reward * self.position * self.entry_price)
        
        info = {
            'portfolio_balance': portfolio_balance ,
            'position': self.position,
            'entry_price': self.entry_price,
            'trade_direction': self.trade_direction
        }
        
        # Close position if TP/SL hit
        if trade_closed:
            self.balance += (reward * self.position * self.entry_price)
            self.position = 0
            self.entry_price = 0
            self.trade_direction = 0
            
        
        self.index += 1
        if self.index >= len(self.data):
            done = True
            truncated = True

        observation = self._get_observation() if not done else None

        return observation, reward, done, truncated, info

    def _get_observation(self):
        data_point = self.data.iloc[self.index]
        
        # Market features
        market_features = np.array([
            data_point['probability'],
            data_point['action'],
            data_point['reward_risk_ratio'],
            data_point['max_gain'],
            data_point['max_drawdown'],
            data_point['impact_score'],
            data_point['news_score'],
            data_point['twitter_score'],
        ], dtype=np.float32)
        
        # Portfolio features
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position_value / (self.balance + self.position_value + 1e-6),  # Position ratio
            self.position  # Absolute position size
        ], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_features])