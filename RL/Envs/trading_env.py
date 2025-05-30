import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .observation_normalizer import ObservationNormalizer

class PatternSentimentEnv(gym.Env):
    def __init__(self, data, initial_balance=100000, reward_type='combined',
                 normalize_observations=True, normalization_range=(-1.0, 1.0),
                 enable_adaptive_scaling=False):
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
        
        # Enhanced tracking for risk metrics
        self.reward_type = reward_type
        self.returns_history = []
        self.max_history_length = 20  # Number of returns to keep for rolling calculations
        self.equity_curve = [initial_balance]
        self.peak_balance = initial_balance
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_amount = 0
        self.loss_amount = 0
        self.max_drawdown = 0
        self.last_trade_idx = -10  # For tracking trade frequency

        # Initialize observation normalizer
        self.normalize_observations = normalize_observations
        self.normalizer = ObservationNormalizer(
            enable_adaptive_scaling=enable_adaptive_scaling, 
            output_range=normalization_range
        )
        
        # Count the number of features we have in the data
        sample_observation = 30
        
        # Observation space: pattern features + technical indicators + time features + portfolio state
        if normalize_observations:
            # Use normalizer to create appropriate observation space
            self.observation_space = self.normalizer.get_normalized_observation_space()
        else:
            # Use unbounded observation space
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(sample_observation,),
                dtype=np.float32
            )
        
        # Enhanced action space:
        # [action_type, position_size, reward_risk_ratio_multiplier]
        # action_type: 0=hold, 1=buy, 2=sell
        # position_size: fraction of portfolio (0.1-1.0)
        # reward_risk_ratio_multiplier: multiplier for risk-reward ratio (0.5-3.0)
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            spaces.Box(
                low=np.array([0.1, 0.5]),  # [position_size, reward_risk_ratio_multiplier]
                high=np.array([1.0, 3.0]),
                dtype=np.float32
            )
        ))

    def calculate_sharpe_reward(self, base_reward, trade_pnl):
        """Calculate reward using Sharpe ratio component"""
        if trade_pnl != 0:  # Only add to history if we had a trade
            self.returns_history.append(base_reward)
            
            # Keep history at max length
            if len(self.returns_history) > self.max_history_length:
                self.returns_history.pop(0)
        
        # Calculate rolling Sharpe ratio (annualized)
        if len(self.returns_history) > 1:
            returns_mean = np.mean(self.returns_history)
            returns_std = np.std(self.returns_history) + 1e-6  # Avoid division by zero
            sharpe = returns_mean / returns_std * np.sqrt(252)  # Annualized
            
            # Sharpe ratio bonus (higher is better)
            sharpe_bonus = np.clip(0.1 * sharpe, -0.5, 0.5)  # Limit impact
            return base_reward + sharpe_bonus
        return base_reward

    def calculate_sortino_reward(self, base_reward, trade_pnl):
        """Calculate reward using Sortino ratio (focusing on downside risk)"""
        if trade_pnl != 0:
            self.returns_history.append(base_reward)
            
            # Keep history at max length
            if len(self.returns_history) > self.max_history_length:
                self.returns_history.pop(0)
        
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

    def calculate_drawdown_penalty(self, trade_pnl):
        """Calculate penalty based on drawdown"""
        if trade_pnl != 0:  # Only update balance if we had a trade
            self.equity_curve.append(self.balance)
            
        # Calculate current drawdown
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

    def calculate_win_rate_bonus(self, trade_pnl):
        """Calculate bonus based on win rate and profit factor"""
        if trade_pnl > 0:
            self.winning_trades += 1
            self.win_amount += trade_pnl
        elif trade_pnl < 0:
            self.losing_trades += 1
            self.loss_amount += abs(trade_pnl)
        
        self.trade_count = self.winning_trades + self.losing_trades
        
        if self.trade_count > 5:  # Need enough trades for meaningful calculation
            win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
            
            # Win rate bonus (scaled to be small but meaningful)
            win_bonus = 0.05 * (win_rate - 0.5) * 2  # -0.05 to 0.05 range
            
            # Profit factor bonus (win amount / loss amount)
            profit_factor = self.win_amount / (self.loss_amount + 1e-6)
            profit_factor_bonus = min(0.05, 0.01 * (profit_factor - 1))  # Cap at 0.05
            
            return win_bonus + profit_factor_bonus
        return 0

    def calculate_consistency_bonus(self, base_reward):
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

    def calculate_overtrading_penalty(self):
        """Penalize excessive trading"""
        if self.index - self.last_trade_idx < 5:  # Less than 5 timesteps since last trade
            return -0.001  # Small penalty for overtrading
        return 0

    def calculate_calmar_reward(self, base_reward, trade_pnl):
        """Calculate reward based on Calmar ratio (return / max drawdown)"""
        if trade_pnl != 0:
            self.returns_history.append(base_reward)
            if len(self.returns_history) > self.max_history_length:
                self.returns_history.pop(0)
        
        if abs(self.max_drawdown) > 0.001 and len(self.returns_history) > 2:
            returns_mean = np.mean(self.returns_history)
            # Calculate Calmar ratio (return / max drawdown)
            calmar = returns_mean / abs(self.max_drawdown + 1e-6) * 252  # Annualized
            
            # Calmar ratio bonus (heavily rewards good return/drawdown ratio)
            calmar_bonus = np.clip(0.2 * calmar, -0.4, 0.4)
            return base_reward + calmar_bonus
        return base_reward

    def calculate_combined_reward(self, base_reward, trade_pnl):
        """Calculate combined risk-adjusted reward"""
        # Base reward from P&L
        adjusted_reward = base_reward
        
        # Add Sharpe ratio component
        sharpe_component = self.calculate_sharpe_reward(base_reward, trade_pnl) - base_reward
        adjusted_reward += (0.2 * sharpe_component)  # 20% weight
        
        # Add sortino ratio component (better handling of downside risk)
        sortino_component = self.calculate_sortino_reward(base_reward, trade_pnl) - base_reward
        adjusted_reward += (0.15 * sortino_component)  # 15% weight
        
        # Add drawdown penalty
        drawdown_penalty = self.calculate_drawdown_penalty(trade_pnl)
        adjusted_reward += drawdown_penalty  # Full impact of drawdown penalty
        
        # Add calmar ratio component
        calmar_component = self.calculate_calmar_reward(base_reward, trade_pnl) - base_reward
        adjusted_reward += (0.15 * calmar_component)  # 15% weight
        
        # Add win rate bonus
        win_bonus = self.calculate_win_rate_bonus(trade_pnl)
        adjusted_reward += (0.3 * win_bonus)  # 30% weight
        
        # Add consistency bonus
        consistency_bonus = self.calculate_consistency_bonus(base_reward)
        adjusted_reward += (0.1 * consistency_bonus)  # 10% weight
        
        # Add overtrading penalty
        overtrading_penalty = self.calculate_overtrading_penalty()
        adjusted_reward += overtrading_penalty
        
        return adjusted_reward
    
    def adaptive_position_sizing(self, base_position_size):
        """Adjust position size based on performance and drawdown"""
        # Start with base position size
        adjusted_size = base_position_size
        
        # 1. Drawdown-based adjustment
        current_equity = self.balance
        peak_equity = self.peak_balance
        drawdown = (current_equity / peak_equity) - 1
        
        # Reduce position size in drawdown
        if drawdown < -0.02:  # 2% drawdown
            adjusted_size *= 0.8  # 20% reduction
        if drawdown < -0.05:  # 5% drawdown
            adjusted_size *= 0.6  # Additional reduction (total: 52% reduction)
        if drawdown < -0.08:  # 8% drawdown
            adjusted_size *= 0.5  # Additional reduction (total: 76% reduction)
            
        # 2. Win/loss streak adjustment
        recent_trades = self.returns_history[-5:] if len(self.returns_history) >= 5 else self.returns_history
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
        current_data = self.data.iloc[self.index]
        if 'atr_ratio' in current_data:
            # Get current ATR ratio (volatility measure)
            atr_ratio = current_data['atr_ratio']
            
            # Baseline ATR ratio (consider this "normal" volatility)
            baseline_atr = 0.002  # Adjust based on your asset
            
            # Adjust position size inversely to volatility
            vol_adjustment = baseline_atr / (atr_ratio + 1e-6)
            vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)  # Limit adjustment range
            
            adjusted_size *= vol_adjustment
        
        # Ensure position size stays within allowed limits (0.1 to 1.0)
        return np.clip(adjusted_size, 0.1, 1.0)

    def reset(self, *, seed=None, options=None):
        self.index = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.tp_price = 0
        self.sl_price = 0
        self.trade_direction = 0
        
        # Reset risk metrics
        self.returns_history = []
        self.equity_curve = [self.initial_balance]
        self.peak_balance = self.initial_balance
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_amount = 0
        self.loss_amount = 0
        self.max_drawdown = 0
        self.last_trade_idx = -10
        
        # Reset normalizer if adaptive scaling is enabled
        self.normalizer.reset()
        
        super().reset(seed=seed)
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        # Always use DataFrame iloc indexing
        # Unpack action
        action_type, continuous_params = action
        base_position_size, risk_reward_multiplier = continuous_params
    
        current_data = self.data.iloc[self.index]
        
        # Use a default price of 1.0 if no price is provided
        # In real markets, you would have actual price data
        self.current_price = 1.0
        
        # Track peak balance for drawdown calculation
        self.peak_balance = max(self.peak_balance, self.balance)
        
        reward = 0
        base_reward = 0
        done = False
        truncated = False
        trade_open = False
        trade_closed = False  # Track if we closed a position this step
        trade_pnl = 0  # Track P&L for this trade
        
        # Apply adaptive position sizing
        position_size = self.adaptive_position_sizing(base_position_size)
        
        max_gain = current_data['max_gain'] if 'max_gain' in current_data else 0.01
        max_drawdown = current_data['max_drawdown'] if 'max_drawdown' in current_data else -0.01
        
        # Open new position
        pattern_action = current_data['action']
        if action_type == 1:  # Go Long
            if pattern_action == 1:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                # For long position agreeing with pattern, use pattern's max_gain for TP and max_drawdown for SL
                self.tp_price = self.entry_price * (1 + (max_gain * risk_reward_multiplier))
                self.sl_price = self.entry_price * (1 + (max_drawdown * (1/risk_reward_multiplier)))
                self.trade_direction = 1
                trade_open = True
                self.last_trade_idx = self.index  # Update last trade time
            elif pattern_action == 2:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                # For long position contradicting pattern's sell signal, we need to ensure:
                # 1. TP is ABOVE entry price (positive value)
                # 2. SL is BELOW entry price (negative value)
                # Since we're going against pattern, we should use abs(max_drawdown) for gain potential and abs(max_gain) for loss potential
                # but with proper signs for long positions
                self.tp_price = self.entry_price * (1 + (abs(max_drawdown) * risk_reward_multiplier)) 
                self.sl_price = self.entry_price * (1 - (abs(max_gain) * (1/risk_reward_multiplier)))
                self.trade_direction = 1
                trade_open = True
                self.last_trade_idx = self.index  # Update last trade time
            
        elif action_type == 2:  # Go Short
            if pattern_action == 2:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                # For short position agreeing with pattern, use pattern's max_gain for TP and max_drawdown for SL
                self.tp_price = self.entry_price * (1 + (max_gain * risk_reward_multiplier))
                self.sl_price = self.entry_price * (1 + (max_drawdown * (1/risk_reward_multiplier)))
                self.trade_direction = -1
                trade_open = True
                self.last_trade_idx = self.index  # Update last trade time
                
            elif pattern_action == 1:
                position_value = self.balance * position_size
                self.position = position_value / self.current_price
                self.entry_price = self.current_price
                # For short position contradicting pattern's buy signal, we need to ensure:
                # 1. TP is BELOW entry price (negative value)
                # 2. SL is ABOVE entry price (positive value)
                # Since we're going against pattern, we should use abs(max_drawdown) for gain potential and abs(max_gain) for loss potential
                # but with proper signs for short positions
                self.tp_price = self.entry_price * (1 - (abs(max_drawdown) * risk_reward_multiplier))
                self.sl_price = self.entry_price * (1 + (abs(max_gain) * (1/risk_reward_multiplier)))
                self.trade_direction = -1
                trade_open = True
                self.last_trade_idx = self.index  # Update last trade time
                
        # Check if trade hit TP/SL
        if trade_open:
            # Get high and low values from DataFrame
            mfe = current_data['mfe'] if 'mfe' in current_data else 0.01
            mae = current_data['mae'] if 'mae' in current_data else -0.01
            
            # Get actual return from DataFrame 
            actual_return = current_data['actual_return']
            
            highest_price = self.entry_price * (1 + mfe)
            lowest_price = self.entry_price * (1 + mae)
            
            if self.trade_direction == 1:  # Long
                if highest_price >= self.tp_price:
                    base_reward = (self.tp_price - self.entry_price)/self.entry_price  # Hit TP
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                elif lowest_price <= self.sl_price:
                    base_reward = (self.sl_price - self.entry_price)/self.entry_price  # Hit SL
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                else: # close the position and the actual return
                    base_reward = actual_return  # Close position at current price
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                    
            elif self.trade_direction == -1:  # Short
                if lowest_price <= self.tp_price:
                    base_reward = -(self.tp_price - self.entry_price)/self.entry_price  # Hit TP
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                elif highest_price >= self.sl_price:
                    base_reward = -(self.sl_price - self.entry_price)/self.entry_price  # Hit SL
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
                else: # close the position and the actual return
                    base_reward = -actual_return
                    trade_pnl = base_reward * self.position * self.entry_price
                    trade_closed = True
        
        # Calculate risk-adjusted reward based on selected type
        if trade_closed:
            if self.reward_type == 'sharpe':
                reward = self.calculate_sharpe_reward(base_reward, trade_pnl)
            elif self.reward_type == 'sortino':
                reward = self.calculate_sortino_reward(base_reward, trade_pnl)
            elif self.reward_type == 'drawdown_focus':
                reward = base_reward + self.calculate_drawdown_penalty(trade_pnl)
            elif self.reward_type == 'calmar':
                reward = self.calculate_calmar_reward(base_reward, trade_pnl)
            elif self.reward_type == 'win_rate':
                reward = base_reward + self.calculate_win_rate_bonus(trade_pnl)
            elif self.reward_type == 'consistency':
                reward = base_reward + self.calculate_consistency_bonus(base_reward)
            elif self.reward_type == 'combined':
                reward = self.calculate_combined_reward(base_reward, trade_pnl)
            else:
                reward = base_reward  # Default to base reward
        else:
            # Small penalty for no action to encourage trading when appropriate
            reward = -0.0001
                    
        # Calculate portfolio value
        self.position_value = self.position * self.current_price
        portfolio_balance = self.balance + (trade_pnl if trade_closed else 0)
        
        info = {
            'portfolio_balance': portfolio_balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'trade_direction': self.trade_direction,
            'trade_pnl': trade_pnl if trade_closed else 0,
            'win_rate': self.winning_trades / self.trade_count if self.trade_count > 0 else 0,
            'max_drawdown': self.max_drawdown
        }
        
        # Close position if TP/SL hit
        if trade_closed:
            self.balance += trade_pnl
            self.position = 0
            self.entry_price = 0
            self.trade_direction = 0
            
            # Update equity curve for drawdown calculation
            self.equity_curve.append(self.balance)
            self.index += 1
        if self.index >= len(self.data):
            done = True
            truncated = True        
            observation = None
        else:
            observation = self._get_observation()

        return observation, reward, done, truncated, info
        
    def _get_observation(self):
        # Always use DataFrame indexing since the data is always in DataFrame format
        data_point = self.data.iloc[self.index]
        features = []
        
        # Add base pattern features (directly from DataFrame)
        base_features = [
            data_point['probability'],
            data_point['action'],
            data_point['reward_risk_ratio'],
            data_point['max_gain'],
            data_point['max_drawdown'],
            data_point['mse'],
            data_point['expected_value']
        ]
        features.extend(base_features)
        
        # Technical indicators - based on actual column names in the dataset
        technical_indicators = ['rsi', 'atr', 'atr_ratio']
        for indicator in technical_indicators:
            if indicator in data_point:
                features.append(data_point[indicator])
            
        # Sentiment features - based on actual column names
        sentiment_features = ['unified_sentiment', 'sentiment_count']
        for feature in sentiment_features:
            if feature in data_point:
                features.append(data_point[feature])
            
        # COT data - based on actual column names in your dataset
        cot_features = [
            'net_noncommercial', 'net_nonreportable',
            'change_nonrept_long', 'change_nonrept_short',
            'change_noncommercial_long', 'change_noncommercial_short'
        ]
        for feature in cot_features:
            if feature in data_point:
                features.append(data_point[feature])
            
        # Time-based features - based on actual column names
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                        'asian_session', 'london_session', 'ny_session']
        for feature in time_features:
            if feature in data_point:
                features.append(data_point[feature])
        
        # Make sure we have at least the base features
        while len(features) < 5:
            features.append(0.0)
            
        # Convert to numpy array
        market_features = np.array(features, dtype=np.float32)
        
        # Portfolio features
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position_value / (self.balance + self.position_value + 1e-6),  # Position ratio
            self.position,  # Absolute position size
            self.max_drawdown,  # Add drawdown as a feature
            self.winning_trades / (self.trade_count + 1e-6),  # Add win rate as a feature
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Apply normalization if enabled
        if self.normalize_observations:
            observation = self.normalizer.normalize_observation(observation)
            
        return observation
