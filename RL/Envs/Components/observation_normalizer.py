import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class ObservationNormalizer:
    """
    A class to handle normalization of observation features in the trading environment.
    Normalizes all features to a standard range (typically [-1, 1] or [0, 1]) for better training stability.
    
    Features are grouped into categories with appropriate normalization methods for each:
    1. Pattern Features (7)
    2. Technical Indicators (3)
    3. Sentiment Features (2)
    4. COT Data (6)
    5. Time Features (7)
    6. Portfolio Features (5)
    """
    
    def __init__(self, 
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 clip_outliers: bool = True):
        """
        Initialize the normalizer with default scaling ranges for each feature type.
        
        Args:
            output_range: The target range for normalized features, default [-1, 1]
            clip_outliers: Whether to clip values outside expected ranges
        """
        
        self.output_range = output_range
        self.clip_outliers = clip_outliers        # Define feature indices for each category
        
        self._min_array = None
        self._max_array = None
        self._range_array = None
        self._out_min = None
        self._out_max = None
        self._out_range = None
        
        # These must match the order in the separated feature structure:
        # Market features (24): Pattern(7) + Technical(3) + Sentiment(1) + COT(6) + Time(7)
        # Portfolio features (6): balance_ratio, portfolio_max_drawdown, win_rate, avg_pnl_per_hour, decisive_exits, recovery_factor
        self.feature_indices = {
            'pattern': list(range(0, 7)),       # 7 features: probability, action, reward_risk_ratio, max_gain, max_drawdown, mse, expected_value
            'technical': list(range(7, 10)),    # 3 features: rsi, atr, atr_ratio  
            'sentiment': list(range(10, 11)),   # 1 feature: unified_sentiment
            'cot': list(range(11, 17)),         # 6 features: change_nonrept_long, change_nonrept_short, change_noncommercial_long, change_noncommercial_short, change_noncommercial_delta, change_nonreportable_delta
            'time': list(range(17, 24)),        # 7 features: hour_sin, hour_cos, day_sin, day_cos, asian_session, london_session, ny_session
            'portfolio': list(range(24, 30))    # 6 features: balance_ratio, portfolio_max_drawdown, win_rate, avg_pnl_per_hour, decisive_exits, recovery_factor
        }
        
        # Define normalization ranges for each feature
        # Format: (min_value, max_value, center_value)
        # Center value is used for scaling features that should be centered around a specific value
        self._init_feature_ranges()
        self._setup_vectorized_normalization()
      
        
    def _init_feature_ranges(self):
        """Initialize default scaling ranges for all features."""
        
        self.feature_ranges = {
            
            # Pattern Features - Based on actual DB values with 20% margin
            0: (0, 1.0, 0.5),        # probability [DB: 0.47-1.0]
            1: (0.0, 2.0, 1.0),          # action [DB: 1-2] (discrete values)
            2: (1.0, 12.0, 5.0),         # reward_risk_ratio [DB: 1.29-11.35]
            3: (-0.1, 0.1, 0.0),       # max_gain [DB: -0.007-0.008]
            4: (-0.1, 0.1, 0.0),     # max_drawdown [DB: -0.0033-0.003]
            5: (0.0, 0.05, 0.0),         # mse [DB: 3.9e-11-0.03]
            6: (0, 100, 50),         # expected_value [DB: 0.55-9.11]
            
            # Technical Indicators
            7: (0, 100, 50.0),       # rsi [DB: 20.53-80.90]
            8: (0, 100, 50),         # atr [DB: 2.43-13.63]
            9: (0.001, 0.01, 0.0055),   # atr_ratio [DB: 0.0012-0.0057]
            
            # Sentiment Features  
            10: (-1.0, 1.0, 0.0),        # unified_sentiment [DB: -0.63-0.79]
            
            # COT Data - Based on actual column names used in observations
            11: (-1000000, 1000000, 0.0),        # change_nonrept_long [DB: -5847-5839]
            12: (-1000000, 1000000, 0.0),        # change_nonrept_short [DB: -5985-7056]
            13: (-1000000, 1000000, 0.0),        # change_noncommercial_long [DB: -29820-49200]
            14: (-1000000, 1000000, 0.0),        # change_noncommercial_short [DB: -21924-19452]
            15: (-1000000, 1000000, 0.0),        # change_noncommercial_delta
            16: (-1000000, 1000000, 0.0),        # change_nonreportable_delta            
            
            # Time Features - Based on DB values, most already normalized
            17: (-1.0, 1.0, 0.0),        # hour_sin [DB: -1.0-1.0]
            18: (-1.0, 1.0, 0.0),        # hour_cos [DB: -1.0-0.97] 
            19: (-1.0, 1.0, 0.0),        # day_sin [DB: -0.43-0.97]
            20: (-1.0, 1.0, 0.0),        # day_cos [DB: -0.90-1.0]
            21: (0.0, 1.0, 0.0),         # asian_session [DB: 0-1]
            22: (0.0, 1.0, 0.0),         # london_session [DB: 0-1]
            23: (0.0, 1.0, 0.0),         # ny_session [DB: 0-1]
            
            # Portfolio Features - Updated feature names and ranges
            24: (0.0, 10.0, 1.0),         # balance_ratio [0, +inf)
            25: (-1.0, 0.0, 0.0),         # portfolio_drawdown [-1, 0] (negative drawdown values)
            26: (0.0, 1.0, 0.5),          # win_rate [0, 1]
            27: (-0.1, 0.1, 0.0),         # avg_pnl_per_hour (P&L per hour efficiency)
            28: (0.0, 1.0, 0.5),          # decisive_exits (ratio of TP/SL vs timeout)
            29: (-1, 1000.0, 1.0),         # recovery_factor (gains/drawdown ratio)
            
        }
        
    def _setup_vectorized_normalization(self):
        """Pre-compute arrays for faster normalization"""
        max_features = max(self.feature_ranges.keys()) + 1
        
        # Pre-compute min and max arrays
        self._min_array = np.array([
            self.feature_ranges.get(i, (0, 1, 0))[0] 
            for i in range(max_features)
        ])
        
        self._max_array = np.array([
            self.feature_ranges.get(i, (0, 1, 0))[1] 
            for i in range(max_features)
        ])
        
        self._range_array = self._max_array - self._min_array 
        
        # Output range values
        self._out_min, self._out_max = self.output_range
        self._out_range = self._out_max - self._out_min
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize the entire observation vector.
        
        Args:
            observation: Raw observation vector from the environment
            
        Returns:
            Normalized observation vector
        """
        
        # Create output array
        normalized = observation.copy()
        
        # Apply clipping if needed
        if self.clip_outliers:
            observation = np.clip(observation, self._min_array, self._max_array)
        
        # Vectorized normalization in one step
        normalized= ((observation - self._min_array) / self._range_array) * self._out_range + self._out_min # (1)
        
        # insure float32 type for compatibility
        normalized = normalized.astype(np.float32)
        
        return normalized

    
    def get_normalized_observation_space(self):
        """
        Return a gym.spaces.Box with normalized bounds for the observation space.
        Useful for defining the environment's observation_space when using this normalizer.
        
        Returns:
            A gym.spaces Box with normalized boundaries
        """
        from gymnasium import spaces
        
        # Use output range for all dimensions
        low_val, high_val = self.output_range
        
        # Use slightly expanded range to allow for numerical precision issues
        low_val -= 0.01
        high_val += 0.01
        
        # Create observation space with normalized values
        return spaces.Box(
            low=low_val,
            high=high_val,
            shape=(len(self.feature_ranges),),
            dtype=np.float32
        )
    
    def reset(self):
        pass
            
