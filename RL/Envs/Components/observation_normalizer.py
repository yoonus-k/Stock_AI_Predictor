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
                 enable_adaptive_scaling: bool = False,
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 clip_outliers: bool = True):
        """
        Initialize the normalizer with default scaling ranges for each feature type.
        
        Args:
            enable_adaptive_scaling: If True, track feature ranges during runtime and adjust normalization
            output_range: The target range for normalized features, default [-1, 1]
            clip_outliers: Whether to clip values outside expected ranges
        """
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.output_range = output_range
        self.clip_outliers = clip_outliers
        
        # Define feature indices for each category
        # These must match the order in _get_observation() method in trading_env.py
        self.feature_indices = {
            'pattern': list(range(0, 7)),
            'technical': list(range(7, 10)),
            'sentiment': list(range(10, 12)),
            'cot': list(range(12, 18)),
            'time': list(range(18, 25)),
            'portfolio': list(range(25, 30))
        }
        
        # Define normalization ranges for each feature
        # Format: (min_value, max_value, center_value)
        # Center value is used for scaling features that should be centered around a specific value
        self._init_feature_ranges()
          # For adaptive scaling (optional)
        self.observed_min = {}
        self.observed_max = {}
        
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
            8: (0, 100, 7.5),         # atr [DB: 2.43-13.63]
            9: (0.001, 0.01, 0.0035),   # atr_ratio [DB: 0.0012-0.0057]
            
            # Sentiment Features
            10: (-1, 1, 0.0),        # unified_sentiment [DB: -0.63-0.79]
            11: (0.0, 100, 3.0),         # sentiment_count [DB: 0-6]
            
            # COT Data - Adding margins for future variation
            12: (-1000000, 1000000, 0),   # net_noncommercial [DB: 131168-315390]
            13: (-1000000, 1000000, 0),     # net_nonreportable [DB: 15499-30285]
            14: (-100000, 100000, 0.0),        # change_nonrept_long [DB: -5847-5839]
            15: (-100000,100000, 0.0),        # change_nonrept_short [DB: -5985-7056]
            16: (-100000, 100000, 0.0),      # change_noncommercial_long [DB: -29820-49200]
            17: (-10000, 100000, 0.0),      # change_noncommercial_short [DB: -21924-19452]
              # Time Features - Based on DB values, most already normalized
            18: (-1.0, 1.0, 0.0),        # hour_sin [DB: -1.0-1.0]
            19: (-1.0, 1.0, 0.0),        # hour_cos [DB: -1.0-0.97] 
            20: (-0.5, 1.0, 0.0),        # day_sin [DB: -0.43-0.97]
            21: (-1.0, 1.1, 0.0),        # day_cos [DB: -0.90-1.0]
            22: (0.0, 1.0, 0.0),         # asian_session [DB: 0-1]
            23: (0.0, 1.0, 0.0),         # london_session [DB: 0-1]
            24: (0.0, 1.0, 0.0),         # ny_session [DB: 0-1]
            
            # Portfolio Features - Using conservative ranges with room for growth
            25: (0.0, 2.0, 1.0),         # balance_ratio [0, +inf)
            26: (0.0, 1.0, 0.5),         # position_ratio [0, 1]
            27: (-1000.0, 1000.0, 0.0),  # position [-inf, +inf]
            28: (-1.0, 0.0, 0.0),        # max_drawdown [-1, 0]
            29: (0.0, 1.0, 0.5)     ,     # win_rate [0, 1]
            30: (0.0,10.0, 5)        # profit_factor [0, +inf)
        }
    
    def normalize_feature(self, value: float, feature_idx: int) -> float:
        """
        Normalize a single feature value based on its index and expected range.
        
        Args:
            value: Raw feature value
            feature_idx: Index of the feature in the observation vector
            
        Returns:
            Normalized feature value in the output range
        """
        if feature_idx not in self.feature_ranges:
            return value  # Return unchanged if we don't have normalization info
            
        min_val, max_val, _ = self._get_feature_range(feature_idx)
        
        # Update observed ranges if adaptive scaling is enabled
        if self.enable_adaptive_scaling:
            self._update_observed_range(feature_idx, value)
        
        # Clip outliers if requested
        if self.clip_outliers:
            value = np.clip(value, min_val, max_val)
            
        # Normalize to [0, 1] first
        norm_val = (value - min_val) / (max_val - min_val + 1e-8)
        
        # Then scale to output range
        out_min, out_max = self.output_range
        return norm_val * (out_max - out_min) + out_min
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize the entire observation vector.
        
        Args:
            observation: Raw observation vector from the environment
            
        Returns:
            Normalized observation vector
        """
        if len(observation) != len(self.feature_ranges):
            # Warning: observation length doesn't match expected features
            # This could happen if the environment changes or feature extraction changes
            print(f"Warning: Observation length {len(observation)} doesn't match expected features {len(self.feature_ranges)}")
            
        normalized = np.zeros_like(observation)
        for i, val in enumerate(observation):
            if i < len(self.feature_ranges):
                normalized[i] = self.normalize_feature(val, i)
            else:
                normalized[i] = val  # Keep as is if we don't have info
                
        return normalized
    
    def normalize_by_category(self, observation: np.ndarray, category: str) -> np.ndarray:
        """
        Normalize features of a specific category.
        
        Args:
            observation: Raw observation vector
            category: Category name ('pattern', 'technical', etc.)
            
        Returns:
            Observation with normalized features for the specified category
        """
        if category not in self.feature_indices:
            return observation  # Return unchanged if category not found
            
        normalized = observation.copy()
        for idx in self.feature_indices[category]:
            if idx < len(observation):
                normalized[idx] = self.normalize_feature(observation[idx], idx)
                
        return normalized
    
    def denormalize_feature(self, value: float, feature_idx: int) -> float:
        """
        Convert a normalized value back to its original scale.
        
        Args:
            value: Normalized feature value
            feature_idx: Feature index
            
        Returns:
            Denormalized value
        """
        if feature_idx not in self.feature_ranges:
            return value  # Return unchanged if we don't have normalization info
            
        min_val, max_val, _ = self._get_feature_range(feature_idx)
        
        # First scale from output range to [0, 1]
        out_min, out_max = self.output_range
        norm_val = (value - out_min) / (out_max - out_min + 1e-8)
        
        # Then back to original range
        return norm_val * (max_val - min_val) + min_val
    
    def denormalize_observation(self, normalized_observation: np.ndarray) -> np.ndarray:
        """
        Convert a normalized observation vector back to its original scale.
        
        Args:
            normalized_observation: Normalized observation vector
            
        Returns:
            Denormalized observation vector
        """
        denormalized = np.zeros_like(normalized_observation)
        for i, val in enumerate(normalized_observation):
            if i < len(self.feature_ranges):
                denormalized[i] = self.denormalize_feature(val, i)
            else:
                denormalized[i] = val  # Keep as is if we don't have info
                
        return denormalized
    
    def _get_feature_range(self, feature_idx: int) -> Tuple[float, float, float]:
        """
        Get the min, max and center values for a feature.
        Uses adaptive ranges if enabled and observed.
        
        Args:
            feature_idx: Feature index
            
        Returns:
            Tuple of (min_value, max_value, center_value)
        """
        default_range = self.feature_ranges[feature_idx]
        
        if not self.enable_adaptive_scaling:
            return default_range
            
        # If adaptive scaling is enabled, use observed ranges when available
        min_val, max_val, center = default_range
        
        if feature_idx in self.observed_min:
            min_val = min(min_val, self.observed_min[feature_idx])
            
        if feature_idx in self.observed_max:
            max_val = max(max_val, self.observed_max[feature_idx])
            
        return (min_val, max_val, center)
    
    def _update_observed_range(self, feature_idx: int, value: float) -> None:
        """
        Update the observed min/max values for a feature.
        
        Args:
            feature_idx: Feature index
            value: Observed value
        """
        # Skip NaN or inf values
        if not np.isfinite(value):
            return
            
        # Update min
        if feature_idx not in self.observed_min or value < self.observed_min[feature_idx]:
            self.observed_min[feature_idx] = value
            
        # Update max
        if feature_idx not in self.observed_max or value > self.observed_max[feature_idx]:
            self.observed_max[feature_idx] = value
    
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
        """Reset adaptive scaling if it's enabled."""
        if self.enable_adaptive_scaling:
            self.observed_min = {}
            self.observed_max = {}
            
    def update_feature_range(self, feature_idx: int, min_val: float, max_val: float, center_val: Optional[float] = None):
        """
        Manually update the normalization range for a specific feature.
        
        Args:
            feature_idx: Feature index
            min_val: Minimum value
            max_val: Maximum value
            center_val: Center value (optional, defaults to current center or (min+max)/2)
        """
        if feature_idx not in self.feature_ranges:
            return
            
        _, _, current_center = self.feature_ranges[feature_idx]
        center = center_val if center_val is not None else current_center
        
        self.feature_ranges[feature_idx] = (min_val, max_val, center)
