"""
Meta-model architecture for aggregating predictions from timeframe-specific models
Combines daily, weekly, and monthly model predictions for more robust trading decisions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import ActorCriticPolicy

class TimeframeModelEnsemble:
    """
    Ensemble of timeframe-specific models (daily, weekly, monthly)
    Provides prediction aggregation and confidence scores
    """
    
    def __init__(
        self,
        daily_model_path: Optional[str] = None,
        weekly_model_path: Optional[str] = None,
        monthly_model_path: Optional[str] = None,
        aggregation_method: str = "weighted_average",
        device: str = "auto"
    ):
        """
        Initialize the timeframe model ensemble
        
        Args:
            daily_model_path: Path to the trained daily model
            weekly_model_path: Path to the trained weekly model
            monthly_model_path: Path to the trained monthly model
            aggregation_method: Method to aggregate predictions (weighted_average, majority_vote, etc.)
            device: Device to run models on (cpu, cuda, auto)
        """
        self.models = {}
        self.timeframes = []
        self.aggregation_method = aggregation_method
        self.device = device
        
        # Load models for each timeframe
        if daily_model_path and os.path.exists(daily_model_path):
            self.models["daily"] = self._load_model(daily_model_path)
            self.timeframes.append("daily")
        
        if weekly_model_path and os.path.exists(weekly_model_path):
            self.models["weekly"] = self._load_model(weekly_model_path)
            self.timeframes.append("weekly")
        
        if monthly_model_path and os.path.exists(monthly_model_path):
            self.models["monthly"] = self._load_model(monthly_model_path)
            self.timeframes.append("monthly")
        
        # Default weights for each timeframe
        self.weights = {
            "daily": 0.5,     # Higher weight for short-term signals
            "weekly": 0.3,    # Medium weight for medium-term signals
            "monthly": 0.2    # Lower weight for long-term signals
        }
        
        # Validate that at least one model is loaded
        if not self.models:
            raise ValueError("No valid models provided for any timeframe")
    
    def _load_model(self, model_path: str):
        """Load a timeframe model from disk"""
        try:
            model = PPO.load(model_path, device=self.device)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    
    def set_aggregation_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for timeframe aggregation
        
        Args:
            weights: Dictionary mapping timeframes to weights
        """
        # Validate weights
        for timeframe, weight in weights.items():
            if timeframe in self.weights:
                self.weights[timeframe] = weight
        
        # Normalize weights
        total = sum(self.weights[tf] for tf in self.timeframes)
        if total > 0:
            for tf in self.timeframes:
                self.weights[tf] /= total
    
    def predict_action(self, observations: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using all available timeframe models
        
        Args:
            observations: Dictionary mapping timeframes to observations
            
        Returns:
            Tuple of (aggregated_action, metadata)
        """
        # Collect predictions from each model
        predictions = {}
        confidences = {}
        
        for timeframe in self.timeframes:
            if timeframe in observations and timeframe in self.models:
                model = self.models[timeframe]
                obs = observations[timeframe]
                
                # Get model prediction
                action, states = model.predict(obs, deterministic=True)
                
                # Get action probabilities for confidence
                obs_tensor = torch.as_tensor(obs).to(model.device)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs_tensor)
                    action_probs = dist.distribution.probs
                    confidence = action_probs.max().item()
                
                predictions[timeframe] = action
                confidences[timeframe] = confidence
        
        # Aggregate predictions based on selected method
        if self.aggregation_method == "weighted_average":
            aggregated_action = self._weighted_average(predictions, confidences)
        elif self.aggregation_method == "majority_vote":
            aggregated_action = self._majority_vote(predictions)
        else:
            # Default to weighted average
            aggregated_action = self._weighted_average(predictions, confidences)
        
        # Prepare metadata
        metadata = {
            "timeframe_predictions": predictions,
            "timeframe_confidences": confidences,
            "aggregation_method": self.aggregation_method,
            "weights": {tf: self.weights[tf] for tf in self.timeframes}
        }
        
        return aggregated_action, metadata
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray], confidences: Dict[str, float]) -> np.ndarray:
        """
        Aggregate predictions using weighted average
        Weights each model's prediction by its confidence and timeframe weight
        """
        # If no predictions, return default action (no trade/hold)
        if not predictions:
            return np.zeros(1)
        
        # Combine weights and confidences
        combined_weights = {}
        for timeframe in predictions:
            combined_weights[timeframe] = self.weights[timeframe] * confidences[timeframe]
        
        # Normalize combined weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            for timeframe in combined_weights:
                combined_weights[timeframe] /= total_weight
        
        # Weighted sum of predictions
        action_shape = next(iter(predictions.values())).shape
        weighted_sum = np.zeros(action_shape)
        
        for timeframe, action in predictions.items():
            weighted_sum += action * combined_weights[timeframe]
        
        return weighted_sum
    
    def _majority_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate predictions using majority vote
        Only works well for discrete action spaces
        """
        # If no predictions, return default action (no trade/hold)
        if not predictions:
            return np.zeros(1)
        
        # For continuous actions, we need to discretize first
        # This is a simplified approach - just count positive vs negative
        votes = {
            "buy": 0,
            "hold": 0,
            "sell": 0
        }
        
        for timeframe, action in predictions.items():
            value = action[0]  # Assume action is 1-dimensional
            
            if value > 0.2:
                votes["buy"] += self.weights[timeframe]
            elif value < -0.2:
                votes["sell"] += self.weights[timeframe]
            else:
                votes["hold"] += self.weights[timeframe]
        
        # Find action with most votes
        max_vote = max(votes.values())
        
        if votes["buy"] == max_vote:
            return np.array([1.0])
        elif votes["sell"] == max_vote:
            return np.array([-1.0])
        else:
            return np.array([0.0])


class MetaModelFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for meta-model
    Processes features from multiple timeframes and model predictions
    """
    
    def __init__(self, observation_space, features_dim=128):
        super(MetaModelFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Input dimension depends on the observation space shape
        n_input_features = int(np.prod(observation_space.shape))
        
        # Neural network architecture for feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        """Extract features from observations"""
        return self.feature_net(observations)


class MetaModelPolicy(ActorCriticPolicy):
    """
    Policy network for the meta-model
    Customized to handle features from multiple timeframe models
    """
    
    def __init__(self, *args, **kwargs):
        # Add custom arguments for meta-model policy if needed
        super(MetaModelPolicy, self).__init__(*args, **kwargs)
