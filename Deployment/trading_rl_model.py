#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reinforcement Learning Model for Hugging Face Hub

This script prepares the RL trading model for deployment on Hugging Face Hub.
It provides functions to:
1. Save the RL model and configurations
2. Load the model from Hugging Face Hub
3. Run inference with the model
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom modules
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RL.Envs.trading_env import TradingEnv
from huggingface_hub import HfApi, hf_hub_download, upload_file

class TradingRLModel:
    """Reinforcement Learning Trading Model wrapper for Hugging Face Hub"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RL trading model
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.model = None
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model to a directory for uploading to HF Hub
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        
        # Save RL model if available
        if self.model is not None:
            model_path = os.path.join(save_directory, "rl_model.zip")
            self.model.save(model_path)
    
    @classmethod
    def from_pretrained(cls, model_id: str, revision: str = "main") -> "TradingRLModel":
        """
        Load the model from Hugging Face Hub
        
        Args:
            model_id: Model ID on Hugging Face Hub
            revision: Model revision
            
        Returns:
            TradingRLModel: Loaded model
        """
        # Download configuration
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", revision=revision)
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize model
        trading_model = cls(config=config)
        
        # Load RL model
        try:
            # Define a temp environment for loading
            # (Note: we'll need to make sure this matches what's needed for inference)
            dummy_env = DummyVecEnv([lambda: TradingEnv([])])
            
            # Download the model file
            model_path = hf_hub_download(repo_id=model_id, filename="rl_model.zip", revision=revision)
            
            # Load the model
            trading_model.model = PPO.load(model_path, env=dummy_env)
        except Exception as e:
            print(f"Error loading RL model: {e}")
            
        return trading_model
    
    def train(self, training_data: List[Dict[str, Any]], eval_data: List[Dict[str, Any]] = None,
              total_timesteps: int = 200000) -> None:
        """
        Train the RL model
        
        Args:
            training_data: Training data
            eval_data: Evaluation data
            total_timesteps: Total timesteps for training
        """
        # Create training environment
        env = TradingEnv(training_data)
        
        # Create evaluation environment if data is provided
        if eval_data:
            eval_env = TradingEnv(eval_data)
        else:
            eval_env = None
        
        # Initialize model
        self.model = PPO("MlpPolicy", env, verbose=1)
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make trading decision based on current state
        
        Args:
            state: Current market and portfolio state
            
        Returns:
            Dict: Trading decision with action type and position size
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded properly")
        
        # Convert state to observation format expected by the model
        observation = self._convert_state_to_observation(state)
        
        # Get action from model
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Interpret action
        action_type = int(action[0])
        position_size = float(action[1])
        
        # Map action type to trading decision
        if action_type == 0:
            decision = "HOLD"
        elif action_type == 1:
            decision = "BUY"
        else:  # action_type == 2
            decision = "SELL"
        
        return {
            "action": decision,
            "position_size": position_size,
            "confidence": self._calculate_confidence(observation)
        }
    
    def _convert_state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert state dictionary to observation array
        
        Args:
            state: State dictionary with pattern, sentiment and portfolio info
            
        Returns:
            np.ndarray: Observation array for the model
        """
        # Extract pattern features
        pattern_features = np.array(state.get("pattern_features", [0.0] * 5), dtype=np.float32)
        
        # Extract sentiment features
        sentiment_features = np.array(state.get("sentiment_features", [0.0] * 3), dtype=np.float32)
        
        # Extract portfolio features
        portfolio_features = np.array([
            state.get("balance", 0.0),
            state.get("position", 0.0),
            state.get("position_value", 0.0)
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([pattern_features, sentiment_features, portfolio_features])
        
        return observation.reshape(1, -1)  # Add batch dimension
    
    def _calculate_confidence(self, observation: np.ndarray) -> float:
        """
        Calculate confidence score for the prediction
        
        Args:
            observation: Observation array
            
        Returns:
            float: Confidence score (0-1)
        """
        # Get the raw action values (before deterministic selection)
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "get_distribution"):
            distribution = self.model.policy.get_distribution(observation)
            if hasattr(distribution, "distribution"):
                # For categorical distributions, get probabilities
                probs = distribution.distribution.probs.cpu().detach().numpy()
                confidence = float(np.max(probs))
            else:
                # For continuous distributions, use a heuristic based on std dev
                confidence = 0.8  # Default confidence
        else:
            confidence = 0.8  # Default confidence
        
        return confidence


def upload_to_hub(model: TradingRLModel, repo_id: str, token: str) -> None:
    """
    Upload the RL trading model to Hugging Face Hub
    
    Args:
        model: RL trading model
        repo_id: Repository ID on Hugging Face Hub
        token: Hugging Face API token
    """
    # Save model locally first
    save_dir = "temp_model"
    model.save_pretrained(save_dir)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create or get repository
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    
    # Upload files
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        token=token
    )
    
    print(f"Model successfully uploaded to {repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Trading Model for Hugging Face Hub")
    parser.add_argument("--upload", action="store_true", help="Upload model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, help="Repository ID on Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    if args.upload:
        if not args.repo_id or not args.token:
            raise ValueError("--repo_id and --token are required for uploading")
        
        # Create sample model
        model = TradingRLModel(config={"example": "config"})
        
        # Upload to hub
        upload_to_hub(model, args.repo_id, args.token)
