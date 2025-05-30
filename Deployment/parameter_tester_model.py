#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Tester Model for Hugging Face Hub

This script prepares the parameter tester model for deployment on Hugging Face Hub.
It provides functions to:
1. Save the parameter tester model and configurations
2. Load the model from Hugging Face Hub
3. Run inference with the model
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom modules
from Experements.ParamTesting.parameter_tester import PARAM_RANGES
from Pattern.pip_pattern_miner import Pattern_Miner
from huggingface_hub import HfApi, hf_hub_download, upload_file

class ParameterTesterModel:
    """Parameter Tester Model wrapper for Hugging Face Hub"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parameter tester model
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.pattern_miner = None
        self.best_params = None
        
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
        
        # Save pattern miner if available
        if self.pattern_miner is not None:
            model_path = os.path.join(save_directory, "pattern_miner.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.pattern_miner, f)
        
        # Save best parameters if available
        if self.best_params is not None:
            params_path = os.path.join(save_directory, "best_params.json")
            with open(params_path, "w") as f:
                json.dump(self.best_params, f)
    
    @classmethod
    def from_pretrained(cls, model_id: str, revision: str = "main") -> "ParameterTesterModel":
        """
        Load the model from Hugging Face Hub
        
        Args:
            model_id: Model ID on Hugging Face Hub
            revision: Model revision
            
        Returns:
            ParameterTesterModel: Loaded model
        """
        # Download configuration
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", revision=revision)
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(config=config)
        
        # Load pattern miner if available
        try:
            model_path = hf_hub_download(repo_id=model_id, filename="pattern_miner.pkl", revision=revision)
            with open(model_path, "rb") as f:
                model.pattern_miner = pickle.load(f)
        except:
            print("Pattern miner not available in the model")
        
        # Load best parameters if available
        try:
            params_path = hf_hub_download(repo_id=model_id, filename="best_params.json", revision=revision)
            with open(params_path, "r") as f:
                model.best_params = json.load(f)
        except:
            print("Best parameters not available in the model")
            
        return model
    
    def train(self, data: pd.DataFrame, stock_id: int, timeframe_id: int) -> Dict[str, Any]:
        """
        Train the parameter tester model
        
        Args:
            data: Price data for training
            stock_id: ID of the stock
            timeframe_id: ID of the timeframe
            
        Returns:
            Dict: Best parameters and performance metrics
        """
        # Create a pattern miner
        self.pattern_miner = Pattern_Miner()
        
        # Initialize parameter search
        param_grid = {
            'n_pips': PARAM_RANGES['n_pips'],
            'lookback': PARAM_RANGES['lookback'],
            'dist_measure': PARAM_RANGES['dist_measure'],
            'min_pattern': PARAM_RANGES['min_pattern']
        }
        
        best_metrics = {}
        best_params = {}
        
        # Grid search through parameters
        for n_pips in param_grid['n_pips']:
            for lookback in param_grid['lookback']:
                for dist_measure in param_grid['dist_measure']:
                    for min_pattern in param_grid['min_pattern']:
                        # Set parameters
                        self.pattern_miner.n_pips = n_pips
                        self.pattern_miner.lookback = lookback
                        self.pattern_miner.dist_measure = dist_measure
                        self.pattern_miner.min_pattern = min_pattern
                        
                        # Extract patterns
                        patterns = self.pattern_miner.extract_patterns(data)
                        
                        # Evaluate patterns
                        metrics = self.pattern_miner.evaluate_patterns(patterns)
                        
                        # Update best parameters if better
                        if not best_metrics or metrics['profit_factor'] > best_metrics['profit_factor']:
                            best_metrics = metrics
                            best_params = {
                                'n_pips': n_pips,
                                'lookback': lookback,
                                'dist_measure': dist_measure,
                                'min_pattern': min_pattern
                            }
        
        # Set best parameters
        self.best_params = best_params
        
        # Return best parameters and metrics
        return {
            'params': best_params,
            'metrics': best_metrics
        }
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using the best parameters
        
        Args:
            data: Price data for prediction
            
        Returns:
            Dict: Prediction results with pattern IDs and future price expectations
        """
        if self.pattern_miner is None or self.best_params is None:
            raise ValueError("Model not trained or loaded properly")
        
        # Set parameters
        self.pattern_miner.n_pips = self.best_params['n_pips']
        self.pattern_miner.lookback = self.best_params['lookback']
        self.pattern_miner.dist_measure = self.best_params['dist_measure']
        self.pattern_miner.min_pattern = self.best_params['min_pattern']
        
        # Extract current pattern
        current_pattern = self.pattern_miner.extract_current_pattern(data)
        
        # Find similar patterns
        similar_patterns = self.pattern_miner.find_similar_patterns(current_pattern)
        
        # Get prediction
        prediction = self.pattern_miner.get_prediction(similar_patterns)
        
        return prediction


def upload_to_hub(model: ParameterTesterModel, repo_id: str, token: str) -> None:
    """
    Upload the parameter tester model to Hugging Face Hub
    
    Args:
        model: Parameter tester model
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
    
    parser = argparse.ArgumentParser(description="Parameter Tester Model for Hugging Face Hub")
    parser.add_argument("--upload", action="store_true", help="Upload model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, help="Repository ID on Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    if args.upload:
        if not args.repo_id or not args.token:
            raise ValueError("--repo_id and --token are required for uploading")
        
        # Create sample model
        model = ParameterTesterModel(config={"example": "config"})
        
        # Upload to hub
        upload_to_hub(model, args.repo_id, args.token)
