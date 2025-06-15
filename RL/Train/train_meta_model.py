"""
Meta-model training script
Combines predictions from timeframe-specific models (daily, weekly, monthly)
into a unified prediction model with MLflow tracking
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Stable-Baselines imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Custom imports
from RL.Envs.meta_environment import MetaTradingEnv
from RL.Utils.meta_model import TimeframeModelEnsemble
from RL.Mlflow.mlflow_manager import MLflowManager
from RL.Mlflow.mlflow_callback import MLflowLoggingCallback
from RL.Envs.Components.callbacks import FeatureImportanceCallback, PortfolioTrackingCallback
from RL.Data.Utils.loader import load_data_from_db


def train_meta_model(
    daily_model_path: Optional[str] = None,
    weekly_model_path: Optional[str] = None,
    monthly_model_path: Optional[str] = None,
    data = None,
    config: Dict[str, Any] = None,
    experiment_name: str = "stock_trading_rl",
    run_name: Optional[str] = None,
    model_save_path: Optional[str] = None,
):
    """
    Train meta-model that combines predictions from multiple timeframe models
    
    Args:
        daily_model_path: Path to trained daily model
        weekly_model_path: Path to trained weekly model
        monthly_model_path: Path to trained monthly model
        data: Optional data to use (if None, will load from DB)
        config: Training configuration dict
        experiment_name: MLflow experiment name
        run_name: MLflow run name (generated if None)
        model_save_path: Path to save the final model
        
    Returns:
        Tuple of (trained model, run_id)
    """
    # Default configuration
    default_config = {
        "timesteps": 100000,
        "eval_freq": 5000,
        "learning_rate": 0.0001,  # Lower learning rate for meta-model
        "batch_size": 128,
        "n_steps": 2048,
        "n_epochs": 10,
        "ent_coef": 0.005,  # Lower entropy coefficient for more exploitation
        "include_model_predictions": True,
        "normalize_observations": True,
        "reward_type": "combined"
    }
    
    # Override defaults with provided config
    if config is None:
        config = {}
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Define save paths
    if model_save_path is None:
        model_save_path = str(project_root / "RL" / "Models" / "meta" / "meta_model.zip")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Set up logs directory
    logs_dir = project_root / "RL" / "Logs" / "meta"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize MLflow tracking
    mlflow_manager = MLflowManager(
        experiment_name=experiment_name,
        timeframe="meta",
        model_type="meta_aggregation"
    )
    
    # Start MLflow run
    run_tags = {
        "model_type": "meta",
        "timeframes_included": ",".join([
            tf for tf, path in {
                "daily": daily_model_path,
                "weekly": weekly_model_path,
                "monthly": monthly_model_path
            }.items() if path is not None
        ])
    }
    
    run = mlflow_manager.start_run(run_name=run_name, tags=run_tags)
    
    # Log configuration parameters
    mlflow_manager.log_params(config)
    mlflow_manager.log_params({
        "daily_model": daily_model_path or "none",
        "weekly_model": weekly_model_path or "none", 
        "monthly_model": monthly_model_path or "none",
        "model_save_path": model_save_path
    })
    
    try:
        # Load or use provided data
        print("Loading or preparing data...")
        
        if data is None:
            # Load data from database
            data = load_data_from_db(timeframe="meta")  # Load data suitable for meta-model training
        
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("No data available for meta-model training")
        
        # Log dataset info
        mlflow_manager.log_dataset_info(data, "full_dataset")
        
        # Split data for training/evaluation
        print("Splitting data for training and evaluation...")
        split_idx = int(len(data) * 0.8)
        training_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        mlflow_manager.log_metrics({
            "training_samples": len(training_data),
            "evaluation_samples": len(eval_data)
        })
        
        print("Creating meta-model training environment...")
        # Create training environment
        train_env = MetaTradingEnv(
            training_data,
            daily_model_path=daily_model_path,
            weekly_model_path=weekly_model_path,
            monthly_model_path=monthly_model_path,
            include_model_predictions=config["include_model_predictions"],
            normalize_observations=config["normalize_observations"],
            reward_type=config["reward_type"]
        )
        
        # Create evaluation environment
        eval_env_base = MetaTradingEnv(
            eval_data,
            daily_model_path=daily_model_path,
            weekly_model_path=weekly_model_path,
            monthly_model_path=monthly_model_path,
            include_model_predictions=config["include_model_predictions"],
            normalize_observations=config["normalize_observations"],
            reward_type=config["reward_type"]
        )
        eval_env = Monitor(eval_env_base)
        
        # Initialize model
        print("Initializing meta-model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            ent_coef=config["ent_coef"],
            verbose=1
        )
        
        # Setup callbacks
        print("Setting up callbacks...")
        mlflow_callback = MLflowLoggingCallback(
            mlflow_manager=mlflow_manager,
            eval_env=eval_env,
            eval_freq=config["eval_freq"],
            log_freq=100,
            verbose=1,
            timeframe="meta"
        )
        
        portfolio_callback = PortfolioTrackingCallback(
            eval_env=eval_env,
            log_path=logs_dir,
            eval_freq=config["eval_freq"],
            n_eval_episodes=1,
            verbose=1
        )
        
        # Create callback list
        callbacks = CallbackList([mlflow_callback, portfolio_callback])
        
        # Train the model
        print(f"Training meta-model for {config['timesteps']} timesteps...")
        model.learn(
            total_timesteps=config["timesteps"],
            callback=callbacks,
            progress_bar=True
        )
        
        # Save the final model
        print(f"Saving meta-model to {model_save_path}...")
        model.save(model_save_path)
        
        # Log final model to MLflow
        mlflow_manager.log_sb3_model(model, "meta_model_final")
        
        # Run final evaluation
        print("Running final evaluation...")
        mean_reward = evaluate_model(model, eval_env, n_episodes=5)
        
        mlflow_manager.log_metrics({
            "final_eval_mean_reward": mean_reward
        })
        
        # Finalize the run
        run_id = run.info.run_id
        mlflow_manager.end_run()
        
        print(f"Training completed successfully! Run ID: {run_id}")
        return model, run_id
        
    except Exception as e:
        print(f"Error in meta-model training: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up MLflow run
        if mlflow_manager.current_run is not None:
            mlflow_manager.end_run(status="FAILED")
        
        return None, None


def evaluate_model(model, env, n_episodes=5):
    """
    Evaluate model performance
    
    Args:
        model: Trained model to evaluate
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Mean reward across episodes
    """
    all_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        all_rewards.append(episode_reward)
    
    mean_reward = np.mean(all_rewards)
    print(f"Evaluation: Mean reward over {n_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train meta-model for Stock AI Predictor")
    
    parser.add_argument("--daily-model", type=str, default=None, 
                        help="Path to trained daily model")
    parser.add_argument("--weekly-model", type=str, default=None, 
                        help="Path to trained weekly model")
    parser.add_argument("--monthly-model", type=str, default=None, 
                        help="Path to trained monthly model")
    parser.add_argument("--timesteps", type=int, default=100000, 
                        help="Number of training timesteps")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save the trained model")
    parser.add_argument("--experiment", type=str, default="stock_trading_rl", 
                        help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Configure with command line arguments
    config = {
        "timesteps": args.timesteps,
    }
    
    daily_model = args.daily_model
    weekly_model = args.weekly_model
    monthly_model = args.monthly_model
    
    if not any([daily_model, weekly_model, monthly_model]):
        print("Error: At least one timeframe model must be provided")
        sys.exit(1)
    
    # Start training
    train_meta_model(
        daily_model_path=daily_model,
        weekly_model_path=weekly_model,
        monthly_model_path=monthly_model,
        config=config,
        experiment_name=args.experiment,
        model_save_path=args.output
    )
