"""
Quick Training Script for RL Trading Model

This script provides a lightweight version of the training process
for quick debugging and experimentation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Envs.trading_env import PatternSentimentEnv
from RL.Envs.action_wrapper import TupleActionWrapper
from RL.Data.loader import load_data_from_db

def quick_train(db_path="RL/Data/samples.db", save_path="RL/Models/quick_test_model", timesteps=1000):
    """
    Quickly train a model with minimal callbacks and parameters for testing
    
    Parameters:
        db_path: Path to database file
        save_path: Where to save the trained model
        timesteps: Number of training timesteps
    """
    print("\n========== QUICK TRAINING FOR TESTING ==========\n")
    
    # Create log directory
    log_path = Path(__file__).parent.parent / "Logs"
    log_path.mkdir(exist_ok=True)
    
    # Load dataset from database
    print("Loading data...")
    rl_dataset = load_data_from_db(db_path)
    
    if rl_dataset.empty:
        print("❌ ERROR: No data found.")
        return
    
    # Split into training and evaluation sets
    split_idx = int(len(rl_dataset) * 0.8)
    training_data = rl_dataset[:split_idx]
    eval_data = rl_dataset[split_idx:]
    
    # Limit data size for quick testing
    training_data = training_data.head(50)
    eval_data = eval_data.head(20)
    
    print(f"Training data: {len(training_data)} records")
    print(f"Evaluation data: {len(eval_data)} records")
    
    # Create environments
    train_env_base = PatternSentimentEnv(training_data, normalize_observations=True)
    train_env = TupleActionWrapper(train_env_base)
    
    eval_env_base = PatternSentimentEnv(eval_data, normalize_observations=True)
    eval_env = Monitor(TupleActionWrapper(eval_env_base))
    
    print(f"Wrapped action space: {train_env.action_space}")
    
    # Create simple evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path),
        log_path=str(log_path),
        eval_freq=max(100, timesteps // 10),
        deterministic=True,
        n_eval_episodes=3,
        verbose=1
    )
    
    # Initialize the model with minimal parameters
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,      # Small number of steps per update
        batch_size=64,
        n_epochs=3,       # Few epochs per update
    )
      # Train the model
    print(f"\nStarting quick training for {timesteps} timesteps...")
    # Always enable progress bar for this quick training script
    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    
    # Save the model
    model.save(save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Quick train RL model for testing")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total training timesteps")
    args = parser.parse_args()
    
    start_time = time.time()
    quick_train(timesteps=args.timesteps)
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time):.2f} seconds")
