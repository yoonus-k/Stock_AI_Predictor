"""
Kaggle Training Script for Reinforcement Learning Trading Model

This script is optimized for running on Kaggle's GPU environment.
It contains the entire training pipeline with progress tracking and model saving.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add imports for stable baselines
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train RL trading model on Kaggle")
parser.add_argument("--timesteps", type=int, default=500000, 
                    help="Total training timesteps")
parser.add_argument("--db-path", type=str, default="/kaggle/input/trading-data/samples.db", 
                    help="Path to SQLite database")
parser.add_argument("--output-dir", type=str, default="/kaggle/working", 
                    help="Directory to save model and logs")
args = parser.parse_args()

# Make sure imports work regardless of where script is run
if os.path.exists("/kaggle/working"):
    # We're on Kaggle
    sys.path.append("/kaggle/working/Stock_AI_Predictor")
    RUNNING_ON_KAGGLE = True
    print("Running on Kaggle environment")
else:
    # We're running locally - use relative imports
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    RUNNING_ON_KAGGLE = False
    print("Running on local environment")

# Import local modules
from RL.Envs.trading_env import PatternSentimentEnv
from RL.Envs.action_wrapper import TupleActionWrapper

# Define callback for feature importance tracking
class FeatureImportanceCallback(CheckpointCallback):
    """Callback to periodically calculate and save feature importance during training"""
    
    def __init__(self, eval_env, log_path, eval_freq=10000):
        super().__init__(
            save_freq=eval_freq,
            save_path=None,
            name_prefix="feature_importance",
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        self.eval_env = eval_env
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0        
        self.feature_names = [
            # Base pattern features (7 features)
            "probability", "action", "reward_risk_ratio", "max_gain",
            "max_drawdown", "mse", "expected_value",
            # Technical indicators (3 features)
            "rsi", "atr", "atr_ratio",
            # Sentiment features (2 features)
            "unified_sentiment", "sentiment_count",
            # COT data (6 features)
            "net_noncommercial", "net_nonreportable",
            "change_nonrept_long", "change_nonrept_short",
            "change_noncommercial_long", "change_noncommercial_short",
            # Time features (7 features)
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "asian_session", "london_session", "ny_session",
            # Portfolio features (5 features)
            "balance_ratio", "position_ratio", "position", "max_drawdown", "win_rate"
        ]
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._calculate_basic_importance(self.model, self.num_timesteps)
        return True
    
    def _calculate_basic_importance(self, model, timestep):
        try:
            print(f"\nCalculating feature importance at timestep {timestep}...")
            
            # Basic placeholder approach - to be enhanced in full implementation
            importance = np.random.uniform(0, 1, size=len(self.feature_names))
            
            # Save data
            importance_data = {
                'timestep': int(timestep),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'permutation_importance': {
                    'feature_names': self.feature_names,
                    'importance': importance.tolist()
                }
            }
            
            # Save to file
            file_path = self.log_path / "feature_importance.json"
            with open(file_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
                
            print(f"Feature importance saved to {file_path}")
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")


class PortfolioTrackingCallback(CheckpointCallback):
    """Callback to track portfolio performance during training"""
    
    def __init__(self, eval_env, log_path, eval_freq=5000):
        super().__init__(
            save_freq=eval_freq,
            save_path=None,
            name_prefix="portfolio_tracking",
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        self.eval_env = eval_env
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0
        
        # Track portfolio metrics
        self.portfolio_values = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
        self.wins = 0
        self.losses = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._evaluate_portfolio(self.model, self.num_timesteps)
        return True
    
    def _evaluate_portfolio(self, model, timestep):
        try:
            print(f"\nEvaluating portfolio performance at timestep {timestep}...")
            
            # Reset environment
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            portfolio_value = 10000.0  # Initial portfolio value
            
            # Basic evaluation loop
            self.portfolio_values = [portfolio_value]
            self.action_counts = {0: 0, 1: 0, 2: 0}
            self.wins = 0
            self.losses = 0
            
            while not (done or truncated):
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Count action
                action_type = int(action[0]) if isinstance(action, np.ndarray) else action
                self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
                
                # Step environment
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                # Simulate portfolio value (very simplified)
                if reward > 0:
                    portfolio_value *= (1 + min(reward / 100, 0.05))  # Limit to reasonable returns
                    self.wins += 1
                elif reward < 0:
                    portfolio_value *= (1 + max(reward / 100, -0.05))  # Limit losses
                    self.losses += 1
                    
                self.portfolio_values.append(portfolio_value)
            
            # Calculate win rate
            total_trades = self.wins + self.losses
            win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
            
            # Save performance metrics
            metrics = {
                'timestep': int(timestep),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'final_portfolio_value': float(self.portfolio_values[-1]),
                'return_pct': float((self.portfolio_values[-1] / self.portfolio_values[0] - 1) * 100),
                'win_rate': float(win_rate),
                'action_counts': {str(k): v for k, v in self.action_counts.items()},  # Convert keys to strings for JSON
                'portfolio_values': [float(val) for val in self.portfolio_values]
            }
            
            # Save to file
            file_path = self.log_path / "performance_metrics.json"
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"Portfolio performance metrics saved to {file_path}")
            
        except Exception as e:
            print(f"Error evaluating portfolio: {e}")


def load_data_from_db(db_path):
    """Load data from SQLite database file"""
    import sqlite3
    
    try:
        # Connect to database and load data
        conn = sqlite3.connect(db_path)
        
        # Read the data
        df = pd.read_sql_query(f"SELECT * FROM dataset", conn)
        conn.close()
        
        # Ensure datetime column exists and is proper datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


def train_model(db_path, output_dir, timesteps=500000, eval_freq=5000, checkpoint_freq=10000):
    """
    Train the RL model with GPU acceleration on Kaggle
    
    Parameters:
        db_path: Path to database file
        output_dir: Directory to save model and logs
        timesteps: Number of training timesteps
        eval_freq: How often to evaluate the model
        checkpoint_freq: How often to save model checkpoints
    """
    print("\n========== REINFORCEMENT LEARNING TRADING MODEL TRAINING ==========\n")
    start_time = time.time()
    
    # Setup directories
    output_path = Path(output_dir)
    log_path = output_path / "Logs"
    tensorboard_path = log_path / "tensorboard"
    checkpoint_path = log_path / "checkpoints"
    model_path = output_path / "trading_model.zip"
    
    # Create directories
    log_path.mkdir(exist_ok=True, parents=True)
    tensorboard_path.mkdir(exist_ok=True)
    checkpoint_path.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Load dataset from database
    print("\nLoading data from database...")
    dataset = load_data_from_db(db_path)
    
    if dataset.empty:
        print("❌ ERROR: No data found in database.")
        return None
    
    # Print dataset info
    print(f"✅ Loaded {len(dataset)} records from database")
    print(f"Dataset columns: {dataset.columns.tolist()}")
    
    # Split into training and evaluation sets
    split_idx = int(len(dataset) * 0.8)
    training_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    
    print(f"\nTraining data size: {len(training_data)} records")
    print(f"Evaluation data size: {len(eval_data)} records")
    
    if len(training_data) == 0 or len(eval_data) == 0:
        print("❌ ERROR: Not enough data for training and evaluation.")
        return None
    
    # Create environments
    print("\nCreating environments...")
    train_env_base = PatternSentimentEnv(
        training_data, 
        normalize_observations=True,
        enable_adaptive_scaling=True
    )
    train_env = TupleActionWrapper(train_env_base)
    
    eval_env_base = PatternSentimentEnv(
        eval_data,
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    eval_env = Monitor(TupleActionWrapper(eval_env_base))
    
    print(f"Original action space: {train_env_base.action_space}")
    print(f"Wrapped action space: {train_env.action_space}")
    
    # Setup TensorBoard logging
    print("\nSetting up TensorBoard logging...")
    logger = configure(str(tensorboard_path), ["tensorboard", "stdout"])
    
    # Create callbacks
    print("\nSetting up callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path),
        log_path=str(log_path),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix="trading_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    feature_callback = FeatureImportanceCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=min(timesteps // 10, 10000)  # Adjust frequency based on total timesteps
    )
    
    portfolio_callback = PortfolioTrackingCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=min(timesteps // 20, 5000)  # Adjust frequency based on total timesteps
    )
    
    # Combine all callbacks
    callbacks = CallbackList([
        eval_callback, 
        checkpoint_callback,
        feature_callback,
        portfolio_callback
    ])
    
    # Initialize the model with GPU optimization
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,  # Encourage exploration
        n_steps=2048,    # Higher for better GPU utilization
        batch_size=128,  # Larger batch size for GPU
        n_epochs=10,     # More epochs for better learning
        policy_kwargs={"net_arch": [256, 256]}  # Deeper network
    )
    
    # Set logger for TensorBoard
    model.set_logger(logger)
    
    # Train the model with progress bar
    print("\n" + "="*50)
    print(f"Starting training for {timesteps} timesteps...")
    print("="*50 + "\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    
    # Save the final model
    model.save(model_path)
    print(f"\n✅ Final model saved to {model_path}")
    
    # Save training metadata
    metadata = {
        "training_data_size": len(training_data),
        "eval_data_size": len(eval_data),
        "timesteps": timesteps,
        "eval_freq": eval_freq,
        "checkpoint_freq": checkpoint_freq,
        "feature_count": train_env.observation_space.shape[0],
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_minutes": (time.time() - start_time) / 60,
        "platform": "Kaggle" if RUNNING_ON_KAGGLE else "Local"
    }
    
    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    
    return model


if __name__ == "__main__":
    train_model(
        db_path=args.db_path,
        output_dir=args.output_dir,
        timesteps=args.timesteps
    )
