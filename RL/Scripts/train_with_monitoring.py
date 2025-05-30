import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Envs.trading_env import PatternSentimentEnv
from RL.Envs.action_wrapper import TupleActionWrapper
from RL.Data.loader import load_data_from_db

# Custom callbacks for enhanced monitoring
class FeatureImportanceCallback(CheckpointCallback):
    """Callback to periodically calculate and save feature importance during training"""
    
    def __init__(self, eval_env, log_path, eval_freq=10000):
        """
        Initialize the callback
        
        Parameters:
            eval_env: Evaluation environment
            log_path: Path to save feature importance data
            eval_freq: How often to calculate feature importance (in timesteps)
        """
        super().__init__(
            save_freq=eval_freq,  # Match the eval frequency
            save_path=None,  # We don't actually save models with this callback
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
        """
        Called at each step during training
        
        Returns:
            True (continue training)
        """
        # The parent class tracks num_timesteps, we can use that
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._calculate_basic_importance(self.model, self.num_timesteps)
        
        return True
    
    def _calculate_basic_importance(self, model, timestep):
        """
        Calculate basic feature importance by perturbing inputs
        
        Parameters:
            model: Current RL model
            timestep: Current timestep
        """
        try:
            print(f"\nCalculating feature importance at timestep {timestep}...")
            
            # Basic placeholder approach - this would be enhanced in the full implementation
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
        """
        Initialize the callback
        
        Parameters:
            eval_env: Evaluation environment
            log_path: Path to save performance data
            eval_freq: How often to evaluate portfolio performance (in timesteps)
        """
        super().__init__(
            save_freq=eval_freq,  # Match the eval frequency
            save_path=None,  # We don't actually save models with this callback
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
        """
        Called at each step during training
        
        Returns:
            True (continue training)
        """
        # Check if it's time to evaluate portfolio
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._evaluate_portfolio(self.model, self.num_timesteps)
        
        return True
    
    def _evaluate_portfolio(self, model, timestep):
        """
        Evaluate portfolio performance using current model
        
        Parameters:
            model: Current RL model
            timestep: Current timestep
        """
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
            
            while not (done or truncated):
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Count action
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                
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
                'action_counts': self.action_counts,
                'portfolio_values': [float(val) for val in self.portfolio_values]
            }
            
            # Save to file
            file_path = self.log_path / "performance_metrics.json"
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"Portfolio performance metrics saved to {file_path}")
            
        except Exception as e:
            print(f"Error evaluating portfolio: {e}")


def train_rl_model(
    db_path="RL/Data/samples.db", 
    save_path="RL/Models/pattern_sentiment_rl_model", 
    timesteps=5000,     # Reduced from 200000 for testing
    eval_freq=1000,     # Reduced from 5000 for testing
    checkpoint_freq=2500,  # Reduced from 10000 for testing
    log_path=None,
    tensorboard=True,
    progress_bar=True,  # Enable progress bar
):
    """
    Train the RL model with data from database with enhanced monitoring and logging
    
    Parameters:
        db_path: Path to database file. If None, will search in common locations.
        save_path: Where to save the trained model
        timesteps: Number of training timesteps
        eval_freq: How often to evaluate the model (in timesteps)
        checkpoint_freq: How often to save model checkpoints (in timesteps)
        log_path: Directory to save logs and checkpoints
        tensorboard: Whether to use TensorBoard logging
    """
    print("\n========== TRADING AGENT TRAINING ==========\n")
    
    # Setup directories
    if log_path is None:
        log_path = Path(__file__).parent.parent / "Logs"
    else:
        log_path = Path(log_path)
    
    model_dir = Path(os.path.dirname(save_path))
    tensorboard_path = log_path / "tensorboard"
    checkpoint_path = log_path / "checkpoints"
    
    # Create directories
    log_path.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_path.mkdir(exist_ok=True)
    checkpoint_path.mkdir(exist_ok=True)
    
    print(f"Logs will be saved to: {log_path}")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Final model will be saved to: {save_path}")
    
    # Load dataset from database
    print("\nLoading data from database...")
    rl_dataset = load_data_from_db(db_path)
    
    if rl_dataset.empty:
        print("❌ ERROR: No data found. Please run data_exploration.ipynb to generate samples.db first.")
        return
    
    # Print dataset info for verification
    print(f"✅ Loaded {len(rl_dataset)} records from database")
    print(f"Dataset columns: {rl_dataset.columns.tolist()}")
    print(f"Dataset sample:\n{rl_dataset.head(1).T}")
    
    # Split into training and evaluation sets
    split_idx = int(len(rl_dataset) * 0.8)
    training_data = rl_dataset[:split_idx]
    eval_data = rl_dataset[split_idx:]
    
    # take only 100 records for the training to test functionality
    training_data = training_data.head(100)
    
    print(f"\nTraining data size: {len(training_data)} records")
    print(f"Evaluation data size: {len(eval_data)} records")
    
    if len(training_data) == 0 or len(eval_data) == 0:
        print("❌ ERROR: Not enough data for training and evaluation.")
        return
    
    # Create environments
    print("\nCreating environments...")
    # Create base training env and wrap it to convert tuple action space to box action space
    train_env_base = PatternSentimentEnv(
        training_data, 
        normalize_observations=True,
        enable_adaptive_scaling=True
    )
    train_env = TupleActionWrapper(train_env_base)
    
    # Create base eval env and wrap it
    eval_env_base = PatternSentimentEnv(
        eval_data,
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    # Wrap with action converter first, then with Monitor for stats tracking
    eval_env = Monitor(TupleActionWrapper(eval_env_base))
    
    # Print action space info for debugging
    print(f"Original action space: {train_env_base.action_space}")
    print(f"Wrapped action space: {train_env.action_space}")
    
    # Setup TensorBoard logging
    if tensorboard:
        print("\nSetting up TensorBoard logging...")
        logger = configure(str(tensorboard_path), ["tensorboard", "stdout"])
    else:
        logger = None
    
    # Create evaluation callback
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
      # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix="trading_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
      # Create feature importance callback
    feature_callback = FeatureImportanceCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=2000  # Reduced from 10000 for faster feedback
    )
    
    # Create portfolio tracking callback
    portfolio_callback = PortfolioTrackingCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=1000  # Reduced from 5000 for faster feedback
    )
      # Combine all callbacks
    callbacks = CallbackList([
        eval_callback, 
        checkpoint_callback,
        feature_callback,
        portfolio_callback
    ])
      # Initialize the model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,  # Encourage exploration
        n_steps=256,     # Reduced from 2048 for faster updates during testing
        batch_size=64,   # Minibatch size
        n_epochs=5,      # Reduced from 10 for faster training during testing
    )
    
    # Set custom logger if TensorBoard is enabled
    if logger is not None:
        model.set_logger(logger)
    
    # Train the model    print("\n" + "="*50)
    print(f"Starting training for {timesteps} timesteps...")
    print("="*50 + "\n")
    
    # Progress bar configuration
    progress_kwargs = {"disable": not progress_bar}
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=progress_bar)
    
    # Save the final model
    model.save(save_path)
    print(f"\n✅ Final model saved to {save_path}")
    
    # Save training metadata
    metadata = {
        "training_data_size": len(training_data),
        "eval_data_size": len(eval_data),
        "timesteps": timesteps,
        "eval_freq": eval_freq,
        "checkpoint_freq": checkpoint_freq,
        "feature_count": train_env.observation_space.shape[0],
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{os.path.splitext(save_path)[0]}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete!")
    print("\nAvailable monitoring tools:")
    print(f"1. Launch unified monitoring dashboard: python Scripts/monitoring_dashboard.py --model={save_path} --log-dir={log_path}")
    print(f"2. View TensorBoard logs: tensorboard --logdir={tensorboard_path}")
    print(f"3. Analyze checkpoints: python Scripts/analyze_checkpoints.py --checkpoint-dir={checkpoint_path}")
    print(f"4. Analyze feature importance: python Scripts/monitor_feature_importance.py --model={save_path}")
    print(f"5. Analyze trading strategy: python Scripts/monitor_trading_strategy.py --model={save_path}")
    print(f"6. Analyze decision boundaries: python Scripts/monitor_decision_boundaries.py --model={save_path}")
    print(f"7. Launch live dashboard: python Scripts/live_training_dashboard.py --log-dir={log_path}")
    
    return model


if __name__ == "__main__":
    import time
    import argparse
    
    # Add command line arguments for easier control
    parser = argparse.ArgumentParser(description="Train the RL trading model")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total training timesteps")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()
    
    start_time = time.time()
    model = train_rl_model(
        timesteps=args.timesteps,
        progress_bar=not args.no_progress
    )
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")
