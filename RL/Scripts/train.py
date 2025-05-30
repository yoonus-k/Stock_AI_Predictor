import sys
from pathlib import Path

from stable_baselines3 import PPO
from RL.Envs.trading_env import PatternSentimentEnv
from RL.Data.loader import load_data_from_db, adapt_to_trading_env  # Use updated loader
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

def train_rl_model(db_path=None, save_path="models/pattern_sentiment_rl_model", 
                   timesteps=200000, eval_freq=5000):
    """
    Train the RL model with data from samples.db
    
    Parameters:
        db_path: Path to samples.db. If None, will search in common locations.
        save_path: Where to save the trained model
        timesteps: Number of training timesteps
        eval_freq: How often to evaluate the model
    """
    # Make sure the models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load dataset from database
    print("Loading data from database...")
    rl_dataset = load_data_from_db()
    
    if rl_dataset.empty:
        print("ERROR: No data found. Please run data_exploration.ipynb to generate samples.db first.")
        return
        
    # Split into training and evaluation sets
    split_idx = int(len(rl_dataset) * 0.8)
    training_data = rl_dataset[:split_idx]
    eval_data = rl_dataset[split_idx:]
    
    print(f"Training data size: {len(training_data)}")
    print(f"Evaluation data size: {len(eval_data)}")
    
    if len(training_data) == 0 or len(eval_data) == 0:
        print("ERROR: Not enough data for training and evaluation.")
        return
    
    # Create environments
    print("Creating environments...")
    env = PatternSentimentEnv(training_data)
    eval_env = Monitor(PatternSentimentEnv(eval_data))
    
    # Create callback for evaluation
    log_path = Path(__file__).parent.parent / "Logs"
    log_path.mkdir(exist_ok=True)
    callback = EvalCallback(eval_env, best_model_save_path=str(log_path),
                            log_path=str(log_path), eval_freq=eval_freq,
                            deterministic=True, render=False)
    
    # Initialize and train model
    print("Training model...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    train_rl_model()
