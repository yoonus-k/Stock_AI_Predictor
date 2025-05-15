# train_rl_agent.py
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent.parent
sys.path.append(str(script_dir))

from stable_baselines3 import PPO
from RL.Envs.trading_env import PatternSentimentEnv
from RL.Data.loader import load_data_from_db  # Assuming you have a function to load your data
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import RL.Envs.trading_env as trading_env

rl_dataset = load_data_from_db()  # You must implement this
# split your data into training and evaluation sets
# For example, you can use the first 80% for training and the rest for evaluation
split_idx = int(len(rl_dataset) * 0.8)
training_data = rl_dataset[:split_idx]
eval_data = rl_dataset[split_idx:]
print(f"Training data size: {len(training_data)}")
print(f"Evaluation data size: {len(eval_data)}")

# Wrap environment
eval_env = Monitor(PatternSentimentEnv(eval_data))  # Use separate data for evaluation
callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                        log_path='./logs/', eval_freq=5000,
                        deterministic=True, render=True)

# Load your historical data
# It should be a list of dicts: [{pattern:..., sentiment:..., price:..., actual_return:...}, ...]

env = PatternSentimentEnv(training_data)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000, callback=callback)

# Save the model
model.save("models/pattern_sentiment_rl_model")
