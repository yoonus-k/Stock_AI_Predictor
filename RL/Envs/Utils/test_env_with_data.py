import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env import TradingEnv

def test_env_with_database():
    """
    Test that the environment can properly load and use data from the database.
    """
    print("Loading data from database...")
    data = load_data_from_db()
    
    if data.empty:
        print("❌ Failed to load data from database.")
        return False
    
    print(f"✅ Loaded {len(data)} rows from database.")
    
    # Print the first row to see the structure
    print("\nFirst row of data:")
    print(data.head(1).to_string(index=False))
    
    # Create environment
    print("\nCreating environment...")
    env = TradingEnv(data)
    
    # Test reset
    print("\nTesting environment reset...")
    obs, info = env.reset()
    print(f"✅ Environment reset successful. Observation shape: {obs.shape}")
    
    # Test step with a sample action
    print("\nTesting environment step...")
    action = env.action_space.sample()
    print(f"Action: {action}")
    
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful.")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Test a few more steps
    print("\nRunning 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward = {reward}, Portfolio Balance = {info['portfolio_balance']:.2f}")
        print(f"Observation: {obs}")  # Print first 5 features of observation
        
        if done:
            print("Episode finished early.")
            break
    
    return True

if __name__ == "__main__":
    success = test_env_with_database()
    if success:
        print("\n✅ Environment test successful!")
    else:
        print("\n❌ Environment test failed.")
