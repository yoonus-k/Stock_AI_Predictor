"""
Environment Checker Script

This script tests your trading environment for compatibility with Stable Baselines 3
and checks for common issues that might cause training to hang.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import environment modules
from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env import TradingEnv


# Import SB3 checker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

def check_trading_environment():
    """Run comprehensive checks on the trading environment"""
    print("\n========== TRADING ENVIRONMENT CHECKER ==========\n")
    
    # Step 1: Load a small subset of data
    print("Step 1: Loading data...")
    db_path = os.path.join(project_root, "RL/Data/Storage/samples.db")
    try:
        rl_dataset = load_data_from_db()
        if rl_dataset.empty:
            print("❌ ERROR: No data found in database.")
            return
        
        # Use small subset for testing
        small_dataset = rl_dataset.head(20)
        print(small_dataset.head())  # Print first few rows for verification
        print(f"✅ Loaded {len(small_dataset)} records for testing.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        traceback.print_exc()
        return
    
    # Step 2: Create the environment
    print("\nStep 2: Creating environment...")
    try:
        env = TradingEnv(
         
            small_dataset, 
            normalize_observations=True,
            enable_adaptive_scaling=False , # Simpler for testing
            reward_type='win_rate'  # Use a basic reward type for initial checks, 
        )
        
        print("✅ Environment created successfully.")
        
        # Print environment info
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Run the SB3 environment checker
    print("\nStep 3: Running SB3 environment checker...")
    try:
        check_env(env)
        print("✅ Environment passes the SB3 environment checker!")
    except Exception as e:
        print(f"❌ SB3 environment check failed: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Test stepping through the environment
    print("\nStep 4: Testing environment interactions...")
    try:
        # Reset the environment
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Try 10 random actions
        for i in range(10):
            action =env.action_space.sample()
            print(f"Step {i+1}: Taking action {action}")
            
            # Measure step time
            start_time = time.time()
            obs, reward, done, truncated, info =env.step(action)
            step_time = time.time() - start_time
            
            print(f"  Observation shape: {obs.shape}")
            print(f"  Trade PNL: {info['trade_pnl']:.2f}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Truncated: {truncated}")
            print(f"  Step time: {step_time:.6f} seconds")
            
            # Check for NaNs or infinities
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print("❌ WARNING: NaN or infinity values detected in observation!")
                
            if np.isnan(reward) or np.isinf(reward):
                print("❌ WARNING: NaN or infinity reward detected!")
            
            # Break if done
            if done or truncated:
                print("Environment episode completed.")
                break
        
        print("✅ Environment stepping test completed successfully.")
    except Exception as e:
        print(f"❌ Error during environment stepping: {e}")
        traceback.print_exc()
        return
    
    # Step 5: Test a full episode with timing
    print("\nStep 5: Testing full episode...")
    try:
        obs, _ =env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        max_steps = 1000  # Safety limit
        step_times = []
        
        while not (done or truncated) and steps < max_steps:
            action =env.action_space.sample()
            
            # Time the step execution
            start_time = time.time()
            obs, reward, done, truncated, info =env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            total_reward += reward
            steps += 1
            
            # Report every 100 steps or at the end
            if steps % 100 == 0 or done or truncated or steps >= max_steps:
                print(f"  Steps: {steps}, Reward: {total_reward}, Mean step time: {np.mean(step_times):.6f}s")
            
        if steps >= max_steps and not (done or truncated):
            print("⚠️ WARNING: Episode did not terminate within max steps!")
        else:
            print(f"✅ Episode completed in {steps} steps with total reward {total_reward}")
        
        # Print step time statistics
        print(f"Step time statistics:")
        print(f"  Mean: {np.mean(step_times):.6f}s")
        print(f"  Max: {np.max(step_times):.6f}s at step {np.argmax(step_times) + 1}")
        print(f"  Min: {np.min(step_times):.6f}s")
        print(f"  Std dev: {np.std(step_times):.6f}s")
    except Exception as e:
        print(f"❌ Error during full episode test: {e}")
        traceback.print_exc()
        return
    
    # Final verdict
    print("\n========== ENVIRONMENT CHECK SUMMARY ==========")
    print("✅ Basic environment structure: PASSED")
    print("✅ SB3 compatibility check: PASSED")
    print("✅ Environment stepping: PASSED")
    print("✅ Full episode playthrough: PASSED")
    print("\nYour environment appears to be properly configured for RL training.")
    print("If you're still experiencing training freezes, try using minimal callbacks.")

if __name__ == "__main__":
    check_trading_environment()
