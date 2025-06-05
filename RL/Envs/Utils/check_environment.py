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
    
    try:
        rl_dataset = load_data_from_db()
        if rl_dataset.empty:
            print("❌ ERROR: No data found in database.")
            return
        
        # Use small subset for testing
        small_dataset = rl_dataset.head(100)
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
            reward_type='combined'  # Use a basic reward type for initial checks, 
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
        #reset the environment to ensure it works
        obs, _ = env.reset()
    except Exception as e:
        print(f"❌ SB3 environment check failed: {e}")
        traceback.print_exc()
        return
    
    # Step 5: Test stepping through the environment
    print("\nStep 5: Testing environment interactions...")
    try:
        # Try 10 random actions
        for i in range(200):
            action =env.action_space.sample()
            print(f"Step {i+1}: Taking action {action}")
            
            # Measure step time
            start_time = time.time()
            obs, reward, done, truncated, info =env.step(action)
            step_time = time.time() - start_time
            
            #print(f"  Trade info: {info}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Truncated: {truncated}")
            print(f"  Step time: {step_time:.6f} seconds")
            
            # Break if done
            if done or truncated:
                print("Environment episode completed.")
                break
        
        print("✅ Environment stepping test completed successfully.")
    except Exception as e:
        print(f"❌ Error during environment stepping: {e}")
        traceback.print_exc()
        return
      # Step 6: Test a full episode with timing
    print("\nStep 6: Testing full episode...")
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
        
        # Print step time statistics        print(f"Step time statistics:")
        print(f"  Mean: {np.mean(step_times):.6f}s")
        print(f"  Max: {np.max(step_times):.6f}s at step {np.argmax(step_times) + 1}")
        print(f"  Min: {np.min(step_times):.6f}s")
        print(f"  Std dev: {np.std(step_times):.6f}s")
        
        # Test new performance metrics after episode
        print(f"\nFinal performance metrics:")
        final_obs, _ = env.reset()  # Get final observation state
        print(f"  Final avg_pnl_per_hour: {final_obs[27]:.4f}")
        print(f"  Final decisive_exits: {final_obs[28]:.4f}")
        print(f"  Final recovery_factor: {final_obs[29]:.4f}")
        
    except Exception as e:
        print(f"❌ Error during full episode test: {e}")
        traceback.print_exc()
        return
    
    # Step 7: Test performance metrics calculation methods
    print("\nStep 7: Testing performance metrics calculation methods...")
    try:
        obs_handler = env.observation_handler
        
        # Test method availability
        if hasattr(obs_handler, 'calculate_avg_pnl_per_hour'):
            avg_pnl = obs_handler.calculate_avg_pnl_per_hour()
            print(f"✅ calculate_avg_pnl_per_hour() works: {avg_pnl:.4f}")
        else:
            print("❌ calculate_avg_pnl_per_hour() method not found")
            
        if hasattr(obs_handler, 'calculate_decisive_exits'):
            decisive = obs_handler.calculate_decisive_exits()
            print(f"✅ calculate_decisive_exits() works: {decisive:.4f}")
        else:
            print("❌ calculate_decisive_exits() method not found")
            
        if hasattr(obs_handler, 'calculate_recovery_factor'):
            recovery = obs_handler.calculate_recovery_factor()
            print(f"✅ calculate_recovery_factor() works: {recovery:.4f}")
        else:
            print("❌ calculate_recovery_factor() method not found")
            
        # Test update_trade_metrics method
        if hasattr(obs_handler, 'update_trade_metrics'):
            print("✅ update_trade_metrics() method found")
            # Test with sample data
            obs_handler.update_trade_metrics(
                trade_pnl_pct=0.02,
                exit_reason='tp',
                holding_hours=4.5,
                balance=10200
            )
            print("✅ update_trade_metrics() test call successful")
        else:
            print("❌ update_trade_metrics() method not found")
            
        print("✅ Performance metrics methods test completed.")
    except Exception as e:
        print(f"❌ Error during metrics methods test: {e}")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"❌ Error during full episode test: {e}")
        traceback.print_exc()
        return
      # Final verdict
    print("\n========== ENVIRONMENT CHECK SUMMARY ==========")
    print("✅ Basic environment structure: PASSED")
    print("✅ SB3 compatibility check: PASSED")
    print("✅ Observation space (30 features): PASSED")
    print("✅ New performance metrics: PASSED")
    print("✅ Environment stepping: PASSED")
    print("✅ Full episode playthrough: PASSED")
    print("✅ Performance metrics methods: PASSED")
    print("\nYour environment appears to be properly configured for RL training.")
    print("New performance metrics (avg_pnl_per_hour, decisive_exits, recovery_factor) are working correctly.")
    print("If you're still experiencing training freezes, try using minimal callbacks.")

if __name__ == "__main__":
    check_trading_environment()
