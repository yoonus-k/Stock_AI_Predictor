import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from RL.Data.Utils.loader import load_data_from_db
from RL.Envs.trading_env_v2 import TradingEnvV2
from RL.Envs.Components.rewards import RewardCalculator
from RL.Envs.Components.observations import ObservationHandler
from RL.Envs.Components.observation_normalizer import ObservationNormalizer
from RL.Envs.Components.trading_state import TradingState

# Import SB3 checker
from stable_baselines3.common.env_checker import check_env

def check_env_with_data():
    """
    Test that the environment can properly load and use data from the database.
    """
    print("Loading data from database...")
    data = load_data_from_db()
    
    if data.empty:
        print("❌ Failed to load data from database.")
        return False
    
    print(f"✅ Loaded {len(data)} rows from database.")
    
    # Create environment
    print("\nCreating environment...")
    env = TradingEnvV2(
        features=data,
        initial_balance=100000,
        reward_type="combined",
        normalize_observations=True,
        commission_rate=0.001,
        enable_short=True,
        max_positions=5,
        verbose=False
    )

   # Step 3: Run the SB3 environment checker
    print("\nStep 3: Running SB3 environment checker...")
    try:
        check_env(env)
        print("✅ Environment passes the SB3 environment checker!")
        #reset the environment to ensure it works
        obs, _ = env.reset()
    except Exception as e:
        print(f"❌ SB3 environment check failed: {e}")
        return

    
    print("\nStep 6: Testing full episode...")
    try:
        obs, _ =env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        max_steps = 5000  # Safety limit
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
                #print(f"  Steps: {steps}, Reward: {total_reward}, Mean step time: {np.mean(step_times):.6f}s")
                pass
            
        if steps >= max_steps and not (done or truncated):
            print("⚠️ WARNING: Episode did not terminate within max steps!")
        else:
            print(f"✅ Episode completed in {steps} steps with total reward {total_reward}")
    except Exception as e:
        print(f"❌ Error during full episode test: {e}")
        return False
    return True

if __name__ == "__main__":

    import line_profiler
    profile = line_profiler.LineProfiler()
    
    profile.add_function(TradingEnvV2.step)
    #profile.add_function(TradingEnvV2._process_trades)
    #profile.add_function(TradingEnvV2._execute_new_trade)
    #profile.add_function(TradingState.calculate_std_return)
    #profile.add_function(TradingEnvV2._calculate_reward)
    #profile.add_function(RewardCalculator.calculate_combined_reward)
    #profile.add_function(RewardCalculator.calculate_sharpe_reward)
    # profile.add_function(RewardCalculator.calculate_consistency_bonus)
    #profile.add_function(ObservationHandler.get_observation)
    #profile.add_function(ObservationNormalizer.normalize_observation)
    
    
    profile.run('check_env_with_data()')
    profile.print_stats()
    
 