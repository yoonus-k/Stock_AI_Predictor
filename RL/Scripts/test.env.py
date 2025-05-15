import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent.parent
sys.path.append(str(script_dir))

from RL.Envs.trading_env import PatternSentimentEnv
import numpy as np
from RL.Data.loader import load_data_from_db 
def test_environment(env):
    # Test 1: Long position that hits TP
    print("\n=== TEST 1: Long with TP Hit ===")
    obs, _ = env.reset()
    action = [0,0.1]  # 50% long
    obs, reward, done, truncated, info = env.step(action)
    print(f"Position opened. Entry: {info['entry_price']} | TP: {env.tp_price} | SL: {env.sl_price} | Position: {info['position']}")
    print(f"Reward: {reward} | {info['portfolio_balance']} ")
    
    # # Test 2: Short position that hits SL
    # print("\n=== TEST 2: Short with SL Hit ===")
    
    # action = {'action_type': 2, 'position_size': np.array([0.1], dtype=np.float32)}  # 30% short
    # obs, reward, done, truncated, info = env.step(action)
    # print(f"Position opened. Entry: {info['entry_price']} | TP: {env.tp_price} | SL: {env.sl_price} | Position: {info['position']}")
    # print(f"Reward: {reward} | {info['portfolio_balance']} ")
    # # Test 3: Multiple consecutive trades
    # print("\n=== TEST 3: Multiple Trades ===")
    
    # for i in range(5):
    #     action_type = 1 if i % 2 == 0 else 2  # Alternate long/short
    #     action = {'action_type': action_type, 'position_size': np.array([0.1], dtype=np.float32)}
    #     obs, reward, done, truncated, info = env.step(action)
    #     print(f"Trade {i+1}: {'Long' if action_type==1 else 'Short'} | Position: {info['position']:.4f} | Entry: {info['entry_price']} | TP: {env.tp_price} | SL: {env.sl_price}")
    #     print(f"Reward: {reward} | {info['portfolio_balance']} \n")
        
if __name__ == "__main__":
    # Load your data here
    rl_dataset = load_data_from_db ()  # You must implement this
    env = PatternSentimentEnv(rl_dataset)
    test_environment(env)