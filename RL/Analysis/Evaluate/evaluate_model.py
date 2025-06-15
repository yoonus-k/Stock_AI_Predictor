import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

"""
Reinforcement Learning Model Evaluation Script
Simplified for direct Box action space:
1. Removed TupleActionWrapper (now obsolete)
2. Simplified action handling for Box action space
3. Enhanced diagnostic outputs for model analysis
4. Safety measures against infinite loops
"""

# Add the project root to path for proper importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL.Envs.trading_env import TradingEnv
from RL.Data.Utils.loader import load_data_from_db

def evaluate_model(model_path,save_path, data=None, reward_type='combined', render_mode=None):
    """
    Evaluate a trained RL model on the given dataset.
    
    Args:
        model_path: Path to the saved model
        data: Data to evaluate on (if None, loads from DB)
        reward_type: Type of reward function to use for evaluation
        render_mode: Optional rendering mode
    
    Returns:
        Dictionary containing performance metrics
    """
    # Load data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        if data is None or len(data) == 0:
            print("Error: Could not load data from database")
            return None
        
    #split data into training and evaluation sets
    print(f"Data loaded with {len(data)} rows")
    split_idx = int(len(data) * 0.5)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    
    # Create evaluation environment
    env = TradingEnv(eval_data, reward_type=reward_type)
    
    # Load the trained model
    model = PPO.load(model_path ,custom_objects={'learning_rate': 0.0003} )
    print(f"Loaded model from {model_path}")
    print(f"Action space: {env.action_space}")
      
    # Run evaluation
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    
    done = False
    rewards = []
    trade_pnls = []
    portfolio_values = [env.balance]
    actions_taken = []
    trade_directions = []
    position_sizes = []
    risk_multiples = []
    action_rewards = {} # Maps action types to rewards
    raw_actions = []  # Track raw action values for analysis
    
    # Tracking trade outcomes
    trade_outcomes = {
        'win': 0,
        'loss': 0,
        'hold': 0,
        'long': 0,
        'short': 0,
    }
    
    step = 0
    max_steps = len(data)   # Set a maximum number of steps (twice the data length as a safety margin)
    print(f"Data length: {len(data)}, Max steps: {max_steps}")
    
    while not done and step < max_steps:
        # Get raw action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Store raw action for analysis        raw_actions.append(action.copy())
        
        # Extract the action type directly - MultiDiscrete already gives us discrete values
        action_type = int(action[0])  # 0=HOLD, 1=BUY, 2=SELL
            
        # Extract and convert position size index to continuous value
        # 0->0.1, 1->0.2, ..., 9->1.0
        position_size = 0.1 + (action[1] * 0.1)
        
        # Extract and convert risk reward index to continuous value
        # 0->0.5, 1->0.75, ..., 9->3.0
        risk_reward_ratio = 0.5 + (action[2] * 0.25)
        
        # Print diagnostic info for first few steps and periodically
        if step == 0 or step < 5 or step % 1000 == 0:
            print(f"\nStep {step} raw action: {action}")
            print(f"Action type: {action_type} ({['HOLD', 'BUY', 'SELL'][action_type]})")
            print(f"Position size: {position_size:.2f}")
            print(f"Risk-reward multiplier: {risk_reward_ratio:.2f}")
        
        # Step environment with raw action array (no need to reformat)
        obs, reward, done, truncated, info = env.step(action)
        
        # Print additional diagnostics every 1000 steps
        if step % 1000 == 0:
            print(f"\nStep {step} environment state:")
            print(f"  Balance: {info['portfolio_balance']}")
            print(f"  Position: {info['position']}")
            print(f"  Trade direction: {info['trade_direction']}")
            print(f"  Trade PnL: {info.get('trade_pnl', 0)}")
            
            # Add action distribution so far to see if we're getting diverse actions
            current_action_counts = pd.Series(actions_taken).value_counts()
            print(f"  Action distribution so far:")
            for act_type in range(3):
                count = current_action_counts.get(act_type, 0)
                pct = (count / len(actions_taken)) * 100 if actions_taken else 0
                print(f"    {act_type} ({['Hold', 'Buy', 'Sell'][act_type]}): {count} ({pct:.1f}%)")
        
        # Record step information
        rewards.append(reward)
        portfolio_values.append(info['portfolio_balance'])
        actions_taken.append(action_type)
        trade_directions.append(info['trade_direction'])
        position_sizes.append(float(position_size))
        risk_multiples.append(float(risk_reward_ratio))
        
        # Track trades
        trade_pnl = info.get('trade_pnl', 0)
        if trade_pnl != 0:
            trade_pnls.append(trade_pnl)
            
            # Record action rewards
            action_key = f"Action {action_type}"
            if action_key not in action_rewards:
                action_rewards[action_key] = []
            action_rewards[action_key].append(reward)
            
            # Record trade outcomes
            if trade_pnl > 0:
                trade_outcomes['win'] += 1
            else:
                trade_outcomes['loss'] += 1
                
            if info['trade_direction'] == 1:
                trade_outcomes['long'] += 1
            elif info['trade_direction'] == -1:
                trade_outcomes['short'] += 1
        else:
            trade_outcomes['hold'] += 1
        
        step += 1
        if step % 500 == 0:
            print(f"Evaluation step {step}/{len(data)}")
    
    # Check if we terminated due to max steps
    if step >= max_steps:
        print(f"WARNING: Evaluation reached maximum step limit ({max_steps}). The environment may not be terminating properly.")
    
    # Calculate performance metrics
    final_balance = portfolio_values[-1]
    initial_balance = env.initial_balance
    total_return = (final_balance / initial_balance - 1) * 100
    
    # Calculate Sharpe ratio
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    # Calculate max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = min(drawdown) * 100
    
    # Calculate win rate
    win_rate = trade_outcomes['win'] / (trade_outcomes['win'] + trade_outcomes['loss']) if trade_outcomes['win'] + trade_outcomes['loss'] > 0 else 0
    
    # Calculate profit factor
    profit_sum = sum(pnl for pnl in trade_pnls if pnl > 0)
    loss_sum = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
    
    # Print results
    print("\n===== Model Evaluation Results =====")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total Trades: {len(trade_pnls)}")
    
    # Print trade stats
    print("\n===== Trade Statistics =====")
    print(f"Wins: {trade_outcomes['win']}")
    print(f"Losses: {trade_outcomes['loss']}")
    print(f"Long Trades: {trade_outcomes['long']}")
    print(f"Short Trades: {trade_outcomes['short']}")
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot portfolio value
    plt.subplot(2, 2, 1)
    plt.plot(portfolio_values)
    plt.title(f'Portfolio Value (Return: {total_return:.2f}%)')
    plt.xlabel('Trading Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 2, 2)
    plt.plot(drawdown * 100)
    plt.title(f'Drawdown (Max: {max_drawdown:.2f}%)')
    plt.xlabel('Trading Steps')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    
    # Plot action distribution
    plt.subplot(2, 2, 3)
    action_counts = pd.Series(actions_taken).value_counts()
    
    # Ensure we have counts for all action types (0, 1, 2)
    for action_type in [0, 1, 2]:
        if action_type not in action_counts.index:
            action_counts[action_type] = 0
    
    # Sort by index to ensure order is [0, 1, 2]
    action_counts = action_counts.sort_index()
    
    # Map indices to meaningful labels
    action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    action_counts.index = [action_labels[i] for i in action_counts.index]
    
    action_counts.plot(kind='bar')
    plt.title('Action Distribution')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Plot position sizes
    plt.subplot(2, 2, 4)
    plt.hist(position_sizes, bins=20)
    plt.title('Position Size Distribution')
    plt.xlabel('Position Size (% of Portfolio)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    
    plot_path = os.path.join(save_path, 'model_evaluation.png')
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    # Create additional plots for trading behavior analysis
    plt.figure(figsize=(15, 10))
    
    # Plot reward distribution by action type
    plt.subplot(2, 2, 1)
    for action, action_rewards_list in action_rewards.items():
        if action_rewards_list:
            plt.hist(action_rewards_list, alpha=0.7, label=action)
    plt.title('Reward Distribution by Action Type')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Plot risk-reward ratio multiplier distribution
    plt.subplot(2, 2, 2)
    plt.hist(risk_multiples, bins=20)
    plt.title('Risk-Reward Ratio Multiplier Distribution')
    plt.xlabel('Multiplier Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot trade P&L distribution
    plt.subplot(2, 2, 3)
    plt.hist(trade_pnls, bins=20)
    plt.title('Trade P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot action frequency instead of 2D histogram
    plt.subplot(2, 2, 4)
    action_counts = pd.Series(actions_taken).value_counts().sort_index()
    action_labels = ['Hold', 'Buy', 'Sell']
    if len(action_counts) <= len(action_labels):
        action_counts.index = [action_labels[i] for i in action_counts.index if i < len(action_labels)]

    action_counts.plot(kind='bar')
    plt.title('Action Frequency Distribution')
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save additional plots
    add_plot_path = os.path.join(save_path, 'trading_behavior_analysis.png')
    plt.savefig(add_plot_path)
    print(f"Trading behavior analysis plot saved to {add_plot_path}")
    
    # Return results dictionary
    return {
        'final_balance': final_balance,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trade_pnls),
        'trade_outcomes': trade_outcomes,
        'portfolio_values': portfolio_values,
        'raw_actions': raw_actions,  # Include raw actions for further analysis
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained RL model')
    parser.add_argument('--model_path', default='RL/Models/Experiments/best_model_continued.zip', type=str, help='Path to the saved model')
    parser.add_argument('--save_path', default='Images/RL/Models', type=str, help='Path to save evaluation results and plots')
    parser.add_argument('--reward_type', type=str, default='combined', help='Type of reward function to use for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    print(f"Evaluating model with direct Box action space")
    evaluate_model(args.model_path,args.save_path, reward_type=args.reward_type)
