import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add the project root to path for proper importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL.Envs.trading_env import PatternSentimentEnv
from RL.Data.loader import load_data_from_db

def evaluate_model(model_path, data=None, reward_type='combined', render_mode=None):
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
    
    # Create evaluation environment
    env = PatternSentimentEnv(data, reward_type=reward_type)
    
    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Run evaluation
    obs, _ = env.reset()
    
    done = False
    rewards = []
    trade_pnls = []
    portfolio_values = [env.balance]
    actions_taken = []
    trade_directions = []
    position_sizes = []
    risk_multiples = []
    action_rewards = {} # Maps action types to rewards
    
    # Tracking trade outcomes
    trade_outcomes = {
        'win': 0,
        'loss': 0,
        'hold': 0,
        'long': 0,
        'short': 0,
        'tp_hit': 0,
        'sl_hit': 0,
    }
    
    step = 0
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action_type, continuous_params = action
        position_size, risk_reward_ratio = continuous_params
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Record step information
        rewards.append(reward)
        portfolio_values.append(info['portfolio_balance'])
        actions_taken.append(action_type)
        trade_directions.append(info['trade_direction'])
        position_sizes.append(position_size)
        risk_multiples.append(risk_reward_ratio)
        
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
    action_counts.index = ['Hold', 'Buy', 'Sell']
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
    plot_dir = os.path.dirname(model_path)
    plot_path = os.path.join(plot_dir, 'model_evaluation.png')
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
    
    # Plot reward vs. position size
    plt.subplot(2, 2, 4)
    trade_position_sizes = []
    trade_rewards = []
    
    # Get only the positions where trades happened
    for i, pnl in enumerate(trade_pnls):
        # Find matching position size
        while len(trade_position_sizes) < len(trade_pnls):
            trade_position_sizes.append(position_sizes[i])
            trade_rewards.append(rewards[i])
    
    plt.scatter(trade_position_sizes, trade_rewards, alpha=0.5)
    plt.title('Reward vs. Position Size')
    plt.xlabel('Position Size')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save additional plots
    add_plot_path = os.path.join(plot_dir, 'trading_behavior_analysis.png')
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
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained RL model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--reward_type', type=str, default='combined', help='Type of reward function to use for evaluation')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, reward_type=args.reward_type)
