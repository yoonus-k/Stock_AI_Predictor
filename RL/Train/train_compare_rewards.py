import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Add the project root to path for proper importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL.Envs.trading_env import TradingEnv
from RL.Data.loader import load_data_from_db

def train_and_evaluate_models(reward_types=['base', 'sharpe', 'sortino', 'drawdown_focus', 'calmar', 'win_rate', 'combined'], 
                             training_timesteps=200000):
    """
    Train multiple models with different reward functions and compare performance.
    
    Args:
        reward_types: List of reward functions to test
        training_timesteps: Number of timesteps to train each model
    
    Returns:
        Dictionary containing trained models and performance metrics
    """
    # Load data from samples.db
    print("Loading data from database...")
    rl_dataset = load_data_from_db()
    
    if rl_dataset is None or len(rl_dataset) == 0:
        print("Error: No data loaded from database. Check your data paths and DB connection.")
        return None, None
    
    print(f"Loaded {len(rl_dataset)} samples from database.")
    
    # Split into training and eval datasets
    split_idx = int(len(rl_dataset) * 0.8)
    train_data = rl_dataset[:split_idx]
    eval_data = rl_dataset[split_idx:]
    
    train_data = train_data.head(100)  # Limit to 10,000 samples for faster training
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Evaluation data: {len(eval_data)} samples")
    
    results = {}
    models = {}
    
    # Create directory for models
    model_dir = os.path.join(os.path.dirname(__file__), "../Models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Train a model for each reward type
    for reward_type in reward_types:
        print(f"\n===== Training model with {reward_type} reward =====")
        
        # Create environment with specific reward function
        env = TradingEnv(train_data, reward_type=reward_type)
        
        # Initialize model with hyperparameters
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, 
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            tensorboard_log=os.path.join(os.path.dirname(__file__), "../Logs")
        )
        
        # Create evaluation callback
        eval_env = TradingEnv(eval_data, reward_type=reward_type)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, f"best_{reward_type}"),
            log_path=os.path.join(os.path.dirname(__file__), "../Logs"),
            eval_freq=10000,
            deterministic=False,
            render=False
        )
        
        # Train model
        model.learn(total_timesteps=training_timesteps, callback=eval_callback)
        
        # Save final model
        model_path = os.path.join(model_dir, f"ppo_trading_{reward_type}")
        model.save(model_path)
        models[reward_type] = model
        
        # Evaluate model performance
        print(f"Evaluating {reward_type} model...")
        eval_env = TradingEnv(eval_data, reward_type='base')  # Use base reward for fair comparison
        
        # Run evaluation
        obs, _ = eval_env.reset()
        done = False
        returns = []
        balances = [eval_env.initial_balance]
        max_drawdowns = []
        win_rates = []
        trades = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            returns.append(reward)
            balances.append(info['portfolio_balance'])
            max_drawdowns.append(info.get('max_drawdown', 0))
            win_rates.append(info.get('win_rate', 0))
            
            # Track actual trades
            if info.get('trade_pnl', 0) != 0:
                trades += 1
        
        # Calculate performance metrics
        final_balance = balances[-1]
        return_pct = (final_balance / eval_env.initial_balance - 1) * 100
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252) if len(returns) > 1 else 0
        max_dd = min(max_drawdowns) * 100 if max_drawdowns else 0
        profit_factor = eval_env.win_amount / (eval_env.loss_amount + 1e-6) if eval_env.loss_amount > 0 else 0
        win_rate = max(win_rates, default=0)
        
        results[reward_type] = {
            'final_balance': final_balance,
            'return_pct': return_pct,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_executed': trades,
            'balance_curve': balances
        }
        
        print(f"Model {reward_type} results:")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Win Rate: {win_rate:.2f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Trades Executed: {trades}")
    
    # Plot performance comparison
    plt.figure(figsize=(14, 10))
    
    # Plot equity curves
    plt.subplot(2, 2, 1)
    for reward_type, metrics in results.items():
        plt.plot(metrics['balance_curve'], label=f"{reward_type} ({metrics['return_pct']:.1f}%)")
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Trading Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot returns
    plt.subplot(2, 2, 2)
    metrics_df = pd.DataFrame([{
        'Reward Type': reward_type,
        'Return (%)': metrics['return_pct'],
        'Sharpe': metrics['sharpe'],
        'Max DD (%)': -metrics['max_drawdown'],
        'Win Rate': metrics['win_rate'],
    } for reward_type, metrics in results.items()])
    
    metrics_df.sort_values('Return (%)', ascending=False).plot(x='Reward Type', y='Return (%)', kind='bar')
    plt.title('Total Return by Reward Type')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # Plot Sharpe ratios
    plt.subplot(2, 2, 3)
    metrics_df.sort_values('Sharpe', ascending=False).plot(x='Reward Type', y='Sharpe', kind='bar')
    plt.title('Sharpe Ratio by Reward Type')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    # Plot Max Drawdown 
    plt.subplot(2, 2, 4)
    metrics_df.sort_values('Max DD (%)', ascending=True).plot(x='Reward Type', y='Max DD (%)', kind='bar')
    plt.title('Maximum Drawdown by Reward Type')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt_path = os.path.join(model_dir, 'reward_comparison.png')
    plt.savefig(plt_path)
    print(f"Performance comparison plot saved to {plt_path}")
    
    # Create performance summary table
    summary = pd.DataFrame(results).T
    summary = summary.drop('balance_curve', axis=1)
    summary_path = os.path.join(model_dir, 'reward_performance_summary.csv')
    summary.to_csv(summary_path)
    print(f"Performance summary saved to {summary_path}")
    
    return models, results

if __name__ == "__main__":
    # Define reward types to test
    reward_types = [
        'base',        # Base reward (no risk adjustment)
        'sharpe',      # Sharpe ratio optimization
        'sortino',     # Sortino ratio (downside risk) optimization
        'drawdown_focus', # Focus on minimizing drawdowns
        'calmar',      # Calmar ratio optimization
        'win_rate',    # Win rate optimization
        'combined'     # Combined risk-adjusted reward
    ]
    
    print("Starting training and evaluation of models with different reward functions")
    print(f"Testing reward types: {reward_types}")
    
    # Train and evaluate models
    models, results = train_and_evaluate_models(reward_types=reward_types, training_timesteps=200000)
    
    if results:
        # Print final comparison
        print("\nFinal Performance Summary:")
        
        # Convert to DataFrame for easier comparison
        summary_df = pd.DataFrame(results).T
        summary_df = summary_df.drop('balance_curve', axis=1)
        
        print(summary_df[['return_pct', 'sharpe', 'max_drawdown', 'win_rate', 'profit_factor', 'trades_executed']])
        
        # Identify best model for different metrics
        best_return = summary_df['return_pct'].idxmax()
        best_sharpe = summary_df['sharpe'].idxmax()
        min_drawdown = summary_df['max_drawdown'].abs().idxmin()
        best_win_rate = summary_df['win_rate'].idxmax()
        
        print("\nBest Models by Metric:")
        print(f"Best Return: {best_return} with {summary_df.loc[best_return, 'return_pct']:.2f}%")
        print(f"Best Sharpe: {best_sharpe} with {summary_df.loc[best_sharpe, 'sharpe']:.2f}")
        print(f"Minimum Drawdown: {min_drawdown} with {summary_df.loc[min_drawdown, 'max_drawdown']:.2f}%")
        print(f"Best Win Rate: {best_win_rate} with {summary_df.loc[best_win_rate, 'win_rate']:.2f}")
    else:
        print("Error occurred during training and evaluation. Check logs for details.")
