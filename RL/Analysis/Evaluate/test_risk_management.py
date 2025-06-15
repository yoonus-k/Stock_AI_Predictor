import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add the project root to path for proper importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL.Envs.trading_env import TradingEnv
from RL.Data.Utils.loader import load_data_from_db

def test_risk_management(model_paths,save_path, data=None, stress_scenarios=['normal', 'drawdown', 'volatility', 'choppy']):
    """
    Test different models in various market stress scenarios.
    
    Args:
        model_paths: Dictionary mapping reward types to model paths
        data: Test data (if None, loads from DB)
        stress_scenarios: List of stress scenarios to test
        
    Returns:
        Dataframe with performance metrics across models and scenarios
    """
    # Load data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        if data is None or len(data) == 0:
            print("Error: Could not load data from database")
            return None
    
    # Define splitting point for different scenarios
    split_idx = int(len(data) * 0.7)
    normal_data = data[:split_idx]  # Use first 70% as "normal" scenario
    
    # Results storage
    all_results = []
    scenario_data = {}
    
    # Create scenarios
    scenario_data['normal'] = normal_data.copy()
    
    # Create drawdown scenario (artificially induce big losses)
    if 'drawdown' in stress_scenarios:
        drawdown_data = normal_data.copy()
        # Modify max_drawdown and actual_return to create a more difficult drawdown scenario
        drawdown_indices = np.random.choice(len(drawdown_data), size=int(len(drawdown_data)*0.3), replace=False)
        for idx in drawdown_indices:
            drawdown_data.iloc[idx, drawdown_data.columns.get_loc('max_drawdown')] *= 2.0  # Double the max drawdown
            drawdown_data.iloc[idx, drawdown_data.columns.get_loc('actual_return')] *= -1.5  # Increase losses
        scenario_data['drawdown'] = drawdown_data
    
    # Create high volatility scenario
    if 'volatility' in stress_scenarios:
        volatility_data = normal_data.copy()
        # Increase volatility by increasing mfe and mae
        vol_indices = np.random.choice(len(volatility_data), size=int(len(volatility_data)*0.5), replace=False)
        for idx in vol_indices:
            if 'mfe' in volatility_data.columns:
                volatility_data.iloc[idx, volatility_data.columns.get_loc('mfe')] *= 2.0  # Increase MFE
            if 'mae' in volatility_data.columns:
                volatility_data.iloc[idx, volatility_data.columns.get_loc('mae')] *= 2.0  # Increase MAE
            if 'atr_ratio' in volatility_data.columns:
                volatility_data.iloc[idx, volatility_data.columns.get_loc('atr_ratio')] *= 3.0  # Increase ATR ratio
        scenario_data['volatility'] = volatility_data
    
    # Create choppy market scenario (low win rate)
    if 'choppy' in stress_scenarios:
        choppy_data = normal_data.copy()
        # Make markets choppy by reducing max_gain and introducing false signals
        choppy_indices = np.random.choice(len(choppy_data), size=int(len(choppy_data)*0.4), replace=False)
        for idx in choppy_indices:
            choppy_data.iloc[idx, choppy_data.columns.get_loc('max_gain')] *= 0.5  # Reduce max gain
            # Randomly flip the action to create false signals
            if np.random.random() > 0.4:
                action = choppy_data.iloc[idx, choppy_data.columns.get_loc('action')]
                choppy_data.iloc[idx, choppy_data.columns.get_loc('action')] = 3 - action  # Flip: 1->2, 2->1
        scenario_data['choppy'] = choppy_data
    
    # Test each model in each scenario
    for reward_type, model_path in model_paths.items():
        print(f"\n===== Testing {reward_type} model =====")
        
        try:
            # Load the model
            model = PPO.load(model_path, custom_objects={'learning_rate': 0.0003})
            print(f"Successfully loaded model from {model_path}")
            
            for scenario_name, scenario_df in scenario_data.items():
                print(f"Testing in {scenario_name} market conditions...")
                
                # Create environment with this scenario data
                env = TradingEnv(scenario_df, reward_type='base')  # Using base reward for fair comparison
                
                # Run evaluation
                obs, _ = env.reset()
                done = False
                rewards = []
                portfolio_values = [env.balance]
                trade_pnls = []
                drawdowns = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    rewards.append(reward)
                    portfolio_values.append(info['portfolio_balance'])
                    
                    # Track trade P&L
                    if info.get('trade_pnl', 0) != 0:
                        trade_pnls.append(info['trade_pnl'])
                        
                    # Calculate current drawdown
                    peak_value = max(portfolio_values)
                    current_value = portfolio_values[-1]
                    current_drawdown = (current_value - peak_value) / peak_value
                    drawdowns.append(current_drawdown)
                
                # Calculate metrics
                final_balance = portfolio_values[-1]
                total_return = (final_balance / env.initial_balance - 1) * 100
                max_drawdown = min(drawdowns) * 100 if drawdowns else 0
                
                # Calculate Sharpe ratio
                pct_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe = np.mean(pct_returns) / (np.std(pct_returns) + 1e-6) * np.sqrt(252) if pct_returns.size > 0 else 0
                
                # Calculate win rate and profit factor
                wins = sum(1 for pnl in trade_pnls if pnl > 0)
                win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else 0
                
                profit_sum = sum([pnl for pnl in trade_pnls if pnl > 0]) if trade_pnls else 0
                loss_sum = abs(sum([pnl for pnl in trade_pnls if pnl < 0])) if trade_pnls else 1  # Avoid division by zero
                profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
                
                # Store results
                all_results.append({
                    'Reward Type': reward_type,
                    'Scenario': scenario_name,
                    'Return (%)': total_return,
                    'Sharpe': sharpe,
                    'Max Drawdown (%)': max_drawdown,
                    'Win Rate': win_rate,
                    'Profit Factor': profit_factor,
                    'Trades': len(trade_pnls),
                    'Portfolio Values': portfolio_values
                })
                
                print(f"  Return: {total_return:.2f}%")
                print(f"  Sharpe: {sharpe:.2f}")
                print(f"  Max Drawdown: {max_drawdown:.2f}%")
                print(f"  Win Rate: {win_rate:.2f}")
                print(f"  Profit Factor: {profit_factor:.2f}")
                print(f"  Trades: {len(trade_pnls)}")
        
        except Exception as e:
            print(f"Error testing {reward_type} model: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create visualization of results
    if not results_df.empty:
        # Create directory for outputs
        output_dir = os.path.join(save_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot equity curves for each model in each scenario
        scenarios = results_df['Scenario'].unique()
        models = results_df['Reward Type'].unique()
        
        for scenario in scenarios:
            plt.figure(figsize=(12, 8))
            
            for reward_type in models:
                subset = results_df[(results_df['Reward Type'] == reward_type) & 
                                  (results_df['Scenario'] == scenario)]
                if not subset.empty:
                    portfolio_values = subset['Portfolio Values'].values[0]
                    plt.plot(portfolio_values, label=f"{reward_type} ({subset['Return (%)'].values[0]:.1f}%)")
            
            plt.title(f'Equity Curves in {scenario.capitalize()} Market')
            plt.xlabel('Trading Steps')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plt_path = os.path.join(output_dir, f'equity_curves_{scenario}.png')
            plt.savefig(plt_path)
            print(f"Saved equity curves for {scenario} scenario to {plt_path}")
        
        # Create comparison charts
        plt.figure(figsize=(15, 15))
        
        # Returns comparison
        plt.subplot(2, 2, 1)
        pivot_returns = results_df.pivot_table(index='Reward Type', columns='Scenario', values='Return (%)')
        pivot_returns.plot(kind='bar', ax=plt.gca())
        plt.title('Returns by Model and Scenario')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Drawdown comparison
        plt.subplot(2, 2, 2)
        pivot_dd = results_df.pivot_table(index='Reward Type', columns='Scenario', values='Max Drawdown (%)')
        pivot_dd.plot(kind='bar', ax=plt.gca())
        plt.title('Max Drawdown by Model and Scenario')
        plt.ylabel('Max Drawdown (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Win Rate comparison
        plt.subplot(2, 2, 3)
        pivot_wr = results_df.pivot_table(index='Reward Type', columns='Scenario', values='Win Rate')
        pivot_wr.plot(kind='bar', ax=plt.gca())
        plt.title('Win Rate by Model and Scenario')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Sharpe comparison
        plt.subplot(2, 2, 4)
        pivot_sharpe = results_df.pivot_table(index='Reward Type', columns='Scenario', values='Sharpe')
        pivot_sharpe.plot(kind='bar', ax=plt.gca())
        plt.title('Sharpe Ratio by Model and Scenario')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = os.path.join(output_dir, 'scenario_model_comparison.png')
        plt.savefig(comparison_path)
        print(f"Saved scenario comparison plot to {comparison_path}")
        
        # Save detailed results to CSV
        csv_path = os.path.join(output_dir, 'stress_test_results.csv')
        results_df.drop('Portfolio Values', axis=1).to_csv(csv_path)
        print(f"Saved detailed results to {csv_path}")
    
    return results_df

if __name__ == "__main__":
    # Directory containing trained models
    model_dir = os.path.join(os.path.dirname(__file__), "../Models")
    save_path = "Images/RL/RiskManagement"
    
    # Find available models
    model_paths = {}
    reward_types = ['base', 'sharpe', 'sortino', 'drawdown_focus', 'calmar', 'win_rate', 'combined']
    
    for reward_type in reward_types:
        path = os.path.join(model_dir, f"ppo_trading_{reward_type}.zip")
        if os.path.exists(path):
            model_paths[reward_type] = path
        else:
            # Try best model path
            best_path = os.path.join(model_dir, f"best_{reward_type}/best_model.zip")
            if os.path.exists(best_path):
                model_paths[reward_type] = best_path
    
    # if not model_paths:
    #     print("No trained models found. Please train models first using train_compare_rewards.py")
    #     sys.exit(1)
    
    print(f"Found {len(model_paths)} trained models: {list(model_paths.keys())}")
    
    # Run stress tests
    results = test_risk_management({"combined":"RL/Models/Experiments/best_model_continued.zip"},save_path)
    
    if results is not None:
        # Print overall rankings
        print("\nOverall Model Performance Rankings:")
        
        # Average return across scenarios
        avg_returns = results.groupby('Reward Type')['Return (%)'].mean().sort_values(ascending=False)
        print("\nBest Average Return:")
        print(avg_returns)
        
        # Best worst-case scenario (max of min returns)
        worst_case = results.groupby('Reward Type')['Return (%)'].min().sort_values(ascending=False)
        print("\nBest Worst-Case Return:")
        print(worst_case)
        
        # Lowest average drawdown
        avg_dd = results.groupby('Reward Type')['Max Drawdown (%)'].mean().sort_values()
        print("\nLowest Average Drawdown:")
        print(avg_dd)
        
        # Overall risk-adjusted ranking (using average Sharpe)
        avg_sharpe = results.groupby('Reward Type')['Sharpe'].mean().sort_values(ascending=False)
        print("\nBest Average Sharpe Ratio:")
        print(avg_sharpe)
