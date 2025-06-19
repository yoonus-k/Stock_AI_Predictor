"""
Test script for the enhanced visualization function
"""

import os
import numpy as np
import pandas as pd
from RL.Envs.trading_env import TradingEnvV2
from RL.Envs.Utils.env_visualization import visualize_env_metrics
from RL.Data.Utils.loader import load_data_from_db

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_test_data(rows=500):
    """Create synthetic data for testing the environment"""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    base_date = pd.Timestamp('2023-01-01')
    dates = [base_date + pd.Timedelta(days=i) for i in range(rows)]
    
    # Generate price data with some trend and volatility
    close = 100.0
    prices = []
    for _ in range(rows):
        # Random walk with drift
        close = close * (1 + np.random.normal(0.0005, 0.01))
        prices.append(close)
    
    # Calculate some basic features
    returns = np.zeros(rows)
    for i in range(1, rows):
        returns[i] = (prices[i] / prices[i-1]) - 1
    
    # Create rolling window features
    ma_5 = pd.Series(prices).rolling(5).mean().values
    ma_20 = pd.Series(prices).rolling(20).mean().values
    std_10 = pd.Series(returns).rolling(10).std().values
      # Create feature DataFrame
    features = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'volume': np.random.randint(1000, 100000, rows),
        'returns': returns,
        'ma_5': ma_5,
        'ma_20': ma_20,
        'std_10': std_10,
        'ema_12': pd.Series(prices).ewm(span=12).mean().values,
        'macd': pd.Series(prices).ewm(span=12).mean().values - pd.Series(prices).ewm(span=26).mean().values
    })
    
    # Set date as index
    features.set_index('date', inplace=True)
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    return features

def main():
    """Main test function"""
    print("Creating test data...")
    features = load_data_from_db()  # Load data from database
    
    print("Initializing environment...")
    env = TradingEnvV2(
        features=features,
        initial_balance=100000,
        reward_type="combined",
        normalize_observations=True,
        commission_rate=0.001,
        enable_short=True,
        max_positions=5,
        verbose=True
    )
    
    print("Running visualization with enhanced reward component analysis...")
    metrics_df = visualize_env_metrics(env, num_steps=1000)
    
    print("Visualization completed successfully")
    print(f"Final equity: ${metrics_df['equity'].iloc[-1]:.2f}")
    print(f"Average reward: {metrics_df['reward'].mean():.6f}")
    print(f"Reward-equity correlation: {metrics_df['reward'].corr(metrics_df['equity_change']):.4f}")
    
    # Check for component correlations if available
    if 'base_reward' in metrics_df.columns:
        components = ['base_reward', 'sharpe_adjustment', 'drawdown_penalty', 'win_rate_bonus', 'consistency_bonus']
        print("\nComponent correlations with total reward:")
        for comp in components:
            corr = metrics_df[comp].corr(metrics_df['reward'])
            print(f"- {comp}: {corr:.4f}")

if __name__ == "__main__":
    main()
