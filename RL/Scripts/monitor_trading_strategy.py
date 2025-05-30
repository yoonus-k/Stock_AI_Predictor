"""
Trading Strategy Analysis and Portfolio Performance Visualization

This script analyzes the trading strategies of trained RL models and visualizes
portfolio performance over time. It helps understand how the model makes trading
decisions and evaluates trading performance metrics.

Key features:
1. Trading action analysis - Analyzes patterns in buy/sell/hold decisions
2. Portfolio performance tracking - Tracks portfolio value over time
3. Trade visualization - Visualizes individual trades on price charts
4. Performance metrics - Calculates key trading metrics (profit factor, win rate, etc.)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from RL.Data.loader import load_data_from_db
from RL.Envs.trading_env import PatternSentimentEnv


class PortfolioTracker:
    """Track portfolio performance during model evaluation"""
    
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset the tracker to initial state"""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_values = []
        self.timestamps = []
        self.trades = []
        self.trade_returns = []
        self.actions = []
        self.positions = []
        self.current_trade = None
        
    def update(self, timestamp, price, position, action, reward):
        """Update tracker with new values"""
        # Update position tracking
        old_position = self.position
        self.position = position
        
        # Calculate portfolio value
        if position == 0:
            portfolio_value = self.balance
        else:
            # For long position
            if position > 0:
                portfolio_value = self.balance + position * price
            # For short position
            else:
                # Short profit/loss is based on difference from entry
                if self.entry_price > 0:
                    price_diff = self.entry_price - price
                    portfolio_value = self.balance + abs(position) * price_diff
                else:
                    portfolio_value = self.balance
        
        # Record action
        self.actions.append({
            'timestamp': timestamp,
            'price': price,
            'action': action,
            'position': position,
            'reward': reward,
            'portfolio_value': portfolio_value
        })
        
        # Track positions separately for analysis
        self.positions.append({
            'timestamp': timestamp,
            'position': position,
            'price': price
        })
        
        # Record portfolio value
        self.portfolio_values.append(portfolio_value)
        self.timestamps.append(timestamp)
        
        # Check for new trade
        if old_position == 0 and position != 0:
            # New trade started
            self.current_trade = {
                'entry_time': timestamp,
                'entry_price': price,
                'position': position,
                'exit_time': None,
                'exit_price': None,
                'profit_loss': 0.0,
                'return_pct': 0.0,
                'duration': 0
            }
            self.entry_price = price
            
        # Check for closed trade
        elif old_position != 0 and (position == 0 or (position * old_position < 0)):
            # Trade closed or reversed
            if self.current_trade is not None:
                # Calculate profit/loss
                if old_position > 0:  # Long position
                    profit_loss = (price - self.current_trade['entry_price']) * abs(old_position)
                else:  # Short position
                    profit_loss = (self.current_trade['entry_price'] - price) * abs(old_position)
                
                # Update trade record
                self.current_trade['exit_time'] = timestamp
                self.current_trade['exit_price'] = price
                self.current_trade['profit_loss'] = profit_loss
                
                # Calculate return percentage
                if old_position > 0:  # Long position
                    self.current_trade['return_pct'] = ((price / self.current_trade['entry_price']) - 1) * 100
                else:  # Short position
                    self.current_trade['return_pct'] = ((self.current_trade['entry_price'] / price) - 1) * 100
                
                # Calculate duration (assuming timestamps are datetime objects)
                if isinstance(timestamp, (pd.Timestamp, datetime)):
                    duration = (timestamp - self.current_trade['entry_time']).total_seconds() / 60  # minutes
                    self.current_trade['duration'] = duration
                
                # Save trade
                self.trades.append(self.current_trade)
                self.trade_returns.append(self.current_trade['return_pct'])
                
                # Update balance based on profit/loss
                self.balance += profit_loss
                
                # If position reversed, start new trade
                if position != 0:
                    self.current_trade = {
                        'entry_time': timestamp,
                        'entry_price': price,
                        'position': position,
                        'exit_time': None,
                        'exit_price': None,
                        'profit_loss': 0.0,
                        'return_pct': 0.0,
                        'duration': 0
                    }
                    self.entry_price = price
                else:
                    self.current_trade = None
                    self.entry_price = 0.0
                    
    def get_performance_metrics(self):
        """Calculate trading performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['initial_balance'] = self.initial_balance
        metrics['final_balance'] = self.balance
        metrics['total_return'] = ((self.balance / self.initial_balance) - 1) * 100
        metrics['total_trades'] = len(self.trades)
        
        # If no trades, return basic metrics only
        if not self.trades:
            return metrics
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        metrics['win_rate'] = (len(winning_trades) / len(self.trades)) * 100
        
        # Calculate average profit/loss
        metrics['avg_profit'] = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
        
        losing_trades = [t for t in self.trades if t['profit_loss'] <= 0]
        metrics['avg_loss'] = np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate profit factor
        total_profit = sum([t['profit_loss'] for t in winning_trades])
        total_loss = abs(sum([t['profit_loss'] for t in losing_trades]))
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate Sharpe Ratio (if we have sufficient data)
        if len(self.trade_returns) > 1:
            avg_return = np.mean(self.trade_returns)
            std_return = np.std(self.trade_returns)
            metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
        
        # Position analysis
        long_trades = [t for t in self.trades if t['position'] > 0]
        short_trades = [t for t in self.trades if t['position'] < 0]
        
        metrics['long_trades'] = len(long_trades)
        metrics['short_trades'] = len(short_trades)
        
        metrics['long_win_rate'] = (len([t for t in long_trades if t['profit_loss'] > 0]) / len(long_trades) * 100) if long_trades else 0
        metrics['short_win_rate'] = (len([t for t in short_trades if t['profit_loss'] > 0]) / len(short_trades) * 100) if short_trades else 0
        
        # Maximum drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        metrics['max_drawdown'] = max_drawdown
        
        return metrics


def run_model_on_data(model, env, data=None, max_steps=None, deterministic=True):
    """
    Run model on environment and collect trading data
    
    Parameters:
        model: Trained RL model
        env: Trading environment
        data: Data to use for evaluation
        max_steps: Maximum number of steps to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        Tuple of (portfolio_tracker, episodes_data)
    """
    # Create environment if needed
    if env is None:
        if data is None:
            raise ValueError("Either env or data must be provided")
        env = PatternSentimentEnv(
            data,
            normalize_observations=True,
            enable_adaptive_scaling=False
        )
    
    # Initialize tracker
    tracker = PortfolioTracker()
    
    # Collect episodes data
    episodes_data = []
    episode_data = []
    
    # Reset environment
    observation, _ = env.reset()
    tracker.reset()
    
    # Setup counters
    step_count = 0
    episode_count = 0
    
    # Run until we reach max steps or finish all data
    done = False
    while (max_steps is None or step_count < max_steps):
        # Get action from model
        action, _states = model.predict(observation, deterministic=deterministic)
        
        # Step environment
        next_observation, reward, done, truncated, info = env.step(action)
        
        # Get current price from environment
        current_price = env.data.iloc[env.index - 1]['close'] if env.index > 0 else env.data.iloc[0]['close']
        timestamp = env.data.index[env.index - 1] if env.index > 0 else env.data.index[0]
        
        # Update tracker
        tracker.update(
            timestamp=timestamp,
            price=current_price,
            position=env.position_size,
            action=action,
            reward=reward
        )
        
        # Record step data
        step_data = {
            'step': step_count,
            'episode': episode_count,
            'timestamp': timestamp,
            'action': action,
            'position': env.position_size,
            'price': current_price,
            'reward': reward,
            'portfolio_value': tracker.portfolio_values[-1]
        }
        
        episode_data.append(step_data)
        
        # Update observation
        observation = next_observation
        
        # Check if episode ended
        if done or truncated:
            # Save episode data
            episodes_data.append(pd.DataFrame(episode_data))
            
            # Reset for next episode
            episode_data = []
            episode_count += 1
            
            # Break if we're out of data
            if env.index >= len(env.data):
                break
                
            # Reset environment
            observation, _ = env.reset()
        
        step_count += 1
    
    # Combine all episode data
    all_data = pd.concat(episodes_data, ignore_index=True) if episodes_data else pd.DataFrame()
    
    return tracker, all_data


def plot_portfolio_performance(tracker, price_data=None, save_path=None):
    """
    Plot portfolio value over time with price chart
    
    Parameters:
        tracker: PortfolioTracker instance
        price_data: DataFrame with timestamp index and 'close' column
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Convert timestamps to datetime if they aren't already
    timestamps = [pd.to_datetime(ts) for ts in tracker.timestamps]
    
    # Plot price chart with trades
    if price_data is not None:
        ax1.plot(price_data.index, price_data['close'], color='black', alpha=0.7, label='Price')
        
        # Plot buy and sell markers
        for trade in tracker.trades:
            if trade['position'] > 0:  # Long trade
                ax1.scatter(trade['entry_time'], trade['entry_price'], color='green', marker='^', s=100)
                ax1.scatter(trade['exit_time'], trade['exit_price'], color='red', marker='v', s=100)
            else:  # Short trade
                ax1.scatter(trade['entry_time'], trade['entry_price'], color='red', marker='v', s=100)
                ax1.scatter(trade['exit_time'], trade['exit_price'], color='green', marker='^', s=100)
    
    # Plot portfolio value
    ax2.plot(timestamps, tracker.portfolio_values, label='Portfolio Value', color='blue')
    
    # Format x-axis
    if len(timestamps) > 0:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add labels and legends
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Add performance metrics as text
    metrics = tracker.get_performance_metrics()
    
    text = (
        f"Total Return: {metrics['total_return']:.2f}%\n"
        f"Trades: {metrics['total_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2f}%\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2f}%"
    )
    
    ax2.text(
        0.02, 0.95, text, transform=ax2.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Portfolio performance plot saved to {save_path}")
    
    plt.show()


def plot_action_distribution(actions_data, save_path=None):
    """
    Plot distribution of trading actions
    
    Parameters:
        actions_data: DataFrame with action data
        save_path: Path to save the plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Count actions by type (0=hold, 1=buy, 2=sell)
    action_counts = actions_data['action'].value_counts().sort_index()
    action_labels = ['Hold', 'Buy', 'Sell']
    
    # Ensure all actions are represented
    for i in range(3):
        if i not in action_counts.index:
            action_counts[i] = 0
    action_counts = action_counts.sort_index()
    
    # Plot action distribution
    axs[0, 0].bar(
        action_labels,
        action_counts.values,
        color=['gray', 'green', 'red']
    )
    axs[0, 0].set_title('Action Distribution')
    axs[0, 0].set_ylabel('Count')
    
    # Plot position over time
    positions_df = pd.DataFrame(actions_data)
    positions_df['timestamp'] = pd.to_datetime(positions_df['timestamp'])
    positions_df = positions_df.set_index('timestamp')
    
    axs[0, 1].plot(positions_df.index, positions_df['position'])
    axs[0, 1].set_title('Position Size Over Time')
    axs[0, 1].set_ylabel('Position Size')
    
    # Plot portfolio value over time
    axs[1, 0].plot(positions_df.index, positions_df['portfolio_value'])
    axs[1, 0].set_title('Portfolio Value Over Time')
    axs[1, 0].set_ylabel('Portfolio Value')
    
    # Plot reward distribution
    axs[1, 1].hist(positions_df['reward'], bins=30, alpha=0.7)
    axs[1, 1].set_title('Reward Distribution')
    axs[1, 1].set_xlabel('Reward')
    axs[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Action distribution plot saved to {save_path}")
    
    plt.show()


def plot_trade_analysis(tracker, save_path=None):
    """
    Plot detailed trade analysis
    
    Parameters:
        tracker: PortfolioTracker instance
        save_path: Path to save the plot
    """
    if not tracker.trades:
        print("No trades to analyze")
        return
    
    # Convert trades to DataFrame for analysis
    trades_df = pd.DataFrame(tracker.trades)
    
    # Set up the plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot return distribution
    axs[0, 0].hist(trades_df['return_pct'], bins=20, alpha=0.7, color='blue')
    axs[0, 0].axvline(x=0, color='red', linestyle='--')
    axs[0, 0].set_title('Trade Return Distribution')
    axs[0, 0].set_xlabel('Return (%)')
    axs[0, 0].set_ylabel('Frequency')
    
    # Plot profit/loss by trade
    trades_df['trade_number'] = range(1, len(trades_df) + 1)
    axs[0, 1].bar(
        trades_df['trade_number'],
        trades_df['profit_loss'],
        color=trades_df['profit_loss'].apply(lambda x: 'green' if x > 0 else 'red')
    )
    axs[0, 1].axhline(y=0, color='black', linestyle='-')
    axs[0, 1].set_title('Profit/Loss by Trade')
    axs[0, 1].set_xlabel('Trade Number')
    axs[0, 1].set_ylabel('Profit/Loss')
    
    # Plot cumulative return
    trades_df['cumulative_return'] = trades_df['return_pct'].cumsum()
    axs[1, 0].plot(trades_df['trade_number'], trades_df['cumulative_return'])
    axs[1, 0].set_title('Cumulative Return')
    axs[1, 0].set_xlabel('Trade Number')
    axs[1, 0].set_ylabel('Cumulative Return (%)')
    
    # Plot trade duration vs return
    try:
        axs[1, 1].scatter(
            trades_df['duration'],
            trades_df['return_pct'],
            alpha=0.7,
            c=trades_df['return_pct'].apply(lambda x: 'green' if x > 0 else 'red')
        )
        axs[1, 1].axhline(y=0, color='black', linestyle='-')
        axs[1, 1].set_title('Trade Duration vs Return')
        axs[1, 1].set_xlabel('Duration (minutes)')
        axs[1, 1].set_ylabel('Return (%)')
    except Exception as e:
        print(f"Error plotting trade duration: {e}")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Trade analysis plot saved to {save_path}")
    
    plt.show()
    
    # Print additional statistics
    metrics = tracker.get_performance_metrics()
    
    print("\n===== TRADE STATISTICS =====\n")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Profit per Winning Trade: ${metrics['avg_profit']:.2f}")
    print(f"Average Loss per Losing Trade: ${metrics['avg_loss']:.2f}")
    
    if metrics['long_trades'] > 0:
        print(f"\nLong Trades: {metrics['long_trades']}")
        print(f"Long Win Rate: {metrics['long_win_rate']:.2f}%")
    
    if metrics['short_trades'] > 0:
        print(f"\nShort Trades: {metrics['short_trades']}")
        print(f"Short Win Rate: {metrics['short_win_rate']:.2f}%")
    
    print(f"\nMaximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Final Portfolio Value: ${tracker.portfolio_values[-1]:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")


def analyze_trading_strategy(model_path, data=None, output_dir=None, episodes=1):
    """
    Analyze trading strategy and portfolio performance
    
    Parameters:
        model_path: Path to the trained model
        data: Optional evaluation data
        output_dir: Directory to save results
        episodes: Number of episodes to evaluate
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "Analysis" / "strategy"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Load evaluation data if not provided
    if data is None:
        print("Loading data from database...")
        data = load_data_from_db()
        # Use only validation portion (20%)
        split_idx = int(len(data) * 0.8)
        data = data[split_idx:]
    
    print(f"Using {len(data)} records for evaluation")
    
    # Create evaluation environment
    env = PatternSentimentEnv(
        data,
        normalize_observations=True,
        enable_adaptive_scaling=False
    )
    
    # Run model on environment
    print("\nRunning model on evaluation data...")
    tracker, actions_data = run_model_on_data(model, env, max_steps=None, deterministic=True)
    
    # Analyze portfolio performance
    print("\nAnalyzing portfolio performance...")
    plot_portfolio_performance(
        tracker,
        price_data=data[['close']],
        save_path=output_dir / "portfolio_performance.png"
    )
    
    # Analyze action distribution
    print("\nAnalyzing action distribution...")
    actions_df = pd.DataFrame([a for a in tracker.actions])
    plot_action_distribution(
        actions_df,
        save_path=output_dir / "action_distribution.png"
    )
    
    # Analyze trades
    print("\nAnalyzing trades...")
    plot_trade_analysis(
        tracker,
        save_path=output_dir / "trade_analysis.png"
    )
    
    # Save trade data
    if tracker.trades:
        trades_df = pd.DataFrame(tracker.trades)
        trades_df.to_csv(output_dir / "trades.csv", index=False)
        print(f"\nTrade data saved to {output_dir / 'trades.csv'}")
    
    # Save portfolio value history
    portfolio_df = pd.DataFrame({
        'timestamp': tracker.timestamps,
        'portfolio_value': tracker.portfolio_values
    })
    portfolio_df.to_csv(output_dir / "portfolio_values.csv", index=False)
    print(f"Portfolio value data saved to {output_dir / 'portfolio_values.csv'}")
    
    # Save metrics
    metrics = tracker.get_performance_metrics()
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "performance_metrics.csv", index=False)
    print(f"Performance metrics saved to {output_dir / 'performance_metrics.csv'}")
    
    print(f"\nTrading strategy analysis complete. Results saved to {output_dir}")
    
    return {
        "tracker": tracker,
        "actions_data": actions_df,
        "metrics": metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trading strategy of a trained RL model")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save analysis results")
    parser.add_argument('--episodes', type=int, default=1, help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    analyze_trading_strategy(
        model_path=args.model,
        output_dir=args.output_dir,
        episodes=args.episodes
    )
