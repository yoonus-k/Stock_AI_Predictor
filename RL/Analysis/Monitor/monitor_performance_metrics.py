"""
Performance Metrics Visualization and Analysis

This script visualizes and analyzes performance metrics from the RL trading model. 
It reads performance metrics from JSON files and generates comprehensive visualizations
and analytics to understand model performance across training iterations.

Features:
- Visualization of portfolio growth and returns
- Analysis of key performance indicators (Sharpe, drawdown, profit factor)
- Action distribution analysis
- Position sizing and risk-reward analysis
- Training evolution visualization
- Performance benchmarking against baseline strategies
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Set plot style
plt.style.use('ggplot')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_performance_metrics(file_path: str) -> Dict[str, Any]:
    """
    Load performance metrics from a JSON file.
    
    Args:
        file_path: Path to the performance metrics JSON file
        
    Returns:
        Dictionary containing performance metrics data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully loaded metrics from {file_path}")
        print(f"Number of timesteps: {len(data['metrics'])}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {"metrics": []}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON.")
        return {"metrics": []}


def extract_metrics_to_dataframe(metrics_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert metrics data to a pandas DataFrame for easier analysis.
    
    Args:
        metrics_data: Dictionary containing metrics data
        
    Returns:
        DataFrame with organized metrics
    """
    data_rows = []
    
    for idx, timestep_data in enumerate(metrics_data['metrics']):
        row = {
            'iteration': idx + 1,
            'timestep': timestep_data['timestep'],
            'date': timestep_data.get('date', ''),
            'final_portfolio_value': timestep_data['final_portfolio_value'],
            'return_pct': timestep_data['return_pct'],
            'win_rate': timestep_data.get('win_rate', 0),
            'sharpe_ratio': timestep_data.get('sharpe_ratio', 0),
            'max_drawdown': timestep_data.get('max_drawdown', 0),
            'profit_factor': timestep_data.get('profit_factor', 0),
            'trade_count': timestep_data.get('trade_count', 0),
            'buy_count': timestep_data.get('buy_count', 0), 
            'sell_count': timestep_data.get('sell_count', 0),
            'hold_count': timestep_data.get('hold_count', 0),
            'avg_profit_per_trade': timestep_data.get('avg_profit_per_trade', 0),
            'avg_loss_per_trade': timestep_data.get('avg_loss_per_trade', 0),
            'max_consecutive_wins': timestep_data.get('max_consecutive_wins', 0),
            'max_consecutive_losses': timestep_data.get('max_consecutive_losses', 0),
            'profitable_trades': timestep_data.get('profitable_trades', 0),
            'losing_trades': timestep_data.get('losing_trades', 0)
        }
        
        # Add action counts
        if 'action_counts' in timestep_data:
            for action, count in timestep_data['action_counts'].items():
                row[f'action_{action}_count'] = count
        
        # Add position sizes counts
        if 'position_sizes_counts' in timestep_data:
            for size, count in timestep_data['position_sizes_counts'].items():
                row[f'position_size_{size}'] = count
                
        # Add risk reward ratios counts
        if 'risk_reward_ratios_counts' in timestep_data:
            for ratio, count in timestep_data['risk_reward_ratios_counts'].items():
                row[f'risk_reward_{ratio}'] = count
        
        # Store only first 50 portfolio values to avoid making the dataframe too large
        if 'portfolio_values' in timestep_data:
            for i, value in enumerate(timestep_data['portfolio_values'][:50]):
                row[f'portfolio_value_{i}'] = value
                
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)


def plot_returns_evolution(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot the evolution of returns and key performance metrics across iterations.
    
    Args:
        metrics_df: DataFrame with metrics data
        save_path: Path to save the plot, if None, plot is displayed
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('RL Model Performance Evolution', fontsize=24, y=0.95)
    
    # Plot 1: Portfolio Value Evolution
    axes[0, 0].plot(metrics_df['timestep'], metrics_df['final_portfolio_value'], marker='o', 
                  linestyle='-', color='#1f77b4', linewidth=2)
    axes[0, 0].set_title('Portfolio Value Evolution', fontsize=18)
    axes[0, 0].set_xlabel('Training Timesteps', fontsize=14)
    axes[0, 0].set_ylabel('Portfolio Value ($)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=10000, linestyle='--', color='red', alpha=0.5, label='Initial Capital')
    axes[0, 0].legend()
    
    # Plot 2: Return Percentage
    axes[0, 1].plot(metrics_df['timestep'], metrics_df['return_pct'], marker='o', 
                   linestyle='-', color='#2ca02c', linewidth=2)
    axes[0, 1].set_title('Return Percentage Evolution', fontsize=18)
    axes[0, 1].set_xlabel('Training Timesteps', fontsize=14)
    axes[0, 1].set_ylabel('Return (%)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, linestyle='--', color='red', alpha=0.5)
    
    # Plot 3: Sharpe Ratio and Max Drawdown
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    line1, = ax1.plot(metrics_df['timestep'], metrics_df['sharpe_ratio'], marker='o', 
                     linestyle='-', color='#ff7f0e', label='Sharpe Ratio', linewidth=2)
    line2, = ax2.plot(metrics_df['timestep'], metrics_df['max_drawdown'] * 100, marker='s', 
                     linestyle='-', color='#d62728', label='Max Drawdown (%)', linewidth=2)
    
    ax1.set_title('Risk-Adjusted Performance Metrics', fontsize=18)
    ax1.set_xlabel('Training Timesteps', fontsize=14)
    ax1.set_ylabel('Sharpe Ratio', fontsize=14, color='#ff7f0e')
    ax2.set_ylabel('Max Drawdown (%)', fontsize=14, color='#d62728')
    ax1.grid(True, alpha=0.3)
    
    lines = [line1, line2]
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper left')
    
    # Plot 4: Win Rate and Profit Factor
    ax3 = axes[1, 1]
    ax4 = ax3.twinx()
    
    line3, = ax3.plot(metrics_df['timestep'], metrics_df['win_rate'], marker='o', 
                     linestyle='-', color='#9467bd', label='Win Rate (%)', linewidth=2)
    line4, = ax4.plot(metrics_df['timestep'], metrics_df['profit_factor'], marker='s', 
                     linestyle='-', color='#8c564b', label='Profit Factor', linewidth=2)
    
    ax3.set_title('Trading Efficiency Metrics', fontsize=18)
    ax3.set_xlabel('Training Timesteps', fontsize=14)
    ax3.set_ylabel('Win Rate (%)', fontsize=14, color='#9467bd')
    ax4.set_ylabel('Profit Factor', fontsize=14, color='#8c564b')
    ax3.grid(True, alpha=0.3)
    
    lines = [line3, line4]
    ax3.legend(lines, [line.get_label() for line in lines], loc='upper left')
    
    # Plot 5: Trade Counts (Buy, Sell, Hold)
    axes[2, 0].bar(metrics_df['timestep'] - 1000, metrics_df['buy_count'], width=1500, 
                 label='Buy', alpha=0.7, color='green')
    axes[2, 0].bar(metrics_df['timestep'], metrics_df['sell_count'], width=1500, 
                 label='Sell', alpha=0.7, color='red')
    axes[2, 0].bar(metrics_df['timestep'] + 1000, metrics_df['hold_count'], width=1500, 
                 label='Hold', alpha=0.7, color='blue')
    
    axes[2, 0].set_title('Action Distribution', fontsize=18)
    axes[2, 0].set_xlabel('Training Timesteps', fontsize=14)
    axes[2, 0].set_ylabel('Count', fontsize=14)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Plot 6: Average Profit/Loss per Trade
    ax5 = axes[2, 1]
    width = 0.35
    x = np.arange(len(metrics_df))
    
    ax5.bar(metrics_df['timestep'] - 1000, metrics_df['avg_profit_per_trade'], 
          width=2000, label='Avg Profit per Trade', alpha=0.7, color='green')
    ax5.bar(metrics_df['timestep'] + 1000, metrics_df['avg_loss_per_trade'], 
          width=2000, label='Avg Loss per Trade', alpha=0.7, color='red')
    
    ax5.set_title('Average Trade Performance', fontsize=18)
    ax5.set_xlabel('Training Timesteps', fontsize=14)
    ax5.set_ylabel('Average Value', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved returns evolution plot to {save_path}")
    else:
        plt.show()



def plot_portfolio_growth_examples(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot detailed examples of portfolio growth for selected iterations.
    
    Args:
        metrics_df: DataFrame with metrics data
        save_path: Path to save the plot, if None, plot is displayed
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Portfolio Growth Analysis (Sample Iterations)', fontsize=24, y=0.95)
    
    # Select iterations with different characteristics
    # For this example, we'll use the first, last, best performing, and worst performing iterations
    
    # Best performing (highest return)
    best_idx = metrics_df['return_pct'].idxmax()
    best_timestep = metrics_df.loc[best_idx, 'timestep']
    best_return = metrics_df.loc[best_idx, 'return_pct']
    
    # Worst performing (lowest return)
    worst_idx = metrics_df['return_pct'].idxmin()
    worst_timestep = metrics_df.loc[worst_idx, 'timestep']
    worst_return = metrics_df.loc[worst_idx, 'return_pct']
    
    # First iteration
    first_timestep = metrics_df.iloc[0]['timestep']
    first_return = metrics_df.iloc[0]['return_pct']
    
    # Last iteration
    last_timestep = metrics_df.iloc[-1]['timestep']
    last_return = metrics_df.iloc[-1]['return_pct']
    
    # Extract portfolio values for each selected iteration
    iterations = [
        ('First Iteration', first_timestep, first_return, 0, 0),
        ('Best Performing', best_timestep, best_return, 0, 1),
        ('Worst Performing', worst_timestep, worst_return, 1, 0),
        ('Latest Iteration', last_timestep, last_return, 1, 1)
    ]
    
    for (title, timestep, return_pct, row, col) in iterations:
        # Find the corresponding row in metrics_df
        row_data = metrics_df[metrics_df['timestep'] == timestep].iloc[0]
        
        # Extract portfolio values
        portfolio_values = []
        for i in range(50):  # We stored only 50 portfolio values in the DataFrame
            col_name = f'portfolio_value_{i}'
            if col_name in row_data:
                portfolio_values.append(row_data[col_name])
        
        # Plot portfolio growth
        axes[row, col].plot(range(len(portfolio_values)), portfolio_values, marker='', 
                          linestyle='-', color='#2ca02c', linewidth=2)
        axes[row, col].set_title(f'{title} (Timestep: {timestep}, Return: {return_pct:.2f}%)', fontsize=16)
        axes[row, col].set_xlabel('Trading Steps', fontsize=14)
        axes[row, col].set_ylabel('Portfolio Value ($)', fontsize=14)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].axhline(y=10000, linestyle='--', color='red', alpha=0.5, label='Initial Capital')
        axes[row, col].legend()
        
        # Highlight key points
        max_value = max(portfolio_values)
        min_value = min(portfolio_values)
        final_value = portfolio_values[-1]
        
        max_idx = portfolio_values.index(max_value)
        min_idx = portfolio_values.index(min_value);
        
        axes[row, col].plot(max_idx, max_value, 'go', markersize=10)
        axes[row, col].plot(min_idx, min_value, 'ro', markersize=10)
        axes[row, col].plot(len(portfolio_values)-1, final_value, 'bo', markersize=10)
        
        axes[row, col].annotate(f'Max: ${max_value:.2f}', xy=(max_idx, max_value), 
                              xytext=(max_idx+1, max_value+300), 
                              arrowprops=dict(facecolor='green', shrink=0.05))
        
        axes[row, col].annotate(f'Min: ${min_value:.2f}', xy=(min_idx, min_value), 
                              xytext=(min_idx+1, min_value-300), 
                              arrowprops=dict(facecolor='red', shrink=0.05))
        
        axes[row, col].annotate(f'Final: ${final_value:.2f}', xy=(len(portfolio_values)-1, final_value), 
                              xytext=(len(portfolio_values)-10, final_value+300), 
                              arrowprops=dict(facecolor='blue', shrink=0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved portfolio growth examples plot to {save_path}")
    else:
        plt.show()


def plot_performance_comparison(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a performance comparison radar chart showing the tradeoff between metrics.
    
    Args:
        metrics_df: DataFrame with metrics data
        save_path: Path to save the plot, if None, plot is displayed
    """
    # Get the data for the first, last, and best performing iterations
    first_data = metrics_df.iloc[0]
    last_data = metrics_df.iloc[-1]
    best_idx = metrics_df['return_pct'].idxmax()
    best_data = metrics_df.loc[best_idx]
    
    # Metrics to compare
    metrics = [
        'return_pct', 'win_rate', 'sharpe_ratio', 
        'max_drawdown', 'profit_factor'
    ]
    
    # Normalize metrics for radar chart
    max_values = {
        'return_pct': max(120, metrics_df['return_pct'].max()),
        'win_rate': 100,
        'sharpe_ratio': max(15, metrics_df['sharpe_ratio'].max()),
        'max_drawdown': 0.3,  # Lower is better, so we'll invert it
        'profit_factor': max(20, metrics_df['profit_factor'].max())
    }
    
    # Prepare data
    first_values = [first_data[m] / max_values[m] for m in metrics]
    last_values = [last_data[m] / max_values[m] for m in metrics]
    best_values = [best_data[m] / max_values[m] for m in metrics]
    
    # Invert max_drawdown since lower is better
    first_values[3] = 1 - first_values[3]
    last_values[3] = 1 - last_values[3]
    best_values[3] = 1 - best_values[3]
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    first_values += first_values[:1]
    last_values += last_values[:1]
    best_values += best_values[:1]
    
    metrics_labels = ['Return %', 'Win Rate', 'Sharpe Ratio', 'Low Drawdown', 'Profit Factor']
    metrics_labels += metrics_labels[:1]
    
    # Plot data
    ax.plot(angles, first_values, 'o-', linewidth=2, label=f'First ({first_data["timestep"]} steps)')
    ax.plot(angles, last_values, 'o-', linewidth=2, label=f'Latest ({last_data["timestep"]} steps)')
    ax.plot(angles, best_values, 'o-', linewidth=2, label=f'Best ({best_data["timestep"]} steps)')
    
    # Fill areas
    ax.fill(angles, first_values, alpha=0.1)
    ax.fill(angles, last_values, alpha=0.1)
    ax.fill(angles, best_values, alpha=0.1)
    
    # Set labels and title
    ax.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, metrics_labels[:-1])
    plt.title('Performance Metrics Comparison', size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set performance metric values
    for i, metric in enumerate(metrics_labels[:-1]):
        ax.text(angles[i], 1.15, 
                f"{metric}:\nFirst: {first_data[metrics[i]]:.2f}\nLatest: {last_data[metrics[i]]:.2f}\nBest: {best_data[metrics[i]]:.2f}", 
                horizontalalignment='center', size=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {save_path}")
    else:
        plt.show()


def plot_win_rate_profit_factor_analysis(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create plots to analyze the relationship between win rate, profit factor, and returns.
    
    Args:
        metrics_df: DataFrame with metrics data
        save_path: Path to save the plot, if None, plot is displayed
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Win Rate and Profit Factor Analysis', fontsize=24, y=0.95)
    
    # Plot 1: Win Rate vs Profit Factor
    scatter = axes[0, 0].scatter(metrics_df['win_rate'], metrics_df['profit_factor'], 
                               s=metrics_df['return_pct']*3, c=metrics_df['return_pct'], 
                               cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Win Rate vs Profit Factor', fontsize=18)
    axes[0, 0].set_xlabel('Win Rate (%)', fontsize=14)
    axes[0, 0].set_ylabel('Profit Factor', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0, 0])
    cbar.set_label('Return (%)', fontsize=12)
    
    # Annotate points
    for i, row in metrics_df.iterrows():
        axes[0, 0].annotate(f"{row['timestep']/1000:.0f}K", 
                          (row['win_rate'], row['profit_factor']),
                          textcoords="offset points", 
                          xytext=(0,10), 
                          ha='center')
    
    # Plot 2: Win Rate vs Return
    axes[0, 1].scatter(metrics_df['win_rate'], metrics_df['return_pct'], 
                     s=metrics_df['trade_count']/2, c=metrics_df['sharpe_ratio'], 
                     cmap='plasma', alpha=0.7)
    axes[0, 1].set_title('Win Rate vs Return', fontsize=18)
    axes[0, 1].set_xlabel('Win Rate (%)', fontsize=14)
    axes[0, 1].set_ylabel('Return (%)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot linear trendline
    z = np.polyfit(metrics_df['win_rate'], metrics_df['return_pct'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(metrics_df['win_rate'], p(metrics_df['win_rate']), "r--", 
                  alpha=0.8, label=f"Trend: y={z[0]:.2f}x{z[1]:+.2f}")
    axes[0, 1].legend()
    
    # Plot 3: Profit Factor vs Return
    axes[1, 0].scatter(metrics_df['profit_factor'], metrics_df['return_pct'], 
                     s=metrics_df['trade_count']/2, c=metrics_df['max_drawdown'], 
                     cmap='RdYlGn_r', alpha=0.7)
    axes[1, 0].set_title('Profit Factor vs Return', fontsize=18)
    axes[1, 0].set_xlabel('Profit Factor', fontsize=14)
    axes[1, 0].set_ylabel('Return (%)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Trade Count vs Win Rate
    axes[1, 1].scatter(metrics_df['trade_count'], metrics_df['win_rate'], 
                     s=metrics_df['return_pct']*2, c=metrics_df['timestep'], 
                     cmap='viridis', alpha=0.7)
    axes[1, 1].set_title('Trade Count vs Win Rate', fontsize=18)
    axes[1, 1].set_xlabel('Trade Count', fontsize=14)
    axes[1, 1].set_ylabel('Win Rate (%)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add timestep annotations
    for i, row in metrics_df.iterrows():
        axes[1, 1].annotate(f"{row['timestep']/1000:.0f}K", 
                          (row['trade_count'], row['win_rate']),
                          textcoords="offset points", 
                          xytext=(0,10), 
                          ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.25)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved win rate and profit factor analysis plot to {save_path}")
    else:
        plt.show()



def analyze_performance_metrics(metrics_file: str, output_dir: Optional[str] = None):
    """
    Main function to analyze performance metrics.
    
    Args:
        metrics_file: Path to the performance metrics JSON file
        output_dir: Directory to save outputs, if None, plots are displayed
    """
    # Load data
    metrics_data = load_performance_metrics(metrics_file)
    
    # Convert to DataFrame
    metrics_df = extract_metrics_to_dataframe(metrics_data)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("Generating performance evolution plot...")
    plot_returns_evolution(
        metrics_df, 
        os.path.join(output_dir, 'returns_evolution.png') if output_dir else None
    )

    
    print("Generating portfolio growth examples plot...")
    plot_portfolio_growth_examples(
        metrics_df, 
        os.path.join(output_dir, 'portfolio_growth_examples.png') if output_dir else None
    )
    
    print("Generating performance comparison plot...")
    plot_performance_comparison(
        metrics_df, 
        os.path.join(output_dir, 'performance_comparison.png') if output_dir else None
    )
    
    print("Generating win rate and profit factor analysis plot...")
    plot_win_rate_profit_factor_analysis(
        metrics_df, 
        os.path.join(output_dir, 'win_rate_profit_factor_analysis.png') if output_dir else None
    )
    

    print("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze RL trading model performance metrics')
    parser.add_argument('--metrics-file', type=str, default='RL/Logs/enhanced_continued/Metrics/performance_metrics.json',
                      help='Path to the performance metrics JSON file')
    parser.add_argument('--output-dir', type=str, default='Images/RL/Performance',
                      help='Directory to save output files')
    
    args = parser.parse_args()
    
    metrics_file = os.path.join(project_root, args.metrics_file)
    output_dir = os.path.join(project_root, args.output_dir)
    
    analyze_performance_metrics(metrics_file, output_dir)
