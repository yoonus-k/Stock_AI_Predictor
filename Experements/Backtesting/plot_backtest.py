"""
Backtest Visualization Module

This module provides visualization capabilities for backtest results.
It includes functions for plotting performance metrics, trade distribution,
and comparative trade analysis.

Usage:
    Import this module to create standardized visualization charts
    for backtest analysis results.
"""

import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_performance_metrics(metrics: Dict[str, float], title: str = "Model Performance Metrics", 
                           save_path: Optional[str] = None) -> None:
    """
    Creates a horizontal bar chart of performance metrics.
    
    Args:
        metrics (dict): Dictionary with metric names as keys and values as values
        title (str): Title for the chart
        save_path (str, optional): Path to save the figure, if None just displays
    """
    # Define colors for visual appeal
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#F44336']
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    plt.title(f"📊 {title}", fontsize=14)
    plt.xlabel("Metric Value")
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Add values to bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.2f}', va='center', ha='left')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_exit_distribution(labels: List[str], sizes: List[int], 
                          title: str = "Trade Exit Reason Distribution",
                          save_path: Optional[str] = None) -> None:
    """
    Creates a donut chart of trade exit reasons.
    
    Args:
        labels (list): List of exit reason labels
        sizes (list): List of counts for each exit reason
        title (str): Title for the chart
        save_path (str, optional): Path to save the figure, if None just displays
    """
    # Define colors for consistent visual identity
    colors = ['#00C49A', '#FF6F61', '#FFA500']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, startangle=140, 
            wedgeprops={'width': 0.4}, autopct='%1.1f%%')
    plt.title(f"🎯 {title}")
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_trade_type_analysis(trade_types: List[str], trades: List[int], win_rates: List[float],
                           title: str = "Buy vs Sell Trade Analysis",
                           save_path: Optional[str] = None) -> None:
    """
    Creates a combined bar and line chart for analyzing trade types.
    
    Args:
        trade_types (list): List of trade type labels (e.g., 'BUY', 'SELL')
        trades (list): Number of trades for each type
        win_rates (list): Win rate percentage for each type
        title (str): Title for the chart
        save_path (str, optional): Path to save the figure, if None just displays
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar = ax1.bar(trade_types, trades, color="#4DB6AC", label="Number of Trades")
    ax2 = ax1.twinx()
    line = ax2.plot(trade_types, win_rates, color="#FF7043", marker='o', label="Win Rate (%)")

    # Add labels and title
    ax1.set_ylabel("Number of Trades")
    ax2.set_ylabel("Win Rate (%)")
    ax1.set_title(f"📈 {title}")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage demonstration
if __name__ == "__main__":
    # Sample data for demonstration
    # In a real scenario, these would come from backtest results
    
    # 1. Performance metrics visualization
    metrics = {
        'Total Return': 68.57,
        'Win Rate': 53.51,
        'Profit Factor': 1.31,
        'Sharpe Ratio': 0.78,
        'Max Drawdown': -9.27
    }
    plot_performance_metrics(metrics, title="Model Forward Test Metrics Summary (2024)")
    
    # 2. Exit reason distribution
    labels = ['TP Hit', 'SL Hit', 'Hold Exit']
    sizes = [553, 289, 225]
    plot_exit_distribution(labels, sizes)
    
    # 3. Trade type analysis
    trade_types = ['BUY', 'SELL']
    trades = [869, 198]
    win_rates = [54.89, 47.47]
    plot_trade_type_analysis(trade_types, trades, win_rates)




