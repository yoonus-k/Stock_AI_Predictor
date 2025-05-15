"""
Performance Metrics Module

This module calculates and stores performance metrics for backtesting results.
It provides standardized metrics calculation and database storage.

Usage:
    Import this module to calculate and store performance metrics in backtests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from datetime import datetime

from Data.Database.db import Database


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Identification
    stock_id: int
    timeframe_id: int
    config_id: int
    start_date: str
    end_date: str
    recognition_technique: str
    
    # Trade statistics
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Risk/Reward metrics
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Additional metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stock_id': self.stock_id,
            'timeframe_id': self.timeframe_id,
            'config_id': self.config_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'recognition_technique': self.recognition_technique,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio,
            'avg_trade_duration': self.avg_trade_duration
        }
    
    def to_metrics_dict(self) -> Dict[str, float]:
        """Convert to metrics dictionary for plotting."""
        return {
            'Total Return (%)': self.total_return_pct,
            'Win Rate (%)': self.win_rate,
            'Profit Factor': self.profit_factor,
            'Sharpe Ratio': self.sharpe_ratio,
            'Max Drawdown (%)': -self.max_drawdown,  # Negate for plotting
            'Sortino Ratio': self.sortino_ratio
        }
    
    def print_summary(self) -> None:
        """Print a summary of the performance metrics."""
        print("\n=== Performance Metrics Summary ===")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Recognition Technique: {self.recognition_technique}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.win_rate:.2f}%")
        print(f"Total Return: {self.total_return_pct:.2f}%")
        print(f"Annualized Return: {self.annualized_return_pct:.2f}%")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"Average Win: {self.avg_win:.2f}%")
        print(f"Average Loss: {self.avg_loss:.2f}%")
        print(f"Max Consecutive Wins: {self.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {self.max_consecutive_losses}")
        print("=====================================")


def calculate_performance_metrics(
    trades_df: pd.DataFrame,
    equity_curve: List[float],
    returns: np.array,
    stock_id: int,
    timeframe_id: int,
    config_id: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    recognition_technique: str
) -> PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        trades_df: DataFrame with trade results
        equity_curve: List of equity values
        returns: Array of daily returns
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        config_id: Configuration ID
        start_date: Start date of the backtest
        end_date: End date of the backtest
        recognition_technique: Name of the recognition technique used
        
    Returns:
        PerformanceMetrics object
    """
    # Basic trade statistics
    num_trades = len(trades_df)
    num_wins = len(trades_df[trades_df['outcome'] == 'win'])
    num_losses = len(trades_df[trades_df['outcome'] == 'loss'])
    win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
    
    # Average win/loss
    avg_win = trades_df[trades_df['outcome'] == 'win']['return_pct'].mean() if num_wins > 0 else 0
    avg_loss = trades_df[trades_df['outcome'] == 'loss']['return_pct'].mean() if num_losses > 0 else 0
    
    # Profit factor
    gross_profit = trades_df[trades_df['outcome'] == 'win']['profit_loss'].sum() if num_wins > 0 else 0
    gross_loss = abs(trades_df[trades_df['outcome'] == 'loss']['profit_loss'].sum()) if num_losses > 0 else 1e-6
    profit_factor = gross_profit / gross_loss
    
    # Calculate consecutive wins/losses
    outcomes = trades_df['outcome'].tolist()
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_consecutive_wins = 0
    current_consecutive_losses = 0
    
    for outcome in outcomes:
        if outcome == 'win':
            current_consecutive_wins += 1
            current_consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
        else:
            current_consecutive_losses += 1
            current_consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
    
    # Drawdown calculations
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min() * 100)
    
    # Return calculations
    total_return_pct = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    
    # Calculate annualized return
    days = (end_date - start_date).days
    years = days / 365
    annualized_return = ((1 + total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Sharpe and Sortino ratios
    daily_returns = pd.Series(returns).dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
    
    risk_free_rate = 0.02 / 252  # Daily risk-free rate (assuming 2% annual)
    excess_returns = daily_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # Sortino ratio uses only negative returns for denominator
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
    
    # Average trade duration
    avg_trade_duration = trades_df['duration'].mean() if 'duration' in trades_df.columns else 0
    
    # Create metrics object
    metrics = PerformanceMetrics(
        stock_id=stock_id,
        timeframe_id=timeframe_id,
        config_id=config_id,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        recognition_technique=recognition_technique,
        total_trades=num_trades,
        win_count=num_wins,
        loss_count=num_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown_pct,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return,
        volatility=volatility * 100,  # Convert to percentage
        calmar_ratio=calmar_ratio,
        avg_trade_duration=avg_trade_duration
    )
    
    return metrics


def store_performance_metrics(db: Database, metrics: PerformanceMetrics) -> int:
    """
    Store performance metrics in the database.
    
    Args:
        db: Database connection
        metrics: PerformanceMetrics object
        
    Returns:
        ID of the stored metrics record
    """
    # Convert metrics to dict for storage
    metrics_dict = metrics.to_dict()
    
    # Check if this config has been tested before
    cursor = db.connection.cursor()
    cursor.execute(
        """SELECT metric_id FROM performance_metrics 
           WHERE stock_id = ? AND timeframe_id = ? AND config_id = ? AND start_date = ? AND end_date = ?""",
        (metrics.stock_id, metrics.timeframe_id, metrics.config_id, metrics.start_date, metrics.end_date)
    )
    existing_id = cursor.fetchone()
    
    if existing_id:
        # Update existing record
        metric_id = existing_id[0]
        
        # Build update statement
        set_clause = ", ".join([f"{key} = ?" for key in metrics_dict.keys()])
        values = list(metrics_dict.values())
        values.append(metric_id)
        
        cursor.execute(
            f"UPDATE performance_metrics SET {set_clause} WHERE metric_id = ?",
            values
        )
        db.connection.commit()
        
        return metric_id
    else:
        # Insert new record
        columns = ", ".join(metrics_dict.keys())
        placeholders = ", ".join(["?"] * len(metrics_dict))
        values = list(metrics_dict.values())
        
        cursor.execute(
            f"INSERT INTO performance_metrics ({columns}) VALUES ({placeholders})",
            values
        )
        db.connection.commit()
        
        return cursor.lastrowid


def compare_techniques(
    metrics_list: List[PerformanceMetrics],
    title: str = "Comparison of Recognition Techniques",
    save_path: Optional[str] = None
) -> None:
    """
    Compare performance metrics across different recognition techniques.
    
    Args:
        metrics_list: List of PerformanceMetrics objects to compare
        title: Title for the chart
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    # Selected metrics for comparison
    metrics_to_compare = ['Total Return (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Profit Factor']
    
    # Extract values for each metric across all techniques
    techniques = [m.recognition_technique for m in metrics_list]
    values = {}
    
    for metric in metrics_to_compare:
        values[metric] = []
        for m in metrics_list:
            metrics_dict = m.to_metrics_dict()
            if metric in metrics_dict:
                values[metric].append(metrics_dict[metric])
            else:
                values[metric].append(0)  # Default value
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    # Plot each metric
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
    
    for i, metric in enumerate(metrics_to_compare):
        bars = axs[i].bar(techniques, values[metric], color=colors[i])
        axs[i].set_title(metric, fontsize=12)
        axs[i].grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add values
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.2f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_performance_report(
    metrics: PerformanceMetrics,
    trades_df: pd.DataFrame,
    equity_curve: List[float],
    test_dates: pd.DatetimeIndex,
    save_path: Optional[str] = None
) -> str:
    """
    Create a comprehensive performance report.
    
    Args:
        metrics: PerformanceMetrics object
        trades_df: DataFrame with trade details
        equity_curve: List of equity values
        test_dates: DatetimeIndex of test dates
        save_path: Optional path to save the report
        
    Returns:
        HTML report as a string
    """
    import matplotlib.pyplot as plt
    import io
    import base64
    from matplotlib.figure import Figure
    
    # Create a report template
    report_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333366; }
            .container { display: flex; flex-wrap: wrap; }
            .card { 
                background-color: #f9f9f9; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 10px; 
                flex: 1 1 300px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric { font-weight: bold; }
            .chart { margin: 20px 0; max-width: 100%; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Backtest Performance Report</h1>
        <p><b>Stock ID:</b> {stock_id} | <b>Timeframe ID:</b> {timeframe_id} | <b>Date Range:</b> {start_date} - {end_date}</p>
        <p><b>Recognition Technique:</b> {recognition_technique} | <b>Config ID:</b> {config_id}</p>
        
        <h2>Performance Summary</h2>
        <div class="container">
            <div class="card">
                <p><span class="metric">Total Return:</span> {total_return_pct:.2f}%</p>
                <p><span class="metric">Annualized Return:</span> {annualized_return_pct:.2f}%</p>
                <p><span class="metric">Volatility:</span> {volatility:.2f}%</p>
                <p><span class="metric">Sharpe Ratio:</span> {sharpe_ratio:.2f}</p>
                <p><span class="metric">Sortino Ratio:</span> {sortino_ratio:.2f}</p>
                <p><span class="metric">Max Drawdown:</span> {max_drawdown:.2f}%</p>
                <p><span class="metric">Calmar Ratio:</span> {calmar_ratio:.2f}</p>
            </div>
            <div class="card">
                <p><span class="metric">Total Trades:</span> {total_trades}</p>
                <p><span class="metric">Win Rate:</span> {win_rate:.2f}%</p>
                <p><span class="metric">Avg Win:</span> {avg_win:.2f}%</p>
                <p><span class="metric">Avg Loss:</span> {avg_loss:.2f}%</p>
                <p><span class="metric">Profit Factor:</span> {profit_factor:.2f}</p>
                <p><span class="metric">Max Consecutive Wins:</span> {max_consecutive_wins}</p>
                <p><span class="metric">Max Consecutive Losses:</span> {max_consecutive_losses}</p>
            </div>
        </div>
        
        <h2>Equity Curve</h2>
        <div class="chart">
            <img src="data:image/png;base64,{equity_curve_img}" width="100%">
        </div>
        
        <h2>Trade Distribution</h2>
        <div class="chart">
            <img src="data:image/png;base64,{trade_dist_img}" width="100%">
        </div>
        
        <h2>Monthly Returns (%)</h2>
        <div class="chart">
            <img src="data:image/png;base64,{monthly_returns_img}" width="100%">
        </div>
        
        <h2>Recent Trades</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Type</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Return</th>
                <th>Outcome</th>
                <th>Reason</th>
                <th>Duration</th>
            </tr>
            {trade_rows}
        </table>
        
        <p>Report generated on {generation_date}</p>
    </body>
    </html>
    """
    
    # Create equity curve chart
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    # 1. Equity curve
    fig_equity = Figure(figsize=(10, 6))
    ax = fig_equity.add_subplot(111)
    ax.plot(test_dates[:len(equity_curve)], equity_curve, 'b-')
    ax.set_title('Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True)
    equity_curve_img = fig_to_base64(fig_equity)
    
    # 2. Trade distribution
    fig_dist = Figure(figsize=(10, 6))
    ax = fig_dist.add_subplot(111)
    if 'type' in trades_df.columns and 'return_pct' in trades_df.columns:
        # Buy trades
        if 'BUY' in trades_df['type'].values:
            buy_returns = trades_df[trades_df['type'] == 'BUY']['return_pct']
            ax.hist(buy_returns, bins=20, alpha=0.5, label='Buy Trades')
        
        # Sell trades
        if 'SELL' in trades_df['type'].values:
            sell_returns = trades_df[trades_df['type'] == 'SELL']['return_pct']
            ax.hist(sell_returns, bins=20, alpha=0.5, label='Sell Trades')
        
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('Trade Return Distribution')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
    trade_dist_img = fig_to_base64(fig_dist)
    
    # 3. Monthly returns heatmap
    if 'entry_time' in trades_df.columns:
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
        trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
        
        # Group by year and month
        monthly_returns = trades_df.groupby(['year', 'month'])['return_pct'].sum().unstack()
        
        fig_monthly = Figure(figsize=(10, 6))
        ax = fig_monthly.add_subplot(111)
        
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        im = ax.imshow(monthly_returns.values, cmap=cmap, aspect='auto')
        
        # Add colorbar
        fig_monthly.colorbar(im, ax=ax, label='Return (%)')
        
        # Configure axis
        ax.set_xticks(np.arange(len(monthly_returns.columns)))
        ax.set_yticks(np.arange(len(monthly_returns.index)))
        ax.set_xticklabels(monthly_returns.columns)
        ax.set_yticklabels(monthly_returns.index)
        
        ax.set_title('Monthly Returns (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Add text annotations with the returns
        for i in range(len(monthly_returns.index)):
            for j in range(len(monthly_returns.columns)):
                if not pd.isna(monthly_returns.values[i, j]):
                    ax.text(j, i, f"{monthly_returns.values[i, j]:.1f}",
                           ha="center", va="center", color="black")
        
        monthly_returns_img = fig_to_base64(fig_monthly)
    else:
        # Create a blank chart if no data
        fig_monthly = Figure(figsize=(10, 6))
        ax = fig_monthly.add_subplot(111)
        ax.text(0.5, 0.5, "Monthly data not available", ha='center', va='center')
        ax.set_axis_off()
        monthly_returns_img = fig_to_base64(fig_monthly)
    
    # Generate trade rows
    trade_rows = ""
    for idx, trade in trades_df.iloc[-20:].iterrows():  # Show last 20 trades
        trade_rows += f"""
        <tr>
            <td>{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d')}</td>
            <td>{trade['type']}</td>
            <td>{trade['entry_price']:.4f}</td>
            <td>{trade['exit_price']:.4f}</td>
            <td>{trade['return_pct']:.2f}%</td>
            <td>{trade['outcome']}</td>
            <td>{trade['reason']}</td>
            <td>{trade['duration']}</td>
        </tr>
        """
    
    # Fill in the template
    report_html = report_template.format(
        stock_id=metrics.stock_id,
        timeframe_id=metrics.timeframe_id,
        start_date=metrics.start_date,
        end_date=metrics.end_date,
        recognition_technique=metrics.recognition_technique,
        config_id=metrics.config_id,
        total_return_pct=metrics.total_return_pct,
        annualized_return_pct=metrics.annualized_return_pct,
        volatility=metrics.volatility,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        max_drawdown=metrics.max_drawdown,
        calmar_ratio=metrics.calmar_ratio,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        avg_win=metrics.avg_win,
        avg_loss=metrics.avg_loss,
        profit_factor=metrics.profit_factor,
        max_consecutive_wins=metrics.max_consecutive_wins,
        max_consecutive_losses=metrics.max_consecutive_losses,
        equity_curve_img=equity_curve_img,
        trade_dist_img=trade_dist_img,
        monthly_returns_img=monthly_returns_img,
        trade_rows=trade_rows,
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    
    # Save the report if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_html)
    
    return report_html
