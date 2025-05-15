"""
Advanced Backtesting Module

This module implements an advanced backtesting framework for trading strategies
based on pattern recognition and cluster matching techniques.
It supports multiple recognition techniques, trade strategies, and performance metrics.

Usage:
    from Experements.Backtesting.backtest_v3 import run_backtest
    
    # Run a backtest with default parameters
    results = run_backtest(
        db=db,
        stock_id=1,
        timeframe_id=2,
        train_start=pd.Timestamp("2019-01-01"),
        train_end=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-01-02"),
        test_end=pd.Timestamp("2023-12-31"),
        recognition_technique="svm"
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))

# Import custom modules
from Data.Database.db import Database
from Experements.Backtesting.backtest_config import BacktestConfig, RecognitionTechnique, ExitStrategy
from Experements.Backtesting.pattern_recognition import (
    extract_pips, create_recognizer, pattern_matcher, PatternRecognizer
)
from Experements.Backtesting.trade_strategies import (
    TradeStrategy, PatternBasedStrategy, VotingEnsembleStrategy,
    TradeType, ExitReason, simulate_trade
)
from Experements.Backtesting.performance_metrics import (
    calculate_performance_metrics, store_performance_metrics,
    PerformanceMetrics, create_performance_report
)
from Experements.Backtesting.database_schema import setup_all_tables


class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, db: Database, config: BacktestConfig):
        """
        Initialize the backtester.
        
        Args:
            db: Database connection
            config: BacktestConfig object
        """
        self.db = db
        self.config = config
        
        # Set up database tables
        setup_all_tables(db.connection)
        
        # Load data
        self.train_df = None
        self.test_df = None
        self.clusters_df = None
        
        # Initialize components
        self.pattern_recognizer = None
        self.trade_strategy = None
        
        # Results
        self.trades_df = None
        self.equity_curve = None
        self.returns = None
        self.metrics = None
        
    def load_data(self) -> None:
        """Load and prepare data for backtesting."""
        print(f"Loading data for Stock ID {self.config.stock_id}, Timeframe ID {self.config.timeframe_id}")
        
        # Get stock data
        df = self.db.get_stock_data(self.config.stock_id, self.config.timeframe_id)
        
        if df.empty:
            raise ValueError(f"No data found for Stock ID {self.config.stock_id}, Timeframe ID {self.config.timeframe_id}")
        
        # Split into train and test sets
        self.train_df = df[(df.index >= self.config.train_start) & (df.index <= self.config.train_end)]
        self.test_df = df[(df.index >= self.config.test_start) & (df.index <= self.config.test_end)]
        
        if self.train_df.empty:
            raise ValueError(f"No training data found for the specified date range")
        
        if self.test_df.empty:
            raise ValueError(f"No test data found for the specified date range")
        
        print(f"Loaded {len(self.train_df)} training records and {len(self.test_df)} test records")
        
        # Load cluster data from DB
        self.clusters_df = self.db.get_clusters_by_stock_id(self.config.stock_id)
        
        if self.clusters_df.empty:
            raise ValueError(f"No clusters found for Stock ID {self.config.stock_id}")
        
        print(f"Loaded {len(self.clusters_df)} clusters")
    
    def setup_pattern_recognizer(self) -> None:
        """Set up the pattern recognizer based on configuration."""
        print(f"Setting up {self.config.recognition_technique.value} pattern recognizer")
        
        # Create recognizer based on technique
        self.pattern_recognizer = create_recognizer(
            self.config.recognition_technique.value,
            **self.config.model_params
        )
        
        # Prepare cluster data for training
        cluster_features = self.clusters_df['AVGPricePoints'].values
        cluster_features = np.array([np.array(x.split(','), dtype=float) for x in cluster_features])
        labels = np.array([i for i in range(len(cluster_features))])
        
        # Train the recognizer
        self.pattern_recognizer.train(cluster_features, labels)
        
        print(f"Pattern recognizer trained with {len(labels)} clusters")
    
    def setup_trade_strategy(self) -> None:
        """Set up the trade strategy based on configuration."""
        print(f"Setting up trade strategy with {self.config.exit_strategy.value} exit strategy")
        
        # Create a pattern matcher function
        def matcher_function(data, i):
            window = data['ClosePrice'].values[i - self.config.lookback:i + 1]
            return pattern_matcher(
                self.pattern_recognizer,
                self.clusters_df,
                window,
                n_pips=self.config.n_pips,
                dist_type=self.config.distance_measure,
                mse_threshold=self.config.mse_threshold
            )
        
        # Create strategy based on exit type
        if self.config.exit_strategy == ExitStrategy.FIXED:
            self.trade_strategy = PatternBasedStrategy(
                pattern_matcher=matcher_function,
                fixed_tp_pct=self.config.fixed_tp_pct,
                fixed_sl_pct=self.config.fixed_sl_pct,
                min_reward_risk=self.config.reward_risk_min,
                name=f"{self.config.recognition_technique.value} Strategy"
            )
        elif self.config.exit_strategy == ExitStrategy.PATTERN_BASED:
            self.trade_strategy = PatternBasedStrategy(
                pattern_matcher=matcher_function,
                min_reward_risk=self.config.reward_risk_min,
                name=f"{self.config.recognition_technique.value} Pattern-Based Strategy"
            )
        else:
            # Default to pattern-based
            self.trade_strategy = PatternBasedStrategy(
                pattern_matcher=matcher_function,
                min_reward_risk=self.config.reward_risk_min,
                name=f"{self.config.recognition_technique.value} Default Strategy"
            )
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest and return results.
        
        Returns:
            Dictionary containing backtest results and metrics
        """
        print(f"Running backtest for {self.config.test_start} to {self.config.test_end}")
        
        # Set up results containers
        prices = self.test_df['ClosePrice'].values
        self.returns = np.zeros(len(self.test_df))
        self.equity_curve = [1.0]  # Start with 1 unit
        trade_results = []
        
        # Variables for tracking unique patterns
        last_pips_x = [0] * self.config.n_pips
        unique_patterns_seen = []
        
        # Backtesting loop
        for i in range(self.config.lookback, len(self.test_df) - self.config.hold_period):
            # Generate trading signal
            trade_type, pattern_info = self.trade_strategy.generate_signal(self.test_df, i)
            
            # Skip if no trade
            if trade_type == TradeType.NONE or not pattern_info:
                self.equity_curve.append(self.equity_curve[-1])
                continue
            
            # Check for pattern uniqueness if needed
            if hasattr(self.config, 'require_unique_patterns') and self.config.require_unique_patterns:
                # Extract PIPs
                window = prices[i - self.config.lookback:i + 1]
                pips_x, pips_y = extract_pips(window, n_pips=self.config.n_pips, dist_type=self.config.distance_measure)
                
                if pips_y is None:
                    self.equity_curve.append(self.equity_curve[-1])
                    continue
                
                # Convert to global indices
                start_i = i - self.config.lookback
                global_pips_x = [x + start_i for x in pips_x]
                
                # Check if this is a unique pattern
                is_unique = True
                for j in range(1, self.config.n_pips - 1):
                    if global_pips_x[j] == last_pips_x[j]:
                        is_unique = False
                        break
                        
                if not is_unique:
                    self.equity_curve.append(self.equity_curve[-1])
                    continue
                    
                # Store this pattern for future comparison
                last_pips_x = global_pips_x
                unique_patterns_seen.append(pips_y.tolist())
            
            # Set up trade parameters
            entry_price = prices[i]
            entry_time = self.test_df.index[i]
            
            # Determine take profit and stop loss levels
            take_profit, stop_loss = self.trade_strategy.calculate_target_and_stop(
                entry_price, trade_type, pattern_info
            )
            
            # Simulate the trade
            use_trailing_stop = self.config.exit_strategy == ExitStrategy.TRAILING
            trailing_pct = self.config.trailing_sl_pct if use_trailing_stop else None
            
            exit_price, profit_loss, exit_reason = simulate_trade(
                prices[i+1:i+self.config.hold_period+1],
                entry_price,
                take_profit,
                stop_loss,
                trade_type,
                self.config.hold_period,
                use_trailing_stop,
                trailing_pct
            )
            
            # Calculate trade duration
            exit_idx = i
            for j in range(1, self.config.hold_period + 1):
                current_price = prices[i + j] if i + j < len(prices) else prices[-1]
                
                # Check for exit conditions
                if (trade_type == TradeType.BUY and 
                    ((current_price >= take_profit) or 
                     (current_price <= stop_loss) or 
                     (j == self.config.hold_period))):
                    exit_idx = i + j
                    break
                elif (trade_type == TradeType.SELL and 
                      ((current_price <= take_profit) or 
                       (current_price >= stop_loss) or 
                       (j == self.config.hold_period))):
                    exit_idx = i + j
                    break
            
            trade_duration = exit_idx - i
            
            # Record trade results
            trade_return_pct = (profit_loss / entry_price) * 100
            
            trade_details = {
                'entry_time': entry_time,
                'exit_time': self.test_df.index[min(i + trade_duration, len(self.test_df) - 1)],
                'type': trade_type.value,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return_pct,
                'profit_loss': profit_loss,
                'outcome': 'win' if profit_loss > 0 else 'loss',
                'reason': exit_reason.value,
                'duration': trade_duration,
                'cluster_id': pattern_info.get('cluster_id'),
                'max_gain': pattern_info.get('max_gain'),
                'max_drawdown': pattern_info.get('max_drawdown'),
                'reward_risk': abs(pattern_info.get('max_gain', 0)) / abs(pattern_info.get('max_drawdown', 1e-6)),
                'confidence': pattern_info.get('confidence')
            }
            
            trade_results.append(trade_details)
            
            # Update returns array
            self.returns[exit_idx] = profit_loss / entry_price
            
            # Update equity curve
            self.equity_curve.append(self.equity_curve[-1] * (1 + self.returns[exit_idx]))
        
        # Create DataFrame from trade results
        self.trades_df = pd.DataFrame(trade_results)
        
        # Calculate performance metrics
        self.metrics = calculate_performance_metrics(
            self.trades_df,
            self.equity_curve,
            self.returns,
            self.config.stock_id,
            self.config.timeframe_id,
            self.config.config_id if hasattr(self.config, 'config_id') else 0,
            self.config.test_start,
            self.config.test_end,
            self.config.recognition_technique.value
        )
        
        # Store metrics in the database
        metric_id = store_performance_metrics(self.db, self.metrics)
        print(f"Performance metrics stored with ID {metric_id}")
        
        # Return results
        return {
            'trades_df': self.trades_df,
            'equity_curve': self.equity_curve,
            'returns': self.returns,
            'metrics': self.metrics,
            'unique_patterns_used': len(unique_patterns_seen) if unique_patterns_seen else 0
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a performance report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            HTML report as a string
        """
        if self.trades_df is None or self.equity_curve is None or self.metrics is None:
            raise ValueError("Backtest must be run before generating a report")
        
        # Generate report
        report_html = create_performance_report(
            self.metrics,
            self.trades_df,
            self.equity_curve,
            self.test_df.index,
            save_path
        )
        
        return report_html
    
    def plot_results(self) -> None:
        """Plot backtest results."""
        if self.trades_df is None or self.equity_curve is None:
            raise ValueError("Backtest must be run before plotting results")
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(self.test_df.index[:len(self.equity_curve)], self.equity_curve, label='Equity Curve')
        axs[0].set_title(f'Backtest Performance - {self.metrics.recognition_technique} Strategy')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Portfolio Value')
        axs[0].grid(True)
        axs[0].legend()
        
        # Add key metrics as text
        metrics_text = (
            f"Total Return: {self.metrics.total_return_pct:.2f}%  |  "
            f"Win Rate: {self.metrics.win_rate:.2f}%  |  "
            f"Profit Factor: {self.metrics.profit_factor:.2f}  |  "
            f"Sharpe: {self.metrics.sharpe_ratio:.2f}  |  "
            f"Max DD: {self.metrics.max_drawdown:.2f}%"
        )
        axs[0].annotate(metrics_text, xy=(0.5, 0.02), xycoords='axes fraction', 
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Drawdown plot
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        axs[1].fill_between(self.test_df.index[:len(drawdown)], drawdown, 0, color='red', alpha=0.3)
        axs[1].plot(self.test_df.index[:len(drawdown)], drawdown, color='red', label='Drawdown')
        axs[1].set_title(f'Drawdown (Max: {self.metrics.max_drawdown:.2f}%)')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Drawdown (%)')
        axs[1].grid(True)
        axs[1].legend()
        
        # Trade distribution
        if not self.trades_df.empty and 'type' in self.trades_df.columns and 'return_pct' in self.trades_df.columns:
            # Split by trade type
            if 'BUY' in self.trades_df['type'].values:
                buy_returns = self.trades_df[self.trades_df['type'] == 'BUY']['return_pct']
                axs[2].hist(buy_returns, bins=20, alpha=0.5, label='Buy Trades')
            
            if 'SELL' in self.trades_df['type'].values:
                sell_returns = self.trades_df[self.trades_df['type'] == 'SELL']['return_pct']
                axs[2].hist(sell_returns, bins=20, alpha=0.5, label='Sell Trades')
            
            # Split by exit reason
            if 'TP Hit' in self.trades_df['reason'].values:
                tp_returns = self.trades_df[self.trades_df['reason'] == 'TP Hit']['return_pct']
                axs[2].hist(tp_returns, bins=20, alpha=0.5, label='TP Hits')
            
            if 'SL Hit' in self.trades_df['reason'].values:
                sl_returns = self.trades_df[self.trades_df['reason'] == 'SL Hit']['return_pct']
                axs[2].hist(sl_returns, bins=20, alpha=0.5, label='SL Hits')
            
            axs[2].axvline(x=0, color='black', linestyle='--')
            axs[2].set_title('Trade Return Distribution')
            axs[2].set_xlabel('Return (%)')
            axs[2].set_ylabel('Frequency')
            axs[2].grid(True)
            axs[2].legend()
        
        plt.tight_layout()
        plt.show()


def run_backtest(
    db: Database,
    stock_id: int,
    timeframe_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    recognition_technique: str = "svm",
    n_pips: int = 5,
    lookback: int = 24,
    hold_period: int = 6,
    config_id: Optional[int] = None,
    save_report: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a backtest with the specified parameters.
    
    Args:
        db: Database connection
        stock_id: Stock ID
        timeframe_id: Timeframe ID
        train_start: Start date for training data
        train_end: End date for training data
        test_start: Start date for test data
        test_end: End date for test data
        recognition_technique: Name of the recognition technique to use
        n_pips: Number of perceptually important points to extract
        lookback: Number of periods to look back for pattern identification
        hold_period: Maximum number of periods to hold a trade
        config_id: Optional configuration ID for stored configuration
        save_report: Whether to save a performance report
        **kwargs: Additional parameters for the backtest
        
    Returns:
        Dictionary containing backtest results and metrics
    """
    # Check if using a stored configuration
    if config_id is not None:
        # Load configuration from database
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM experiment_configs WHERE config_id = ?", (config_id,))
        config_data = cursor.fetchone()
        
        if config_data is None:
            raise ValueError(f"Configuration with ID {config_id} not found")
        
        # Create config object
        config = BacktestConfig(
            stock_id=stock_id,
            timeframe_id=timeframe_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            recognition_technique=config_data['recognition_technique'] or recognition_technique,
            n_pips=config_data['n_pips'] or n_pips,
            lookback=config_data['lookback'] or lookback,
            hold_period=config_data['hold_period'] or hold_period,
            config_id=config_id
        )
        
        # Add additional parameters
        for attr in ['returns_hold_period', 'distance_measure', 'mse_threshold',
                   'fixed_tp_pct', 'fixed_sl_pct', 'trailing_sl_pct', 
                   'time_exit_periods', 'reward_risk_min']:
            if attr in config_data and config_data[attr] is not None:
                setattr(config, attr, config_data[attr])
        
        # Parse model params if available
        if 'model_params' in config_data and config_data['model_params']:
            try:
                config.model_params = eval(config_data['model_params'])
            except:
                pass
    else:
        # Create a new configuration
        config = BacktestConfig(
            stock_id=stock_id,
            timeframe_id=timeframe_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            recognition_technique=recognition_technique,
            n_pips=n_pips,
            lookback=lookback,
            hold_period=hold_period,
            **kwargs
        )
    
    # Create and run backtester
    backtester = Backtester(db, config)
    backtester.load_data()
    backtester.setup_pattern_recognizer()
    backtester.setup_trade_strategy()
    results = backtester.run_backtest()
    
    # Plot results
    backtester.plot_results()
    
    # Generate and save report if requested
    if save_report:
        report_dir = os.path.join(current_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            report_dir, 
            f"backtest_stock{stock_id}_tf{timeframe_id}_{recognition_technique}_{timestamp}.html"
        )
        
        backtester.generate_report(report_path)
        print(f"Performance report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    db = Database()
    
    results = run_backtest(
        db=db,
        stock_id=1,  # Gold
        timeframe_id=2,  # 1-hour timeframe
        train_start=pd.Timestamp("2019-01-01"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-01"),
        test_end=pd.Timestamp("2023-12-31"),
        recognition_technique="svm",
        n_pips=5,
        lookback=24,
        hold_period=6,
        mse_threshold=0.03,
        reward_risk_min=1.2,
        exit_strategy=ExitStrategy.PATTERN_BASED,
        save_report=True
    )
    
    # Print metrics summary
    results['metrics'].print_summary()
    
    # Close database connection
    db.close()
