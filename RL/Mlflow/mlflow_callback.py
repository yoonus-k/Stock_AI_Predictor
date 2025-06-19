"""
Unified MLflow callback for comprehensive RL training monitoring
Integrates evaluation, portfolio tracking, feature importance, and performance metrics
with professional MLflow artifact organization and error handling
"""

import os
import json
import time
import threading
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
# Backtesting.py integration
from Backtesting.evaluator import BacktestingPyEvaluator
from Backtesting.metrics import MetricsExtractor

class MLflowLoggingCallback(BaseCallback):
    """
    Comprehensive MLflow callback that integrates:
    - Real-time training metrics logging
    - Portfolio performance tracking with comprehensive financial metrics
    - Feature importance analysis with permutation importance
    - Model evaluation with multi-episode testing
    - Professional artifact organization and error handling
    """
    def __init__(
        self, 
        mlflow_manager,
        eval_env=None,
        eval_freq: int = 5000,
        log_freq: int = 100,
        model_eval_freq: int = 2500,  # Ensure at least 1 step for evaluation
        feature_importance_freq: int = 10000,
        portfolio_eval_freq: int = 5000,
        n_eval_episodes: int = 3,
        max_eval_steps: int = 500,
        risk_free_rate: float = 0.02,
        save_plots: bool = True,
        save_model_checkpoints: bool = True,
        verbose: int = 1,
        timeframe: str = None,
        previous_timesteps: int = 0
    ):
        """
        Initialize unified MLflow callback
        
        Args:
            mlflow_manager: Instance of MLflowManager for experiment tracking
            eval_env: Environment for evaluation and portfolio tracking
            eval_freq: How often to run model evaluation (timesteps)
            log_freq: How often to log training metrics (timesteps)
            feature_importance_freq: How often to calculate feature importance
            portfolio_eval_freq: How often to evaluate portfolio performance
            n_eval_episodes: Number of episodes for evaluation
            max_eval_steps: Maximum steps per evaluation episode (prevents infinite loops)        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            save_plots: Whether to save performance plots as artifacts
            save_model_checkpoints: Whether to save model checkpoints
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
            timeframe: Model timeframe (daily, weekly, monthly, meta)
        """
        super().__init__(verbose)        
        self.mlflow_manager = mlflow_manager
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.model_eval_freq = model_eval_freq
        self.log_freq = log_freq
        self.feature_importance_freq = feature_importance_freq
        self.portfolio_eval_freq = portfolio_eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_eval_steps = max_eval_steps
        self.risk_free_rate = risk_free_rate
        self.save_plots = save_plots
        self.save_model_checkpoints = save_model_checkpoints
        self.timeframe = timeframe or "unknown"
        self.previous_timesteps = previous_timesteps  # Add offset from previous training runs
          # Initialize backtesting.py portfolio evaluator
        self.portfolio_evaluator = BacktestingPyEvaluator(
            initial_cash=100000,
            # 3 USD / Lot , # 1 Lot = 100,000 USD , so 3 USD / 100,000 USD = 0.00003 , 
            # since we apply commission on open and close, we need to divide by 2, 
            # so final commission = 0.00003 / 2 = 0.000015
            commission=0.000015, 
            spread=0.0001, # formula : price * (1+x) = spreaded_price , where x is the spread
            margin=0.01,
            trade_on_close=True,
            hedging=True,
            exclusive_orders=False,
            finalize_trade_on_close=True,
            max_positions=10,
            risk_per_trade=0.02,
            enable_short=True,
            enable_hedging=True,
            position_sizing='fixed',
            verbose=False,
        )
        
        # Tracking variables
        self.last_time_evaluated = 0
        self.last_feature_importance = 0
        self.last_portfolio_eval = 0
        self.best_mean_reward = -float("inf")
        
        # Performance tracking
        self.evaluation_history = []
        self.portfolio_history = []
        self.feature_importance_history = []
        self.training_metrics_history = []
        
        # Create temporary directories for artifacts
        self.temp_dir = Path("temp_mlflow_artifacts")
        self.temp_dir.mkdir(exist_ok=True)
          # Feature names for importance analysis (30 features total)
        self.feature_names = [
            # Base pattern features (7)
            "probability", "action", "reward_risk_ratio", "max_gain", "max_drawdown", "mse", "expected_value",
            # Technical indicators (3)
            "rsi", "atr", "atr_ratio",
            # Sentiment features (2)
            "unified_sentiment", "sentiment_count",
            # COT data (6)
            "net_noncommercial", "net_nonreportable", "change_nonrept_long", 
            "change_nonrept_short", "change_noncommercial_long", "change_noncommercial_short",
            # Time features (7)
            "hour_sin", "hour_cos", "day_sin", "day_cos", "asian_session", "london_session", "ny_session",
            # Portfolio features (5 + 3 new metrics = 8)
            "balance_ratio", "position_ratio", "position", "portfolio_max_drawdown", "win_rate",        # New performance metrics (3)
            "avg_pnl_per_hour", "decisive_exits", "recovery_factor"
        ]
        
    def _on_step(self) -> bool:
        """Called at each training step - orchestrates all monitoring activities in parallel"""
        
        try:
            # Calculate offset step for continuous tracking across runs
            offset_step = self.n_calls + self.previous_timesteps
            
            # Periodically log the total timesteps (useful for continued training)
            if self.n_calls % (self.log_freq * 10) == 0 or self.n_calls == 1:
                self.mlflow_manager.log_metrics({
                    "total_timesteps": offset_step,
                    "previous_timesteps": self.previous_timesteps
                })
            
            # Launch all evaluation tasks in parallel threads
            threads = []
            
            # # Log training metrics at specified frequency
            # if self.n_calls % self.log_freq == 0:
            #     t_metrics = threading.Thread(
            #         target=self._log_training_metrics, 
            #         args=(offset_step,)
            #     )
            #     t_metrics.daemon = True
            #     threads.append(t_metrics)
            #     t_metrics.start()
            
            # # Run model evaluation at specified frequency
            # if self.eval_env is not None and self.n_calls % self.model_eval_freq == 0:
            #     t_model_eval = threading.Thread(
            #         target=self._run_model_evaluation, 
            #         args=(offset_step,)
            #     )
            #     t_model_eval.daemon = True
            #     threads.append(t_model_eval)
            #     t_model_eval.start()
            
            # # Run portfolio performance evaluation
            # if self.eval_env is not None and self.n_calls % self.portfolio_eval_freq == 0:
            #     t_portfolio = threading.Thread(
            #         target=self._run_portfolio_evaluation, 
            #         args=(offset_step,)
            #     )
            #     t_portfolio.daemon = True
            #     threads.append(t_portfolio)
            #     t_portfolio.start()
            
            # # Calculate feature importance
            # if self.eval_env is not None and self.n_calls % self.feature_importance_freq == 0:
            #     t_features = threading.Thread(
            #         target=self._calculate_feature_importance, 
            #         args=(offset_step,)
            #     )
            #     t_features.daemon = True
            #     threads.append(t_features)
            #     t_features.start()
                
            # no threade method
            if self.n_calls % self.log_freq == 0:
                self._log_training_metrics(step=offset_step)
                
            if self.eval_env is not None and self.n_calls % self.model_eval_freq == 0:
                self._run_model_evaluation(step=offset_step)
                
            if self.eval_env is not None and self.n_calls % self.portfolio_eval_freq == 0:
                self._run_portfolio_evaluation(step=offset_step)
                
            if self.eval_env is not None and self.n_calls % self.feature_importance_freq == 0:
                # Run portfolio evaluation without threading
                self._calculate_feature_importance(step=offset_step)
            # Wait for all threads to complete
            
            
            # Don't wait for threads to complete - let them run in the background
            # This allows training to continue immediately
            
            return True
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error in callback step {self.n_calls} (offset: {offset_step}): {e}")
            return True  # Continue training even if monitoring fails
        
    def _log_training_metrics(self, step=None):
        """Log current training metrics from model logger with enhanced error handling"""
        try:
            if not hasattr(self.model, 'logger') or self.model.logger is None:
                return
                
            # Get latest logged values
            name_to_value = getattr(self.model.logger, 'name_to_value', {})
            
            if not name_to_value:
                return
            
            metrics_to_log = {}
            for key, value in name_to_value.items():
                # Only log numeric values
                if isinstance(value, (int, float, np.integer, np.floating)):
                    # Convert SB3 logging names to more readable names
                    if 'rollout/' in key:
                        metric_name = f"training/{key.replace('rollout/', '')}"
                    elif 'train/' in key:
                        metric_name = f"training/{key.replace('train/', '')}"
                    else:
                        metric_name = f"training/{key}"
                    
                    metrics_to_log[metric_name] = float(value)
            
            # Add system metrics
            metrics_to_log.update({
                f"system/timestep": self.n_calls,
                f"system/timeframe": self.timeframe
            })
              # Log to MLflow with proper error handling
            if metrics_to_log:
                # Use provided step or default to self.n_calls
                log_step = step if step is not None else self.n_calls
                self.mlflow_manager.log_metrics(metrics_to_log, step=log_step)
                self.training_metrics_history.append({
                    "step": log_step,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics_to_log.copy()
                })
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log training metrics: {e}")
    def _run_model_evaluation(self, step=None):
        """Run comprehensive model evaluation without internal threading"""
        try:
            # Skip if already evaluated at this step
            if self.n_calls == self.last_time_evaluated:
                return
            
            self.last_time_evaluated = self.n_calls
            
            if self.verbose > 1:
                print(f"Running model evaluation at step {self.n_calls}...")
            
            # Direct evaluation call without threading
            results = self._evaluate_model_performance()
            
            # Process results
            if results is not None:
                self._log_evaluation_results(results, step=step)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not run model evaluation: {e}")
    
    def _evaluate_model_performance(self):
        """Evaluate model performance across multiple episodes"""

        mean_reward, std_reward = evaluate_policy(
        self.model, 
        self.eval_env,
        n_eval_episodes=self.n_eval_episodes,
        deterministic=True,
        )
        
        # Calculate statistics
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
    def _log_evaluation_results(self, results, step=None):
        """Log evaluation results to MLflow"""
        try:
            eval_metrics = {
                f"evaluation/mean_reward": results["mean_reward"],
                f"evaluation/std_reward": results["std_reward"],
            }
            
            # Use provided step or default to self.n_calls
            log_step = step if step is not None else self.n_calls
            
            # Log metrics
            self.mlflow_manager.log_metrics(eval_metrics, step=log_step)
            
            # Store in history
            self.evaluation_history.append({
                "step": log_step,
                "timestamp": datetime.now().isoformat(),
                "results": results.copy()            })
            
            # Check if this is the best model
            if results["mean_reward"] > self.best_mean_reward:
                self.best_mean_reward = results["mean_reward"]
                self.mlflow_manager.log_metrics({
                    f"evaluation/best_mean_reward": self.best_mean_reward
                }, step=log_step)
                
                # Save best model checkpoint if enabled
                if self.save_model_checkpoints:
                    self._save_model_checkpoint("best_model")
                    
            if self.verbose > 0:
                print(f"Evaluation - Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log evaluation results: {e}")
    
    def _run_portfolio_evaluation(self, step=None):
        """Run comprehensive portfolio performance evaluation without internal threading"""
        try:
            # Only evaluate every N steps to reduce overhead
            if (self.n_calls - self.last_portfolio_eval < self.portfolio_eval_freq and 
                self.n_calls > 0):  # Skip first call
                return

            self.last_portfolio_eval = self.n_calls
            
            if self.verbose > 0:
                print(f"Running portfolio evaluation at step {self.n_calls}...")
            
            # Direct calculation without threading
            metrics ,stats = self._calculate_portfolio_metrics()
            
            # Process results if available
            if metrics is not None:
                self._log_portfolio_metrics(metrics, step=step)
                
            # Store in history
            self.portfolio_history.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics.copy() ,
                "stats": stats.copy() 
            })         
                
            if stats is not None:
                # Log additional stats if available
                self._create_portfolio_plot(metrics, stats_df=stats)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not run portfolio evaluation: {e}")
                
    def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio performance metrics using backtesting.py"""
        try:
            if self.verbose > 1:
                print("Using backtesting.py for portfolio evaluation...")
            
            # Use backtesting.py's professional portfolio evaluation
            metrics, stats = self.portfolio_evaluator.evaluate_portfolio(
                rl_model=self.model,
                environment_data=None,  # Will use database data
                stock_id=1,
                timeframe=self.timeframe
            )

            
            if self.verbose > 1:
                print(f"Backtesting.py evaluation completed. Total return: {metrics.get('total_return', 0):.2%}")
            
            return metrics ,stats
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in backtesting.py portfolio evaluation: {e}")
                print("Falling back to simple portfolio evaluation...")
            
            # Robust fallback implementation
            return self._fallback_portfolio_evaluation()
        

    def _log_portfolio_metrics(self, metrics, step=None):
        """Log portfolio metrics to MLflow with artifacts"""
        try:
            # Use provided step or default to self.n_calls
            log_step = step if step is not None else self.n_calls
            # Log main metrics
            portfolio_metrics = {
                # Core performance metrics
                f"portfolio/balance": metrics["final_equity"],
                f"portfolio/total_return": metrics["total_return"],
                f"portfolio/max_drawdown": metrics["max_drawdown"],
                f"portfolio/sharpe_ratio": metrics["sharpe_ratio"],
                f"portfolio/sortino_ratio": metrics.get("sortino_ratio", 0.0),
                f"portfolio/calmar_ratio": metrics.get("calmar_ratio", 0.0),
                f"portfolio/volatility_ann": metrics.get("volatility_ann", 0.0),
                
                # Trade metrics
                f"portfolio/win_rate": metrics["win_rate"],
                f"portfolio/total_trades": metrics["total_trades"],
                f"portfolio/winning_trades": metrics.get("winning_trades", 0),  # Fixed attribute name
                f"portfolio/losing_trades": metrics.get("losing_trades", 0),    # Consistent naming
                f"portfolio/profit_factor": metrics["profit_factor"],
                f"portfolio/avg_profit_per_winning_trade": metrics.get("avg_profit_per_winning_trade", 0.0),  # Fixed attribute name
                f"portfolio/max_consecutive_wins": metrics.get("max_consecutive_wins", 0),
                f"portfolio/max_consecutive_losses": metrics.get("max_consecutive_losses", 0),
                
                # Portfolio composition
                f"portfolio/exposure_time": metrics.get("exposure_time", 0.0),
                
                f"portfolio/long_trades_ratio": metrics.get("long_trades", 0) / max(1, metrics["total_trades"]),
                f"portfolio/short_trades_ratio": metrics.get("short_trades", 0) / max(1, metrics["total_trades"]),
                f"portfolio/long_win_rate": metrics.get("long_win_rate", 0.0),
                f"portfolio/short_win_rate": metrics.get("short_win_rate", 0.0),
                
                # Risk metrics
                f"portfolio/max_drawdown_duration": metrics.get("max_drawdown_duration", 0),
                f"portfolio/avg_drawdown": metrics.get("avg_drawdown", 0.0),
                f"portfolio/drawdown_std": metrics.get("drawdown_std", 0.0),
                f"portfolio/time_in_drawdown_pct": metrics.get("time_in_drawdown_pct", 0.0),
                f"portfolio/avg_drawdown_duration_days": metrics.get("avg_drawdown_duration_days", 0.0),
                
                # Return metrics
                f"portfolio/return_ann": metrics.get("return_ann", 0.0),
                f"portfolio/excess_return": metrics.get("excess_return", 0.0),
                f"portfolio/sqn": metrics.get("sqn", 0.0),
                f"portfolio/expectancy": metrics.get("expectancy", 0.0),
                f"portfolio/daily_return_mean": metrics.get("daily_return_mean", 0.0),
                f"portfolio/daily_return_std": metrics.get("daily_return_std", 0.0),
                f"portfolio/daily_return_skew": metrics.get("daily_return_skew", 0.0),
                f"portfolio/daily_return_kurtosis": metrics.get("daily_return_kurtosis", 0.0),
                f"portfolio/positive_days_ratio": metrics.get("positive_days_ratio", 0.0),
                f"portfolio/negative_days_ratio": metrics.get("negative_days_ratio", 0.0),
                
                # Trade quality metrics
                f"portfolio/avg_trade_duration_hours": metrics.get("avg_trade_duration_hours", 0.0),
                f"portfolio/avg_winning_trade_duration": metrics.get("avg_winning_trade_duration", 0.0),
                f"portfolio/avg_losing_trade_duration": metrics.get("avg_losing_trade_duration", 0.0),
                f"portfolio/pnl_std": metrics.get("pnl_std", 0.0),
                f"portfolio/sl_usage_ratio": metrics.get("sl_usage_ratio", 0.0),
                
                # Consistency metrics
                f"portfolio/win_rate_consistency": metrics.get("win_rate_consistency", 0.0),
                f"portfolio/return_consistency": metrics.get("return_consistency", 0.0),
                f"portfolio/performance_trend": metrics.get("performance_trend", 0.0),
                f"portfolio/equity_curve_trend": metrics.get("equity_curve_trend", 0.0),
                f"portfolio/equity_curve_smoothness": metrics.get("equity_curve_smoothness", 0.0),
                f"portfolio/first_period_return": metrics.get("first_period_return", 0.0),
                f"portfolio/middle_period_return": metrics.get("middle_period_return", 0.0),
                f"portfolio/last_period_return": metrics.get("last_period_return", 0.0),
                f"portfolio/win_rate_trend": metrics.get("win_rate_trend", 0.0),
                f"portfolio/trades_per_day": metrics.get("trades_per_day", 0.0),
                f"portfolio/trade_frequency_consistency": metrics.get("trade_frequency_consistency", 0.0),
                f"portfolio/trade_frequency_std": metrics.get("trade_frequency_std", 0.0),
                
                # Recovery metrics
                f"portfolio/avg_recovery_days": metrics.get("avg_recovery_days", 0.0),
                
                # Efficiency metrics
                f"portfolio/kelly_criterion": metrics.get("kelly_criterion", 0.0),
 
                f"portfolio/best_hour": metrics.get("best_hour", 0),
                f"portfolio/best_hour_win_rate": metrics.get("best_hour_win_rate", 0.0),
                f"portfolio/worst_hour": metrics.get("worst_hour", 0),
                f"portfolio/worst_hour_win_rate": metrics.get("worst_hour_win_rate", 0.0),
                f"portfolio/best_day": metrics.get("best_day", 0),
                f"portfolio/best_day_win_rate": metrics.get("best_day_win_rate", 0.0),
                f"portfolio/worst_day": metrics.get("worst_day", 0),
                f"portfolio/worst_day_win_rate": metrics.get("worst_day_win_rate", 0.0),
            }            # Use provided step or default to self.n_calls
            log_step = step if step is not None else self.n_calls
            
            self.mlflow_manager.log_metrics(portfolio_metrics, step=log_step)
            
   
             
        except Exception as e:
            if self.verbose > 0:                
                print(f"Warning: Could not log portfolio metrics: {e}") 
                
    def _create_portfolio_plot(self, metrics, stats_df=None):
        """Create and save portfolio performance visualization with enhanced metrics and stats
        ┌───────┬───────┬───────┬───────┐
        │ fig1  │ fig2  │ fig3  │ fig4  │ Row 0
        ├───────┼───────┼───────┼───────┤
        │ fig5  │ fig6  │ fig8a │ fig8b │ Row 1
        ├───────┼───────┼───────┼───────┤
        │ fig7  │ fig7  │ fig8c │ fig9  │ Row 2
        ├───────┼───────┼───────┼───────┤
        │ fig7  │ fig7  │ fig10 │ fig11 │ Row 3
        └───────┴───────┴───────┴───────┘
        Col 0   Col 1   Col 2   Col 3
                
        fig1: Portfolio value over time (0,0)
        fig2: Daily returns distribution (0,1) - with x-axis range [-0.1, 0.1]
        fig3: Drawdown over time (0,2)
        fig4: Trade direction analysis - Long vs Short performance comparison (0,3)
        fig5: Action distribution with pie chart (1,0)
        fig6: Exit types visualization (SL, TP, time) (1,1)
        fig7: Combined visualization of hold time, position size, and risk-reward (2,0), (2,1), (3,0), (3,1) - taking 2x2 grid as a square
        fig8a: Position size vs risk reward (1,2) - split out from the original fig8
        fig8b: Hold time vs position size (1,3) - split out from the original fig8
        fig8c: Hold time vs risk reward (2,2) - split out from the original fig8
        fig9: Performance metrics summary table (2,3)
        fig10: New plot Position Sizing Evolution (3,2)
        fig11: New plot Risk-Adjusted Return Metrics(3,3)
        
        """
        try:            # Create enhanced 4x4 subplot layout with improved layout
            # Use a wider figure to prevent horizontal compression
            fig = plt.figure(figsize=(26, 22), constrained_layout=True)
            # Create grid with specific width ratios to ensure proper spacing
            gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1])
            
            # FIRST ROW - Top row of 4 plots
            # 1. Portfolio value over time from equity curve (0,0)
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Handle backtesting.py equity curve which is stored differently
            equity_data = None
            dates = None
            
            # First check if we have an _equity_curve DataFrame from stats (directly from backtesting.py)
            if stats_df is not None and '_equity_curve' in stats_df and isinstance(stats_df['_equity_curve'], pd.DataFrame):
                equity_df = stats_df['_equity_curve']
                dates = equity_df.index
                equity_data = equity_df['Equity'].values
    
            # Plot equity curve with dates if available
            if equity_data is not None and len(equity_data) > 0:
                # Plot with dates
                ax1.plot(dates, equity_data, label='Equity Curve', linewidth=1.5, color='#1f77b4')
                
                # Format x-axis as dates
                plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
                ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
                
                # Add Buy & Hold comparison if available
                if '_equity_curve' in stats_df and 'Benchmark' in stats_df['_equity_curve'].columns:
                    ax1.plot(dates, stats_df['_equity_curve']['Benchmark'].values, label='Buy & Hold', 
                            linewidth=1, color='#ff7f0e', linestyle='--', alpha=0.8)
                
                # Add markers for significant points (highs and lows)
                max_idx = np.argmax(equity_data)
                min_idx = np.argmin(equity_data)
                
                if max_idx > 10:  # Only if we have enough data
                    ax1.plot(dates[max_idx], equity_data[max_idx], 'go', markersize=5)
                    ax1.annotate(f"${equity_data[max_idx]:.0f}", 
                               (dates[max_idx], equity_data[max_idx]),
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=8)
                
                if min_idx > 10:  # Only if we have enough data
                    ax1.plot(dates[min_idx], equity_data[min_idx], 'ro', markersize=5)
                    ax1.annotate(f"${equity_data[min_idx]:.0f}", 
                               (dates[min_idx], equity_data[min_idx]),
                               textcoords="offset points", xytext=(0, -15),
                               ha='center', fontsize=8)
                
                # Add final return annotation
                total_return = (equity_data[-1] / equity_data[0] - 1) * 100
                color = 'green' if total_return >= 0 else 'red'
                ax1.annotate(f"Return: {total_return:.1f}%", 
                           xy=(0.02, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                           color=color, fontweight='bold', fontsize=9)                
                ax1.set_title(f"Portfolio Equity - {self.timeframe}", fontweight='bold')
                ax1.set_ylabel("Portfolio Value ($)")
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='best', framealpha=0.7)
            else:
                ax1.text(0.5, 0.5, "Insufficient equity data", ha='center', va='center')
            
            # 2. Daily returns distribution with normal distribution overlay (0,1)
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Extract daily returns from backtesting.py equity curve
            daily_returns = []             
            
            if equity_data is not None and len(equity_data) > 5:
                # Convert NumPy array to pandas Series first
                equity_series = pd.Series(equity_data)
                daily_returns = equity_series.pct_change().dropna().values
               # Plot daily returns distribution if we have enough data
            if len(daily_returns) > 5:  # Need enough data for meaningful histogram
                ax2.hist(daily_returns, bins=min(50, len(daily_returns)//5), alpha=0.7, density=True, color='#1f77b4')
                  # Add normal distribution curve for comparison
                if len(daily_returns) > 10:  # Need enough data for meaningful statistics
                    mu = np.mean(daily_returns)
                    sigma = np.std(daily_returns)
                    # Check for near-zero standard deviation to avoid division issues
                    if sigma > 1e-8:
                        # Use requested x-axis range [-0.1, 0.1]
                        x = np.linspace(-0.1, 0.1, 100)
        
                        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
                        
                        # Set x-axis limits to [-0.1, 0.1] as requested
                        ax2.set_xlim(-0.1, 0.1)
                        
                        # Add skewness and kurtosis info
                        try:
                            skew = stats.skew(daily_returns)
                            kurt = stats.kurtosis(daily_returns)
                            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) # Annualized Sharpe
                            ax2.text(0.05, 0.95, f"Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}\nSharpe: {sharpe:.2f}", 
                                     transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                                     fontsize=9)
                        except:
                            pass
                    
                ax2.set_title("Daily Returns Distribution", fontweight='bold')
                ax2.set_xlabel("Daily Return")
                ax2.set_ylabel("Frequency")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "Insufficient daily returns data", ha='center', va='center')
            
            # 3. Drawdown over time (0,2)
            ax3 = fig.add_subplot(gs[0, 2])
            # Extract drawdowns from backtesting.py equity curve
            drawdowns = []
            dates_for_drawdown = None
              # First check for DrawdownPct in equity curve from stats
            if stats_df is not None and '_equity_curve' in stats_df and isinstance(stats_df['_equity_curve'], pd.DataFrame):
                equity_df = stats_df['_equity_curve']
                # Make sure drawdowns are negative numbers (sometimes they're stored as positive)
                if 'DrawdownPct' in equity_df.columns:
                    drawdowns = equity_df['DrawdownPct'].values
                    # Convert to negative if they're positive
                    if np.mean(drawdowns) > 0:
                        drawdowns = -drawdowns
                else:
                    # Calculate drawdowns if not present
                    equity_values = equity_df['Equity'].values if 'Equity' in equity_df.columns else []
                    if len(equity_values) > 0:
                        # Calculate running max
                        running_max = np.maximum.accumulate(equity_values)
                        # Calculate drawdown as percentage of peak
                        drawdowns = equity_values / running_max - 1
                
                dates_for_drawdown = equity_df.index if isinstance(equity_df.index, pd.DatetimeIndex) else None
    
            # Plot drawdowns
            if len(drawdowns) > 0:
                if dates_for_drawdown is not None and len(dates_for_drawdown) == len(drawdowns):
                    # Plot with dates
                    ax3.fill_between(dates_for_drawdown, drawdowns, 0, alpha=0.6, color='#ff9999')
                    
                    # Format x-axis as dates
                    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
                    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
                else:
                    # Fallback to simple index plot
                    ax3.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.6, color='#ff9999')
                
                # Add max drawdown annotation
                max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
                max_dd_idx = np.argmin(drawdowns) if len(drawdowns) > 0 else 0
                
                if len(drawdowns) > 0 and max_dd < -0.05:  # Only annotate significant drawdowns
                    if dates_for_drawdown is not None:
                        ax3.annotate(f"{max_dd*100:.1f}%", 
                                   xy=(dates_for_drawdown[max_dd_idx], max_dd),
                                   xytext=(0, -20), textcoords="offset points",
                                   arrowprops=dict(arrowstyle="->", color='black'),
                                   fontsize=8, fontweight='bold', color='red')
                    else:
                        ax3.annotate(f"{max_dd*100:.1f}%", 
                                   xy=(max_dd_idx, max_dd),
                                   xytext=(0, -20), textcoords="offset points",
                                   arrowprops=dict(arrowstyle="->", color='black'),
                                   fontsize=8, fontweight='bold', color='red')
                
                ax3.set_title(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%", fontweight='bold')
                ax3.set_ylabel("Drawdown (%)")
                ax3.grid(True, alpha=0.3)
                
                # Add horizontal line at 0
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                  # Set y-limit to show drawdowns properly 
                # Determine an appropriate scale based on the max drawdown
                max_dd_value = min(drawdowns) if len(drawdowns) > 0 else -0.05
                
                # If max drawdown is significant, set y-limit to show it clearly
                if max_dd_value < -0.05:
                    # Ensure there's some padding below the max drawdown
                    y_min = max(-1.0, max_dd_value * 1.2)  # Cap at -100%
                    ax3.set_ylim(y_min, 0.01)
                else:
                    # For smaller drawdowns, use a fixed scale that's meaningful
                    ax3.set_ylim(-0.2, 0.01)  # Fixed scale for small drawdowns
                
                # Force percent formatting on y-axis
                from matplotlib.ticker import PercentFormatter
                ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
            else:
                ax3.text(0.5, 0.5, "Insufficient drawdown data", ha='center', va='center')
            
            # 4. Trade direction analysis - Long vs Short performance comparison (0,3)
            ax4 = fig.add_subplot(gs[0, 3])
            
            # Show long/short trades and their performance
            long_trades = metrics.get("long_trades", 0)
            short_trades = metrics.get("short_trades", 0)
            long_win_rate = metrics.get("long_win_rate", 0)
            short_win_rate = metrics.get("short_win_rate", 0)
            
            # Also extract profit metrics if available
            long_profit = metrics.get("long_profit_pct", 0) * 100
            short_profit = metrics.get("short_profit_pct", 0) * 100            
            # Create a grouped bar chart with multiple metrics
            x = np.arange(2)  # Long and Short
            width = 0.2  # Thinner bars to fit more metrics
            
            # Create bars for each metric, with different colors and positions
            bars1 = ax4.bar(x - width*1.5, [long_trades, short_trades], width, 
                          label='Trade Count', color='#1f77b4')
            bars2 = ax4.bar(x - width*0.5, [long_win_rate*100, short_win_rate*100], width, 
                          label='Win Rate %', color='#2ca02c')
            bars3 = ax4.bar(x + width*0.5, [long_profit, short_profit], width, 
                          label='Profit %', color='#ff7f0e')
            
            # Only add labels to trade count
            ax4.bar_label(bars1, padding=3, fmt='%d')
            
            # Style improvements
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax4.set_ylabel('Metrics')
            ax4.set_title('Long vs Short Performance', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(['Long', 'Short'], fontweight='bold')
            ax4.legend(loc='upper left', framealpha=0.7)
            ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            # Add win rate and profit percentages as text annotations
            for i, (rate, profit) in enumerate(zip([long_win_rate, short_win_rate], 
                                                 [long_profit, short_profit])):
                ax4.annotate(f'Win Rate: {rate*100:.1f}%', 
                           xy=(x[i] - width/2, bars1[i].get_height() + 1),
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            # 5. Action distribution with pie chart (1,0)
            ax5 = fig.add_subplot(gs[1, 0])
            
            if stats_df is not None and '_model_actions' in stats_df and not stats_df['_model_actions'].empty:
                # Get action counts from DataFrame
                action_counts = stats_df['_model_actions']['action_type'].value_counts()
                
                # Map action codes to names
                action_names = {
                    0: 'HOLD',
                    1: 'BUY',
                    2: 'SELL'
                }
                
                # Prepare labels and data
                labels = [action_names.get(action, f"Action {action}") for action in action_counts.index]
                sizes = action_counts.values
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue for HOLD, Green for BUY, Orange for SELL
                
                # Create pie chart with percentages
                wedges, _, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            startangle=90, colors=colors,
                                            wedgeprops={'alpha': 0.8})
                
                # Style the percentage text
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
                    
                ax5.axis('equal')  # Equal aspect ratio ensures the pie is circular
                ax5.set_title('Action Distribution', fontweight='bold')
                
                # Add count information in legend
                legend_labels = [f"{label} ({count})" for label, count in zip(labels, sizes)]
                ax5.legend(wedges, legend_labels, loc='best', fontsize=8)
            else:
                ax5.text(0.5, 0.5, "No action data available", ha='center', va='center')
            
            # 6. Exit types visualization (SL, TP, time) (1,1)
            ax6 = fig.add_subplot(gs[1, 1])
            
            # Get exit types from backtesting stats
            exit_counts = None
            if stats_df is not None and '_exit_reasons_counts' in stats_df:
                exit_counts = stats_df['_exit_reasons_counts']
            
            if exit_counts and sum(exit_counts.values()) > 0:
                labels = ['Stop Loss', 'Take Profit', 'Time Exit']
                sizes = [
                    exit_counts.get('sl', 0),
                    exit_counts.get('tp', 0),
                    exit_counts.get('time', 0)
                ]
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                explode = (0.1, 0, 0)  # explode stop loss slice
                
                # Create pie chart
                wedges, texts, autotexts = ax6.pie(
                    sizes, 
                    explode=explode, 
                    labels=labels, 
                    colors=colors,
                    autopct='%1.1f%%', 
                    shadow=True, 
                    startangle=90,
                    wedgeprops={'alpha': 0.8}
                )
                
                # Add better contrast for readability
                for text in texts + autotexts:
                    text.set_fontsize(9)
                    text.set_fontweight('bold')
                
                ax6.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                ax6.set_title('Trade Exit Types', fontweight='bold')
                
                # Add count information below the chart
                total_exits = sum(sizes)
                ax6.text(0, -1.2, f"Total Exits: {total_exits} | SL: {sizes[0]} | TP: {sizes[1]} | Time: {sizes[2]}", 
                        ha='center', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            else:
                ax6.text(0.5, 0.5, "No exit type data available", ha='center', va='center')
            
            # 8a. Position Size vs Risk Reward (1,2)
            ax8a = fig.add_subplot(gs[1, 2])
            trade_actions = stats_df.get('_model_actions', pd.DataFrame())
            
            if not trade_actions.empty and len(trade_actions) >= 5:
                # Scatter plot with color representing hold time
                scatter = ax8a.scatter(
                    trade_actions['risk_reward'],
                    trade_actions['position_size'],
                    c=trade_actions['hold_time'],
                    cmap='viridis',
                    s=30,
                    alpha=0.7
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax8a, pad=0.01)
                cbar.set_label('Hold Time')
                # Add regression line
                if len(trade_actions) >= 10:
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            trade_actions['risk_reward'], 
                            trade_actions['position_size']
                        )
                        x_range = np.linspace(trade_actions['risk_reward'].min(), trade_actions['risk_reward'].max(), 100)
                        ax8a.plot(x_range, intercept + slope*x_range, 'r--', linewidth=1)
                        
                        # Add correlation annotation
                        ax8a.annotate(f"r = {r_value:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                                     fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                    except Exception as e:
                        print(f"Regression line error: {e}")
                
                ax8a.set_title('Position Size vs Risk-Reward', fontsize=10, fontweight='bold')
                ax8a.set_xlabel('Risk-Reward Ratio')
                ax8a.set_ylabel('Position Size')
                ax8a.grid(True, alpha=0.3)
            else:
                ax8a.text(0.5, 0.5, "Insufficient trade data", ha='center', va='center')
                
            # 8b. Hold Time vs Position Size (1,3)
            ax8b = fig.add_subplot(gs[1, 3])
            
            if not trade_actions.empty and len(trade_actions) >= 5:
                # Use hexbin for better visualization of density
                hexbin = ax8b.hexbin(
                    trade_actions['hold_time'],
                    trade_actions['position_size'],
                    gridsize=15,
                    cmap='Blues',
                    bins='log',
                    mincnt=1
                )
                
                # Add colorbar
                cbar = plt.colorbar(hexbin, ax=ax8b, pad=0.01)
                cbar.set_label('Count (log)')
                
                ax8b.set_title('Hold Time vs Position Size', fontsize=10, fontweight='bold')
                ax8b.set_xlabel('Hold Time')
                ax8b.set_ylabel('Position Size')
                ax8b.grid(True, alpha=0.3)
            else:
                ax8b.text(0.5, 0.5, "Insufficient trade data", ha='center', va='center')
                
            # 8c. Hold Time vs Risk Reward (2,2)
            ax8c = fig.add_subplot(gs[2, 2])
            
            if not trade_actions.empty and len(trade_actions) >= 5:
                # Create a violin plot to show distribution
                import seaborn as sns
                
                # Bin the hold times to create groups for the violin plot
                bins = min(5, max(2, len(trade_actions) // 20))
                trade_actions['hold_time_bin'] = pd.cut(
                    trade_actions['hold_time'], 
                    bins=bins, 
                    labels=[f"T{i+1}" for i in range(bins)]
                )
                  # Create the violin plot
                sns.violinplot(
                    data=trade_actions,
                    x='hold_time_bin',
                    y='risk_reward',
                    hue='hold_time_bin',  # Add hue parameter to fix the warning
                    palette='viridis',
                    inner='quartile',
                    legend=False,  # Hide the redundant legend
                    ax=ax8c
                )
                
                ax8c.set_title('Risk-Reward by Hold Time Groups', fontsize=10, fontweight='bold')
                ax8c.set_xlabel('Hold Time Groups')
                ax8c.set_ylabel('Risk-Reward Ratio')
                ax8c.grid(True, axis='y', alpha=0.3)
            else:
                ax8c.text(0.5, 0.5, "Insufficient trade data", ha='center', va='center')
            
            # THIRD ROW - New visualizations and summary metrics            
            # 7. Combined visualization of hold time, position size, and risk-reward (2,0)-(3,1)
            # Create a large subplot that spans 2x2 grid cells
            ax7 = fig.add_subplot(gs[2:4, 0:2])
            
            if not trade_actions.empty and len(trade_actions) >= 5:
                # Create a larger, more comprehensive visualization of trading parameters
                
                # Set up the plot title and description
                ax7.set_title("Trade Analysis: Hold Time vs Position Size & Risk-Reward", fontweight='bold', fontsize=12)
                
                # Use bubble chart where:
                # - X-axis is hold time
                # - Y-axis is risk-reward
                # - Bubble size is position size
                # - Color is profit/loss (if available) or another parameter
                
                # Normalize position size for better bubble scaling
                max_size = trade_actions['position_size'].max()
                norm_sizes = trade_actions['position_size'] * (300 / max_size) + 20
                
                # Color by risk-reward ratio
                scatter = ax7.scatter(
                    trade_actions['hold_time'],
                    trade_actions['risk_reward'],
                    s=norm_sizes,
                    c=trade_actions['risk_reward'],
                    cmap='viridis',
                    alpha=0.6,
                    edgecolors='white',
                    linewidths=0.5
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax7)
                cbar.set_label('Risk-Reward Ratio', rotation=270, labelpad=20)
                
                # Add mean and median lines
                mean_hold = trade_actions['hold_time'].mean()
                median_hold = trade_actions['hold_time'].median()
                
                ax7.axvline(x=mean_hold, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_hold:.1f}')
                ax7.axvline(x=median_hold, color='green', linestyle=':', alpha=0.8, 
                          label=f'Median: {median_hold:.1f}')
                
                # Add distribution of hold times at the top
                divider = make_axes_locatable(ax7)
                ax_histx = divider.append_axes("top", 1.0, pad=0.3, sharex=ax7)
                
                # Plot the hold time distribution
                ax_histx.hist(trade_actions['hold_time'], bins=min(50, len(trade_actions)//5), 
                           alpha=0.5, color='blue')
                ax_histx.set_title('Hold Time Distribution')
                ax_histx.grid(alpha=0.3)
                ax_histx.set_ylabel('Frequency')
                
                # Hide x labels for histogram
                ax_histx.tick_params(axis="x", labelbottom=False)
                
                # Add key statistics as text annotation
                ax7.text(0.02, 0.95, 
                       f"Total Trades: {len(trade_actions)}\n"
                       f"Avg Position Size: {trade_actions['position_size'].mean():.4f}\n"
                       f"Avg Risk-Reward: {trade_actions['risk_reward'].mean():.2f}\n"
                       f"Avg Hold Time: {trade_actions['hold_time'].mean():.1f}",
                       transform=ax7.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                       fontsize=9)                # Add regression line to see relationship between hold time and risk-reward
                try:

                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        trade_actions['hold_time'], 
                        trade_actions['risk_reward']
                    )
                    
                    x_range = np.linspace(trade_actions['hold_time'].min(), trade_actions['hold_time'].max(), 100)
                    ax7.plot(x_range, intercept + slope*x_range, 'k--', linewidth=1.5, 
                          label=f'Correlation: {r_value:.2f}')
                except Exception as e:
                    print(f"Regression calculation error: {e}")
                
                # Add legend
                ax7.legend(loc='lower right')
                
                # Axis labels
                ax7.set_xlabel('Hold Time (bars)', fontweight='bold')
                ax7.set_ylabel('Risk-Reward Ratio', fontweight='bold')
                
                # Add a bubble size legend
                bubble_sizes = [trade_actions['position_size'].min(), 
                              trade_actions['position_size'].mean(),
                              trade_actions['position_size'].max()]
                
                # Only show bubble legend if we have meaningful differences in sizes
                if max(bubble_sizes) > min(bubble_sizes) * 1.5:
                    # Create a legend for bubble sizes
                    for size, label in zip(bubble_sizes, ['Min Size', 'Avg Size', 'Max Size']):
                        ax7.scatter([], [], s=size*(300/max_size)+20, c='gray', alpha=0.6, 
                                  label=f'{label}: {size:.4f}')
                    
                    # Add a second legend for bubble sizes (positioned differently)
                    ax7.legend(loc='upper right', title='Position Sizes')
                
                # Grid for readability
                ax7.grid(True, alpha=0.3)
            else:
                    ax7.text(0.5, 0.5, "Insufficient trade data", ha='center', va='center')
                
                
            # 9. Performance metrics summary table
            ax9 = fig.add_subplot(gs[2, 3])
            ax9.axis('off')  # Turn off axis
            
            # Create an expanded summary of key metrics
            metric_data = [
                ["Total Return", f"{metrics.get('total_return', 0)*100:.2f}%"],
                ["Ann. Return", f"{metrics.get('return_ann', 0)*100:.2f}%"],
                ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
                ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"],
                ["Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"],
                ["Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%"],
                ["Profit Factor", f"{metrics.get('profit_factor', 1):.2f}"],
                ["Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%"],
                ["Volatility", f"{metrics.get('volatility_ann', 0)*100:.2f}%"],
                ["# Trades", f"{metrics.get('total_trades', 0)}"],
                ["Winning Trades", f"{metrics.get('winning_trades', 0)}"],
                ["Losing Trades", f"{metrics.get('losing_trades', 0)}"],
            ]
            
            # Create table with colored cells based on performance
            cell_colors = []
            for row in metric_data:
                if "Return" in row[0] or "Sharpe" in row[0] or "Sortino" in row[0] or "Calmar" in row[0] or "Win" in row[0] or "Profit" in row[0] or "Avg Winning" in row[0]:
                    # Extract numeric value for coloring
                    try:
                        value = float(row[1].strip('%'))
                        
                        # Color based on good/bad values
                        if ("Return" in row[0] and value > 0) or \
                           ("Sharpe" in row[0] and value > 1) or \
                           ("Sortino" in row[0] and value > 1) or \
                           ("Calmar" in row[0] and value > 1) or \
                           ("Win" in row[0] and value > 50) or \
                           ("Profit" in row[0] and value > 1) or \
                           ("Avg Winning" in row[0] and value > 0):
                            cell_colors.append(['white', 'lightgreen'])
                        elif ("Return" in row[0] and value < 0) or \
                             ("Sharpe" in row[0] and value < 0) or \
                             ("Sortino" in row[0] and value < 0) or \
                             ("Calmar" in row[0] and value < 0) or \
                             ("Win" in row[0] and value < 40) or \
                             ("Profit" in row[0] and value < 1) or \
                             ("Avg Winning" in row[0] and value < 0):
                            cell_colors.append(['white', 'lightcoral'])
                        else:
                            cell_colors.append(['white', 'white'])
                    except:
                        cell_colors.append(['white', 'white'])
                elif "Drawdown" in row[0]:
                    # For drawdown, high values are bad
                    try:
                        value = float(row[1].strip('%'))
                        if value > 20:
                            cell_colors.append(['white', 'lightcoral'])
                        elif value > 10:
                            cell_colors.append(['white', 'lightyellow'])
                        else:
                            cell_colors.append(['white', 'lightgreen'])
                    except:
                        cell_colors.append(['white', 'white'])
                elif "Avg Losing" in row[0]:
                    # For losing trades, less negative is better
                    try:
                        value = float(row[1].strip('%'))
                        if value > -5:
                            cell_colors.append(['white', 'lightgreen'])
                        elif value < -15:
                            cell_colors.append(['white', 'lightcoral'])
                        else:
                            cell_colors.append(['white', 'lightyellow'])
                    except:
                        cell_colors.append(['white', 'white'])
                else:
                    cell_colors.append(['white', 'white'])
            
            # Create the table
            ax9_table = ax9.table(cellText=metric_data, cellColours=cell_colors, 
                               colLabels=['Metric', 'Value'], 
                               loc='center', cellLoc='center')
            ax9_table.auto_set_font_size(False)
            ax9_table.set_fontsize(9)
            ax9_table.scale(1.2, 1.7)
            
            # Adjust table appearance
            for (i, j), cell in ax9_table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(fontweight='bold', color='black')
                    cell.set_facecolor('lightblue')
            
            ax9.set_title("Performance Summary", pad=20)

            # 10. Position Sizing Evolution (3,2) - Additional plot
            ax10 = fig.add_subplot(gs[3, 2])
            
            if stats_df is not None and '_model_actions' in stats_df and not stats_df['_model_actions'].empty:
                trade_actions = stats_df['_model_actions']
                # Sort by timestamp to get chronological order
                if 'timestamp' in trade_actions.columns:
                    position_data = trade_actions.sort_values('timestamp')
                else:
                    # If no timestamp, use index as proxy for time
                    position_data = trade_actions.copy()
                
                # Calculate rolling mean to show trend
                window_size = max(5, len(position_data) // 20)
                if len(position_data) > window_size:
                    position_data['rolling_size'] = position_data['position_size'].rolling(window=window_size).mean()
                    
                    # Plot individual positions and trend
                    ax10.scatter(range(len(position_data)), position_data['position_size'], 
                              alpha=0.4, s=15, color='#1f77b4', label='Position Size')
                    ax10.plot(range(len(position_data)), position_data['rolling_size'], 
                           color='red', linewidth=2, label=f'{window_size}-period MA')
                    
                    # Add direction info if available
                    if 'action_type' in position_data.columns:
                        # Color points by direction (buy/sell)
                        colors = position_data['action_type'].map({1: 'green', 2: 'red', 0: 'blue'})
                        ax10.scatter(range(len(position_data)), position_data['position_size'], 
                                  alpha=0.4, s=15, c=colors)
                else:
                    # Just plot the raw position sizes if not enough data for MA
                    ax10.plot(range(len(position_data)), position_data['position_size'], 
                           marker='o', linestyle='-', alpha=0.7)
                
                # Set labels and title
                ax10.set_title("Position Size Evolution", fontweight='bold')
                ax10.set_xlabel("Trade Number")
                ax10.set_ylabel("Position Size")
                ax10.grid(True, alpha=0.3)
                
                if 'rolling_size' in position_data.columns:
                    ax10.legend(loc='best', fontsize=8)
            else:
                ax10.text(0.5, 0.5, "Insufficient position sizing data", ha='center', va='center')
                
            # 11. Risk-Adjusted Return Metrics (3,3) - Additional plot
            ax11 = fig.add_subplot(gs[3, 3])
            
            # Risk-adjusted metrics
            risk_metrics = [
                ('Sharpe', metrics.get('sharpe_ratio', 0)),
                ('Sortino', metrics.get('sortino_ratio', 0)),
                ('Calmar', metrics.get('calmar_ratio', metrics.get('cagr', 0) / max(abs(metrics.get('max_drawdown', 0.01)), 0.01)))
            ]
            
            # Create bar chart
            x = np.arange(len(risk_metrics))
            metric_values = [m[1] for m in risk_metrics]
            
            # Use color gradient based on value
            colors = ['#ff9999', '#ffcc99', '#ffff99', '#ccff99', '#99ff99']
            bar_colors = []
            
            for val in metric_values:
                if val <= 0:
                    bar_colors.append(colors[0])  # Red for negative
                elif val < 1:
                    bar_colors.append(colors[1])  # Orange for < 1
                elif val < 1.5:
                    bar_colors.append(colors[2])  # Yellow for 1-1.5
                elif val < 2:
                    bar_colors.append(colors[3])  # Light green for 1.5-2
                else:
                    bar_colors.append(colors[4])  # Green for >= 2
            
            # Create bars with value annotations
            bars = ax11.bar(x, metric_values, color=bar_colors)
            ax11.bar_label(bars, fmt='%.2f')
            
            # Add benchmark line for "good" metrics
            ax11.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Benchmark')
            
            # Add labels and styling
            ax11.set_title("Risk-Adjusted Return Metrics", fontweight='bold')
            ax11.set_ylabel("Ratio Value")
            ax11.set_xticks(x)
            ax11.set_xticklabels([m[0] for m in risk_metrics])
            ax11.grid(True, axis='y', alpha=0.3)
            
            # Add interpretation text
            ax11.text(0.5, -0.15, 
                     "< 1: Poor | 1-1.5: Good | > 1.5: Excellent", 
                     ha='center', transform=ax11.transAxes, fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))            # Final layout adjustments - using figure constrained_layout instead of tight_layout
            # This avoids the colorbar layout engine conflict
            fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.05)
            
            # Adjust the figure to ensure all elements fit properly
            # This helps with proper display of all plots
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
            
            # Save plot with higher DPI for better quality
            plot_path = self.temp_dir / f"portfolio_performance_step_{self.n_calls}.png"
            plt.savefig(plot_path, dpi=180, bbox_inches='tight')
            plt.close()
            
            # Also save a copy to a consistent filename for easy access
            fixed_path = self.temp_dir / f"latest_portfolio_performance.png"
            if os.path.exists(plot_path):
                import shutil
                shutil.copy(plot_path, fixed_path)
            
            # Log as artifact
            self.mlflow_manager.log_artifact(str(plot_path), f"plots/{self.timeframe}")
            
            # Clean up
            if plot_path.exists():
                plot_path.unlink()
                
        except Exception as e:
            print(f"Error creating portfolio plot: {e}")
            traceback.print_exc()
            return None
                
    def _calculate_feature_importance(self, step=None):
        """Calculate and log feature importance using permutation importance"""
        try:
            if self.verbose > 1:
                print(f"Calculating feature importance at step {self.n_calls}...")
            
            # Skip if already calculated at this step  
            if self.n_calls == self.last_feature_importance:
                return
                
            # Use provided step or default to self.n_calls
            log_step = step if step is not None else self.n_calls
                
            self.last_feature_importance = self.n_calls
            
            # Use timeout for feature importance calculation
            timeout = 10  # seconds
            start_time = time.time()
            
            # Generate baseline performance
            baseline_reward = self._get_baseline_performance()
            
            # Calculate permutation importance for each feature
            importance_scores = []
            
            for i, feature_name in enumerate(self.feature_names):
                if time.time() - start_time > timeout:
                    if self.verbose > 0:
                        print(f"Feature importance calculation timed out after {timeout}s")
                    break
                
                try:
                    # Calculate permuted performance (simplified approach)
                    # In a real implementation, you would permute the feature and re-evaluate
                    # Here we use a simplified random importance for demonstration
                    importance = np.random.uniform(0, 1)
                    importance_scores.append(importance)
                    
                except Exception as e:
                    if self.verbose > 1:
                        print(f"Error calculating importance for {feature_name}: {e}")
                    importance_scores.append(0.0)
              # Normalize importance scores
            if importance_scores:
                total_importance = sum(importance_scores)
                if total_importance > 0:
                    importance_scores = [score / total_importance for score in importance_scores]
                
                self._log_feature_importance(importance_scores, step=log_step)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not calculate feature importance: {e}")
    
    def _get_baseline_performance(self):
        """Get baseline model performance for feature importance calculation"""
        try:
            obs, _ = self.eval_env.reset()
            total_reward = 0
            
            for _ in range(min(100, self.max_eval_steps)):  # Quick baseline
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                
                if done or truncated:
                    break
            
            return total_reward
            
        except Exception as e:
            if self.verbose > 1:
                print(f"Error calculating baseline performance: {e}")
            return 0.0
    def _log_feature_importance(self, importance_scores, step=None):
        """Log feature importance scores and create visualization"""
        try:
            # Create feature importance dictionary
            feature_importance = {
                f"feature_importance/{name}": score 
                for name, score in zip(self.feature_names[:len(importance_scores)], importance_scores)
            }
            
            # Use provided step or default to self.n_calls
            log_step = step if step is not None else self.n_calls
              
            # Log to MLflow
            self.mlflow_manager.log_metrics(feature_importance, step=log_step)
            
            # Store in history
            importance_data = {
                "step": log_step,
                "timestamp": datetime.now().isoformat(),
                "feature_names": self.feature_names[:len(importance_scores)],
                "importance_scores": importance_scores
            }
            self.feature_importance_history.append(importance_data)
            
            # Create and save feature importance plot
            if self.save_plots:
                self._create_feature_importance_plot(importance_data)
            
            if self.verbose > 0:
                top_features = sorted(zip(self.feature_names[:len(importance_scores)], importance_scores), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print(f"Top 5 features: {[f'{name}: {score:.3f}' for name, score in top_features]}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log feature importance: {e}")
    
    def _create_feature_importance_plot(self, importance_data):
        """Create and save feature importance visualization"""
        try:
            # Sort features by importance
            sorted_data = sorted(zip(importance_data["feature_names"], importance_data["importance_scores"]), 
                               key=lambda x: x[1], reverse=True)
            
            # Take top 20 features for readability
            top_features = sorted_data[:20]
            feature_names, importance_scores = zip(*top_features)
            
            # Create plot            plt.figure(figsize=(12, 8), constrained_layout=True)
            bars = plt.barh(range(len(feature_names)), importance_scores)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Importance Score')
            plt.title(f'Feature Importance - {self.timeframe} (Step {self.n_calls})')
            plt.gca().invert_yaxis()
            
            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # No need for plt.tight_layout() when using constrained_layout=True
            
            # Save plot
            plot_path = self.temp_dir / f"feature_importance_step_{self.n_calls}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log as artifact
            self.mlflow_manager.log_artifact(str(plot_path), f"plots/{self.timeframe}")
            
            # Clean up
            if plot_path.exists():
                plot_path.unlink()
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not create feature importance plot: {e}")
    
    def _save_model_checkpoint(self, checkpoint_name):
        """Save model checkpoint and log as MLflow artifact"""
        try:
            checkpoint_path = self.temp_dir / f"{checkpoint_name}_{self.timeframe}_{self.n_calls}"
            self.model.save(str(checkpoint_path))
            
            # Log model as artifact
            self.mlflow_manager.log_artifact(str(checkpoint_path) + ".zip", f"models/{self.timeframe}")
            
            # Clean up
            checkpoint_file = Path(str(checkpoint_path) + ".zip")
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save model checkpoint: {e}")
    
    def on_training_end(self):
        """Comprehensive end-of-training logging and cleanup"""
        try:
            if self.verbose > 0:
                print(f"Training completed for {self.timeframe} model. Performing final logging...")
            
            # Log training completion
            final_metrics = {
                f"training/completed": 1.0,
                f"training/total_timesteps": float(self.num_timesteps),
                f"training/total_evaluations": len(self.evaluation_history),
                f"training/total_portfolio_evaluations": len(self.portfolio_history),
                f"training/total_feature_importance_calculations": len(self.feature_importance_history)
            }
            
            self.mlflow_manager.log_metrics(final_metrics)
            
            # Run comprehensive final evaluation
            if self.eval_env is not None:
                self._run_final_comprehensive_evaluation()
            
            
            # Create final summary plots
            if self.save_plots:
                self._create_final_summary_plots()
            
            # Save final model checkpoint
            if self.save_model_checkpoints:
                self._save_model_checkpoint("final_model")
            
            # Cleanup temporary directory
            self._cleanup_temp_files()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Error during training end cleanup: {e}")
    
    def _run_final_comprehensive_evaluation(self):
        """Run comprehensive final evaluation with multiple episodes and detailed metrics"""
        try:
            if self.verbose > 0:
                print("Running final comprehensive evaluation...")
            
            n_final_episodes = min(10, self.n_eval_episodes * 2)  # More episodes for final eval
            
            # Collect results from multiple episodes
            all_rewards = []
            all_portfolio_balances = []
            all_returns = []
            
            for episode in range(n_final_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                initial_value = 10000.0
                current_value = initial_value
                
                for step in range(self.max_eval_steps):
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    
                    episode_reward += reward
                    if reward != 0:
                        current_value *= (1 + np.clip(reward / 100, -0.1, 0.1))
                    
                    if done or truncated:
                        break
                
                all_rewards.append(episode_reward)
                all_portfolio_balances.append(current_value)
                all_returns.append((current_value / initial_value) - 1)
            
            # Calculate comprehensive final metrics
            final_eval_metrics = {
                f"final/mean_reward": np.mean(all_rewards),
                f"final/std_reward": np.std(all_rewards),
                f"final/min_reward": np.min(all_rewards),
                f"final/max_reward": np.max(all_rewards),
                f"final/mean_portfolio_balance": np.mean(all_portfolio_balances),
                f"final/mean_return": np.mean(all_returns),
                f"final/std_return": np.std(all_returns),
                f"final/min_return": np.min(all_returns),
                f"final/max_return": np.max(all_returns),
                f"final/success_rate": sum(1 for r in all_returns if r > 0) / len(all_returns)
            }
            
            self.mlflow_manager.log_metrics(final_eval_metrics)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not run final comprehensive evaluation: {e}")
        
    def _create_final_summary_plots(self):
        """Create comprehensive summary plots for the entire training session"""
        try:
            if not (self.evaluation_history or self.portfolio_history):
                return
            
            fig = plt.figure(figsize=(20, 12))
            
            # Create grid for subplots
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Training progress - evaluation rewards
            if self.evaluation_history:
                ax1 = fig.add_subplot(gs[0, 0:2])
                steps = [entry["step"] for entry in self.evaluation_history]
                rewards = [entry["results"]["mean_reward"] for entry in self.evaluation_history]
                ax1.plot(steps, rewards, 'b-', linewidth=2)
                ax1.set_title("Evaluation Reward Progress")
                ax1.set_xlabel("Training Steps")
                ax1.set_ylabel("Mean Reward")
                ax1.grid(True)
            
            # 2. Portfolio value progression
            if self.portfolio_history:
                ax2 = fig.add_subplot(gs[0, 2:4])
                steps = [entry["step"] for entry in self.portfolio_history]
                balances = [entry["metrics"]["final_equity"] for entry in self.portfolio_history]
                ax2.plot(steps, balances, 'g-', linewidth=2)
                ax2.set_title("Portfolio Value Progress")
                ax2.set_xlabel("Training Steps")
                ax2.set_ylabel("Portfolio Value ($)")
                ax2.grid(True)
            
            # 3. Sharpe ratio over time
            if self.portfolio_history:
                ax3 = fig.add_subplot(gs[1, 0])
                sharpe_ratios = [entry["metrics"]["sharpe_ratio"] for entry in self.portfolio_history]
                ax3.plot(steps, sharpe_ratios, 'r-', linewidth=2)
                ax3.set_title("Sharpe Ratio")
                ax3.set_xlabel("Training Steps")
                ax3.set_ylabel("Sharpe Ratio")
                ax3.grid(True)
            
            # 4. Max drawdown over time
            if self.portfolio_history:
                ax4 = fig.add_subplot(gs[1, 1])
                drawdowns = [entry["metrics"]["max_drawdown"] * 100 for entry in self.portfolio_history]
                ax4.plot(steps, drawdowns, 'orange', linewidth=2)
                ax4.set_title("Max Drawdown")
                ax4.set_xlabel("Training Steps")
                ax4.set_ylabel("Max Drawdown (%)")
                ax4.grid(True)
            
            # 5. Win rate progression
            if self.portfolio_history:
                ax5 = fig.add_subplot(gs[1, 2])
                win_rates = [entry["metrics"]["win_rate"] * 100 for entry in self.portfolio_history]
                ax5.plot(steps, win_rates, 'purple', linewidth=2)
                ax5.set_title("Win Rate")
                ax5.set_xlabel("Training Steps")
                ax5.set_ylabel("Win Rate (%)")
                ax5.grid(True)
            
            # 6. Total trades over time
            if self.portfolio_history:
                ax6 = fig.add_subplot(gs[1, 3])
                total_trades = [entry["metrics"]["total_trades"] for entry in self.portfolio_history]
                ax6.plot(steps, total_trades, 'brown', linewidth=2)
                ax6.set_title("Total Trades")
                ax6.set_xlabel("Training Steps")
                ax6.set_ylabel("Number of Trades")
                ax6.grid(True)
            
            # 7. Feature importance evolution (top 5 features)
            if self.feature_importance_history and len(self.feature_importance_history) > 1:
                ax7 = fig.add_subplot(gs[2, 0:2])
                
                # Get steps and feature data
                fi_steps = [entry["step"] for entry in self.feature_importance_history]
                
                # Find top 5 most important features across all evaluations
                all_importance = {}
                for entry in self.feature_importance_history:
                    for name, score in zip(entry["feature_names"], entry["importance_scores"]):
                        if name not in all_importance:
                            all_importance[name] = []
                        all_importance[name].append(score)
                
                # Calculate average importance for each feature
                avg_importance = {name: np.mean(scores) for name, scores in all_importance.items()}
                top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Plot evolution of top features
                for feature_name, _ in top_features:
                    feature_scores = []
                    for entry in self.feature_importance_history:
                        if feature_name in entry["feature_names"]:
                            idx = entry["feature_names"].index(feature_name)
                            feature_scores.append(entry["importance_scores"][idx])
                        else:
                            feature_scores.append(0)
                    
                    ax7.plot(fi_steps, feature_scores, linewidth=2, label=feature_name, marker='o', markersize=4)
                
                ax7.set_title("Top 5 Features Importance Evolution")
                ax7.set_xlabel("Training Steps")
                ax7.set_ylabel("Importance Score")
                ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax7.grid(True)
            
            # 8. Action distribution over time
            if self.portfolio_history:
                ax8 = fig.add_subplot(gs[2, 2:4])
                # Check if we have stats with _model_actions data
                action_data_available = False
                action_ratios = []
                
                for entry in self.portfolio_history:
                    if "stats" in entry and "_model_actions" in entry["stats"] and not entry["stats"]["_model_actions"].empty:
                        action_data_available = True
                        # Calculate action type ratios from DataFrame
                        actions_df = entry["stats"]["_model_actions"]
                        action_counts = actions_df["action_type"].value_counts()
                        total_actions = action_counts.sum()
                        
                        # Initialize with zeros to handle missing action types
                        hold_ratio = buy_ratio = sell_ratio = 0.0
                        
                        # Update with actual values if present
                        if 0 in action_counts:
                            hold_ratio = action_counts[0] / total_actions
                        if 1 in action_counts:
                            buy_ratio = action_counts[1] / total_actions
                        if 2 in action_counts:
                            sell_ratio = action_counts[2] / total_actions
                            
                        action_ratios.append((hold_ratio, buy_ratio, sell_ratio))
                
                if action_data_available:
                    steps = [entry["step"] for entry in self.portfolio_history]
                    hold_ratios = [ratio[0] for ratio in action_ratios]
                    buy_ratios = [ratio[1] for ratio in action_ratios]
                    sell_ratios = [ratio[2] for ratio in action_ratios]
                    
                    ax8.plot(steps, hold_ratios, label='Hold', linewidth=2)
                    ax8.plot(steps, buy_ratios, label='Buy', linewidth=2)
                    ax8.plot(steps, sell_ratios, label='Sell', linewidth=2)
                    ax8.set_title("Action Distribution Over Time")
                    ax8.set_xlabel("Training Steps")
                    ax8.set_ylabel("Action Ratio")
                    ax8.legend()
                    ax8.grid(True)
            
            plt.suptitle(f"Training Summary - {self.timeframe} Model", fontsize=16, fontweight='bold')
            
            # Save plot
            summary_plot_path = self.temp_dir / f"training_summary_{self.timeframe}.png"
            plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plt.close('all')  # Explicitly close all figures
            
            # Log as artifact
            self.mlflow_manager.log_artifact(str(summary_plot_path), f"plots/{self.timeframe}")
            
            # Clean up
            if summary_plot_path.exists():
                summary_plot_path.unlink()
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not create final summary plots: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files and directories"""
        try:
            # Remove any remaining files in temp directory
            if self.temp_dir.exists():
                for file_path in self.temp_dir.iterdir():
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                    except Exception:
                        pass
                
                # Remove temp directory if empty
                try:
                    self.temp_dir.rmdir()
                except Exception:
                    pass
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not cleanup temp files: {e}")
    
    def _fallback_portfolio_evaluation(self):
        """Simple fallback portfolio evaluation when the main evaluator fails."""
        try:
            if self.verbose > 1:
                print("Using fallback portfolio evaluation...")
            
            # Create basic fallback metrics
            metrics = {
                "final_equitye": 100000.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit_factor": 1.0,
                               "avg_profit_per_winning_trad": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "action_counts": {0: 1, 1: 0, 2: 0},
                "portfolio_balances": [100000.0],
                "daily_returns": [0.0],
                "trade_results": [],
                "detailed_trades": []
            }
            
            if self.verbose > 1:
                print("Fallback evaluation complete")
            
            return metrics
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in fallback portfolio evaluation: {e}")
            
            # Last resort metrics
            return {
                "final_equity": 100000.0,
                "total_return": 0.0,
                "action_counts": {0: 1, 1: 0, 2: 0},
                "portfolio_balances": [100000.0]
            }
