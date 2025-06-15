"""
Comprehensive Metrics Extractor for Backtesting.py

Extracts and processes built-in metrics from backtesting.py results
and formats them for MLflow integration and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """
    Professional metrics extractor for backtesting.py results.
    
    Extracts built-in metrics from backtesting.py and adds minimal
    RL-specific metrics for comprehensive performance analysis.
    """
    def __init__(self):
        """Initialize metrics extractor."""
        self.risk_free_rate = 0.02  # Default 2% risk-free rate
        
    def extract_all_metrics(self, stats: pd.Series) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from backtesting.py results.
        
        Leverages backtesting.py's built-in metrics directly and adds derived metrics.
        
        Args:
            stats: backtesting.py statistics Series (contains all built-in metrics)
            
        Returns:
            Dict with all extracted metrics
        """
        try:
            # Validate inputs
            if not isinstance(stats, (pd.Series, dict)):
                logger.warning(f"Expected stats to be Series or dict, got {type(stats)}")
                stats = pd.Series(stats) if stats else pd.Series()
            
            # Use built-in metrics directly
            core_metrics = self._extract_builtin_metrics(stats)
            
            # Extract additional derived metrics
            derived_metrics = self.get_derived_metrics(stats)
            
            # Combine all metrics
            all_metrics = {**core_metrics, **derived_metrics}
            
            logger.info(f"Extracted {len(all_metrics)} metrics ({len(core_metrics)} built-in, {len(derived_metrics)} derived)")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return self._get_fallback_metrics()
    
    def _extract_builtin_metrics(self, stats: pd.Series) -> Dict[str, Any]:
        """
        Extract all built-in metrics from backtesting.py results directly.
        """
        builtin_metrics = {}
        
        # Direct extraction of all built-in metrics
        metric_mappings = {
            # Portfolio performance
            'start_value': 'Start',
            'end_value': 'End', 
            'final_equity': 'Equity Final [$]',
            'equity_peak': 'Equity Peak [$]',
            
            # Returns
            'total_return': 'Return [%]',
            'return_ann': 'Return (Ann.) [%]',
            'buy_hold_return': 'Buy & Hold Return [%]',
            'exposure_time': 'Exposure Time [%]',
            
            # Risk metrics
            'max_drawdown': 'Max. Drawdown [%]',
            'avg_drawdown': 'Avg. Drawdown [%]',
            'max_drawdown_duration': 'Max. Drawdown Duration',
            'volatility_ann': 'Volatility (Ann.) [%]',
            
            # Risk-adjusted returns
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio', 
            'calmar_ratio': 'Calmar Ratio',
            
            # Trade metrics
            'total_trades': '# Trades',
            'win_rate': 'Win Rate [%]',
            'best_trade': 'Best Trade [%]',
            'worst_trade': 'Worst Trade [%]',
            'avg_trade': 'Avg. Trade [%]',
            'avg_trade_duration': 'Avg. Trade Duration',
            'max_trade_duration': 'Max. Trade Duration',
            'profit_factor': 'Profit Factor',
            'expectancy': 'Expectancy [%]',
            'sqn': 'SQN',
            'kelly_criterion': 'Kelly Criterion',
            
        }
        
        # Extract all mapped metrics with safe conversion
        for metric_name, stats_key in metric_mappings.items():
            value = stats.get(stats_key, 0)
            
            # Convert percentages to decimals for consistency
            if '[%]' in stats_key and isinstance(value, (int, float)):
                builtin_metrics[metric_name] = float(value) / 100.0
            else:
                builtin_metrics[metric_name] = (value)
                
        # Add calculated fields for compatibility
        total_trades = builtin_metrics.get('total_trades', 0) or 0  # Ensure not None
        win_rate = builtin_metrics.get('win_rate', 0) or 0  # Ensure not None
        
        # Handle NaN and None values safely
        if pd.isna(total_trades):
            total_trades = 0
        if pd.isna(win_rate):
            win_rate = 0
            
        winning_trades = int(total_trades * win_rate)
        losing_trades = int(total_trades * (1 - win_rate))
        
        builtin_metrics.update({
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'excess_return': (builtin_metrics.get('total_return', 0) or 0) - (builtin_metrics.get('buy_hold_return', 0) or 0),
            'final_portfolio_value': builtin_metrics.get('final_equity', 0) or 0,
        })
        
        return builtin_metrics
    
      # function to get the derived metrics
    def get_derived_metrics(self, stats: pd.Series) -> Dict[str, Any]:
        """
        Extract derived metrics from backtesting.py results.
        
        Args:
            stats: backtesting.py statistics Series
            
        Returns:
            Dict with derived metrics
        """
        try:
            # Validate inputs
            if not isinstance(stats, (pd.Series, dict)):
                logger.warning(f"Expected stats to be Series or dict, got {type(stats)}")
                stats = pd.Series(stats) if stats else pd.Series()
            
            derived_metrics = {}
            
            # Process trade data if available
            if '_trades' in stats and not stats['_trades'].empty:
                trades_df = stats['_trades']
                derived_metrics.update(self._extract_trade_metrics(trades_df))
            
            # Process equity curve if available
            if '_equity_curve' in stats and not stats['_equity_curve'].empty:
                equity_curve_df = stats['_equity_curve']
                derived_metrics.update(self._extract_equity_curve_metrics(equity_curve_df))
            
            # Add strategy consistency metrics
            if '_trades' in stats and '_equity_curve' in stats:
                derived_metrics.update(self._extract_consistency_metrics(
                    stats['_trades'], 
                    stats['_equity_curve']
                ))
            
            return derived_metrics
            
        except Exception as e:
            logger.error(f"Error extracting derived metrics: {e}")
            return self._get_fallback_metrics()
    
    def _extract_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract advanced metrics from trade data."""
        metrics = {}
        
        try:
            if trades_df.empty:
                return metrics
              # Convert duration to numeric for calculations (in hours)
            if 'Duration' in trades_df.columns:
                # Make a copy of the DataFrame and extract hours from timedelta
                trades_df = trades_df.copy()  # Create an explicit copy to avoid SettingWithCopyWarning
                trades_df.loc[:, 'duration_hours'] = trades_df['Duration'].apply(
                    lambda x: x.total_seconds() / 3600 if pd.notna(x) else 0
                )
            
            # Position analysis
            metrics['long_trades'] = len(trades_df[trades_df['Size'] > 0])
            metrics['short_trades'] = len(trades_df[trades_df['Size'] < 0])
            metrics['long_ratio'] = metrics['long_trades'] / max(1, len(trades_df))
            metrics['short_ratio'] = metrics['short_trades'] / max(1, len(trades_df))
            
            # Profit/loss analysis
            profitable_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]
            
            # Profitability by position type
            if metrics['long_trades'] > 0:
                long_profitable = len(profitable_trades[profitable_trades['Size'] > 0])
                metrics['long_win_rate'] = long_profitable / metrics['long_trades']
            else:
                metrics['long_win_rate'] = 0
                
            if metrics['short_trades'] > 0:
                short_profitable = len(profitable_trades[profitable_trades['Size'] < 0])
                metrics['short_win_rate'] = short_profitable / metrics['short_trades']
            else:
                metrics['short_win_rate'] = 0
            
            # Average trade duration
            metrics['avg_trade_duration_hours'] = trades_df['duration_hours'].mean() if 'duration_hours' in trades_df else 0
            metrics['avg_winning_trade_duration'] = profitable_trades['duration_hours'].mean() if 'duration_hours' in trades_df and not profitable_trades.empty else 0
            metrics['avg_losing_trade_duration'] = losing_trades['duration_hours'].mean() if 'duration_hours' in trades_df and not losing_trades.empty else 0
            
            # Risk management metrics
            if 'SL' in trades_df.columns and 'TP' in trades_df.columns:
                trades_with_sl = trades_df[pd.notna(trades_df['SL'])]
                trades_with_tp = trades_df[pd.notna(trades_df['TP'])]
                
                metrics['sl_usage_ratio'] = len(trades_with_sl) / max(1, len(trades_df))
                metrics['tp_usage_ratio'] = len(trades_with_tp) / max(1, len(trades_df))
                if not trades_with_sl.empty and not trades_with_tp.empty:
                    # Calculate average risk-reward ratio
                    long_trades = trades_df[trades_df['Size'] > 0]
                    short_trades = trades_df[trades_df['Size'] < 0]
                    
                    if not long_trades.empty:
                        # Filter for trades with SL and TP
                        long_trades_with_sl_tp = long_trades[pd.notna(long_trades['SL']) & pd.notna(long_trades['TP'])].copy()
                        if not long_trades_with_sl_tp.empty:
                            # Create a copy and use proper .loc accessor to avoid SettingWithCopyWarning
                            long_trades_with_sl_tp.loc[:, 'risk'] = abs(long_trades_with_sl_tp['EntryPrice'] - long_trades_with_sl_tp['SL'])
                            long_trades_with_sl_tp.loc[:, 'reward'] = abs(long_trades_with_sl_tp['TP'] - long_trades_with_sl_tp['EntryPrice'])
                            long_trades_with_sl_tp.loc[:, 'risk_reward_ratio'] = long_trades_with_sl_tp['reward'] / long_trades_with_sl_tp['risk'].replace(0, np.nan)
                            metrics['long_avg_risk_reward'] = long_trades_with_sl_tp['risk_reward_ratio'].mean()
                    
                    if not short_trades.empty:
                        # Filter for trades with SL and TP
                        short_trades_with_sl_tp = short_trades[pd.notna(short_trades['SL']) & pd.notna(short_trades['TP'])].copy()
                        if not short_trades_with_sl_tp.empty:
                            # Create a copy and use proper .loc accessor to avoid SettingWithCopyWarning
                            short_trades_with_sl_tp.loc[:, 'risk'] = abs(short_trades_with_sl_tp['EntryPrice'] - short_trades_with_sl_tp['SL'])
                            short_trades_with_sl_tp.loc[:, 'reward'] = abs(short_trades_with_sl_tp['EntryPrice'] - short_trades_with_sl_tp['TP'])
                            short_trades_with_sl_tp.loc[:, 'risk_reward_ratio'] = short_trades_with_sl_tp['reward'] / short_trades_with_sl_tp['risk'].replace(0, np.nan)
                            metrics['short_avg_risk_reward'] = short_trades_with_sl_tp['risk_reward_ratio'].mean()
            
            # PnL distribution metrics
            if 'PnL' in trades_df.columns:
                metrics['pnl_std'] = trades_df['PnL'].std()
                metrics['pnl_skew'] = trades_df['PnL'].skew()
                metrics['avg_profit_per_winning_trade'] = profitable_trades['PnL'].mean() if not profitable_trades.empty else 0
                metrics['avg_loss_per_losing_trade'] = losing_trades['PnL'].mean() if not losing_trades.empty else 0
                  # Maximum consecutive winning and losing trades
                if 'ReturnPct' in trades_df.columns:
                    # Create a fresh copy for sorting to avoid SettingWithCopyWarning
                    trades_df_sorted = trades_df.copy().sort_values('EntryBar')
                    win_loss = (trades_df_sorted['ReturnPct'] > 0).astype(int)
                    
                    # Count streaks
                    win_loss_shift = win_loss.shift(1).fillna(0).astype(int)
                    streak_start = (win_loss != win_loss_shift) | (win_loss_shift == 0)
                    streak_id = streak_start.cumsum()
                    
                    streak_df = pd.DataFrame({
                        'win': win_loss,
                        'streak_id': streak_id
                    })
                    
                    streak_lengths = streak_df.groupby(['streak_id', 'win']).size().reset_index(name='length')
                    
                    win_streaks = streak_lengths[streak_lengths['win'] == 1]['length']
                    loss_streaks = streak_lengths[streak_lengths['win'] == 0]['length']
                    
                    metrics['max_consecutive_wins'] = win_streaks.max() if not win_streaks.empty else 0
                    metrics['max_consecutive_losses'] = loss_streaks.max() if not loss_streaks.empty else 0
                    metrics['avg_win_streak'] = win_streaks.mean() if not win_streaks.empty else 0
                    metrics['avg_loss_streak'] = loss_streaks.mean() if not loss_streaks.empty else 0
              # Time-based metrics
            if 'EntryTime' in trades_df.columns:
                # Ensure we're working with a copy to avoid SettingWithCopyWarning
                trades_df = trades_df.copy() 
                trades_df.loc[:, 'EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
                trades_df.loc[:, 'entry_hour'] = trades_df['EntryTime'].dt.hour
                trades_df.loc[:, 'entry_day_of_week'] = trades_df['EntryTime'].dt.dayofweek
                
                # Group trades by hour and calculate win rate
                hourly_performance = trades_df.groupby('entry_hour')['PnL'].agg(['mean', 'count', lambda x: (x > 0).mean()]).reset_index()
                hourly_performance.columns = ['hour', 'avg_pnl', 'trade_count', 'win_rate']
                
                # Find best and worst trading hours
                best_hour = hourly_performance.loc[hourly_performance['win_rate'].idxmax()]
                worst_hour = hourly_performance.loc[hourly_performance['win_rate'].idxmin()]
                
                metrics['best_hour'] = int(best_hour['hour'])
                metrics['best_hour_win_rate'] = float(best_hour['win_rate'])
                metrics['worst_hour'] = int(worst_hour['hour'])
                metrics['worst_hour_win_rate'] = float(worst_hour['win_rate'])
                
                # Day of week analysis
                daily_performance = trades_df.groupby('entry_day_of_week')['PnL'].agg(['mean', 'count', lambda x: (x > 0).mean()]).reset_index()
                daily_performance.columns = ['day', 'avg_pnl', 'trade_count', 'win_rate']
                
                best_day = daily_performance.loc[daily_performance['win_rate'].idxmax()]
                worst_day = daily_performance.loc[daily_performance['win_rate'].idxmin()]
                
                metrics['best_day'] = int(best_day['day'])
                metrics['best_day_win_rate'] = float(best_day['win_rate'])
                metrics['worst_day'] = int(worst_day['day'])
                metrics['worst_day_win_rate'] = float(worst_day['win_rate'])
            
        except Exception as e:
            logger.error(f"Error extracting trade metrics: {e}")
        
        return metrics
    
    def _extract_equity_curve_metrics(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metrics from equity curve data."""
        metrics = {}
        
        try:
            if equity_df.empty:
                return metrics
              # Calculate daily returns
            if isinstance(equity_df.index, pd.DatetimeIndex):
                # Make a copy of the DataFrame to avoid SettingWithCopyWarning
                equity_df = equity_df.copy()
                equity_df.loc[:, 'daily_return'] = equity_df['Equity'].pct_change()
                
                # Skip first row which will be NaN
                daily_returns = equity_df['daily_return'].iloc[1:]
                
                # Remove outliers for more accurate volatility calculation
                q1 = daily_returns.quantile(0.25)
                q3 = daily_returns.quantile(0.75)
                iqr = q3 - q1
                outlier_mask = (daily_returns >= q1 - 1.5 * iqr) & (daily_returns <= q3 + 1.5 * iqr)
                filtered_returns = daily_returns[outlier_mask]
                
                metrics['daily_return_mean'] = daily_returns.mean()
                metrics['daily_return_median'] = daily_returns.median()
                metrics['daily_return_std'] = daily_returns.std()
                metrics['daily_return_skew'] = daily_returns.skew()
                metrics['daily_return_kurtosis'] = daily_returns.kurtosis()
                
                # Calculate positive vs negative days
                positive_days = (daily_returns > 0).sum()
                negative_days = (daily_returns < 0).sum()
                total_days = len(daily_returns)
                
                metrics['positive_days_ratio'] = positive_days / total_days if total_days > 0 else 0
                metrics['negative_days_ratio'] = negative_days / total_days if total_days > 0 else 0
                
                # Calculate maximum drawdown duration in days
                if 'DrawdownDuration' in equity_df.columns:
                    max_dd_duration = equity_df['DrawdownDuration'].max()
                    if pd.notna(max_dd_duration):
                        metrics['max_drawdown_days'] = max_dd_duration.days
                
                # Calculate recovery periods
                if 'DrawdownPct' in equity_df.columns:
                    # Find peaks in equity curve (where drawdown is 0)
                    peaks = equity_df[equity_df['DrawdownPct'] == 0].index
                    if len(peaks) > 1:
                        # Calculate time between peaks
                        peak_diffs = [(peaks[i+1] - peaks[i]).days for i in range(len(peaks)-1)]
                        metrics['avg_recovery_days'] = sum(peak_diffs) / len(peak_diffs) if peak_diffs else 0
                
                # Calculate drawdown statistics
                if 'DrawdownPct' in equity_df.columns:
                    drawdowns = equity_df['DrawdownPct']
                    metrics['drawdown_std'] = drawdowns.std()
                    metrics['time_in_drawdown_pct'] = (drawdowns > 0).mean()
                    metrics['avg_drawdown_duration_days'] = equity_df['DrawdownDuration'].dropna().mean().days if 'DrawdownDuration' in equity_df else 0
            
            # Calculate equity curve stability
            equity_values = equity_df['Equity'].values
            equity_log_returns = np.diff(np.log(equity_values))
            metrics['equity_curve_smoothness'] = 1.0 / (1.0 + np.std(equity_log_returns)) if len(equity_log_returns) > 0 else 0
            
            # Calculate linear regression on equity curve to assess trend strength
            if len(equity_values) > 2:
                x = np.arange(len(equity_values)).reshape(-1, 1)
                y = equity_values
                
                # Simple linear regression
                try:
                    slope = np.polyfit(x.flatten(), y, 1)[0]
                    metrics['equity_curve_trend'] = slope
                except:
                    metrics['equity_curve_trend'] = 0
            
        except Exception as e:
            logger.error(f"Error extracting equity curve metrics: {e}")
        
        return metrics
    
    def _extract_consistency_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract consistency and strategy behavior metrics."""
        metrics = {}
        
        try:
            if trades_df.empty or equity_df.empty:
                return metrics
            
            # Split the equity curve into segments and analyze performance in each
            if isinstance(equity_df.index, pd.DatetimeIndex) and len(equity_df) > 10:
                # Divide the equity curve into 3 equal time periods
                segment_size = len(equity_df) // 3
                
                first_segment = equity_df.iloc[:segment_size]
                middle_segment = equity_df.iloc[segment_size:segment_size*2]
                last_segment = equity_df.iloc[segment_size*2:]
                
                # Calculate returns in each segment
                if 'Equity' in first_segment.columns and 'Equity' in middle_segment.columns and 'Equity' in last_segment.columns:
                    first_return = (first_segment['Equity'].iloc[-1] / first_segment['Equity'].iloc[0]) - 1 if first_segment['Equity'].iloc[0] > 0 else 0
                    middle_return = (middle_segment['Equity'].iloc[-1] / middle_segment['Equity'].iloc[0]) - 1 if middle_segment['Equity'].iloc[0] > 0 else 0
                    last_return = (last_segment['Equity'].iloc[-1] / last_segment['Equity'].iloc[0]) - 1 if last_segment['Equity'].iloc[0] > 0 else 0
                    
                    metrics['first_period_return'] = first_return
                    metrics['middle_period_return'] = middle_return
                    metrics['last_period_return'] = last_return
                    
                    # Calculate return consistency
                    returns = [first_return, middle_return, last_return]
                    metrics['return_consistency'] = 1.0 / (1.0 + np.std(returns)) if len(returns) > 0 and np.std(returns) > 0 else 0
                    
                    # Check if performance is improving or deteriorating
                    metrics['performance_trend'] = np.polyfit([1, 2, 3], returns, 1)[0] if len(returns) == 3 else 0
              # Trade frequency consistency
            if 'EntryTime' in trades_df.columns:
                # Create a copy to avoid SettingWithCopyWarning
                trades_df = trades_df.copy()
                trades_df.loc[:, 'EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
                
                if trades_df['EntryTime'].min() != trades_df['EntryTime'].max():
                    # Calculate number of days in the trading period
                    days_elapsed = (trades_df['EntryTime'].max() - trades_df['EntryTime'].min()).total_seconds() / (24 * 3600)
                    
                    if days_elapsed > 0:
                        # Calculate trades per day
                        trades_per_day = len(trades_df) / days_elapsed
                        metrics['trades_per_day'] = trades_per_day
                        
                        # Group trades by day
                        trades_df.loc[:, 'trade_date'] = trades_df['EntryTime'].dt.date
                        trades_by_day = trades_df.groupby('trade_date').size()
                        
                        # Calculate trade frequency consistency
                        metrics['trade_frequency_std'] = trades_by_day.std()
                        metrics['trade_frequency_consistency'] = 1.0 / (1.0 + metrics['trade_frequency_std']) if metrics['trade_frequency_std'] > 0 else 1.0
              # Win rate stability over time
            if 'EntryTime' in trades_df.columns and 'ReturnPct' in trades_df.columns:
                # Create a copy and sort to avoid SettingWithCopyWarning
                trades_df_sorted = trades_df.copy().sort_values('EntryTime')
                
                # Split trades into 3 equal chunks by time
                chunk_size = len(trades_df_sorted) // 3
                if chunk_size > 0:
                    first_chunk = trades_df_sorted.iloc[:chunk_size]
                    middle_chunk = trades_df_sorted.iloc[chunk_size:chunk_size*2]
                    last_chunk = trades_df_sorted.iloc[chunk_size*2:]
                    
                    first_win_rate = (first_chunk['ReturnPct'] > 0).mean() if not first_chunk.empty else 0
                    middle_win_rate = (middle_chunk['ReturnPct'] > 0).mean() if not middle_chunk.empty else 0
                    last_win_rate = (last_chunk['ReturnPct'] > 0).mean() if not last_chunk.empty else 0
                    
                    metrics['first_period_win_rate'] = first_win_rate
                    metrics['middle_period_win_rate'] = middle_win_rate
                    metrics['last_period_win_rate'] = last_win_rate
                    
                    win_rates = [first_win_rate, middle_win_rate, last_win_rate]
                    metrics['win_rate_consistency'] = 1.0 / (1.0 + np.std(win_rates)) if np.std(win_rates) > 0 else 1.0
                    metrics['win_rate_trend'] = np.polyfit([1, 2, 3], win_rates, 1)[0] if len(win_rates) == 3 else 0
            
        except Exception as e:
            logger.error(f"Error extracting consistency metrics: {e}")
        
        return metrics
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Return fallback metrics in case of errors."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'final_equity': 0.0,
            'volatility_ann': 0.0,
            'error': 'Fallback metrics due to extraction error'
        }
