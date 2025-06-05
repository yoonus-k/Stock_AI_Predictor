"""
Advanced Metrics Extractor for Backtrader RL Trading

This module provides comprehensive metrics extraction and calculation for RL trading strategies.
It computes 50+ professional-grade trading metrics including risk-adjusted returns,
drawdown analysis, trade quality metrics, and RL-specific performance indicators.

Key Features:
- 50+ institutional-grade trading metrics
- Risk-adjusted performance measures
- Advanced drawdown and recovery analysis
- Trade quality and consistency metrics
- RL-specific action pattern analysis
- Market regime adaptation metrics
- Portfolio construction analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """
    Comprehensive metrics extractor for RL trading strategies.
    
    Computes institutional-grade performance metrics including:
    - Return metrics (total, annualized, risk-adjusted)
    - Risk metrics (volatility, VaR, CVaR, max drawdown)
    - Trade metrics (win rate, profit factor, trade quality)
    - Advanced metrics (Sortino, Calmar, Kelly, Information ratio)
    - RL-specific metrics (action consistency, regime adaptation)
    """
    
    def __init__(self):
        """Initialize the metrics extractor."""
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days_per_year = 252
        self.hours_per_trading_day = 24  # For 24/7 forex/crypto markets
        
    def extract_comprehensive_metrics(
        self,
        trade_log: List[Dict],
        equity_curve: List[Dict],
        base_metrics: Dict[str, Any] = None,
        benchmark_returns: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive performance metrics.
        
        Args:
            trade_log: List of trade records
            equity_curve: List of equity curve points
            base_metrics: Base metrics from analyzers
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary containing 50+ comprehensive metrics
        """
        try:
            metrics = base_metrics.copy() if base_metrics else {}
            
            # Convert to DataFrames for easier processing
            trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
            equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
            
            # Basic validation
            if trades_df.empty and equity_df.empty:
                return self._get_minimal_metrics(metrics)
            
            # Extract different categories of metrics
            metrics.update(self._calculate_return_metrics(equity_df, trades_df))
            metrics.update(self._calculate_risk_metrics(equity_df, trades_df))
            metrics.update(self._calculate_trade_metrics(trades_df))
            metrics.update(self._calculate_drawdown_metrics(equity_df))
            metrics.update(self._calculate_risk_adjusted_metrics(equity_df))
            metrics.update(self._calculate_consistency_metrics(trades_df, equity_df))
            metrics.update(self._calculate_efficiency_metrics(trades_df))
            metrics.update(self._calculate_rl_specific_metrics(trades_df))
            
            if benchmark_returns:
                metrics.update(self._calculate_relative_metrics(equity_df, benchmark_returns))
            
            # Calculate composite scores
            metrics.update(self._calculate_composite_scores(metrics))
            
            logger.info(f"Extracted {len(metrics)} comprehensive metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive metrics: {e}")
            return base_metrics if base_metrics else {}
    
    def _calculate_return_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate return-based performance metrics."""
        metrics = {}
        
        try:
            if not equity_df.empty:
                # Basic return calculations
                initial_equity = equity_df['equity'].iloc[0]
                final_equity = equity_df['equity'].iloc[-1]
                
                total_return = (final_equity - initial_equity) / initial_equity
                metrics['total_return'] = total_return
                
                # Time-based returns
                if 'datetime' in equity_df.columns:
                    duration_days = (equity_df['datetime'].iloc[-1] - equity_df['datetime'].iloc[0]).days
                    if duration_days > 0:
                        annualized_return = (1 + total_return) ** (365.25 / duration_days) - 1
                        metrics['annualized_return'] = annualized_return
                
                # Period returns
                equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
                
                if len(equity_df) > 1:
                    metrics.update({
                        'average_return': equity_df['returns'].mean(),
                        'median_return': equity_df['returns'].median(),
                        'geometric_mean_return': stats.gmean(1 + equity_df['returns'][equity_df['returns'] > -1]) - 1,
                        'compound_annual_growth_rate': (final_equity / initial_equity) ** (365.25 / max(duration_days, 1)) - 1
                    })
            
            # Trade-based returns
            if not trades_df.empty:
                metrics.update({
                    'average_trade_return': trades_df['pnl_pct'].mean(),
                    'median_trade_return': trades_df['pnl_pct'].median(),
                    'best_trade_return': trades_df['pnl_pct'].max(),
                    'worst_trade_return': trades_df['pnl_pct'].min(),
                    'trade_return_std': trades_df['pnl_pct'].std()
                })
            
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
        
        return metrics
    
    def _calculate_risk_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-based performance metrics."""
        metrics = {}
        
        try:
            if not equity_df.empty and len(equity_df) > 1:
                equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
                returns = equity_df['returns']
                
                # Volatility metrics
                daily_vol = returns.std()
                metrics.update({
                    'volatility_daily': daily_vol,
                    'volatility_annualized': daily_vol * np.sqrt(365.25),
                    'downside_volatility': returns[returns < 0].std(),
                    'upside_volatility': returns[returns > 0].std()
                })
                
                # VaR and CVaR
                var_95 = returns.quantile(0.05)
                var_99 = returns.quantile(0.01)
                cvar_95 = returns[returns <= var_95].mean()
                cvar_99 = returns[returns <= var_99].mean()
                
                metrics.update({
                    'var_95': var_95,
                    'var_99': var_99,
                    'cvar_95': cvar_95,
                    'cvar_99': cvar_99,
                    'expected_shortfall': cvar_95
                })
                
                # Distribution metrics
                metrics.update({
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'jarque_bera_stat': stats.jarque_bera(returns)[0],
                    'jarque_bera_pvalue': stats.jarque_bera(returns)[1]
                })
            
            # Trade-based risk metrics
            if not trades_df.empty:
                losing_trades = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct']
                
                if len(losing_trades) > 0:
                    metrics.update({
                        'average_loss': losing_trades.mean(),
                        'worst_loss': losing_trades.min(),
                        'loss_standard_deviation': losing_trades.std(),
                        'tail_ratio': abs(trades_df['pnl_pct'].quantile(0.95) / trades_df['pnl_pct'].quantile(0.05))
                    })
                
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-specific performance metrics."""
        metrics = {}
        
        try:
            if trades_df.empty:
                return metrics
            
            # Basic trade statistics
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl_pct'] > 0]
            losing_trades = trades_df[trades_df['pnl_pct'] < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            metrics.update({
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_count / total_trades if total_trades > 0 else 0,
                'loss_rate': loss_count / total_trades if total_trades > 0 else 0
            })
            
            # Profit factor and related metrics
            gross_profit = winning_trades['pnl_pct'].sum() if win_count > 0 else 0
            gross_loss = abs(losing_trades['pnl_pct'].sum()) if loss_count > 0 else 0.001  # Avoid division by zero
            
            metrics.update({
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
                'net_profit': gross_profit - gross_loss
            })
            
            # Average trade metrics
            if win_count > 0:
                metrics.update({
                    'average_winning_trade': winning_trades['pnl_pct'].mean(),
                    'largest_winning_trade': winning_trades['pnl_pct'].max(),
                    'winning_trade_std': winning_trades['pnl_pct'].std()
                })
            
            if loss_count > 0:
                metrics.update({
                    'average_losing_trade': losing_trades['pnl_pct'].mean(),
                    'largest_losing_trade': losing_trades['pnl_pct'].min(),
                    'losing_trade_std': losing_trades['pnl_pct'].std()
                })
            
            # Trade duration metrics
            if 'hold_time_hours' in trades_df.columns:
                metrics.update({
                    'average_trade_duration': trades_df['hold_time_hours'].mean(),
                    'median_trade_duration': trades_df['hold_time_hours'].median(),
                    'max_trade_duration': trades_df['hold_time_hours'].max(),
                    'min_trade_duration': trades_df['hold_time_hours'].min()
                })
            
            # Consecutive trade analysis
            trade_results = trades_df['pnl_pct'] > 0
            consecutive_wins = self._calculate_consecutive_runs(trade_results, True)
            consecutive_losses = self._calculate_consecutive_runs(trade_results, False)
            
            metrics.update({
                'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
                'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
                'average_consecutive_wins': np.mean(consecutive_wins) if consecutive_wins else 0,
                'average_consecutive_losses': np.mean(consecutive_losses) if consecutive_losses else 0
            })
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
        
        return metrics
    
    def _calculate_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown and recovery metrics."""
        metrics = {}
        
        try:
            if equity_df.empty or len(equity_df) < 2:
                return metrics
            
            equity = equity_df['equity'].values
            
            # Calculate running maximum (peak)
            peak = np.maximum.accumulate(equity)
            
            # Calculate drawdown
            drawdown = (equity - peak) / peak
            
            # Basic drawdown metrics
            max_drawdown = drawdown.min()
            max_drawdown_idx = np.argmin(drawdown)
            
            metrics.update({
                'max_drawdown': abs(max_drawdown),
                'max_drawdown_pct': abs(max_drawdown) * 100,
                'current_drawdown': abs(drawdown[-1]),
                'average_drawdown': abs(drawdown[drawdown < 0]).mean() if any(drawdown < 0) else 0
            })
            
            # Drawdown duration analysis
            if 'datetime' in equity_df.columns:
                timestamps = equity_df['datetime'].values
                
                # Find drawdown periods
                drawdown_periods = self._find_drawdown_periods(drawdown, timestamps)
                
                if drawdown_periods:
                    durations = [(end - start).days for start, end, _ in drawdown_periods]
                    depths = [depth for _, _, depth in drawdown_periods]
                    
                    metrics.update({
                        'max_drawdown_duration_days': max(durations) if durations else 0,
                        'average_drawdown_duration_days': np.mean(durations) if durations else 0,
                        'number_of_drawdown_periods': len(drawdown_periods),
                        'average_drawdown_depth': np.mean(depths) if depths else 0
                    })
            
            # Recovery analysis
            recovery_factors = []
            underwater_curve = drawdown < -0.001  # Consider significant drawdowns
            
            if any(underwater_curve):
                # Time to recovery metrics
                recovery_times = self._calculate_recovery_times(drawdown, equity_df.get('datetime'))
                if recovery_times:
                    metrics.update({
                        'average_recovery_time_days': np.mean(recovery_times),
                        'max_recovery_time_days': max(recovery_times),
                        'recovery_factor': abs(max_drawdown) / np.std(drawdown) if np.std(drawdown) > 0 else 0
                    })
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        metrics = {}
        
        try:
            if equity_df.empty or len(equity_df) < 2:
                return metrics
            
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            returns = equity_df['returns']
            
            if len(returns) == 0 or returns.std() == 0:
                return metrics
            
            # Risk-free rate adjustment (daily)
            risk_free_daily = (1 + self.risk_free_rate) ** (1/365) - 1
            excess_returns = returns - risk_free_daily
            
            # Sharpe Ratio
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(365.25)
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino Ratio (using downside deviation)
            downside_returns = returns[returns < risk_free_daily]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std()
                sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(365.25)
                metrics['sortino_ratio'] = sortino_ratio
            
            # Calmar Ratio
            if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
                annual_return = returns.mean() * 365.25
                calmar_ratio = annual_return / metrics['max_drawdown']
                metrics['calmar_ratio'] = calmar_ratio
            
            # Information Ratio (vs zero benchmark)
            tracking_error = returns.std()
            information_ratio = returns.mean() / tracking_error if tracking_error > 0 else 0
            metrics['information_ratio'] = information_ratio
            
            # Treynor Ratio (assuming beta = 1 for simplicity)
            treynor_ratio = excess_returns.mean()
            metrics['treynor_ratio'] = treynor_ratio
            
            # Sterling Ratio
            if 'average_drawdown' in metrics and metrics['average_drawdown'] > 0:
                sterling_ratio = returns.mean() * 365.25 / metrics['average_drawdown']
                metrics['sterling_ratio'] = sterling_ratio
            
            # Burke Ratio
            if len(downside_returns) > 0:
                burke_ratio = excess_returns.mean() / np.sqrt(np.sum(downside_returns ** 2))
                metrics['burke_ratio'] = burke_ratio
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
        
        return metrics
    
    def _calculate_consistency_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate consistency and stability metrics."""
        metrics = {}
        
        try:
            # Trade consistency
            if not trades_df.empty:
                # Rolling win rates
                if len(trades_df) >= 10:
                    win_indicators = (trades_df['pnl_pct'] > 0).astype(int)
                    rolling_win_rates = win_indicators.rolling(window=10).mean().dropna()
                    
                    metrics.update({
                        'win_rate_consistency': 1 - rolling_win_rates.std() if len(rolling_win_rates) > 0 else 0,
                        'win_rate_stability': rolling_win_rates.min() if len(rolling_win_rates) > 0 else 0
                    })
                
                # Profit consistency
                if 'pnl_pct' in trades_df.columns:
                    profit_consistency = 1 - (trades_df['pnl_pct'].std() / abs(trades_df['pnl_pct'].mean())) if trades_df['pnl_pct'].mean() != 0 else 0
                    metrics['profit_consistency'] = max(0, profit_consistency)
            
            # Equity curve consistency
            if not equity_df.empty and len(equity_df) > 10:
                equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
                
                # Rolling Sharpe ratios
                rolling_sharpe = equity_df['returns'].rolling(window=20).apply(
                    lambda x: x.mean() / x.std() if x.std() > 0 else 0
                ).dropna()
                
                if len(rolling_sharpe) > 0:
                    metrics.update({
                        'sharpe_consistency': 1 - rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else 0,
                        'performance_stability': rolling_sharpe.min()
                    })
                
                # Regime analysis (simple trend detection)
                if len(equity_df) >= 50:
                    equity_trend = equity_df['equity'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
                    trend_consistency = equity_trend.mean()
                    metrics['trend_consistency'] = trend_consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency metrics: {e}")
        
        return metrics
    
    def _calculate_efficiency_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading efficiency metrics."""
        metrics = {}
        
        try:
            if trades_df.empty:
                return metrics
            
            # Trade efficiency
            total_pnl = trades_df['pnl_pct'].sum()
            total_trades = len(trades_df)
            
            if total_trades > 0:
                trade_efficiency = total_pnl / total_trades
                metrics['trade_efficiency'] = trade_efficiency
            
            # Position sizing efficiency
            if 'position_size_pct' in trades_df.columns:
                # Correlation between position size and returns
                position_return_corr = trades_df['position_size_pct'].corr(trades_df['pnl_pct'])
                metrics['position_sizing_efficiency'] = position_return_corr if not np.isnan(position_return_corr) else 0
            
            # Time efficiency
            if 'hold_time_hours' in trades_df.columns:
                hourly_returns = trades_df['pnl_pct'] / trades_df['hold_time_hours']
                metrics.update({
                    'average_hourly_return': hourly_returns.mean(),
                    'time_efficiency': hourly_returns.std() / abs(hourly_returns.mean()) if hourly_returns.mean() != 0 else 0
                })
            
            # Kelly Criterion calculation
            if len(trades_df) > 10:
                win_rate = len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df)
                if win_rate > 0 and win_rate < 1:
                    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean()
                    avg_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean())
                    
                    if avg_loss > 0:
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                        metrics['kelly_criterion'] = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
        
        return metrics
    
    def _calculate_rl_specific_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RL-specific performance metrics."""
        metrics = {}
        
        try:
            if trades_df.empty:
                return metrics
            
            # Action consistency
            if 'action_type' in trades_df.columns:
                action_distribution = trades_df['action_type'].value_counts(normalize=True)
                action_entropy = -sum(p * np.log(p) for p in action_distribution if p > 0)
                metrics['action_entropy'] = action_entropy
                metrics['action_diversity'] = len(action_distribution) / 3.0  # Normalized by max actions (BUY, SELL, HOLD)
            
            # Risk-reward ratio utilization
            if 'risk_reward' in trades_df.columns:
                rr_efficiency = trades_df.groupby('risk_reward')['pnl_pct'].mean()
                if not rr_efficiency.empty:
                    optimal_rr = rr_efficiency.idxmax()
                    metrics['optimal_risk_reward'] = optimal_rr
                    metrics['risk_reward_efficiency'] = rr_efficiency.max()
            
            # Position sizing analysis
            if 'position_size_pct' in trades_df.columns:
                size_performance = trades_df.groupby(pd.cut(trades_df['position_size_pct'], bins=5))['pnl_pct'].mean()
                size_consistency = 1 - size_performance.std() / abs(size_performance.mean()) if size_performance.mean() != 0 else 0
                metrics['position_size_consistency'] = max(0, size_consistency)
            
            # Exit reason analysis
            if 'exit_reason' in trades_df.columns:
                exit_reasons = trades_df['exit_reason'].value_counts(normalize=True)
                
                # Calculate performance by exit reason
                exit_performance = trades_df.groupby('exit_reason')['pnl_pct'].mean()
                
                metrics.update({
                    'stop_loss_rate': exit_reasons.get('stop_loss', 0),
                    'take_profit_rate': exit_reasons.get('take_profit', 0),
                    'time_exit_rate': exit_reasons.get('time_exit', 0),
                    'stop_loss_performance': exit_performance.get('stop_loss', 0),
                    'take_profit_performance': exit_performance.get('take_profit', 0),
                    'time_exit_performance': exit_performance.get('time_exit', 0)
                })
            
            # Learning progression (if entry_time available)
            if 'entry_time' in trades_df.columns and len(trades_df) > 20:
                # Split into early and late periods
                mid_point = len(trades_df) // 2
                early_trades = trades_df.iloc[:mid_point]
                late_trades = trades_df.iloc[mid_point:]
                
                early_performance = early_trades['pnl_pct'].mean()
                late_performance = late_trades['pnl_pct'].mean()
                
                learning_improvement = late_performance - early_performance
                metrics['learning_progression'] = learning_improvement
                
                # Win rate progression
                early_win_rate = len(early_trades[early_trades['pnl_pct'] > 0]) / len(early_trades)
                late_win_rate = len(late_trades[late_trades['pnl_pct'] > 0]) / len(late_trades)
                metrics['win_rate_progression'] = late_win_rate - early_win_rate
            
        except Exception as e:
            logger.error(f"Error calculating RL-specific metrics: {e}")
        
        return metrics
    
    def _calculate_relative_metrics(self, equity_df: pd.DataFrame, benchmark_returns: List[float]) -> Dict[str, Any]:
        """Calculate metrics relative to benchmark."""
        metrics = {}
        
        try:
            if equity_df.empty or not benchmark_returns:
                return metrics
            
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            strategy_returns = equity_df['returns'].values
            
            # Align lengths
            min_length = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
            
            # Active returns
            active_returns = strategy_returns - benchmark_returns
            
            # Information ratio
            tracking_error = np.std(active_returns)
            information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            # Alpha calculation
            risk_free_daily = (1 + self.risk_free_rate) ** (1/365) - 1
            alpha = np.mean(strategy_returns) - (risk_free_daily + beta * (np.mean(benchmark_returns) - risk_free_daily))
            
            metrics.update({
                'information_ratio_vs_benchmark': information_ratio,
                'tracking_error': tracking_error,
                'beta': beta,
                'alpha': alpha,
                'correlation_with_benchmark': np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
            })
            
        except Exception as e:
            logger.error(f"Error calculating relative metrics: {e}")
        
        return metrics
    
    def _calculate_composite_scores(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite performance scores."""
        scores = {}
        
        try:
            # Overall Performance Score (0-100)
            score_components = []
            
            # Return component (25%)
            if 'total_return' in metrics:
                return_score = min(100, max(0, metrics['total_return'] * 100 + 50))
                score_components.append(('return', return_score, 0.25))
            
            # Risk component (25%)
            if 'sharpe_ratio' in metrics:
                sharpe_score = min(100, max(0, metrics['sharpe_ratio'] * 20 + 50))
                score_components.append(('risk', sharpe_score, 0.25))
            
            # Consistency component (25%)
            if 'win_rate' in metrics:
                consistency_score = metrics['win_rate'] * 100
                score_components.append(('consistency', consistency_score, 0.25))
            
            # Efficiency component (25%)
            if 'profit_factor' in metrics:
                efficiency_score = min(100, max(0, (metrics['profit_factor'] - 1) * 50))
                score_components.append(('efficiency', efficiency_score, 0.25))
            
            # Calculate weighted score
            if score_components:
                total_score = sum(score * weight for _, score, weight in score_components)
                scores['overall_performance_score'] = total_score
                
                # Individual component scores
                for component, score, _ in score_components:
                    scores[f'{component}_score'] = score
            
            # Risk-Adjusted Score
            if 'sharpe_ratio' in metrics and 'max_drawdown' in metrics:
                risk_adjusted_score = (
                    (metrics['sharpe_ratio'] * 20) - 
                    (metrics['max_drawdown'] * 100)
                )
                scores['risk_adjusted_score'] = risk_adjusted_score
            
            # RL Strategy Score (specific to RL performance)
            rl_components = []
            
            if 'learning_progression' in metrics:
                learning_score = min(100, max(0, metrics['learning_progression'] * 1000 + 50))
                rl_components.append(learning_score)
            
            if 'action_diversity' in metrics:
                diversity_score = metrics['action_diversity'] * 100
                rl_components.append(diversity_score)
            
            if rl_components:
                scores['rl_strategy_score'] = np.mean(rl_components)
            
        except Exception as e:
            logger.error(f"Error calculating composite scores: {e}")
        
        return scores
    
    def _calculate_consecutive_runs(self, series: pd.Series, target_value: bool) -> List[int]:
        """Calculate consecutive runs of a target value."""
        runs = []
        current_run = 0
        
        for value in series:
            if value == target_value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        # Add final run if it ends with target value
        if current_run > 0:
            runs.append(current_run)
        
        return runs
    
    def _find_drawdown_periods(self, drawdown: np.ndarray, timestamps: np.ndarray) -> List[Tuple]:
        """Find periods of drawdown with start, end, and depth."""
        periods = []
        in_drawdown = False
        start_idx, max_dd_depth = None, 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_idx = i
                max_dd_depth = dd
            elif dd < max_dd_depth and in_drawdown:  # Deeper drawdown
                max_dd_depth = dd
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_idx is not None:
                    periods.append((timestamps[start_idx], timestamps[i], abs(max_dd_depth)))
        
        return periods
    
    def _calculate_recovery_times(self, drawdown: np.ndarray, timestamps: Optional[pd.Series]) -> List[float]:
        """Calculate recovery times from drawdowns."""
        if timestamps is None:
            return []
        
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001 and not in_drawdown:
                in_drawdown = True
                drawdown_start = timestamps.iloc[i] if i < len(timestamps) else None
            elif dd >= -0.001 and in_drawdown and drawdown_start:
                in_drawdown = False
                recovery_end = timestamps.iloc[i] if i < len(timestamps) else None
                if recovery_end:
                    recovery_time = (recovery_end - drawdown_start).days
                    recovery_times.append(recovery_time)
        
        return recovery_times
    
    def _get_minimal_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return minimal metrics when data is insufficient."""
        minimal = base_metrics.copy() if base_metrics else {}
        
        # Ensure essential metrics exist
        essential_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'overall_performance_score': 0.0
        }
        
        for key, default_value in essential_metrics.items():
            if key not in minimal:
                minimal[key] = default_value
        
        return minimal
