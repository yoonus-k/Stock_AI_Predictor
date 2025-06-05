"""
Custom Backtrader Analyzers for RL Trading Strategy Performance Analysis

This module provides comprehensive analyzers that calculate 50+ metrics specifically
designed for reinforcement learning trading strategies. These analyzers integrate
seamlessly with Backtrader and provide detailed performance insights for MLflow logging.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import backtrader as bt
import warnings

# Suppress numpy runtime warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
np.seterr(divide='ignore', invalid='ignore')


class RLPerformanceAnalyzer(bt.Analyzer):
    """
    Comprehensive performance analyzer for RL trading strategies
    Calculates 50+ metrics including returns, risk, drawdown, and trading-specific metrics
    """
    
    def __init__(self):
        super().__init__()
        
        # Core tracking variables
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        self.action_counts = defaultdict(int)
        self.position_history = []
        self.cash_history = []
        
        # Risk metrics tracking
        self.drawdown_history = []
        self.peak_values = []
        self.underwater_periods = []
        
        # Trade-specific tracking
        self.trade_durations = []
        self.winning_streaks = []
        self.losing_streaks = []
        self.current_streak = 0
        self.current_streak_type = None
        
        # Action sequence tracking
        self.action_sequences = []
        self.position_changes = []
        
    def start(self):
        """Initialize analyzer at strategy start"""
        self.start_value = self.strategy.broker.getvalue()
        self.start_cash = self.strategy.broker.getcash()
        self.peak_value = self.start_value
        
    def next(self):
        """Called on each bar - track portfolio metrics"""
        current_value = self.strategy.broker.getvalue()
        current_cash = self.strategy.broker.getcash()
        
        # Track portfolio values
        self.portfolio_values.append(current_value)
        self.cash_history.append(current_cash)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (current_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
        
        # Track drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.drawdown_history.append(drawdown)
        self.peak_values.append(self.peak_value)
        
        # Track position
        position_size = self.strategy.position.size
        self.position_history.append(position_size)
        
        # Track action if available from strategy
        if hasattr(self.strategy, 'last_action'):
            action = self.strategy.last_action
            self.action_counts[action] += 1
            self.action_sequences.append(action)
            
        # Track position changes
        if len(self.position_history) > 1:
            if self.position_history[-1] != self.position_history[-2]:
                self.position_changes.append({
                    'bar': len(self.position_history),
                    'from_position': self.position_history[-2],
                    'to_position': self.position_history[-1],
                    'portfolio_value': current_value
                })
    
    def notify_trade(self, trade):
        """Called when a trade is closed"""
        if trade.isclosed:
            # Calculate trade metrics
            trade_info = {
                'entry_bar': trade.dtopen,
                'exit_bar': trade.dtclose,
                'duration': (trade.dtclose - trade.dtopen).days if trade.dtclose and trade.dtopen else 0,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl / abs(trade.value) if trade.value != 0 else 0,
                'commission': trade.commission,
                'size': trade.size,
                'entry_price': trade.price,
                'exit_price': trade.price if hasattr(trade, 'exit_price') else None,
                'direction': 'long' if trade.size > 0 else 'short'
            }
            
            self.trades.append(trade_info)
            self.trade_durations.append(trade_info['duration'])
            
            # Track winning/losing streaks
            is_winner = trade.pnl > 0
            if self.current_streak_type is None:
                self.current_streak_type = is_winner
                self.current_streak = 1
            elif self.current_streak_type == is_winner:
                self.current_streak += 1
            else:
                # Streak ended, record it
                if self.current_streak_type:
                    self.winning_streaks.append(self.current_streak)
                else:
                    self.losing_streaks.append(self.current_streak)
                
                # Start new streak
                self.current_streak_type = is_winner
                self.current_streak = 1
    
    def get_analysis(self):
        """Return comprehensive analysis results"""
        if not self.portfolio_values:
            return {}
        
        analysis = {}
        
        # Basic return metrics
        analysis.update(self._calculate_return_metrics())
        
        # Risk metrics
        analysis.update(self._calculate_risk_metrics())
        
        # Trade metrics
        analysis.update(self._calculate_trade_metrics())
        
        # Drawdown metrics
        analysis.update(self._calculate_drawdown_metrics())
        
        # Action and position metrics
        analysis.update(self._calculate_action_metrics())
        
        # Advanced performance metrics
        analysis.update(self._calculate_advanced_metrics())
        
        # Time-based metrics
        analysis.update(self._calculate_time_metrics())
        
        return analysis
    
    def _calculate_return_metrics(self):
        """Calculate return-based metrics"""
        final_value = self.portfolio_values[-1]
        
        return {
            'total_return': (final_value - self.start_value) / self.start_value,
            'total_return_pct': ((final_value - self.start_value) / self.start_value) * 100,
            'annualized_return': self._annualize_return(),
            'cumulative_return': final_value / self.start_value - 1,
            'absolute_return': final_value - self.start_value,
            'final_portfolio_value': final_value,
            'initial_portfolio_value': self.start_value,        }
    
    def _calculate_risk_metrics(self):
        """Calculate risk-based metrics"""
        if len(self.daily_returns) < 2:
            return {}
            
        returns_array = np.array(self.daily_returns)
        
        # Handle CVaR calculation safely
        try:
            var_95_threshold = np.percentile(returns_array, 5)
            cvar_values = returns_array[returns_array <= var_95_threshold]
            cvar_95 = np.mean(cvar_values) if len(cvar_values) > 0 else 0
        except:
            cvar_95 = 0
        
        return {
            'volatility': np.std(returns_array),
            'annualized_volatility': np.std(returns_array) * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'downside_deviation': self._calculate_downside_deviation(),
            'var_95': np.percentile(returns_array, 5),
            'var_99': np.percentile(returns_array, 1),
            'cvar_95': cvar_95,
            'skewness': self._calculate_skewness(),
            'kurtosis': self._calculate_kurtosis(),
        }
    
    def _calculate_trade_metrics(self):
        """Calculate trade-specific metrics"""
        if not self.trades:
            return {}
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'loss_rate': len(losing_trades) / len(self.trades),
            'avg_trade_pnl': np.mean([t['pnl'] for t in self.trades]),
            'avg_winning_trade': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_losing_trade': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'best_trade': max([t['pnl'] for t in self.trades]) if self.trades else 0,
            'worst_trade': min([t['pnl'] for t in self.trades]) if self.trades else 0,
            'profit_factor': self._calculate_profit_factor(),
            'payoff_ratio': self._calculate_payoff_ratio(),
            'avg_trade_duration': np.mean(self.trade_durations) if self.trade_durations else 0,
            'max_trade_duration': max(self.trade_durations) if self.trade_durations else 0,
            'min_trade_duration': min(self.trade_durations) if self.trade_durations else 0,
        }
    
    def _calculate_drawdown_metrics(self):
        """Calculate drawdown-specific metrics"""
        if not self.drawdown_history:
            return {}
        
        drawdowns = np.array(self.drawdown_history)
        
        return {
            'max_drawdown': np.max(drawdowns),
            'max_drawdown_pct': np.max(drawdowns) * 100,
            'avg_drawdown': np.mean(drawdowns[drawdowns > 0]),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(),
            'recovery_factor': self._calculate_recovery_factor(),
            'lake_ratio': self._calculate_lake_ratio(),
            'pain_index': np.mean(drawdowns),
            'ulcer_index': np.sqrt(np.mean(drawdowns ** 2)),
        }
    
    def _calculate_action_metrics(self):
        """Calculate action and position-specific metrics"""
        total_actions = sum(self.action_counts.values())
        
        metrics = {
            'total_actions': total_actions,
            'hold_ratio': self.action_counts.get(0, 0) / max(total_actions, 1),
            'buy_ratio': self.action_counts.get(1, 0) / max(total_actions, 1),
            'sell_ratio': self.action_counts.get(2, 0) / max(total_actions, 1),
            'position_changes': len(self.position_changes),
            'avg_position_hold_time': self._calculate_avg_position_hold_time(),
        }
        
        # Add action counts
        for action, count in self.action_counts.items():
            metrics[f'action_{action}_count'] = count
        
        return metrics
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        return {
            'information_ratio': self._calculate_information_ratio(),
            'treynor_ratio': self._calculate_treynor_ratio(),
            'jensen_alpha': self._calculate_jensen_alpha(),
            'tracking_error': self._calculate_tracking_error(),
            'max_consecutive_wins': max(self.winning_streaks) if self.winning_streaks else 0,
            'max_consecutive_losses': max(self.losing_streaks) if self.losing_streaks else 0,
            'avg_winning_streak': np.mean(self.winning_streaks) if self.winning_streaks else 0,
            'avg_losing_streak': np.mean(self.losing_streaks) if self.losing_streaks else 0,
            'expectancy': self._calculate_expectancy(),
            'kelly_criterion': self._calculate_kelly_criterion(),
        }
    
    def _calculate_time_metrics(self):
        """Calculate time-based metrics"""
        trading_days = len(self.portfolio_values)
        
        return {
            'trading_days': trading_days,
            'annual_trading_days': 252,
            'trading_period_years': trading_days / 252,
            'trades_per_day': len(self.trades) / max(trading_days, 1),
            'trades_per_year': len(self.trades) / max(trading_days / 252, 1),
            'action_frequency': len(self.action_sequences) / max(trading_days, 1),
        }
    
    # Helper methods for complex calculations
    def _annualize_return(self):
        """Calculate annualized return"""
        if len(self.portfolio_values) < 2:
            return 0
        
        trading_days = len(self.portfolio_values)
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        years = trading_days / 252
        
        if years <= 0:
            return 0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 2:
            return 0
        
        excess_returns = np.array(self.daily_returns) - (risk_free_rate / 252)
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, risk_free_rate=0.02):
        """Calculate Sortino ratio"""
        if len(self.daily_returns) < 2:
            return 0
        
        returns = np.array(self.daily_returns)
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio"""
        annualized_return = self._annualize_return()
        max_dd = max(self.drawdown_history) if self.drawdown_history else 0
        
        return annualized_return / max_dd if max_dd > 0 else 0
    
    def _calculate_downside_deviation(self, target_return=0):
        """Calculate downside deviation"""
        if len(self.daily_returns) < 2:
            return 0
        
        returns = np.array(self.daily_returns)
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0
        
        return np.std(downside_returns)
    def _calculate_skewness(self):
        """Calculate skewness of returns"""
        if len(self.daily_returns) < 3:
            return 0
        
        returns = np.array(self.daily_returns)
        std_returns = np.std(returns)
        
        # Check for division by zero or invalid values
        if std_returns == 0 or np.isnan(std_returns) or np.isinf(std_returns):
            return 0
            
        try:
            skewness = ((returns - np.mean(returns)) ** 3).mean() / (std_returns ** 3)
            return skewness if np.isfinite(skewness) else 0
        except (ZeroDivisionError, RuntimeWarning):
            return 0
    
    def _calculate_kurtosis(self):
        """Calculate kurtosis of returns"""
        if len(self.daily_returns) < 4:
            return 0
        
        returns = np.array(self.daily_returns)
        std_returns = np.std(returns)
        
        # Check for division by zero or invalid values
        if std_returns == 0 or np.isnan(std_returns) or np.isinf(std_returns):
            return 0
            
        try:
            kurtosis = ((returns - np.mean(returns)) ** 4).mean() / (std_returns ** 4) - 3
            return kurtosis if np.isfinite(kurtosis) else 0
        except (ZeroDivisionError, RuntimeWarning):
            return 0
    
    def _calculate_profit_factor(self):
        """Calculate profit factor"""
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        return total_profit / total_loss if total_loss > 0 else float('inf')
    
    def _calculate_payoff_ratio(self):
        """Calculate average win to average loss ratio"""
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        return avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    def _calculate_max_drawdown_duration(self):
        """Calculate maximum drawdown duration in days"""
        if not self.drawdown_history:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in self.drawdown_history:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_recovery_factor(self):
        """Calculate recovery factor"""
        total_return = self._calculate_return_metrics()['total_return']
        max_dd = max(self.drawdown_history) if self.drawdown_history else 0
        
        return total_return / max_dd if max_dd > 0 else 0
    
    def _calculate_lake_ratio(self):
        """Calculate lake ratio (average drawdown length / total periods)"""
        if not self.drawdown_history:
            return 0
        
        underwater_days = sum(1 for dd in self.drawdown_history if dd > 0)
        return underwater_days / len(self.drawdown_history)
    
    def _calculate_avg_position_hold_time(self):
        """Calculate average position hold time"""
        if len(self.position_changes) < 2:
            return 0
        
        hold_times = []
        for i in range(len(self.position_changes) - 1):
            if self.position_changes[i]['to_position'] != 0:
                hold_time = self.position_changes[i+1]['bar'] - self.position_changes[i]['bar']
                hold_times.append(hold_time)
        
        return np.mean(hold_times) if hold_times else 0
    
    def _calculate_information_ratio(self, benchmark_return=0.05):
        """Calculate information ratio vs benchmark"""
        if len(self.daily_returns) < 2:
            return 0
        
        portfolio_return = self._annualize_return()
        excess_return = portfolio_return - benchmark_return
        tracking_error = self._calculate_tracking_error()
        
        return excess_return / tracking_error if tracking_error > 0 else 0
    
    def _calculate_treynor_ratio(self, beta=1.0, risk_free_rate=0.02):
        """Calculate Treynor ratio"""
        portfolio_return = self._annualize_return()
        excess_return = portfolio_return - risk_free_rate
        
        return excess_return / beta if beta != 0 else 0
    
    def _calculate_jensen_alpha(self, beta=1.0, market_return=0.08, risk_free_rate=0.02):
        """Calculate Jensen's alpha"""
        portfolio_return = self._annualize_return()
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        return portfolio_return - expected_return
    
    def _calculate_tracking_error(self, benchmark_returns=None):
        """Calculate tracking error"""
        if benchmark_returns is None:
            # Use a simple market return assumption
            benchmark_returns = [0.08 / 252] * len(self.daily_returns)
        
        if len(self.daily_returns) != len(benchmark_returns):
            return np.std(self.daily_returns) if self.daily_returns else 0
        
        excess_returns = np.array(self.daily_returns) - np.array(benchmark_returns)
        return np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_expectancy(self):
        """Calculate expectancy per trade"""
        if not self.trades:
            return 0
        
        win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in self.trades) else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in self.trades) else 0
        
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    def _calculate_kelly_criterion(self):
        """Calculate Kelly criterion optimal bet size"""
        if not self.trades:
            return 0
        
        win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in self.trades) else 0
        avg_loss = abs(np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0])) if any(t['pnl'] < 0 for t in self.trades) else 1
        
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0


class RLActionAnalyzer(bt.Analyzer):
    """
    Specialized analyzer for RL action patterns and decision quality
    """
    
    def __init__(self):
        super().__init__()
        self.action_sequences = []
        self.decision_points = []
        self.market_regime_actions = defaultdict(list)
        
    def next(self):
        """Track action patterns"""
        if hasattr(self.strategy, 'last_action'):
            action = self.strategy.last_action
            price_change = 0
            
            if len(self.strategy.data) > 1:
                price_change = (self.strategy.data.close[0] - self.strategy.data.close[-1]) / self.strategy.data.close[-1]
            
            self.action_sequences.append({
                'action': action,
                'bar': len(self.strategy.data),
                'price': self.strategy.data.close[0],
                'price_change': price_change,
                'portfolio_value': self.strategy.broker.getvalue(),
                'position_size': self.strategy.position.size
            })
    
    def get_analysis(self):
        """Return action analysis"""
        if not self.action_sequences:
            return {}
        
        return {
            'action_consistency': self._calculate_action_consistency(),
            'market_timing_score': self._calculate_market_timing(),
            'decision_frequency': len(self.action_sequences),
            'action_distribution': self._get_action_distribution(),
            'regime_adaptation': self._calculate_regime_adaptation()
        }
    
    def _calculate_action_consistency(self):
        """Calculate consistency of actions"""
        actions = [seq['action'] for seq in self.action_sequences]
        if len(actions) < 2:
            return 0
        
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        return 1 - (changes / (len(actions) - 1))
    
    def _calculate_market_timing(self):
        """Calculate market timing effectiveness"""
        if len(self.action_sequences) < 2:
            return 0
        
        correct_decisions = 0
        for i, seq in enumerate(self.action_sequences[1:], 1):
            prev_action = self.action_sequences[i-1]['action']
            price_change = seq['price_change']
            
            # Simple timing check: buy before price goes up, sell before price goes down
            if (prev_action == 1 and price_change > 0) or (prev_action == 2 and price_change < 0):
                correct_decisions += 1
        
        return correct_decisions / (len(self.action_sequences) - 1)
    
    def _get_action_distribution(self):
        """Get distribution of actions"""
        actions = [seq['action'] for seq in self.action_sequences]
        total = len(actions)
        
        return {
            'hold_pct': actions.count(0) / total * 100,
            'buy_pct': actions.count(1) / total * 100,
            'sell_pct': actions.count(2) / total * 100
        }
    
    def _calculate_regime_adaptation(self):
        """Calculate adaptation to market regimes"""
        # Simple regime classification based on price trends
        regimes = []
        for seq in self.action_sequences:
            if seq['price_change'] > 0.01:
                regimes.append('bullish')
            elif seq['price_change'] < -0.01:
                regimes.append('bearish')
            else:
                regimes.append('neutral')
        
        # Calculate action appropriateness for each regime
        regime_scores = {}
        for regime in ['bullish', 'bearish', 'neutral']:
            regime_actions = [self.action_sequences[i]['action'] for i, r in enumerate(regimes) if r == regime]
            if regime_actions:
                if regime == 'bullish':
                    # Should prefer buying in bullish markets
                    regime_scores[regime] = regime_actions.count(1) / len(regime_actions)
                elif regime == 'bearish':
                    # Should prefer selling in bearish markets
                    regime_scores[regime] = regime_actions.count(2) / len(regime_actions)
                else:
                    # Should prefer holding in neutral markets
                    regime_scores[regime] = regime_actions.count(0) / len(regime_actions)
            else:
                regime_scores[regime] = 0
        
        return regime_scores


# Export all analyzers
__all__ = ['RLPerformanceAnalyzer', 'RLActionAnalyzer']
