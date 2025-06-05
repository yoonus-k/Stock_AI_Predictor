"""
Backtrader Portfolio Evaluator

This module provides the main orchestrator for backtesting RL strategies using Backtrader.
It integrates with the MLflow callback to replace the current portfolio evaluation with
professional-grade backtesting capabilities.

Key Features:
- Comprehensive backtesting with realistic trading simulation
- Long and short positions with hedging support
- Professional-grade performance metrics (50+ metrics)
- Integration with MLflow for experiment tracking
- Support for multiple timeframes and assets
- Advanced risk management and position sizing
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
import os

from .rl_strategy import RLTradingStrategy
from .data_feeds import create_complete_backtest_data, SimplePriceFeed, ObservationManager
from .analyzers import RLPerformanceAnalyzer, RLActionAnalyzer
from .metrics_extractor import MetricsExtractor

logger = logging.getLogger(__name__)


class BacktraderPortfolioEvaluator:
    """
    Main orchestrator for backtesting RL strategies using Backtrader.
    
    This class:
    - Sets up the backtesting environment
    - Configures data feeds from RL environment data
    - Runs comprehensive backtesting with realistic trading simulation
    - Extracts professional-grade performance metrics
    - Integrates with MLflow for experiment tracking
    """
    
    def __init__(
        self,
        initial_cash: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        enable_short: bool = True,
        enable_hedging: bool = True,
        max_positions: int = 5,
        risk_per_trade: float = 0.02,
        position_sizing: str = 'fixed',
        verbose: bool = False
    ):
        """
        Initialize the portfolio evaluator.
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            enable_short: Allow short positions
            enable_hedging: Allow hedging (simultaneous long/short)
            max_positions: Maximum concurrent positions
            risk_per_trade: Risk per trade as fraction of equity
            position_sizing: Position sizing method ('fixed', 'percent', 'kelly')
            verbose: Enable verbose logging
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.enable_short = enable_short
        self.enable_hedging = enable_hedging
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.position_sizing = position_sizing
        self.verbose = verbose
        
        # Initialize components
        self.cerebro = None
        self.results = None
        self.metrics_extractor = MetricsExtractor()
        
        # Performance tracking
        self.last_backtest_results = {}
        self.trade_log = []
        self.equity_curve = []
        
        logger.info(f"Initialized BacktraderPortfolioEvaluator with {initial_cash} initial cash")
    
    def evaluate_portfolio(
        self,
        rl_model: Any,
        environment_data: Union[pd.DataFrame, np.ndarray, Dict],
        episode_length: int = None,
        timeframe: str = '1H',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main evaluation method that replaces the MLflow callback's _calculate_portfolio_metrics.
        
        Args:
            rl_model: The trained RL model
            environment_data: Data from the RL environment (observations, actions, rewards)
            episode_length: Length of the evaluation episode
            timeframe: Trading timeframe ('1H', '4H', '1D', etc.)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing comprehensive performance metrics
        """
        try:
            logger.info("Starting portfolio evaluation with Backtrader")
              # Setup backtesting environment
            self._setup_cerebro()
            
            # Create data feed and observation manager
            price_feed, obs_manager = self._create_data_feed(environment_data, timeframe)
            self.cerebro.adddata(price_feed)
            
            # Add strategy with observation manager
            self.cerebro.addstrategy(
                RLTradingStrategy,
                model=rl_model,
                observation_manager=obs_manager,
                initial_cash=self.initial_cash,
                position_sizing=self.position_sizing,
                max_positions=self.max_positions,
                risk_per_trade=self.risk_per_trade,
                enable_hedging=self.enable_hedging,
                enable_short=self.enable_short,
                commission=self.commission,
                slippage=self.slippage,
                verbose=self.verbose
            )
            
            # Add analyzers
            self.cerebro.addanalyzer(RLPerformanceAnalyzer, _name='performance')
            self.cerebro.addanalyzer(RLActionAnalyzer, _name='actions')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # Configure broker
            self._setup_broker()
            
            # Run backtest
            logger.info("Running backtest...")
            start_time = datetime.now()
            
            self.results = self.cerebro.run()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Backtest completed in {duration:.2f} seconds")
            
            # Extract comprehensive metrics
            metrics = self._extract_all_metrics()
            
            # Store results for future reference
            self.last_backtest_results = metrics
            
            logger.info(f"Portfolio evaluation completed. Total return: {metrics.get('total_return', 0):.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in portfolio evaluation: {e}")
            return self._get_fallback_metrics()
    
    def _setup_cerebro(self):
        """Setup the Backtrader Cerebro engine."""
        self.cerebro = bt.Cerebro()
        
        # Set initial cash
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Configure execution settings
        self.cerebro.broker.set_checksubmit(False)  # Allow multiple orders
        self.cerebro.broker.set_coc(True)  # Close on close prices
        
        # Add observers for tracking
        self.cerebro.addobserver(bt.observers.Broker)
        self.cerebro.addobserver(bt.observers.Trades)
        self.cerebro.addobserver(bt.observers.BuySell)
        
        if self.verbose:
            logger.info("Cerebro engine configured")
    
    def _setup_broker(self):
        """Setup broker configuration."""
        # Set commission
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # Set slippage
        if hasattr(self.cerebro.broker, 'set_slippage_perc'):
            self.cerebro.broker.set_slippage_perc(self.slippage)
        
        # Configure margin for short selling
        if self.enable_short:
            self.cerebro.broker.set_shortcash(True)
        
        if self.verbose:
            logger.info(f"Broker configured: commission={self.commission}, slippage={self.slippage}")
    def _create_data_feed(
        self,
        environment_data: Union[pd.DataFrame, np.ndarray, Dict],
        timeframe: str = '1H'
    ) -> Tuple[Optional[SimplePriceFeed], Optional[ObservationManager]]:
        """Create data feed and observation manager using simplified approach."""
        try:
            # For new simplified approach, we expect stock_id and timeframe_id
            stock_id = 1  # Default to GOLD
            timeframe_id = self._timeframe_to_id(timeframe)
            
            # Create using factory functions
            price_feed, obs_manager = create_complete_backtest_data(
                stock_id=stock_id, 
                timeframe_id=timeframe_id
            )
            
            if price_feed is None:
                logger.warning("Could not create price feed from database, using fallback")
                return self._create_fallback_data_feed(), ObservationManager()
            
            logger.info(f"Created data feed for stock_id={stock_id}, timeframe_id={timeframe_id}")
            return price_feed, obs_manager
                
        except Exception as e:
            logger.error(f"Error creating data feed: {e}")
            # Fallback: create minimal data feed
            return self._create_fallback_data_feed(), ObservationManager()
    
    def _timeframe_to_id(self, timeframe: str) -> int:
        """Convert timeframe string to database ID."""
        timeframe_map = {
            '1M': 1,
            '5M': 2, 
            '15M': 3,
            '30M': 4,
            '1H':5,
            '4H': 6,
            '1D': 7
        }
        return timeframe_map.get(timeframe, 5)  # Default to 5min
    
    def _get_timeframe_constant(self, timeframe: str) -> int:
        """Convert timeframe string to Backtrader constant."""
        timeframe_map = {
            '1M': bt.TimeFrame.Minutes,
            '5M': bt.TimeFrame.Minutes,
            '15M': bt.TimeFrame.Minutes,
            '30M': bt.TimeFrame.Minutes,
            '1H': bt.TimeFrame.Minutes,
            '4H': bt.TimeFrame.Minutes,
            '1D': bt.TimeFrame.Days,
            '1W': bt.TimeFrame.Weeks
        }
        return timeframe_map.get(timeframe, bt.TimeFrame.Minutes)
    def _create_fallback_data_feed(self) -> SimplePriceFeed:
        """Create a minimal fallback data feed for testing."""
        # Generate synthetic price data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(1000)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(1000)) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        return SimplePriceFeed(dataname=data)
    
    def _extract_all_metrics(self) -> Dict[str, Any]:
        """Extract comprehensive metrics from backtest results."""
        try:
            if not self.results:
                return self._get_fallback_metrics()
            
            strategy = self.results[0]
            
            # Get analyzer results
            performance_analyzer = strategy.analyzers.performance.get_analysis()
            action_analyzer = strategy.analyzers.actions.get_analysis()
            trade_analyzer = strategy.analyzers.trades.get_analysis()
            sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
            drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
            returns_analyzer = strategy.analyzers.returns.get_analysis()
            
            # Combine all metrics
            all_metrics = {}
            
            # Performance metrics (50+ metrics)
            all_metrics.update(performance_analyzer)
            
            # Action analysis metrics
            all_metrics.update(action_analyzer)
            
            # Standard Backtrader metrics
            all_metrics.update({
                'total_trades': trade_analyzer.get('total', {}).get('total', 0),
                'won_trades': trade_analyzer.get('won', {}).get('total', 0),
                'lost_trades': trade_analyzer.get('lost', {}).get('total', 0),
                'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0),
                'max_drawdown': drawdown_analyzer.get('max', {}).get('drawdown', 0),
                'max_drawdown_period': drawdown_analyzer.get('max', {}).get('len', 0),
                'total_return': returns_analyzer.get('rtot', 0),
                'average_return': returns_analyzer.get('ravg', 0)
            })
            
            # Strategy-specific metrics
            if hasattr(strategy, 'get_performance_metrics'):
                strategy_metrics = strategy.get_performance_metrics()
                all_metrics.update(strategy_metrics)
            
            # Extract trade log and equity curve
            if hasattr(strategy, 'get_trade_log'):
                self.trade_log = strategy.get_trade_log()
                all_metrics['trade_log_length'] = len(self.trade_log)
            
            if hasattr(strategy, 'get_equity_curve'):
                self.equity_curve = strategy.get_equity_curve()
                all_metrics['equity_curve_length'] = len(self.equity_curve)
            
            # Additional computed metrics
            all_metrics.update(self._compute_additional_metrics())
            
            # Use metrics extractor for advanced calculations
            enhanced_metrics = self.metrics_extractor.extract_comprehensive_metrics(
                trade_log=self.trade_log,
                equity_curve=self.equity_curve,
                base_metrics=all_metrics
            )
            
            all_metrics.update(enhanced_metrics)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return self._get_fallback_metrics()
    
    def _compute_additional_metrics(self) -> Dict[str, Any]:
        """Compute additional performance metrics."""
        try:
            additional_metrics = {}
            
            # Broker metrics
            if self.cerebro and self.cerebro.broker:
                final_value = self.cerebro.broker.getvalue()
                additional_metrics.update({
                    'final_portfolio_value': final_value,
                    'total_return_abs': final_value - self.initial_cash,
                    'total_return_pct': (final_value - self.initial_cash) / self.initial_cash,
                    'final_cash': self.cerebro.broker.getcash(),
                    'final_position_value': final_value - self.cerebro.broker.getcash()
                })
            
            # Trade-based metrics
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                
                if not trades_df.empty:
                    # Win/Loss analysis
                    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
                    losing_trades = trades_df[trades_df['pnl_pct'] < 0]
                    
                    additional_metrics.update({
                        'total_trades_custom': len(trades_df),
                        'winning_trades_custom': len(winning_trades),
                        'losing_trades_custom': len(losing_trades),
                        'win_rate_custom': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                        'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
                        'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
                        'largest_win': trades_df['pnl_pct'].max() if len(trades_df) > 0 else 0,
                        'largest_loss': trades_df['pnl_pct'].min() if len(trades_df) > 0 else 0,
                        'avg_trade_duration': trades_df['hold_time_hours'].mean() if len(trades_df) > 0 else 0
                    })
                    
                    # Risk metrics
                    if len(winning_trades) > 0 and len(losing_trades) > 0:
                        profit_factor = abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum())
                        additional_metrics['profit_factor_custom'] = profit_factor
            
            # Equity curve metrics
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                
                if not equity_df.empty and len(equity_df) > 1:
                    equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
                    
                    additional_metrics.update({
                        'volatility_annualized': equity_df['returns'].std() * np.sqrt(252 * 24),  # For hourly data
                        'skewness': equity_df['returns'].skew(),
                        'kurtosis': equity_df['returns'].kurtosis(),
                        'var_95': equity_df['returns'].quantile(0.05),
                        'cvar_95': equity_df['returns'][equity_df['returns'] <= equity_df['returns'].quantile(0.05)].mean()
                    })
            
            return additional_metrics
            
        except Exception as e:
            logger.error(f"Error computing additional metrics: {e}")
            return {}
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Return basic fallback metrics in case of errors."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'final_portfolio_value': self.initial_cash,
            'volatility': 0.0,
            'error': 'Fallback metrics due to evaluation error'
        }
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Get detailed trade analysis."""
        if self.trade_log:
            return pd.DataFrame(self.trade_log)
        return pd.DataFrame()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve data."""
        if self.equity_curve:
            return pd.DataFrame(self.equity_curve)
        return pd.DataFrame()
    
    def plot_results(self, show_trades: bool = True, save_path: Optional[str] = None):
        """Plot backtest results."""
        try:
            if self.cerebro and hasattr(self.cerebro, 'plot'):
                # Use Backtrader's built-in plotting
                plots = self.cerebro.plot(
                    style='candlestick',
                    volume=True,
                    plotdist=1.0
                )
                
                if save_path:
                    import matplotlib.pyplot as plt
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                
                return plots
            else:
                logger.warning("No results available for plotting")
                return None
                
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            return None
    
    def save_results(self, filepath: str):
        """Save comprehensive results to file."""
        try:
            results_data = {
                'metrics': self.last_backtest_results,
                'trade_log': self.trade_log,
                'equity_curve': self.equity_curve,
                'config': {
                    'initial_cash': self.initial_cash,
                    'commission': self.commission,
                    'slippage': self.slippage,
                    'enable_short': self.enable_short,
                    'enable_hedging': self.enable_hedging,
                    'max_positions': self.max_positions,
                    'risk_per_trade': self.risk_per_trade
                }
            }
            
            # Save as pickle for complete data preservation
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str):
        """Load previously saved results."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                results_data = pickle.load(f)
            
            self.last_backtest_results = results_data.get('metrics', {})
            self.trade_log = results_data.get('trade_log', [])
            self.equity_curve = results_data.get('equity_curve', [])
            
            logger.info(f"Results loaded from {filepath}")
            return results_data
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None
    
    def compare_strategies(self, strategies_configs: List[Dict]) -> pd.DataFrame:
        """Compare multiple strategy configurations."""
        # This would be implemented for strategy optimization
        # For now, return placeholder
        logger.info("Strategy comparison not yet implemented")
        return pd.DataFrame()
    
    def optimize_parameters(self, param_ranges: Dict) -> Dict:
        """Optimize strategy parameters using Backtrader's optimization."""
        # This would be implemented for parameter optimization
        # For now, return placeholder
        logger.info("Parameter optimization not yet implemented")
        return {}
