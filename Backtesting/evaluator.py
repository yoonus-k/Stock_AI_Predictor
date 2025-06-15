"""
Main Portfolio Evaluator for Backtesting.py

Professional backtesting orchestrator that replaces Backtrader with backtesting.py.
Provides seamless integration with existing RL infrastructure while leveraging
all built-in features of backtesting.py for superior performance and metrics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import backtesting.py
try:
    from Lib.Backtesting.backtesting import Backtest
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("⚠️ backtesting.py not installed. Install with: pip install backtesting")

# Import local modules
from Backtesting.data_provider import DataProvider
from Backtesting.strategies import RLTradingStrategy
from Backtesting.metrics import MetricsExtractor

logger = logging.getLogger(__name__)


class BacktestingPyEvaluator:
    """
    Professional portfolio evaluator using backtesting.py framework.
    
    This class provides a clean, efficient replacement for BacktraderPortfolioEvaluator
    with enhanced capabilities:
    
    - 10x simpler API than Backtrader
    - Built-in comprehensive metrics (50+)
    - Vectorized execution (faster performance)
    - Interactive visualizations
    - Native RL/ML integration
    - Professional MLflow integration
    """
    
    def __init__(
        self,
        initial_cash: float = 100000,
        commission: float = 0.0005,
        spread: float = 0.0001,
        margin: float = 1.0,
        trade_on_close: bool = False,
        hedging: bool = True,
        exclusive_orders: bool = True,
        finalize_trade_on_close: bool =True,
        # RL Strategy parameters (same as Backtrader version)
        max_positions: int = 5,
        risk_per_trade: float = 0.02,
        enable_short: bool = True,
        enable_hedging: bool = True,
        position_sizing: str = 'fixed',
        
        # Additional parameters
        verbose: bool = False
    ):
        """
        Initialize the backtesting.py evaluator.
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission rate (0.001 = 0.1%)
            margin: Margin requirement (1.0 = no leverage)
            trade_on_close: Execute orders on close vs open of next bar
            hedging: Allow simultaneous long/short positions
            exclusive_orders: Prevent multiple orders on same bar
            
            max_positions: Maximum concurrent positions (RL strategy)
            risk_per_trade: Risk per trade as fraction of equity
            enable_short: Allow short positions
            enable_hedging: Allow hedging in strategy
            position_sizing: Position sizing method
            
            verbose: Enable verbose logging
        """
        if not BACKTESTING_AVAILABLE:
            raise ImportError("backtesting.py not available. Install with: pip install backtesting")
        
        # Backtesting.py parameters
        self.initial_cash = initial_cash
        self.commission = commission
        
        self.spread = spread  # Spread is not used directly in backtesting.py, but can be set as commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        self.finalize_trade_on_close = finalize_trade_on_close
        
        # RL Strategy parameters
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.enable_short = enable_short
        self.enable_hedging = enable_hedging
        self.position_sizing = position_sizing
        
        self.verbose = verbose
        
        # Initialize components
        self.data_provider = DataProvider(verbose=verbose)
        self.metrics_extractor = MetricsExtractor()
        
        # Results storage
        self.last_backtest = None
        self.last_stats = None
        self.last_strategy = None
        self.last_results = {}
        
        logger.info(f"🚀 BacktestingPy Evaluator initialized:")
        logger.info(f"   Initial cash: ${initial_cash:,.2f}")
        logger.info(f"   Commission: {commission:.3%}")
        logger.info(f"   Max positions: {max_positions}")
        logger.info(f"   Risk per trade: {risk_per_trade:.1%}")
    
    def evaluate_portfolio(
        self,
        rl_model: Any,
        environment_data: Union[pd.DataFrame, Dict, Any] = None,
        stock_id: int = 1,
        timeframe: str = "1H",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) :
        """
        Main evaluation method - replaces BacktraderPortfolioEvaluator.evaluate_portfolio().
        Uses backtesting.py's built-in metrics and functionality.
        
        Args:
            rl_model: Trained RL model for predictions
            environment_data: Optional data (if None, loads from database)
            stock_id: Stock identifier for database loading
            timeframe: Trading timeframe
            start_date: Start date for evaluation
            end_date: End date for evaluation
            **kwargs: Additional parameters
            
        Returns:
            Dict containing comprehensive performance metrics
        """
        try:
            logger.info("🚀 Starting portfolio evaluation with backtesting.py")
            start_time = datetime.now()
            
            # Prepare data using improved data provider
            price_data, observation_manager, data_info = self._prepare_evaluation_data(
                environment_data, stock_id, timeframe, start_date, end_date
            )
            
            if price_data is None or len(price_data) < 50:
                logger.warning("Insufficient data for backtesting")
                return self._get_fallback_metrics()
            
            logger.info(f"📊 Data loaded: {data_info.get('total_bars', 0)} bars, "
                      f"${data_info.get('price_stats', {}).get('min_price', 0):.2f} - "
                      f"${data_info.get('price_stats', {}).get('max_price', 0):.2f}")
            
            
            # Create backtest with strategy parameters
            backtest = Backtest(
                data=price_data,
                strategy=RLTradingStrategy,
                cash=self.initial_cash,
                commission=self.commission,
                spread=self.spread,
                margin=self.margin,
                trade_on_close=self.trade_on_close,
                hedging=self.hedging,
                exclusive_orders=self.exclusive_orders , 
                finalize_trades=self.finalize_trade_on_close
            )
            
            # Run backtest with RL model and parameters
            stats = backtest.run(
                model=rl_model,
                observation_manager=observation_manager,
                max_positions=self.max_positions,
                enable_short=self.enable_short,
                enable_hedging=self.enable_hedging,
                position_sizing=self.position_sizing
            )
            
            # print the trades from the stats
            # trades = stats['_trades']
            #print(trades)
            # save a CSV file with the trades
            # trades.to_csv(f"trades_{stock_id}_{timeframe}.csv", index=False)
            #print(trades.head(10))  # Print first few trades
            equity_curve = stats['_equity_curve']
            #print(equity_curve.head())  # Print first few rows of equity curve

            # print(stats)
            
            # to CSV
            #equity_curve.to_csv(f"equity_curve_{stock_id}_{timeframe}.csv", index=False)
            # print(trades)
          
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # plot results
            if self.verbose:
                backtest.plot(
                    filename=None,  # No file output by default
                    open_browser=True,  # Do not open browser automatically
                    relative_equity=True,  # Show relative equity changes
                    resample=False,  # No resampling
                    superimpose=False,  # Separate panels for each plot
                    show_legend=True,  # Show legend
                    plot_return=False,  # Plot return curve
                    plot_pl=True,  # Plot P&L
                    plot_volume=False,  # Plot volume
                    plot_drawdown=True,  # Plot drawdown
                    smooth_equity=False,  # No smoothing by default
                    plot_trades=True,  # Plot trades on chart
                )
            
            
            strategy = stats['_strategy']
            
            # add model action history to stats
            model_actions_df = pd.DataFrame(strategy.model_action_history)
            stats['_model_actions'] = model_actions_df
            
            # add the exit type counts to stats
            stats['_exit_reasons_counts'] = strategy.exit_reasons_counts
          
            all_metrics = self.metrics_extractor.extract_all_metrics(stats)
            
            # Add basic execution info
            all_metrics.update({
                'execution_time_seconds': duration,
                'data_bars': len(price_data),
                'timeframe': timeframe,
                'stock_id': stock_id
            })
        
            # Store results for future reference
            self.last_backtest = backtest
            self.last_stats = stats
            self.last_strategy = strategy
            self.last_results = all_metrics
            
            return all_metrics , stats
            
        except Exception as e:
            logger.error(f"❌ Portfolio evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_metrics()
    
    def _prepare_evaluation_data(
        self,
        environment_data: Union[pd.DataFrame, Dict, Any],
        stock_id: int,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[Optional[pd.DataFrame], Optional[Any], Dict[str, Any]]:
        """Prepare data for evaluation."""
        
        # If environment_data is provided, try to use it
        if environment_data is not None:
            logger.info("Using provided environment data")
            # TODO: Handle environment_data conversion if needed
            # For now, fall back to database loading
        
        # Load from database
        return self.data_provider.prepare_complete_dataset(
            stock_id=stock_id,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Return fallback metrics in case of errors."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'final_equity': self.initial_cash,
            'volatility_ann': 0.0,
            'execution_time_seconds': 0.0,
            'error': 'Fallback metrics due to evaluation error'
        }
        
    def _create_action_distribution_plots(self, model_actions_df, exit_reasons_counts, filename_prefix):
        """Create visualizations for model action distributions and save them as files"""
        if model_actions_df is None or model_actions_df.empty:
            return None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plots = {}
        
        try:
            # 1. Position Size Distribution Plot
            plt.figure(figsize=(10, 6))
            trade_actions = model_actions_df[model_actions_df['action_type'] > 0]
            
            if not trade_actions.empty:
                sns.histplot(data=trade_actions, x='position_size', kde=True, bins=10)
                plt.title('Position Size Distribution')
                plt.xlabel('Position Size')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Save to file
                position_size_file = f"position_size_dist_{filename_prefix}.png"
                plt.savefig(position_size_file, dpi=150, bbox_inches='tight')
                plots['position_size_plot'] = position_size_file
            plt.close()
            
            # 2. Risk Reward Distribution Plot
            plt.figure(figsize=(10, 6))
            if not trade_actions.empty:
                sns.histplot(data=trade_actions, x='risk_reward', kde=True, bins=10)
                plt.title('Risk-Reward Distribution')
                plt.xlabel('Risk-Reward Ratio')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Save to file
                risk_reward_file = f"risk_reward_dist_{filename_prefix}.png"
                plt.savefig(risk_reward_file, dpi=150, bbox_inches='tight')
                plots['risk_reward_plot'] = risk_reward_file
            plt.close()
            
            # 3. Hold Time Distribution Plot
            plt.figure(figsize=(10, 6))
            if not trade_actions.empty:
                sns.histplot(data=trade_actions, x='hold_time', kde=True, bins=10)
                plt.title('Hold Time Distribution')
                plt.xlabel('Hold Time (hours)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Save to file
                hold_time_file = f"hold_time_dist_{filename_prefix}.png"
                plt.savefig(hold_time_file, dpi=150, bbox_inches='tight')
                plots['hold_time_plot'] = hold_time_file
            plt.close()
            
            # 4. Trade Exit Types Pie Chart
            if exit_reasons_counts:
                plt.figure(figsize=(10, 8))
                labels = ['Stop Loss', 'Take Profit', 'Time Exit']
                sizes = [
                    exit_reasons_counts.get('sl', 0),
                    exit_reasons_counts.get('tp', 0),
                    exit_reasons_counts.get('time', 0)
                ]
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                explode = (0.1, 0, 0)  # explode stop loss slice
                
                plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
                plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                plt.title('Trade Exit Types Distribution')
                
                # Save to file
                exit_types_file = f"exit_types_{filename_prefix}.png"
                plt.savefig(exit_types_file, dpi=150, bbox_inches='tight')
                plots['exit_types_plot'] = exit_types_file
            plt.close()
            
            # 5. Action Distribution Heatmap (combining multiple dimensions)
            if not trade_actions.empty and len(trade_actions) >= 10:
                plt.figure(figsize=(12, 10))
                
                # Create pivot table for position size vs risk reward
                pivot_data = trade_actions.pivot_table(
                    index=pd.cut(trade_actions['position_size'], bins=5),
                    columns=pd.cut(trade_actions['risk_reward'], bins=5),
                    values='timestamp',
                    aggfunc='count',
                    fill_value=0
                )
                
                # Create heatmap
                sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Count'})
                plt.title('Position Size vs Risk-Reward Distribution')
                plt.ylabel('Position Size')
                plt.xlabel('Risk-Reward Ratio')
                
                # Save to file
                heatmap_file = f"action_heatmap_{filename_prefix}.png"
                plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
                plots['action_heatmap_plot'] = heatmap_file
            plt.close('all')  # Close all remaining plots
            
            return plots
        
        except Exception as e:
            logger.error(f"Error creating action distribution plots: {e}")
            return None