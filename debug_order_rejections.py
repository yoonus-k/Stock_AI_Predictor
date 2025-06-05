"""
Debug order rejections systematically.

This script will test various scenarios to identify why orders are being rejected
even when they should be valid.
"""

import backtrader as bt
import numpy as np
import logging
from datetime import datetime
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from Backtrader.data_feeds import create_complete_backtest_data
from Backtrader.rl_strategy import RLTradingStrategy


class MinimalTestStrategy(bt.Strategy):
    """Ultra minimal strategy to test basic order execution"""
    
    def __init__(self):
        self.order_count = 0
        
    def notify_order(self, order):
        """Track all order notifications"""
        status_names = {
            0: 'Created', 1: 'Submitted', 2: 'Accepted', 3: 'Partial',
            4: 'Completed', 5: 'Canceled', 6: 'Margin', 7: 'Rejected'
        }
        status_name = status_names.get(order.status, f'Unknown({order.status})')
        
        logger.info(f"MINIMAL ORDER {order.ref}: {status_name}")
        logger.info(f"  Size: {order.size}, Price: {order.price}")
        logger.info(f"  Cash: ${self.broker.get_cash():.2f}")
        logger.info(f"  Value: ${self.broker.get_value():.2f}")
        
        if order.status == 7:  # Rejected
            logger.error(f"MINIMAL ORDER REJECTED!")
            
    def next(self):
        """Place one simple order"""
        if len(self.data) == 5 and self.order_count == 0:  # Only place one order
            current_price = self.data.close[0]
            cash = self.broker.get_cash()
            
            # Calculate very conservative position size
            max_affordable = int(cash * 0.1 / current_price)  # Only 10% of cash
            size = min(max_affordable, 1)  # At most 1 share
            
            logger.info(f"MINIMAL ORDER ATTEMPT:")
            logger.info(f"  Price: ${current_price:.2f}")
            logger.info(f"  Cash: ${cash:.2f}")
            logger.info(f"  Max affordable: {max_affordable}")
            logger.info(f"  Order size: {size}")
            
            if size > 0:
                order = self.buy(size=size)
                self.order_count += 1
                logger.info(f"  Order placed: {order.ref if order else 'None'}")
            else:
                logger.warning("  Cannot afford even 1 share")


def test_minimal_orders():
    """Test minimal orders to see if any basic order works"""
    print("\n" + "="*60)
    print("TESTING MINIMAL ORDERS")
    print("="*60)
    
    # Get data
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No data available")
        return False
    
    # Test with different cash amounts
    cash_amounts = [10000, 50000, 100000]
    
    for cash in cash_amounts:
        print(f"\nTesting with ${cash:,} cash:")
        
        cerebro = bt.Cerebro()
        cerebro.adddata(price_feed)
        cerebro.addstrategy(MinimalTestStrategy)
        
        # Configure broker
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.001)
        
        logger.info(f"Starting with ${cash:,} cash")
        
        # Run limited test
        results = cerebro.run(maxbars=10)
        
        final_cash = cerebro.broker.get_cash()
        final_value = cerebro.broker.get_value()
        
        print(f"  Final cash: ${final_cash:,.2f}")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Trade executed: {final_value != cash}")


def test_order_parameters():
    """Test different order parameters to find the issue"""
    print("\n" + "="*60)
    print("TESTING ORDER PARAMETERS")
    print("="*60)
    
    # Get data
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No data available")
        return False
    
    # Get first price for calculations
    price_data = []
    for i, line in enumerate(price_feed):
        if i >= 5:  # Get first 5 prices
            break
        price_data.append(line[3])  # close price
    
    if not price_data:
        print("❌ No price data")
        return False
        
    first_price = price_data[0]
    logger.info(f"First price: ${first_price:.2f}")
    
    # Test different position sizes with $100K
    test_sizes = [1, 2, 5, 10, 20]
    cash = 100000
    
    for size in test_sizes:
        required_cash = size * first_price
        percent_of_cash = (required_cash / cash) * 100
        
        print(f"Size {size}: ${required_cash:,.2f} ({percent_of_cash:.1f}% of cash)")
        
        if required_cash > cash:
            print("  ❌ Exceeds available cash")
        else:
            print("  ✅ Should be affordable")


def test_rl_strategy_issues():
    """Test the actual RL strategy with debugging"""
    print("\n" + "="*60)
    print("TESTING RL STRATEGY ISSUES")
    print("="*60)
    
    # Get data
    price_feed, obs_manager = create_complete_backtest_data(stock_id=1, timeframe_id=5)
    
    if price_feed is None:
        print("❌ No data available")
        return False
    
    class DiagnosticModel:
        """Model that provides detailed diagnostics"""
        def __init__(self):
            self.call_count = 0
            
        def predict(self, observation, deterministic=True):
            self.call_count += 1
            logger.info(f"MODEL CALL {self.call_count}:")
            logger.info(f"  Observation shape: {observation.shape if hasattr(observation, 'shape') else 'Unknown'}")
            
            # Return a very conservative action
            action = np.array([1, 1, 2, 2])  # BUY, small size, low risk/reward
            logger.info(f"  Action: {action}")
            return action, None
    
    model = DiagnosticModel()
    
    # Create RL strategy with detailed logging
    cerebro = bt.Cerebro()
    cerebro.adddata(price_feed)
    
    cerebro.addstrategy(
        RLTradingStrategy,
        model=model,
        observation_manager=obs_manager,
        initial_cash=100000,
        verbose=True,  # Enable all logging
        risk_per_trade=0.005,  # Very small risk (0.5%)
        enable_short=False
    )
    
    # Configure broker conservatively
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    logger.info("RUNNING RL STRATEGY DIAGNOSTIC:")
    logger.info(f"  Cash: ${cerebro.broker.get_cash():,}")
    logger.info(f"  Risk per trade: 0.5%")
    
    # Run with limited bars
    results = cerebro.run(maxbars=10)
    strategy = results[0]
    
    final_cash = cerebro.broker.get_cash()
    final_value = cerebro.broker.get_value()
    
    print(f"\nRL STRATEGY RESULTS:")
    print(f"  Model calls: {model.call_count}")
    print(f"  Final cash: ${final_cash:,.2f}")
    print(f"  Final value: ${final_value:,.2f}")
    print(f"  Return: {((final_value - 100000) / 100000) * 100:.2f}%")
    
    # Check strategy internals
    print(f"  Pending orders: {len(strategy.pending_orders)}")
    print(f"  Current positions: {len(strategy.current_positions)}")
    print(f"  Trade log entries: {len(strategy.trade_log)}")


def main():
    """Run all diagnostic tests"""
    print("DEBUGGING ORDER REJECTION ISSUES")
    print("=" * 80)
    
    # Test 1: Minimal orders
    test_minimal_orders()
    
    # Test 2: Order parameters
    test_order_parameters()
    
    # Test 3: RL strategy issues
    test_rl_strategy_issues()
    
    print("\nDIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
