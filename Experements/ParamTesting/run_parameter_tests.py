#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Testing Runner Script

This script provides a command-line interface to run parameter tests
for stock pattern mining using either the local multithreaded implementation
or the Google Colab optimized version.

Examples:
    # Test all parameters for Gold on 1-minute timeframe
    python run_parameter_tests.py --stock GOLD --timeframe M1 --test-all

    # Test all timeframes for Gold
    python run_parameter_tests.py --stock GOLD --test-all

    # Test all stocks and all timeframes
    python run_parameter_tests.py --test-all
    
    # Run on Colab if available
    python run_parameter_tests.py --stock GOLD --timeframe M1 --test-all --use-colab
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Navigate up to project root
sys.path.append(str(project_root))

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run parameter tests for stock pattern mining"
    )
    
    # Stock and timeframe arguments
    parser.add_argument('--stock', help='Stock to test (symbol or ID)')
    parser.add_argument('--timeframe', help='Timeframe to test (name or ID)')
    
    # Date range arguments
    parser.add_argument('--start-date', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (YYYY-MM-DD)')
    
    # Testing options
    parser.add_argument('--test-all', action='store_true', help='Test all parameter combinations')
    parser.add_argument('--hold-strategy', default='timeframe', choices=['timeframe', 'formula'],
                        help='Strategy for determining hold periods')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with default parameters')
    parser.add_argument('--compare', action='store_true', help='Compare hold period strategies')
    
    # Environment options
    parser.add_argument('--use-colab', action='store_true', help='Use Colab-optimized version if available')
    parser.add_argument('--threads', type=int, help='Number of worker threads to use (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    # Choose which parameter tester to use
    try:
        if args.use_colab:
            try:
                # Try to import Colab-optimized version
                from Experements.ParamTesting.colab_parameter_tester import ColabParameterTester
                tester_class = ColabParameterTester
                print("Using Colab-optimized parameter tester")
            except ImportError:
                # Fall back to multithreaded version
                from Experements.ParamTesting.parameter_tester_multithreaded import MultiThreadedParameterTester
                tester_class = MultiThreadedParameterTester
                print("Colab version not available, using multithreaded parameter tester")
        else:
            # Use the multithreaded version
            from Experements.ParamTesting.parameter_tester_multithreaded import MultiThreadedParameterTester
            tester_class = MultiThreadedParameterTester
            print("Using multithreaded parameter tester")
        
        # Create tester instance
        tester = tester_class(num_workers=args.threads)
        
        # Run tests based on command-line arguments
        if args.compare:
            # Compare hold period strategies
            if not args.stock:
                print("Error: --stock is required for --compare")
                return 1
                
            tester.compare_hold_period_strategies(args.stock)
        
        elif args.quick_test:
            # Run quick test with default parameters
            if not args.stock or not args.timeframe:
                print("Error: --stock and --timeframe are required for --quick-test")
                return 1
                
            tester.run_quick_test(args.stock, args.timeframe, args.start_date, args.end_date)
        
        else:
            # Run full tests
            if args.stock and args.timeframe:
                # Test specific stock and timeframe
                if hasattr(tester, 'run_parameter_testing_for_stock_timeframe_parallel'):
                    # Colab version
                    tester.run_parameter_testing_for_stock_timeframe_parallel(
                        args.stock, args.timeframe, start_date=args.start_date, end_date=args.end_date
                    )
                else:
                    # Standard version
                    stock_id, stock_symbol = tester.get_stock_by_symbol_or_id(args.stock)
                    timeframe_id, minutes, timeframe_name = tester.get_timeframe_by_name_or_id(args.timeframe)
                    
                    results = tester.run_parameter_test(
                        stock_id, timeframe_id, args.start_date, args.end_date,
                        args.hold_strategy, args.test_all
                    )
                    
                    if results is not None and not results.empty:
                        tester.plot_results(results, stock_symbol, timeframe_name)
                        tester.generate_report(results, stock_id, stock_symbol, 
                                              timeframe_id, timeframe_name)
            
            elif args.stock:
                # Test all timeframes for a specific stock
                tester.run_all_tests(
                    args.stock, args.test_all, args.hold_strategy,
                    args.start_date, args.end_date
                )
            
            else:
                # Test all stocks and timeframes
                tester.run_all_tests(
                    None, args.test_all, args.hold_strategy,
                    args.start_date, args.end_date
                )
        
    except Exception as e:
        print(f"Error during parameter testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up resources
        if 'tester' in locals():
            tester.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
