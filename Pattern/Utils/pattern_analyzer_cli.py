"""
Advanced Pattern Analysis and Visualization Tool

This script provides a command-line interface for analyzing and visualizing
price patterns and clusters from the database. It supports batch operations,
exporting data, and generating comprehensive reports.
"""

import sys
import os
import argparse
import json
import pandas as pd
from datetime import datetime
from Pattern.Utils.pattern_visualizer import PatternVisualizer

class PatternAnalyzer:
    """
    Command-line tool for pattern analysis and visualization.
    """
    
    def __init__(self):
        """Initialize the PatternAnalyzer."""
        self.visualizer = PatternVisualizer()
    
    def list_stocks(self):
        """List all available stocks in the database."""
        stocks = self.visualizer.get_all_stocks()
        
        print("\n=== Available Stocks ===")
        print(f"{'ID':<5} {'Symbol':<10} {'Name':<30} {'Sector':<20}")
        print("="*65)
        
        for stock in stocks:
            print(f"{stock['stock_id']:<5} {stock['symbol']:<10} {stock['name']:<30} {stock['sector']:<20}")
    
    def list_timeframes(self):
        """List all available timeframes in the database."""
        timeframes = self.visualizer.get_all_timeframes()
        
        print("\n=== Available Timeframes ===")
        print(f"{'ID':<5} {'Minutes':<10} {'Name':<20} {'Description':<30}")
        print("="*65)
        
        for tf in timeframes:
            print(f"{tf['timeframe_id']:<5} {tf['minutes']:<10} {tf['name']:<20} {tf['description']:<30}")
    
    def list_configs(self, stock_id):
        """
        List all available configuration settings for a specific stock.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        """
        configs = self.visualizer.get_configs_for_stock(stock_id)
        
        if not configs:
            print(f"No configurations found for stock ID {stock_id}")
            return
        
        print("\n=== Available Configurations ===")
        print(f"{'ID':<5} {'Name':<20} {'PIPs':<5} {'Lookback':<10} {'Hold':<5} {'Dist':<5} {'Description':<30}")
        print("="*80)
        
        for config in configs:
            print(f"{config['config_id']:<5} {config['name']:<20} {config['n_pips']:<5} "
                  f"{config['lookback']:<10} {config['hold_period']:<5} {config['distance_measure']:<5} "
                  f"{config['description'][:30]:<30}")
    
    def list_clusters(self, stock_id, timeframe_id, config_id):
        """
        List all clusters for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        clusters = self.visualizer.get_all_clusters(stock_id, timeframe_id, config_id)
        
        if not clusters:
            print(f"No clusters found for stock ID {stock_id}, timeframe ID {timeframe_id}, config ID {config_id}")
            return
        
        print("\n=== Available Clusters ===")
        print(f"{'ID':<5} {'Pattern Count':<15} {'Outcome':<10} {'Max Gain':<10} {'Max DD':<10} {'RR Ratio':<10} {'Label':<10}")
        print("="*70)
        
        for cluster in clusters:
            outcome = f"{cluster['outcome']:.4f}" if cluster['outcome'] is not None else "N/A"
            max_gain = f"{cluster['max_gain']:.4f}" if cluster['max_gain'] is not None else "N/A"
            max_dd = f"{cluster['max_drawdown']:.4f}" if cluster['max_drawdown'] is not None else "N/A"
            rr_ratio = f"{cluster['reward_risk_ratio']:.2f}" if cluster['reward_risk_ratio'] is not None else "N/A"
            
            print(f"{cluster['cluster_id']:<5} {cluster['pattern_count']:<15} {outcome:<10} "
                  f"{max_gain:<10} {max_dd:<10} {rr_ratio:<10} {cluster['label']:<10}")
    
    def visualize_cluster(self, stock_id, timeframe_id, config_id, cluster_id):
        """
        Visualize a specific cluster.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        cluster_id : int
            ID of the cluster
        """
        print(f"Visualizing cluster {cluster_id}...")
        self.visualizer.plot_cluster_center(cluster_id, stock_id, timeframe_id, config_id)
        
        user_input = input("Would you like to see the patterns in this cluster? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.plot_cluster_patterns(cluster_id, stock_id, timeframe_id, config_id)
        
        user_input = input("Would you like to see the histogram of outcomes? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.plot_cluster_histogram(cluster_id, stock_id, timeframe_id, config_id)
        
        user_input = input("Would you like to see the MFE/MAE analysis? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.plot_mfe_mae_analysis(stock_id, timeframe_id, config_id, cluster_id)
    
    def compare_clusters(self, stock_id, timeframe_id, config_id):
        """
        Compare all clusters for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        print(f"Comparing all clusters...")
        
        # Plot all cluster centers
        self.visualizer.plot_all_clusters(stock_id, timeframe_id, config_id)
        
        # Plot performance comparison
        self.visualizer.plot_cluster_performance_comparison(stock_id, timeframe_id, config_id)
        
        user_input = input("Would you like to see the MFE/MAE analysis for all clusters? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.plot_mfe_mae_analysis(stock_id, timeframe_id, config_id)
    
    def generate_report(self, stock_id, timeframe_id, config_id):
        """
        Generate a comprehensive report for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        print(f"Generating report...")
        report = self.visualizer.generate_pattern_report(stock_id, timeframe_id, config_id)
        
        # Generate a default filename
        stock_info = self.visualizer.get_stock_info(stock_id)
        timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pattern_report_{stock_info['symbol']}_{timeframe_info['name']}_{timestamp}.txt"
        
        # Ask the user if they want to save the report
        print("\n" + "="*80)
        print(report)
        print("="*80 + "\n")
        
        user_input = input(f"Would you like to save this report to {filename}? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.save_report_to_file(report, filename)
    
    def export_data(self, stock_id, timeframe_id, config_id):
        """
        Export cluster data to a CSV file.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        print(f"Exporting data...")
        
        # Generate a default filename
        stock_info = self.visualizer.get_stock_info(stock_id)
        timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clusters_{stock_info['symbol']}_{timeframe_info['name']}_{timestamp}.csv"
        
        user_input = input(f"Export to {filename}? (y/n): ")
        if user_input.lower() == 'y':
            self.visualizer.export_cluster_data(stock_id, timeframe_id, config_id, filename)
    
    def batch_analysis(self, stock_ids, timeframe_ids, config_ids, export=False, report=False):
        """
        Perform batch analysis on multiple stocks, timeframes, and configurations.
        
        Parameters:
        -----------
        stock_ids : list
            List of stock IDs
        timeframe_ids : list
            List of timeframe IDs
        config_ids : list
            List of configuration IDs
        export : bool, optional
            Whether to export data to CSV
        report : bool, optional
            Whether to generate reports
        """
        for stock_id in stock_ids:
            for timeframe_id in timeframe_ids:
                for config_id in config_ids:
                    try:
                        # Get basic info
                        stock_info = self.visualizer.get_stock_info(stock_id)
                        timeframe_info = self.visualizer.get_timeframe_info(timeframe_id)
                        config_info = self.visualizer.get_config_info(config_id)
                        
                        print(f"\nAnalyzing {stock_info['symbol']} on {timeframe_info['name']} with config {config_info['name']}...")
                        
                        # Compare clusters
                        self.visualizer.plot_all_clusters(stock_id, timeframe_id, config_id)
                        self.visualizer.plot_cluster_performance_comparison(stock_id, timeframe_id, config_id)
                        
                        # Export data if requested
                        if export:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"clusters_{stock_info['symbol']}_{timeframe_info['name']}_{timestamp}.csv"
                            self.visualizer.export_cluster_data(stock_id, timeframe_id, config_id, filename)
                        
                        # Generate report if requested
                        if report:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"report_{stock_info['symbol']}_{timeframe_info['name']}_{timestamp}.txt"
                            report_text = self.visualizer.generate_pattern_report(stock_id, timeframe_id, config_id)
                            self.visualizer.save_report_to_file(report_text, filename)
                    
                    except Exception as e:
                        print(f"Error analyzing {stock_id}/{timeframe_id}/{config_id}: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pattern Analysis and Visualization Tool')
    
    # Main options
    parser.add_argument('--list-stocks', action='store_true', help='List all available stocks')
    parser.add_argument('--list-timeframes', action='store_true', help='List all available timeframes')
    parser.add_argument('--list-configs', type=int, metavar='STOCK_ID', help='List all available configurations for a stock')
    parser.add_argument('--list-clusters', nargs=3, type=int, metavar=('STOCK_ID', 'TIMEFRAME_ID', 'CONFIG_ID'),
                        help='List all clusters for a specific stock, timeframe, and configuration')
    
    # Visualization options
    parser.add_argument('--visualize-cluster', nargs=4, type=int, 
                        metavar=('STOCK_ID', 'TIMEFRAME_ID', 'CONFIG_ID', 'CLUSTER_ID'),
                        help='Visualize a specific cluster')
    parser.add_argument('--compare-clusters', nargs=3, type=int, 
                        metavar=('STOCK_ID', 'TIMEFRAME_ID', 'CONFIG_ID'),
                        help='Compare all clusters for a specific stock, timeframe, and configuration')
    
    # Report and export options
    parser.add_argument('--generate-report', nargs=3, type=int, 
                        metavar=('STOCK_ID', 'TIMEFRAME_ID', 'CONFIG_ID'),
                        help='Generate a comprehensive report')
    parser.add_argument('--export-data', nargs=3, type=int, 
                        metavar=('STOCK_ID', 'TIMEFRAME_ID', 'CONFIG_ID'),
                        help='Export cluster data to CSV')
    
    # Batch analysis options
    parser.add_argument('--batch-analysis', action='store_true',
                        help='Perform batch analysis on multiple stocks, timeframes, and configurations')
    parser.add_argument('--stocks', type=int, nargs='+', help='Stock IDs for batch analysis')
    parser.add_argument('--timeframes', type=int, nargs='+', help='Timeframe IDs for batch analysis')
    parser.add_argument('--configs', type=int, nargs='+', help='Configuration IDs for batch analysis')
    parser.add_argument('--export', action='store_true', help='Export data during batch analysis')
    parser.add_argument('--report', action='store_true', help='Generate reports during batch analysis')
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = PatternAnalyzer()
    
    # Execute the requested command
    if args.list_stocks:
        analyzer.list_stocks()
    
    elif args.list_timeframes:
        analyzer.list_timeframes()
    
    elif args.list_configs is not None:
        analyzer.list_configs(args.list_configs)
    
    elif args.list_clusters is not None:
        analyzer.list_clusters(*args.list_clusters)
    
    elif args.visualize_cluster is not None:
        analyzer.visualize_cluster(*args.visualize_cluster)
    
    elif args.compare_clusters is not None:
        analyzer.compare_clusters(*args.compare_clusters)
    
    elif args.generate_report is not None:
        analyzer.generate_report(*args.generate_report)
    
    elif args.export_data is not None:
        analyzer.export_data(*args.export_data)
    
    elif args.batch_analysis:
        if not all([args.stocks, args.timeframes, args.configs]):
            print("Error: Batch analysis requires --stocks, --timeframes, and --configs")
            return
        
        analyzer.batch_analysis(args.stocks, args.timeframes, args.configs, 
                               args.export, args.report)
    
    else:
        # If no command is specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
