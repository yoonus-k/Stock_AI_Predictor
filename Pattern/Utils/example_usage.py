"""
Example script for using the PatternVisualizer class.

This script demonstrates how to use the PatternVisualizer class to visualize
price patterns and clusters from the database.
"""

import sys
import os
import argparse
from Pattern.Utils.pattern_visualizer import PatternVisualizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pattern Visualization Utility')
    parser.add_argument('--stock_id', type=int, default=1, help='Stock ID')
    parser.add_argument('--timeframe_id', type=int, default=1, help='Timeframe ID')
    parser.add_argument('--config_id', type=int, default=1, help='Configuration ID')
    parser.add_argument('--cluster_id', type=int, default=0, help='Cluster ID')
    parser.add_argument('--show_all', action='store_true', help='Show all visualizations')
    parser.add_argument('--report', action='store_true', help='Generate a text report')
    
    args = parser.parse_args()
    
    # Create the visualizer
    visualizer = PatternVisualizer()
    
    try:
        # Get some basic info about what we're visualizing
        stock_info = visualizer.get_stock_info(args.stock_id)
        timeframe_info = visualizer.get_timeframe_info(args.timeframe_id)
        config_info = visualizer.get_config_info(args.config_id)
        
        print(f"Analyzing patterns for {stock_info['symbol']} on {timeframe_info['name']} timeframe with configuration {config_info['name']}")
        
        if args.show_all:
            # Perform all visualizations
            print("Displaying cluster center...")
            visualizer.plot_cluster_center(args.cluster_id, args.stock_id, args.timeframe_id, args.config_id)
            
            print("Displaying patterns in cluster...")
            visualizer.plot_cluster_patterns(args.cluster_id, args.stock_id, args.timeframe_id, args.config_id)
            
            print("Displaying cluster histogram...")
            visualizer.plot_cluster_histogram(args.cluster_id, args.stock_id, args.timeframe_id, args.config_id)
            
            print("Displaying all cluster centers...")
            visualizer.plot_all_clusters(args.stock_id, args.timeframe_id, args.config_id)
            
            print("Displaying cluster performance comparison...")
            visualizer.plot_cluster_performance_comparison(args.stock_id, args.timeframe_id, args.config_id)
            
            print("Displaying MFE/MAE analysis...")
            visualizer.plot_mfe_mae_analysis(args.stock_id, args.timeframe_id, args.config_id, args.cluster_id)
        else:
            # Just display the cluster center
            visualizer.plot_cluster_center(args.cluster_id, args.stock_id, args.timeframe_id, args.config_id)
        
        if args.report:
            # Generate and save a report
            print("Generating report...")
            report = visualizer.generate_pattern_report(args.stock_id, args.timeframe_id, args.config_id)
            visualizer.save_report_to_file(report)
            
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()
