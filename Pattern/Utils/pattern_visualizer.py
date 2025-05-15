"""
Pattern and Cluster Visualization Utilities

This module provides functions for visualizing price patterns and clusters identified by the
Pattern_Miner class. It retrieves data from the database and creates various visualizations
to help analyze price patterns, clusters, and their predictive performance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.gridspec import GridSpec
import sqlite3
from typing import List, Dict, Tuple, Optional, Union, Any
import seaborn as sns
from datetime import datetime

# Add parent directory to path to import Pattern_Miner
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pip_pattern_miner import Pattern_Miner


class PatternVisualizer:
    """
    A class for visualizing price patterns and clusters from the database.
    
    This class provides methods to visualize patterns, clusters, and their attributes
    such as MFE (Maximum Favorable Excursion), MAE (Maximum Adverse Excursion),
    returns, and other statistical properties.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the PatternVisualizer.
        
        Parameters:
        -----------
        db_path : str, optional
            Path to the SQLite database. If None, uses the default database.
        """
        if db_path is None:
            # Default path to the database
            self.db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'Data','Storage', 'data.db'
            )
        else:
            self.db_path = db_path
            self.conn = None
        self.cursor = None
        self._connect_to_db()
        
        # Set default style for visualization
        try:
            # Try newer style name format first
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                # Fall back to more generic style
                plt.style.use('ggplot')
            except:
                # If all else fails, use default style
                pass
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def _connect_to_db(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise
            
    def _disconnect_from_db(self):
        """Disconnect from the SQLite database."""
        if self.conn:
            self.conn.close()
            print("Disconnected from database")
            
    def __del__(self):
        """Clean up database connection when object is deleted."""
        self._disconnect_from_db()
        
    #----------------------------------------------------------------------------------------
    # Database Query Methods
    #----------------------------------------------------------------------------------------
    
    def get_stock_info(self, stock_id: int) -> Dict[str, Any]:
        """
        Get information about a stock from the database.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock to retrieve
            
        Returns:
        --------
        dict
            Dictionary containing stock information
        """
        query = """
        SELECT stock_id, symbol, name, sector 
        FROM stocks 
        WHERE stock_id = ?
        """
        self.cursor.execute(query, (stock_id,))
        result = self.cursor.fetchone()
        
        if result:
            return {
                'stock_id': result[0],
                'symbol': result[1],
                'name': result[2],
                'sector': result[3]
            }
        else:
            raise ValueError(f"No stock found with ID {stock_id}")
    
    def get_timeframe_info(self, timeframe_id: int) -> Dict[str, Any]:
        """
        Get information about a timeframe from the database.
        
        Parameters:
        -----------
        timeframe_id : int
            ID of the timeframe to retrieve
            
        Returns:
        --------
        dict
            Dictionary containing timeframe information
        """
        query = """
        SELECT timeframe_id, minutes, name, description 
        FROM timeframes 
        WHERE timeframe_id = ?
        """
        self.cursor.execute(query, (timeframe_id,))
        result = self.cursor.fetchone()
        
        if result:
            return {
                'timeframe_id': result[0],
                'minutes': result[1],
                'name': result[2],
                'description': result[3]
            }
        else:
            raise ValueError(f"No timeframe found with ID {timeframe_id}")
    
    def get_config_info(self, config_id: int) -> Dict[str, Any]:
        """
        Get information about an experiment configuration from the database.
        
        Parameters:
        -----------
        config_id : int
            ID of the configuration to retrieve
            
        Returns:
        --------
        dict
            Dictionary containing configuration information
        """
        query = """
        SELECT config_id, name, n_pips, lookback, hold_period, 
               returns_hold_period, distance_measure, description 
        FROM experiment_configs 
        WHERE config_id = ?
        """
        self.cursor.execute(query, (config_id,))
        result = self.cursor.fetchone()
        
        if result:
            return {
                'config_id': result[0],
                'name': result[1],
                'n_pips': result[2],
                'lookback': result[3],
                'hold_period': result[4],
                'returns_hold_period': result[5],
                'distance_measure': result[6],
                'description': result[7]
            }
        else:
            raise ValueError(f"No configuration found with ID {config_id}")
    
    def get_cluster_info(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int) -> Dict[str, Any]:
        """
        Get information about a specific cluster from the database.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
            
        Returns:
        --------
        dict
            Dictionary containing cluster information
        """
        query = """
        SELECT cluster_id, stock_id, timeframe_id, config_id, description, 
               avg_price_points_json, avg_volume, outcome, label, 
               probability_score_dir, probability_score_stat, 
               pattern_count, max_gain, max_drawdown, 
               reward_risk_ratio, profit_factor, created_at
        FROM clusters 
        WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
        """
        self.cursor.execute(query, (cluster_id, stock_id, timeframe_id, config_id))
        result = self.cursor.fetchone()
        
        if result:
            # Parse the JSON string of price points if it exists
            avg_price_points = json.loads(result[5]) if result[5] else None
            
            return {
                'cluster_id': result[0],
                'stock_id': result[1],
                'timeframe_id': result[2],
                'config_id': result[3],
                'description': result[4],
                'avg_price_points': avg_price_points,
                'avg_volume': result[6],
                'outcome': result[7],
                'label': result[8],
                'probability_score_dir': result[9],
                'probability_score_stat': result[10],
                'pattern_count': result[11],
                'max_gain': result[12],
                'max_drawdown': result[13],
                'reward_risk_ratio': result[14],
                'profit_factor': result[15],
                'created_at': result[16]
            }
        else:
            raise ValueError(f"No cluster found with ID {cluster_id} for stock {stock_id}, timeframe {timeframe_id}, config {config_id}")
    
    def get_patterns_by_cluster(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int) -> List[Dict[str, Any]]:
        """
        Get all patterns belonging to a specific cluster from the database.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
            
        Returns:
        --------
        list
            List of dictionaries containing pattern information
        """
        query = """
        SELECT pattern_id, stock_id, timeframe_id, config_id, cluster_id, 
               price_points_json, volume, outcome, max_gain, max_drawdown
        FROM patterns 
        WHERE cluster_id = ? AND stock_id = ? AND timeframe_id = ? AND config_id = ?
        """
        self.cursor.execute(query, (cluster_id, stock_id, timeframe_id, config_id))
        results = self.cursor.fetchall()
        
        patterns = []
        for result in results:
            # Parse the JSON string of price points if it exists
            price_points = json.loads(result[5]) if result[5] else None
            
            patterns.append({
                'pattern_id': result[0],
                'stock_id': result[1],
                'timeframe_id': result[2],
                'config_id': result[3],
                'cluster_id': result[4],
                'price_points': price_points,
                'volume': result[6],
                'outcome': result[7],
                'max_gain': result[8],
                'max_drawdown': result[9]
            })
            
        return patterns
    
    def get_all_clusters(self, stock_id: int, timeframe_id: int, config_id: int) -> List[Dict[str, Any]]:
        """
        Get all clusters for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
            
        Returns:
        --------
        list
            List of dictionaries containing cluster information
        """
        query = """
        SELECT cluster_id, stock_id, timeframe_id, config_id, description, 
               avg_price_points_json, outcome, label, 
               probability_score_dir, pattern_count, max_gain, max_drawdown, 
               reward_risk_ratio, profit_factor
        FROM clusters 
        WHERE stock_id = ? AND timeframe_id = ? AND config_id = ?
        ORDER BY cluster_id
        """
        self.cursor.execute(query, (stock_id, timeframe_id, config_id))
        results = self.cursor.fetchall()
        
        clusters = []
        for result in results:
            # Parse the JSON string of price points if it exists
            avg_price_points = json.loads(result[5]) if result[5] else None
            
            clusters.append({
                'cluster_id': result[0],
                'stock_id': result[1],
                'timeframe_id': result[2],
                'config_id': result[3],
                'description': result[4],
                'avg_price_points': avg_price_points,
                'outcome': result[6],
                'label': result[7],
                'probability_score_dir': result[8],
                'pattern_count': result[9],
                'max_gain': result[10],
                'max_drawdown': result[11],
                'reward_risk_ratio': result[12],
                'profit_factor': result[13]
            })
            
        return clusters
    
    def get_stock_data(self, stock_id: int, timeframe_id: int, limit: int = 1000) -> pd.DataFrame:
        """
        Get price data for a specific stock and timeframe.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        limit : int, optional
            Maximum number of records to retrieve
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing price data
        """
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM stock_data
        WHERE stock_id = ? AND timeframe_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=(stock_id, timeframe_id, limit))
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()  # Ensure chronological order
        
        # Rename columns for mplfinance
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def get_configs_for_stock(self, stock_id: int) -> List[Dict[str, Any]]:
        """
        Get all configuration settings used for a specific stock.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
            
        Returns:
        --------
        list
            List of dictionaries containing configuration information
        """
        query = """
        SELECT DISTINCT e.config_id, e.name, e.n_pips, e.lookback, e.hold_period, 
                        e.returns_hold_period, e.distance_measure, e.description
        FROM experiment_configs e
        INNER JOIN clusters c ON e.config_id = c.config_id
        WHERE c.stock_id = ?
        ORDER BY e.config_id
        """
        self.cursor.execute(query, (stock_id,))
        results = self.cursor.fetchall()
        
        configs = []
        for result in results:
            configs.append({
                'config_id': result[0],
                'name': result[1],
                'n_pips': result[2],
                'lookback': result[3],
                'hold_period': result[4],
                'returns_hold_period': result[5],
                'distance_measure': result[6],
                'description': result[7]
            })
            
        return configs
    
    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """
        Get information about all stocks in the database.
        
        Returns:
        --------
        list
            List of dictionaries containing stock information
        """
        query = """
        SELECT stock_id, symbol, name, sector
        FROM stocks
        ORDER BY symbol
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        stocks = []
        for result in results:
            stocks.append({
                'stock_id': result[0],
                'symbol': result[1],
                'name': result[2],
                'sector': result[3]
            })
            
        return stocks
    
    def get_all_timeframes(self) -> List[Dict[str, Any]]:
        """
        Get information about all timeframes in the database.
        
        Returns:
        --------
        list
            List of dictionaries containing timeframe information
        """
        query = """
        SELECT timeframe_id, minutes, name, description
        FROM timeframes
        ORDER BY minutes
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        timeframes = []
        for result in results:
            timeframes.append({
                'timeframe_id': result[0],
                'minutes': result[1],
                'name': result[2],
                'description': result[3]
            })
            
        return timeframes
    
    #----------------------------------------------------------------------------------------
    # Visualization Methods - Cluster Analysis
    #----------------------------------------------------------------------------------------
    
    def plot_cluster_center(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int,
                           display_stats: bool = True):
        """
        Plot a cluster center pattern.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        display_stats : bool, optional
            Whether to display statistics alongside the pattern
        """
        try:
            # Get cluster information
            cluster_info = self.get_cluster_info(cluster_id, stock_id, timeframe_id, config_id)
            
            # Get stock, timeframe, and config info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            config_info = self.get_config_info(config_id)
            
            # Create a figure
            fig = plt.figure(figsize=(12, 8))
            
            if display_stats:
                # Create a grid with two columns
                gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
            else:
                ax1 = fig.add_subplot(111)
            
            # Plot the pattern on the first subplot
            if cluster_info['avg_price_points']:
                x = np.arange(len(cluster_info['avg_price_points']))
                ax1.plot(x, cluster_info['avg_price_points'], marker='o', linewidth=2,
                       color='blue', label=f"Cluster {cluster_id}")
                
                # Add labels and title
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Normalized Price')
                ax1.set_title(f"Cluster {cluster_id} Pattern for {stock_info['symbol']} ({timeframe_info['name']})")
                ax1.grid(True)
                
                # Highlight expected outcome direction
                outcome = cluster_info['outcome']
                if outcome > 0:
                    ax1.axhspan(cluster_info['avg_price_points'][-1], 
                              cluster_info['avg_price_points'][-1] + 0.2, 
                              alpha=0.2, color='green', label='Expected Bullish Outcome')
                elif outcome < 0:
                    ax1.axhspan(cluster_info['avg_price_points'][-1] - 0.2, 
                              cluster_info['avg_price_points'][-1], 
                              alpha=0.2, color='red', label='Expected Bearish Outcome')
                
                ax1.legend()
                
                # Display statistics on the second subplot if requested
                if display_stats:
                    # Clear the second subplot for text
                    ax2.axis('off')
                    
                    # Construct the statistics text
                    stats_text = f"Cluster Statistics:\n\n"
                    stats_text += f"Stock: {stock_info['symbol']} - {stock_info['name']}\n"
                    stats_text += f"Timeframe: {timeframe_info['name']}\n"
                    stats_text += f"Config: {config_info['name']}\n\n"
                    stats_text += f"Pattern Count: {cluster_info['pattern_count']}\n"
                    stats_text += f"Outcome: {cluster_info['outcome']:.4f}\n"
                    stats_text += f"Label: {cluster_info['label']}\n"
                    stats_text += f"Probability Score (Dir): {cluster_info['probability_score_dir']:.4f}\n"
                    stats_text += f"Max Gain: {cluster_info['max_gain']:.4f}\n"
                    stats_text += f"Max Drawdown: {cluster_info['max_drawdown']:.4f}\n"
                    stats_text += f"Reward/Risk Ratio: {cluster_info['reward_risk_ratio']:.4f}\n"
                    stats_text += f"Profit Factor: {cluster_info['profit_factor']:.4f}\n"
                    
                    # Add configuration details
                    stats_text += f"\nConfiguration Details:\n"
                    stats_text += f"PIPs: {config_info['n_pips']}\n"
                    stats_text += f"Lookback: {config_info['lookback']}\n"
                    stats_text += f"Hold Period: {config_info['hold_period']}\n"
                    stats_text += f"Returns Hold Period: {config_info['returns_hold_period']}\n"
                    stats_text += f"Distance Measure: {config_info['distance_measure']}\n"
                    
                    # Display the text
                    ax2.text(0, 0.95, stats_text, va='top', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting cluster center: {e}")
    
    def plot_cluster_patterns(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int,
                            max_patterns: int = 10):
        """
        Plot all patterns within a cluster.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        max_patterns : int, optional
            Maximum number of patterns to display
        """
        try:
            # Get cluster information and all patterns in the cluster
            cluster_info = self.get_cluster_info(cluster_id, stock_id, timeframe_id, config_id)
            patterns = self.get_patterns_by_cluster(cluster_id, stock_id, timeframe_id, config_id)
            
            # Get stock, timeframe, and config info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Plot the cluster center pattern
            if cluster_info['avg_price_points']:
                x = np.arange(len(cluster_info['avg_price_points']))
                plt.plot(x, cluster_info['avg_price_points'], linewidth=3, color='black', 
                       label='Cluster Center', marker='o')
            
            # Plot each pattern (up to max_patterns)
            patterns_to_plot = min(len(patterns), max_patterns)
            for i in range(patterns_to_plot):
                if patterns[i]['price_points']:
                    x = np.arange(len(patterns[i]['price_points']))
                    plt.plot(x, patterns[i]['price_points'], alpha=0.5, 
                           label=f"Pattern {patterns[i]['pattern_id']}")
            
            # Add labels and title
            plt.xlabel('Time Steps')
            plt.ylabel('Normalized Price')
            plt.title(f"Patterns in Cluster {cluster_id} for {stock_info['symbol']} ({timeframe_info['name']})")
            plt.grid(True)
            
            # Add legend with optimal placement
            if patterns_to_plot < 10:
                plt.legend()
            else:
                plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting cluster patterns: {e}")
    
    def plot_cluster_histogram(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int):
        """
        Plot histograms of outcomes within a cluster.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        try:
            # Get all patterns in the cluster
            patterns = self.get_patterns_by_cluster(cluster_id, stock_id, timeframe_id, config_id)
            
            if not patterns:
                print("No patterns found for this cluster.")
                return
            
            # Extract outcomes, max gains, and max drawdowns
            outcomes = [p['outcome'] for p in patterns if p['outcome'] is not None]
            max_gains = [p['max_gain'] for p in patterns if p['max_gain'] is not None]
            max_drawdowns = [p['max_drawdown'] for p in patterns if p['max_drawdown'] is not None]
            
            # Get stock, timeframe info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot histograms
            if outcomes:
                ax1.hist(outcomes, bins=20, alpha=0.7, color='blue')
                ax1.axvline(np.mean(outcomes), color='r', linestyle='dashed', linewidth=1, 
                          label=f'Mean: {np.mean(outcomes):.4f}')
                ax1.set_title('Pattern Outcomes')
                ax1.set_xlabel('Outcome')
                ax1.set_ylabel('Frequency')
                ax1.legend()
            
            if max_gains:
                ax2.hist(max_gains, bins=20, alpha=0.7, color='green')
                ax2.axvline(np.mean(max_gains), color='r', linestyle='dashed', linewidth=1, 
                          label=f'Mean: {np.mean(max_gains):.4f}')
                ax2.set_title('Maximum Gains (MFE)')
                ax2.set_xlabel('Max Gain')
                ax2.set_ylabel('Frequency')
                ax2.legend()
            
            if max_drawdowns:
                ax3.hist(max_drawdowns, bins=20, alpha=0.7, color='red')
                ax3.axvline(np.mean(max_drawdowns), color='blue', linestyle='dashed', linewidth=1, 
                          label=f'Mean: {np.mean(max_drawdowns):.4f}')
                ax3.set_title('Maximum Drawdowns (MAE)')
                ax3.set_xlabel('Max Drawdown')
                ax3.set_ylabel('Frequency')
                ax3.legend()
            
            plt.suptitle(f"Cluster {cluster_id} Statistics for {stock_info['symbol']} ({timeframe_info['name']})")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting cluster histogram: {e}")
    
    def plot_cluster_candlestick_examples(self, cluster_id: int, stock_id: int, timeframe_id: int, config_id: int,
                                       max_examples: int = 6):
        """
        Plot candlestick examples of patterns within a cluster.
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        max_examples : int, optional
            Maximum number of examples to display
        """
        try:
            # Get all patterns in the cluster
            patterns = self.get_patterns_by_cluster(cluster_id, stock_id, timeframe_id, config_id)
            
            if not patterns:
                print("No patterns found for this cluster.")
                return
                
            # Get config information to determine lookback period
            config_info = self.get_config_info(config_id)
            lookback = config_info['lookback']
            hold_period = config_info['hold_period']
            
            # Get stock data
            stock_data = self.get_stock_data(stock_id, timeframe_id)
            
            # Get stock, timeframe info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Determine grid size based on number of examples
            examples_count = min(len(patterns), max_examples)
            grid_size = int(np.ceil(np.sqrt(examples_count)))
            
            # Create a figure with subplots
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            
            # Flatten axes array for easy indexing
            if examples_count > 1:
                axes = axes.flatten()
            
            # Loop through patterns and create candlestick charts
            for i in range(examples_count):
                pattern = patterns[i]
                
                # Determine the data slice to plot
                # This is a simplified example - in a real implementation, you'd need
                # to match pattern_id to actual timestamps in the stock data
                # Here we're just taking random slices for demonstration
                if len(stock_data) > lookback:
                    start_idx = np.random.randint(0, len(stock_data) - lookback - hold_period)
                    end_idx = start_idx + lookback + hold_period
                    data_slice = stock_data.iloc[start_idx:end_idx]
                    
                    # Plot candlestick chart
                    if examples_count == 1:
                        ax = axes
                    else:
                        ax = axes[i]
                    
                    mpf.plot(data_slice, type='candle', style='charles', ax=ax, 
                            title=f"Pattern {pattern['pattern_id']}")
                    
                    # Add outcome annotation
                    outcome = pattern['outcome']
                    if outcome is not None:
                        outcome_color = 'green' if outcome > 0 else 'red'
                        ax.annotate(f"Outcome: {outcome:.4f}", 
                                  xy=(0.05, 0.95), xycoords='axes fraction',
                                  color=outcome_color, fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            if examples_count > 1:
                for j in range(examples_count, len(axes)):
                    axes[j].axis('off')
            
            plt.suptitle(f"Candlestick Examples for Cluster {cluster_id} - {stock_info['symbol']} ({timeframe_info['name']})")
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting candlestick examples: {e}")
    
    #----------------------------------------------------------------------------------------
    # Visualization Methods - Multi-Cluster Analysis
    #----------------------------------------------------------------------------------------
    
    def plot_all_clusters(self, stock_id: int, timeframe_id: int, config_id: int):
        """
        Plot all cluster centers for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        try:
            # Get all clusters
            clusters = self.get_all_clusters(stock_id, timeframe_id, config_id)
            
            if not clusters:
                print("No clusters found for this configuration.")
                return
            
            # Get stock, timeframe info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Plot each cluster center
            for cluster in clusters:
                if cluster['avg_price_points']:
                    x = np.arange(len(cluster['avg_price_points']))
                    label = f"Cluster {cluster['cluster_id']}"
                    
                    # Add outcome direction to label
                    if cluster['outcome'] is not None:
                        if cluster['outcome'] > 0:
                            label += " (Bullish)"
                        elif cluster['outcome'] < 0:
                            label += " (Bearish)"
                        else:
                            label += " (Neutral)"
                    
                    plt.plot(x, cluster['avg_price_points'], marker='o', label=label)
            
            # Add labels and title
            plt.xlabel('Time Steps')
            plt.ylabel('Normalized Price')
            plt.title(f"All Cluster Centers for {stock_info['symbol']} ({timeframe_info['name']})")
            plt.grid(True)
            
            # Add legend with optimal placement
            if len(clusters) <= 10:
                plt.legend()
            else:
                plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting all clusters: {e}")
    
    def plot_cluster_performance_comparison(self, stock_id: int, timeframe_id: int, config_id: int):
        """
        Plot a comparison of performance metrics across all clusters.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        """
        try:
            # Get all clusters
            clusters = self.get_all_clusters(stock_id, timeframe_id, config_id)
            
            if not clusters:
                print("No clusters found for this configuration.")
                return
            
            # Get stock, timeframe info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Extract metrics for comparison
            cluster_ids = [c['cluster_id'] for c in clusters]
            outcomes = [c['outcome'] if c['outcome'] is not None else 0 for c in clusters]
            max_gains = [c['max_gain'] if c['max_gain'] is not None else 0 for c in clusters]
            max_drawdowns = [c['max_drawdown'] if c['max_drawdown'] is not None else 0 for c in clusters]
            reward_risk_ratios = [c['reward_risk_ratio'] if c['reward_risk_ratio'] is not None else 0 for c in clusters]
            pattern_counts = [c['pattern_count'] if c['pattern_count'] is not None else 0 for c in clusters]
            
            # Convert cluster IDs to strings for plotting
            cluster_ids_str = [f"C{c}" for c in cluster_ids]
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot outcome comparison
            axes[0, 0].bar(cluster_ids_str, outcomes, color=[
                'green' if o > 0 else 'red' if o < 0 else 'gray' for o in outcomes
            ])
            axes[0, 0].set_title('Expected Outcomes by Cluster')
            axes[0, 0].set_xlabel('Cluster ID')
            axes[0, 0].set_ylabel('Expected Outcome')
            axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot max gain and drawdown comparison
            axes[0, 1].bar(cluster_ids_str, max_gains, color='green', label='Max Gain')
            axes[0, 1].bar(cluster_ids_str, max_drawdowns, color='red', label='Max Drawdown')
            axes[0, 1].set_title('Max Gain vs Max Drawdown by Cluster')
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].legend()
            
            # Plot reward/risk ratio
            axes[1, 0].bar(cluster_ids_str, reward_risk_ratios, color=[
                'green' if r > 1 else 'orange' if r > 0 else 'red' for r in reward_risk_ratios
            ])
            axes[1, 0].set_title('Reward/Risk Ratio by Cluster')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('Reward/Risk Ratio')
            axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.5)
            
            # Plot pattern count
            axes[1, 1].bar(cluster_ids_str, pattern_counts, color='blue')
            axes[1, 1].set_title('Pattern Count by Cluster')
            axes[1, 1].set_xlabel('Cluster ID')
            axes[1, 1].set_ylabel('Count')
            
            plt.suptitle(f"Cluster Performance Comparison for {stock_info['symbol']} ({timeframe_info['name']})")
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting cluster performance comparison: {e}")
    
    def plot_mfe_mae_analysis(self, stock_id: int, timeframe_id: int, config_id: int, cluster_id: Optional[int] = None):
        """
        Plot a detailed analysis of Maximum Favorable Excursion (MFE) and 
        Maximum Adverse Excursion (MAE) for patterns.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
        cluster_id : int, optional
            ID of the cluster to analyze. If None, analyzes all clusters.
        """
        try:
            # Get stock, timeframe info for labels
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            
            # Query for patterns
            if cluster_id is not None:
                # Get patterns for a specific cluster
                patterns = self.get_patterns_by_cluster(cluster_id, stock_id, timeframe_id, config_id)
                title_suffix = f" for Cluster {cluster_id}"
            else:
                # Get all patterns for the configuration
                query = """
                SELECT pattern_id, cluster_id, outcome, max_gain, max_drawdown
                FROM patterns
                WHERE stock_id = ? AND timeframe_id = ? AND config_id = ?
                """
                self.cursor.execute(query, (stock_id, timeframe_id, config_id))
                results = self.cursor.fetchall()
                
                patterns = []
                for result in results:
                    patterns.append({
                        'pattern_id': result[0],
                        'cluster_id': result[1],
                        'outcome': result[2],
                        'max_gain': result[3],
                        'max_drawdown': result[4]
                    })
                
                title_suffix = " for All Clusters"
            
            if not patterns:
                print("No patterns found for this configuration.")
                return
            
            # Extract MFE and MAE values
            mfe_values = [p['max_gain'] for p in patterns if p['max_gain'] is not None]
            mae_values = [p['max_drawdown'] for p in patterns if p['max_drawdown'] is not None]
            outcomes = [p['outcome'] for p in patterns if p['outcome'] is not None]
            
            # Create a figure with multiple visualizations
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Scatter plot of MFE vs MAE
            ax1 = fig.add_subplot(gs[0, 0])
            scatter = ax1.scatter(mae_values, mfe_values, c=outcomes, cmap='RdYlGn', 
                                alpha=0.6, edgecolors='k', linewidth=0.5)
            ax1.set_xlabel('Maximum Adverse Excursion (MAE)')
            ax1.set_ylabel('Maximum Favorable Excursion (MFE)')
            ax1.set_title('MFE vs MAE Scatter Plot')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add a diagonal line for reference
            lims = [
                min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
                max(ax1.get_xlim()[1], ax1.get_ylim()[1])
            ]
            ax1.plot(lims, lims, 'k--', alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Outcome')
            
            # 2. Histogram of MFE
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(mfe_values, bins=20, alpha=0.7, color='green')
            ax2.axvline(np.mean(mfe_values), color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean: {np.mean(mfe_values):.4f}')
            ax2.set_title('MFE Distribution')
            ax2.set_xlabel('Maximum Favorable Excursion (MFE)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # 3. Histogram of MAE
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(mae_values, bins=20, alpha=0.7, color='red')
            ax3.axvline(np.mean(mae_values), color='blue', linestyle='dashed', linewidth=1, 
                      label=f'Mean: {np.mean(mae_values):.4f}')
            ax3.set_title('MAE Distribution')
            ax3.set_xlabel('Maximum Adverse Excursion (MAE)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            
            # 4. MFE/MAE Ratio
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Calculate MFE/MAE ratio (avoiding division by zero)
            ratios = []
            for i in range(len(mfe_values)):
                if mae_values[i] != 0 and mae_values[i] is not None:
                    ratios.append(abs(mfe_values[i] / mae_values[i]))
                else:
                    # If MAE is zero, use a large value to represent "infinite" ratio
                    if mfe_values[i] > 0:
                        ratios.append(10)  # arbitrary large value
                    else:
                        ratios.append(0)
            
            ax4.hist(ratios, bins=20, alpha=0.7, color='purple')
            ax4.axvline(np.mean(ratios), color='r', linestyle='dashed', linewidth=1, 
                      label=f'Mean: {np.mean(ratios):.4f}')
            ax4.set_title('MFE/MAE Ratio Distribution')
            ax4.set_xlabel('Absolute MFE/MAE Ratio')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            
            plt.suptitle(f"MFE/MAE Analysis for {stock_info['symbol']} ({timeframe_info['name']}){title_suffix}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting MFE/MAE analysis: {e}")
    
    #----------------------------------------------------------------------------------------
    # Utility Methods
    #----------------------------------------------------------------------------------------
    
    def generate_pattern_report(self, stock_id: int, timeframe_id: int, config_id: int):
        """
        Generate a comprehensive text report about all clusters and patterns
        for a specific stock, timeframe, and configuration.
        
        Parameters:
        -----------
        stock_id : int
            ID of the stock
        timeframe_id : int
            ID of the timeframe
        config_id : int
            ID of the configuration
            
        Returns:
        --------
        str
            Text report
        """
        try:
            # Get relevant information
            stock_info = self.get_stock_info(stock_id)
            timeframe_info = self.get_timeframe_info(timeframe_id)
            config_info = self.get_config_info(config_id)
            clusters = self.get_all_clusters(stock_id, timeframe_id, config_id)
            
            if not clusters:
                return "No clusters found for this configuration."
            
            # Generate report header
            report = f"==================================================\n"
            report += f"PATTERN ANALYSIS REPORT\n"
            report += f"==================================================\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Asset and configuration information
            report += f"ASSET INFORMATION:\n"
            report += f"Symbol: {stock_info['symbol']}\n"
            report += f"Name: {stock_info['name']}\n"
            report += f"Sector: {stock_info['sector']}\n\n"
            
            report += f"TIMEFRAME: {timeframe_info['name']} ({timeframe_info['minutes']} minutes)\n\n"
            
            report += f"CONFIGURATION:\n"
            report += f"Name: {config_info['name']}\n"
            report += f"PIPs: {config_info['n_pips']}\n"
            report += f"Lookback: {config_info['lookback']}\n"
            report += f"Hold Period: {config_info['hold_period']}\n"
            report += f"Returns Hold Period: {config_info['returns_hold_period']}\n"
            report += f"Distance Measure: {config_info['distance_measure']}\n\n"
            
            # Overall statistics
            total_patterns = 0
            bullish_clusters = 0
            bearish_clusters = 0
            neutral_clusters = 0
            
            for cluster in clusters:
                total_patterns += cluster['pattern_count'] if cluster['pattern_count'] else 0
                if cluster['outcome'] > 0:
                    bullish_clusters += 1
                elif cluster['outcome'] < 0:
                    bearish_clusters += 1
                else:
                    neutral_clusters += 1
            
            report += f"OVERALL STATISTICS:\n"
            report += f"Total Clusters: {len(clusters)}\n"
            report += f"Total Patterns: {total_patterns}\n"
            report += f"Bullish Clusters: {bullish_clusters}\n"
            report += f"Bearish Clusters: {bearish_clusters}\n"
            report += f"Neutral Clusters: {neutral_clusters}\n\n"
            
            # Top performing clusters
            bullish_sorted = sorted([c for c in clusters if c['outcome'] > 0], 
                                  key=lambda x: x['outcome'], reverse=True)
            bearish_sorted = sorted([c for c in clusters if c['outcome'] < 0], 
                                   key=lambda x: x['outcome'])
            
            report += f"TOP BULLISH CLUSTERS:\n"
            for i, cluster in enumerate(bullish_sorted[:3]):
                report += f"{i+1}. Cluster {cluster['cluster_id']}: Outcome={cluster['outcome']:.4f}, "
                report += f"Patterns={cluster['pattern_count']}, RR Ratio={cluster['reward_risk_ratio']:.2f}\n"
            report += "\n"
            
            report += f"TOP BEARISH CLUSTERS:\n"
            for i, cluster in enumerate(bearish_sorted[:3]):
                report += f"{i+1}. Cluster {cluster['cluster_id']}: Outcome={cluster['outcome']:.4f}, "
                report += f"Patterns={cluster['pattern_count']}, RR Ratio={cluster['reward_risk_ratio']:.2f}\n"
            report += "\n"
            
            # Individual cluster details
            report += f"DETAILED CLUSTER INFORMATION:\n"
            for cluster in clusters:
                report += f"--------------------------------------------------\n"
                report += f"Cluster {cluster['cluster_id']}:\n"
                report += f"  Label: {cluster['label']}\n"
                report += f"  Pattern Count: {cluster['pattern_count']}\n"
                report += f"  Outcome: {cluster['outcome']:.4f}\n"
                report += f"  Probability Score: {cluster['probability_score_dir']:.4f}\n"
                report += f"  Max Gain: {cluster['max_gain']:.4f}\n"
                report += f"  Max Drawdown: {cluster['max_drawdown']:.4f}\n"
                report += f"  Reward/Risk Ratio: {cluster['reward_risk_ratio']:.4f}\n"
                report += f"  Profit Factor: {cluster['profit_factor']:.4f}\n"
                report += f"  Description: {cluster['description']}\n"
            
            report += f"==================================================\n"
            report += f"END OF REPORT\n"
            report += f"==================================================\n"
            
            return report
            
        except Exception as e:
            print(f"Error generating pattern report: {e}")
            return f"Error generating report: {str(e)}"

    def save_report_to_file(self, report: str, filename: str = None):
        """
        Save a text report to a file.
        
        Parameters:
        -----------
        report : str
            Text report to save
        filename : str, optional
            Name of the file to save to. If None, generates a default filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to {filename}")
        except Exception as e:
            print(f"Error saving report: {e}")

    def export_cluster_data(self, stock_id: int, timeframe_id: int, config_id: int, filename: str = None):
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
        filename : str, optional
            Name of the file to save to. If None, generates a default filename.
        """
        try:
            # Get all clusters
            clusters = self.get_all_clusters(stock_id, timeframe_id, config_id)
            
            if not clusters:
                print("No clusters found for this configuration.")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(clusters)
            
            # Remove the price points JSON to keep the CSV clean
            if 'avg_price_points' in df.columns:
                df = df.drop(columns=['avg_price_points'])
            
            # Generate filename if not provided
            if filename is None:
                stock_info = self.get_stock_info(stock_id)
                timeframe_info = self.get_timeframe_info(timeframe_id)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clusters_{stock_info['symbol']}_{timeframe_info['name']}_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Cluster data exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting cluster data: {e}")


# Example usage
if __name__ == "__main__":
    visualizer = PatternVisualizer()
    
    # Example: Visualize a specific cluster
    stock_id = 1  # Bitcoin
    timeframe_id = 1  # 60-minute
    config_id = 1  # Default config
    cluster_id = 0  # First cluster
    
    # Display cluster center with statistics
    visualizer.plot_cluster_center(cluster_id, stock_id, timeframe_id, config_id)
    
    # Display all patterns in the cluster
    visualizer.plot_cluster_patterns(cluster_id, stock_id, timeframe_id, config_id)
    
    # Display histogram of outcomes
    visualizer.plot_cluster_histogram(cluster_id, stock_id, timeframe_id, config_id)
    
    # Compare all clusters
    visualizer.plot_all_clusters(stock_id, timeframe_id, config_id)
    
    # Performance comparison
    visualizer.plot_cluster_performance_comparison(stock_id, timeframe_id, config_id)
    
    # MFE/MAE analysis
    visualizer.plot_mfe_mae_analysis(stock_id, timeframe_id, config_id, cluster_id)
    
    # Generate and save a report
    report = visualizer.generate_pattern_report(stock_id, timeframe_id, config_id)
    print(report)
