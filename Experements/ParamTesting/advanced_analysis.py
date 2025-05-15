#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Parameter Testing Analysis

This script provides comprehensive data visualizations for the parameter testing results,
exploring relationships between pattern count, profit factor, win rates, and other metrics
across different parameter combinations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import traceback
import argparse

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # Navigate up to project root
sys.path.append(str(project_root))

# Output directory
OUTPUT_DIR = project_root / "Images" / "ParamTesting" / "AdvancedAnalysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to the database
DB_PATH = project_root / "Data" / "Storage" / "data.db"

class AdvancedAnalyzer:
    """Class for advanced analysis and visualization of parameter testing results."""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize the analyzer with database connection."""
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
        
        # Create high-resolution figure settings
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        
        # Set seaborn style
        sns.set(style="whitegrid")
    
    def get_timeframe_suffix(self, timeframe_id=None):
        """Get timeframe suffix for output filenames.
        
        Args:
            timeframe_id: Optional timeframe ID
            
        Returns:
            str: Filename suffix based on timeframe
        """
        # Get timeframe name if provided
        timeframe_name = None
        if timeframe_id:
            result = self.cursor.execute(
                "SELECT name FROM timeframes WHERE timeframe_id = ?", 
                (timeframe_id,)
            ).fetchone()
            if result:
                timeframe_name = result[0]
        
        # Create filename suffix based on timeframe
        return f"_{timeframe_name}" if timeframe_name else "_all_timeframes"
        
    def load_cluster_data(self, stock_id=None, timeframe_id=None):
        """Load cluster data from the database.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            DataFrame with cluster data
        """
        # Build query
        query = """
        SELECT c.cluster_id, c.stock_id, c.timeframe_id, s.symbol, t.name as timeframe_name,
               c.probability_score_dir, c.probability_score_stat, 
               c.pattern_count, c.max_gain, c.max_drawdown,
               c.reward_risk_ratio, c.profit_factor, c.label,
               ec.n_pips, ec.lookback, ec.hold_period
        FROM clusters c
        JOIN experiment_configs ec ON c.config_id = ec.config_id
        JOIN stocks s ON c.stock_id = s.stock_id
        JOIN timeframes t ON c.timeframe_id = t.timeframe_id
        """
        
        # Add filters if specified
        conditions = []
        params = []
        
        if stock_id:
            conditions.append("c.stock_id = ?")
            params.append(stock_id)
        
        if timeframe_id:
            conditions.append("c.timeframe_id = ?")
            params.append(timeframe_id)
        
        # Add WHERE clause if there are conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute query
        if params:
            df = pd.read_sql_query(query, self.conn, params=params)
        else:
            df = pd.read_sql_query(query, self.conn)
        
        # Debug: Log the columns of the DataFrame
        print("Loaded DataFrame columns:", df.columns.tolist())
        
        # Add calculated fields
        if not df.empty:
            # Add parameter combination string for easier identification
            df['param_combo'] = df.apply(
                lambda row: f"P{row['n_pips']}_L{row['lookback']}_H{row['hold_period']}", 
                axis=1
            )
            
            # Add win rate as derived from reward risk ratio
            # Win rate = reward risk ratio / (1 + reward risk ratio)
            # This is an approximation formula
            df['win_rate'] = df['reward_risk_ratio'] / (1 + df['reward_risk_ratio']) * 100
        
        return df
    
    def load_performance_metrics(self, stock_id=None, timeframe_id=None):
        """Load performance metrics from the database.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            DataFrame with performance metrics
        """
        # Build query
        query = """
        SELECT pm.metric_id, pm.stock_id, pm.timeframe_id, s.symbol, t.name as timeframe_name,
               pm.config_id, pm.win_rate, pm.profit_factor, pm.total_trades,
               ec.n_pips, ec.lookback, ec.hold_period
        FROM performance_metrics pm
        JOIN experiment_configs ec ON pm.config_id = ec.config_id
        JOIN stocks s ON pm.stock_id = s.stock_id
        JOIN timeframes t ON pm.timeframe_id = t.timeframe_id
        """
        
        # Add filters if specified
        conditions = []
        params = []
        
        if stock_id:
            conditions.append("pm.stock_id = ?")
            params.append(stock_id)
        
        if timeframe_id:
            conditions.append("pm.timeframe_id = ?")
            params.append(timeframe_id)
        
        # Add WHERE clause if there are conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute query
        if params:
            df = pd.read_sql_query(query, self.conn, params=params)
        else:
            df = pd.read_sql_query(query, self.conn)
        
        # Add calculated fields
        if not df.empty:
            # Add parameter combination string for easier identification
            df['param_combo'] = df.apply(
                lambda row: f"P{row['n_pips']}_L{row['lookback']}_H{row['hold_period']}", 
                axis=1
            )
        
        return df

    def analyze_pattern_count_vs_profit_factor(self, stock_id=None, timeframe_id=None):
       
        # Load data
        df = self.load_cluster_data(stock_id, timeframe_id)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get suffix for output filenames
        filename_suffix = self.get_timeframe_suffix(timeframe_id)
        
        # Filter out extreme outliers for better visualization
        df = df[df['profit_factor'] <= 1000]
        
        # Create a scatter plot with cluster labels
        plt.figure(figsize=(14, 10))
        
        # Define colors for each label
        colors = {'Buy': 'green', 'Sell': 'red', 'Neutral': 'gray'}
        
        # Plot points by label
        for label, color in colors.items():
            subset = df[df['label'] == label]
            if not subset.empty:
                plt.scatter(
                    subset['pattern_count'],
                    subset['profit_factor'],
                    c=color,
                    label=label,
                    alpha=0.6,
                    s=subset['win_rate'] * 2  # Size based on win rate
                )
        
        # Add trendline
        z = np.polyfit(df['pattern_count'], df['profit_factor'], 1)
        p = np.poly1d(z)
        plt.plot(
            df['pattern_count'],
            p(df['pattern_count']),
            "r--",
            alpha=0.8,
            label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}"
        )
        
        # Calculate correlation
        correlation = df['pattern_count'].corr(df['profit_factor'])
        
        # Add annotation for correlation
        plt.annotate(
            f"Correlation: {correlation:.3f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        )
        
        # Add title and labels
        plt.title('Relationship Between Pattern Count and Profit Factor', fontsize=16)
        plt.xlabel('Pattern Count', fontsize=14)
        plt.ylabel('Profit Factor', fontsize=14)
        
        # Add legend
        plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
        
        # Save the figure
        output_file = OUTPUT_DIR / f"pattern_count_vs_profit_factor{filename_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved pattern count vs profit factor analysis to {output_file}")
        
        # Create interactive Plotly visualization
        fig = px.scatter(
            df, 
            x='pattern_count',
            y='profit_factor',
            color='label',
            size='win_rate',
            hover_data=['param_combo', 'symbol', 'timeframe_name', 'probability_score_stat'],
            color_discrete_map={'Buy': 'green', 'Sell': 'red', 'Neutral': 'gray'},
            title='Relationship Between Pattern Count and Profit Factor',
            labels={
                'pattern_count': 'Pattern Count',
                'profit_factor': 'Profit Factor',
                'win_rate': 'Win Rate (%)',
                'param_combo': 'Parameter Combination',
                'symbol': 'Stock Symbol',
                'timeframe_name': 'Timeframe',
                'probability_score_stat': 'Probability Score (Stat)'
            }
        )
        
        # Add trend line
        fig.add_traces(
            px.scatter(
                df, 
                x='pattern_count', 
                y='profit_factor',
                trendline='ols'
            ).data[1]
        )
        
        # Update layout
        fig.update_layout(
            width=1200,
            height=800,
            font=dict(size=14),
            legend=dict(title='Cluster Label', font=dict(size=12))
        )
        
        # Save interactive visualization
        interactive_file = OUTPUT_DIR / f"pattern_count_vs_profit_factor_interactive{filename_suffix}.html"
        fig.write_html(str(interactive_file))
        
        print(f"Saved interactive pattern count vs profit factor analysis to {interactive_file}")
        
        # Create additional views - Histogram of pattern counts by label
        plt.figure(figsize=(14, 8))
        for label, color in zip(['Buy', 'Sell', 'Neutral'], ['green', 'red', 'gray']):
            subset = df[df['label'] == label]
            if not subset.empty:
                sns.histplot(
                    data=subset,
                    x='pattern_count',
                    color=color,
                    alpha=0.5,
                    label=label,
                    kde=True
                )
        
        plt.title('Distribution of Pattern Counts by Cluster Label', fontsize=16)
        plt.xlabel('Pattern Count', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
        
        # Save histogram
        histogram_file = OUTPUT_DIR / f"pattern_count_distribution_by_label{filename_suffix}.png"
        plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved pattern count distribution to {histogram_file}")
        
        # Return statistics
        stats_dict = {
            'mean_pattern_count': df['pattern_count'].mean(),
            'median_pattern_count': df['pattern_count'].median(),
            'mean_profit_factor': df['profit_factor'].mean(),
            'median_profit_factor': df['profit_factor'].median(),
            'correlation': correlation,
            'pattern_count_by_label': df.groupby('label')['pattern_count'].mean().to_dict(),
            'profit_factor_by_label': df.groupby('label')['profit_factor'].mean().to_dict()
        }
        
        return stats_dict

    def analyze_win_rate_by_parameters(self, stock_id=None, timeframe_id=None):
        """Analyze win rates across different parameter combinations.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            dict: Statistics about win rates by parameters
        """
        # Load data
        df = self.load_cluster_data(stock_id, timeframe_id)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get suffix for output filenames
        filename_suffix = self.get_timeframe_suffix(timeframe_id)
        
        # Create visualizations for win rates by each parameter
        parameters = ['n_pips', 'lookback', 'hold_period']
        
        # Create boxplots for each parameter
        for param in parameters:
            plt.figure(figsize=(14, 8))
            
            # Group by parameter and compute mean win rate
            param_win_rates = df.groupby(param)['win_rate'].mean().reset_index()
            
            # Sort by parameter value
            param_win_rates = param_win_rates.sort_values(param)
            
            # Create bar chart
            plt.bar(
                param_win_rates[param].astype(str),
                param_win_rates['win_rate'],
                color='skyblue'
            )
            
            # Add data labels
            for i, v in enumerate(param_win_rates['win_rate']):
                plt.text(
                    i,
                    v + 0.5,
                    f'{v:.1f}%',
                    ha='center',
                    fontsize=10
                )
            
            # Add title and labels
            plt.title(f'Win Rate by {param}', fontsize=16)
            plt.xlabel(param, fontsize=14)
            plt.ylabel('Win Rate (%)', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            
            # Save figure
            output_file = OUTPUT_DIR / f"win_rate_by_{param}{filename_suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved win rate analysis by {param} to {output_file}")
        
        # Create correlation heatmap between parameters and win rate
        param_corr = df[parameters + ['win_rate']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            param_corr,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Parameter Correlation with Win Rate', fontsize=16)
        plt.tight_layout()
        
        # Save heatmap
        corr_file = OUTPUT_DIR / f"parameter_win_rate_correlation{filename_suffix}.png"
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parameter correlation analysis to {corr_file}")
        
        # Return statistics
        stats_dict = {
            'win_rate_by_parameter': {
                param: df.groupby(param)['win_rate'].mean().to_dict()
                for param in parameters
            },
            'parameter_correlations': param_corr['win_rate'].to_dict()
        }
        
        return stats_dict
    
    def analyze_probability_score_stat_relationships(self, stock_id=None, timeframe_id=None):
        """Analyze relationships between probability scores and statistical measures.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            dict: Statistics about probability score relationships
        """
        # Load data
        df = self.load_cluster_data(stock_id, timeframe_id)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get suffix for output filenames
        filename_suffix = self.get_timeframe_suffix(timeframe_id)
        
        # Check if probability_score_stat exists
        if 'probability_score_stat' not in df.columns:
            print("Probability score statistics not found in data.")
            return None
        
        # Create scatter plot of probability score vs win rate
        plt.figure(figsize=(14, 8))
        
        # Create scatter plot with different colors for each label
        for label, color in zip(['Buy', 'Sell', 'Neutral'], ['green', 'red', 'gray']):
            label_df = df[df['label'] == label]
            if not label_df.empty:
                plt.scatter(
                    label_df['probability_score_stat'],
                    label_df['win_rate'],
                    alpha=0.7,
                    s=100,
                    c=color,
                    label=label
                )
        
        # Calculate correlation
        correlation = df['probability_score_stat'].corr(df['win_rate'])
        
        # Add annotation for correlation
        plt.annotate(
            f"Correlation: {correlation:.3f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        )
        
        # Add trend line
        z = np.polyfit(df['probability_score_stat'], df['win_rate'], 1)
        p = np.poly1d(z)
        plt.plot(
            df['probability_score_stat'],
            p(df['probability_score_stat']),
            "r--",
            alpha=0.8,
            linewidth=2
        )
        
        # Add title and labels
        plt.title('Relationship Between Probability Score and Win Rate', fontsize=16)
        plt.xlabel('Probability Score (Statistical)', fontsize=14)
        plt.ylabel('Win Rate (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
        
        # Save figure
        output_file = OUTPUT_DIR / f"probability_score_vs_win_rate{filename_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved probability score vs win rate analysis to {output_file}")
        
        # Create scatter plot of probability score vs profit factor
        plt.figure(figsize=(14, 8))
        
        # Create scatter plot with different colors for each label
        for label, color in zip(['Buy', 'Sell', 'Neutral'], ['green', 'red', 'gray']):
            label_df = df[df['label'] == label]
            if not label_df.empty:
                plt.scatter(
                    label_df['probability_score_stat'],
                    label_df['profit_factor'],
                    alpha=0.7,
                    s=100,
                    c=color,
                    label=label
                )
        
        # Calculate correlation
        pf_correlation = df['probability_score_stat'].corr(df['profit_factor'])
        
        # Add annotation for correlation
        plt.annotate(
            f"Correlation: {pf_correlation:.3f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
        )
        
        # Add trend line
        z = np.polyfit(df['probability_score_stat'], df['profit_factor'], 1)
        p = np.poly1d(z)
        plt.plot(
            df['probability_score_stat'],
            p(df['probability_score_stat']),
            "r--",
            alpha=0.8,
            linewidth=2
        )
        
        # Add title and labels
        plt.title('Relationship Between Probability Score and Profit Factor', fontsize=16)
        plt.xlabel('Probability Score (Statistical)', fontsize=14)
        plt.ylabel('Profit Factor', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
        
        # Save figure
        pf_output_file = OUTPUT_DIR / f"probability_score_vs_profit_factor{filename_suffix}.png"
        plt.savefig(pf_output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved probability score vs profit factor analysis to {pf_output_file}")
        
        # Create distribution of probability scores
        plt.figure(figsize=(14, 8))
        
        # Create histogram with different colors for each label
        for label, color in zip(['Buy', 'Sell', 'Neutral'], ['green', 'red', 'gray']):
            label_df = df[df['label'] == label]
            if not label_df.empty:
                plt.hist(
                    label_df['probability_score_stat'],
                    alpha=0.6,
                    bins=20,
                    color=color,
                    label=label
                )
        
        # Add title and labels
        plt.title('Distribution of Probability Scores by Cluster Label', fontsize=16)
        plt.xlabel('Probability Score (Statistical)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
        
        # Save figure
        hist_output_file = OUTPUT_DIR / f"probability_score_distribution{filename_suffix}.png"
        plt.savefig(hist_output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved probability score distribution to {hist_output_file}")
        
        # Create interactive visualization with plotly
        fig = px.scatter(
            df, 
            x='probability_score_stat',
            y='win_rate',
            color='label',
            size='pattern_count',
            hover_data=['param_combo', 'symbol', 'timeframe_name', 'profit_factor'],
            color_discrete_map={'Buy': 'green', 'Sell': 'red', 'Neutral': 'gray'},
            title='Relationship Between Probability Score and Win Rate',
            labels={
                'probability_score_stat': 'Probability Score (Statistical)',
                'win_rate': 'Win Rate (%)',
                'pattern_count': 'Pattern Count',
                'param_combo': 'Parameter Combination',
                'symbol': 'Stock Symbol',
                'timeframe_name': 'Timeframe',
                'profit_factor': 'Profit Factor'
            }
        )
        
        # Add trend line
        fig.add_traces(
            px.scatter(
                df, 
                x='probability_score_stat', 
                y='win_rate',
                trendline='ols'
            ).data[1]
        )
        
        # Update layout
        fig.update_layout(
            width=1200,
            height=800,
            font=dict(size=14),
            legend=dict(title='Cluster Label', font=dict(size=12))
        )
        
        # Save interactive visualization
        interactive_file = OUTPUT_DIR / f"probability_score_interactive{filename_suffix}.html"
        fig.write_html(str(interactive_file))
        
        print(f"Saved interactive probability score analysis to {interactive_file}")
        
        # Calculate threshold analysis
        # Find optimal probability score thresholds for maximizing win rate
        thresholds = np.linspace(df['probability_score_stat'].min(), df['probability_score_stat'].max(), 20)
        threshold_results = []
        
        for threshold in thresholds:
            high_prob_df = df[df['probability_score_stat'] >= threshold]
            
            if not high_prob_df.empty:
                avg_win_rate = high_prob_df['win_rate'].mean()
                avg_profit_factor = high_prob_df['profit_factor'].mean()
                pattern_count = high_prob_df['pattern_count'].sum()
                percent_patterns = (pattern_count / df['pattern_count'].sum()) * 100
                
                threshold_results.append({
                    'threshold': threshold,
                    'avg_win_rate': avg_win_rate,
                    'avg_profit_factor': avg_profit_factor,
                    'pattern_count': pattern_count,
                    'percent_patterns': percent_patterns
                })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        # Create threshold analysis plot
        plt.figure(figsize=(14, 8))
        
        # Create line plot for win rate
        ax1 = plt.gca()
        ax1.plot(
            threshold_df['threshold'],
            threshold_df['avg_win_rate'],
            'b-',
            linewidth=2,
            label='Win Rate (%)'
        )
        ax1.set_xlabel('Probability Score Threshold', fontsize=14)
        ax1.set_ylabel('Win Rate (%)', fontsize=14, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Add second y-axis for pattern count percentage
        ax2 = ax1.twinx()
        ax2.plot(
            threshold_df['threshold'],
            threshold_df['percent_patterns'],
            'r-',
            linewidth=2,
            label='Pattern Coverage (%)'
        )
        ax2.set_ylabel('Pattern Coverage (%)', fontsize=14, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add title
        plt.title('Win Rate and Pattern Coverage by Probability Score Threshold', fontsize=16)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Save figure
        threshold_file = OUTPUT_DIR / f"probability_score_threshold_analysis{filename_suffix}.png"
        plt.savefig(threshold_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved probability score threshold analysis to {threshold_file}")
        
        # Find the optimal threshold that maximizes both win rate and coverage
        # Simple approach: find where the product of win rate and pattern coverage is maximized
        threshold_df['combined_score'] = threshold_df['avg_win_rate'] * threshold_df['percent_patterns'] / 100
        optimal_row = threshold_df.loc[threshold_df['combined_score'].idxmax()]
        optimal_threshold = optimal_row['threshold']
        
        print(f"Optimal probability score threshold: {optimal_threshold:.3f}")
        print(f"At this threshold: Win Rate = {optimal_row['avg_win_rate']:.1f}%, Pattern Coverage = {optimal_row['percent_patterns']:.1f}%")
        
        # Return statistics
        stats_dict = {
            'win_rate_correlation': correlation,
            'profit_factor_correlation': pf_correlation,
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_df.to_dict('records')
        }
        
        return stats_dict

    def analyze_label_distributions(self, stock_id=None, timeframe_id=None):
        """Analyze the distribution of labels (Buy/Sell/Neutral) across different parameters.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            dict: Statistics about label distributions
        """
        # Load data
        df = self.load_cluster_data(stock_id, timeframe_id)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get suffix for output filenames
        filename_suffix = self.get_timeframe_suffix(timeframe_id)
        
        # Count the distribution of labels
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        
        # Calculate percentages
        total = label_counts['Count'].sum()
        label_counts['Percentage'] = (label_counts['Count'] / total) * 100
        
        # Create pie chart of label distribution
        plt.figure(figsize=(12, 8))
        
        # Define colors for pie chart
        colors = ['green', 'red', 'gray']
        
        plt.pie(
            label_counts['Count'],
            labels=label_counts['Label'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=True,
            explode=[0.1 if label == label_counts['Label'].iloc[0] else 0 for label in label_counts['Label']]
        )
        
        plt.title('Distribution of Cluster Labels', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save pie chart
        pie_file = OUTPUT_DIR / f"label_distribution_pie{filename_suffix}.png"
        plt.savefig(pie_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved label distribution pie chart to {pie_file}")
        
        # Create a comparison of metrics by label
        metrics = ['win_rate', 'profit_factor', 'pattern_count', 'probability_score_stat']
        
        # Group by label
        label_metrics = df.groupby('label')[metrics].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten multi-level column index
        label_metrics.columns = ['_'.join(col).strip('_') if col[0] != 'label' else 'label' for col in label_metrics.columns]
        
        # Create bar chart comparing key metrics across labels
        plt.figure(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.25
        
        # Set position of bars on x axis
        r1 = np.arange(len(label_metrics))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        plt.bar(
            r1,
            label_metrics['win_rate_mean'],
            width=bar_width,
            color='skyblue',
            edgecolor='black',
            label='Win Rate (%)'
        )
        
        plt.bar(
            r2,
            label_metrics['profit_factor_mean'],
            width=bar_width,
            color='lightgreen',
            edgecolor='black',
            label='Profit Factor'
        )
        
        plt.bar(
            r3,
            label_metrics['probability_score_stat_mean'] * 100,  # Scale for better visualization
            width=bar_width,
            color='lightcoral',
            edgecolor='black',
            label='Probability Score (x100)'
        )
        
        # Add labels and title
        plt.xlabel('Cluster Label', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.title('Comparison of Key Metrics by Cluster Label', fontsize=16)
        plt.xticks([r + bar_width for r in range(len(label_metrics))], label_metrics['label'])
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save comparison chart
        comparison_file = OUTPUT_DIR / f"label_metrics_comparison{filename_suffix}.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved label metrics comparison to {comparison_file}")
        
        # Create analysis of parameter distribution by label
        parameters = ['n_pips', 'lookback', 'hold_period']
        
        # Create parameter distribution plots for each label
        for param in parameters:
            plt.figure(figsize=(14, 8))
            
            # Create a grouped histogram
            for label, color in zip(['Buy', 'Sell', 'Neutral'], ['green', 'red', 'gray']):
                label_df = df[df['label'] == label]
                if not label_df.empty:
                    plt.hist(
                        label_df[param],
                        alpha=0.6,
                        bins=min(20, len(df[param].unique())),
                        color=color,
                        label=label
                    )
            
            # Add title and labels
            plt.title(f'Distribution of {param} by Cluster Label', fontsize=16)
            plt.xlabel(param, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Cluster Label', fontsize=12, title_fontsize=12)
            
            # Save figure
            param_dist_file = OUTPUT_DIR / f"{param}_distribution_by_label{filename_suffix}.png"
            plt.savefig(param_dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {param} distribution by label to {param_dist_file}")
        
        # Create interactive comparison visualization
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                           subplot_titles=("Label Distribution", "Key Metrics by Label"))
        
        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=label_counts['Label'],
                values=label_counts['Count'],
                hole=0.4,
                marker_colors=colors
            ),
            row=1, col=1
        )
        
        # Add bar chart for metrics
        for i, (metric, color) in enumerate(zip(['win_rate_mean', 'profit_factor_mean', 'probability_score_stat_mean'],
                                             ['skyblue', 'lightgreen', 'lightcoral'])):
            y_values = label_metrics[metric]
            if metric == 'probability_score_stat_mean':
                y_values = y_values * 100  # Scale for better visualization
                
            fig.add_trace(
                go.Bar(
                    x=label_metrics['label'],
                    y=y_values,
                    name=metric.replace('_mean', '').replace('_', ' ').title(),
                    marker_color=color
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Cluster Label Analysis",
            width=1200,
            height=600
        )
        
        # Save interactive visualization
        interactive_file = OUTPUT_DIR / f"label_analysis_interactive{filename_suffix}.html"
        fig.write_html(str(interactive_file))
        
        print(f"Saved interactive label analysis to {interactive_file}")
        
        # Additional analysis: How labels relate to stock symbols
        if 'symbol' in df.columns:
            # Create a cross-tabulation of symbols and labels
            symbol_label_counts = pd.crosstab(df['symbol'], df['label'])
            
            # Convert to percentages
            symbol_label_pct = symbol_label_counts.div(symbol_label_counts.sum(axis=1), axis=0) * 100
            
            # Create heatmap
            plt.figure(figsize=(14, max(8, len(symbol_label_pct) * 0.4)))
            
            sns.heatmap(
                symbol_label_pct,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={'label': 'Percentage (%)'}
            )
            
            plt.title('Label Distribution by Stock Symbol (%)', fontsize=16)
            plt.ylabel('Stock Symbol', fontsize=14)
            plt.xlabel('Cluster Label', fontsize=14)
            plt.tight_layout()
            
            # Save heatmap
            symbol_heatmap_file = OUTPUT_DIR / f"label_distribution_by_symbol{filename_suffix}.png"
            plt.savefig(symbol_heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved label distribution by symbol to {symbol_heatmap_file}")
        
        # Return statistics
        stats_dict = {
            'label_distribution': label_counts.to_dict('records'),
            'label_metrics': {
                label: {
                    'win_rate': df[df['label'] == label]['win_rate'].mean(),
                    'profit_factor': df[df['label'] == label]['profit_factor'].mean(),
                    'pattern_count': df[df['label'] == label]['pattern_count'].sum(),
                    'probability_score': df[df['label'] == label]['probability_score_stat'].mean() if 'probability_score_stat' in df.columns else None
                }
                for label in df['label'].unique()
            }
        }
        
        return stats_dict

    def create_parameter_effectiveness_analysis(self, stock_id=None, timeframe_id=None):
        """Analyze the effectiveness of different parameter combinations.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
            
        Returns:
            dict: Statistics about parameter effectiveness
        """
        # Load data
        df = self.load_cluster_data(stock_id, timeframe_id)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get suffix for output filenames
        filename_suffix = self.get_timeframe_suffix(timeframe_id)
        
        # Group by parameter combination
        param_combos = df['param_combo'].unique()
        grouped = df.groupby('param_combo').agg({
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'pattern_count': 'sum',
            'max_gain': 'mean',
            'max_drawdown': 'mean',
            'reward_risk_ratio': 'mean',
            'n_pips': 'first',
            'lookback': 'first',
            'hold_period': 'first'
        }).reset_index()
        
        # Sort by win rate to find top combinations
        top_combinations = grouped.sort_values('win_rate', ascending=False).head(10)
        
        # Create visualization of top parameters
        plt.figure(figsize=(14, 10))
        
        # Create a new dataframe for the heatmap with parameter combinations as index
        heatmap_df = top_combinations.copy()
        heatmap_df['param_combo'] = heatmap_df.apply(
            lambda row: f"P{int(row['n_pips'])}_L{int(row['lookback'])}_H{int(row['hold_period'])}", 
            axis=1
        )
        heatmap_df = heatmap_df.set_index('param_combo')
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'Win Rate (%)': heatmap_df['win_rate'],
            'Profit Factor': heatmap_df['profit_factor'],
            'Pattern Count': heatmap_df['pattern_count']
        })
        
        # Normalize for better visualization
        normalized_data = heatmap_data.copy()
        for col in normalized_data.columns:
            max_val = normalized_data[col].max()
            if max_val > 0:
                normalized_data[col] = normalized_data[col] / max_val
        
        # Plot heatmap
        ax = sns.heatmap(
            normalized_data.T,  # Transpose for better display
            annot=heatmap_data.T.round(2),  # Show actual values
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={'label': 'Normalized Value'}
        )
        
        plt.title('Top 10 Parameter Combinations by Win Rate', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        output_file = OUTPUT_DIR / f"top_parameter_combinations{filename_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved top parameter combinations to {output_file}")
        
        # Create radar charts for top 5 combinations
        top5 = top_combinations.head(5)
        
        # Prepare data for radar chart
        metrics = ['win_rate', 'profit_factor', 'reward_risk_ratio', 'pattern_count', 'max_gain']
        
        # Normalize metrics for radar chart
        radar_data = top5[metrics].copy()
        for metric in metrics:
            max_val = radar_data[metric].max()
            if max_val > 0:
                radar_data[metric] = radar_data[metric] / max_val
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set up the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each parameter combination
        for i, row in top5.iterrows():
            values = radar_data.loc[i, metrics].tolist()
            values += values[:1]  # Close the loop
            
            # Plot radar path
            ax.plot(angles, values, linewidth=2, label=row['param_combo'])
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], size=12)
        ax.set_title('Top 5 Parameter Combinations Performance Comparison', size=16)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save radar chart
        radar_file = OUTPUT_DIR / f"top_param_combo_radar{filename_suffix}.png"
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parameter combination radar chart to {radar_file}")
        
        # Create 3D scatter plot of parameter space
        fig = px.scatter_3d(
            grouped,
            x='n_pips',
            y='lookback',
            z='hold_period',
            color='win_rate',
            size='pattern_count',
            hover_data=['param_combo', 'profit_factor', 'reward_risk_ratio'],
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Parameter Space Exploration: Win Rate and Pattern Count',
            labels={
                'n_pips': 'N Pips',
                'lookback': 'Lookback Period',
                'hold_period': 'Hold Period',
                'win_rate': 'Win Rate (%)',
                'pattern_count': 'Pattern Count',
                'param_combo': 'Parameter Combination',
                'profit_factor': 'Profit Factor',
                'reward_risk_ratio': 'Reward/Risk Ratio'
            }
        )
        
        # Update layout
        fig.update_layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis_title='N Pips',
                yaxis_title='Lookback Period',
                zaxis_title='Hold Period'
            )
        )
        
        # Save interactive visualization
        interactive_file = OUTPUT_DIR / f"parameter_space_exploration{filename_suffix}.html"
        fig.write_html(str(interactive_file))
        
        print(f"Saved interactive parameter space exploration to {interactive_file}")
        
        # Analyze parameter interactions
        # Create a multi-parameter interaction analysis
        print("Analyzing parameter interactions...")
        
        # Create parameter interaction matrices
        interactions = {}
        parameters = ['n_pips', 'lookback', 'hold_period']
        
        for i, param1 in enumerate(parameters):
            for param2 in parameters[i+1:]:
                # Create a pivot table showing how these two parameters interact to affect win rate
                pivot = df.pivot_table(
                    values='win_rate',
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".1f",
                    cmap="YlGnBu",
                    linewidths=0.5,
                    cbar_kws={'label': 'Win Rate (%)'}
                )
                
                plt.title(f'Win Rate by {param1.title()} and {param2.title()}', fontsize=16)
                plt.xlabel(param2.title(), fontsize=14)
                plt.ylabel(param1.title(), fontsize=14)
                plt.tight_layout()
                
                # Save interaction heatmap
                interaction_file = OUTPUT_DIR / f"interaction_{param1}_{param2}{filename_suffix}.png"
                plt.savefig(interaction_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {param1}-{param2} interaction analysis to {interaction_file}")
                
                # Store interaction matrix for statistics
                interactions[f"{param1}_{param2}"] = pivot.to_dict()
        
        # Calculate parameter importance through correlation
        param_importance = {}
        for param in parameters:
            correlation = df[param].corr(df['win_rate'])
            param_importance[param] = abs(correlation)
        
        # Create parameter importance bar chart
        plt.figure(figsize=(10, 6))
        
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(param_importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        
        bars = plt.bar(
            range(len(sorted_importance)),
            list(sorted_importance.values()),
            tick_label=list(sorted_importance.keys())
        )
        
        # Add importance values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                fontsize=12
            )
        
        plt.title('Parameter Importance (Correlation with Win Rate)', fontsize=16)
        plt.xlabel('Parameter', fontsize=14)
        plt.ylabel('Absolute Correlation', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.0)
        
        # Save parameter importance chart
        importance_file = OUTPUT_DIR / f"parameter_importance{filename_suffix}.png"
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parameter importance analysis to {importance_file}")
        
        # Analyze parameter robustness
        # For each parameter combination, calculate the standard deviation of win rates
        # across different stocks and timeframes to find the most robust combinations
        
        # Group by param_combo and calculate stats for each combination
        robustness = df.groupby('param_combo').agg({
            'win_rate': ['mean', 'std', 'count'],
            'profit_factor': 'mean',
            'n_pips': 'first',
            'lookback': 'first',
            'hold_period': 'first'
        })
        
        # Flatten multi-level column index
        robustness.columns = ['_'.join(col).strip('_') for col in robustness.columns.values]
        
        # Calculate coefficient of variation (CV) for win rate
        # Lower CV means more consistent performance
        robustness['win_rate_cv'] = (robustness['win_rate_std'] / robustness['win_rate_mean']) * 100
        
        # Sort by win rate mean (descending) and CV (ascending) to find high and stable win rates
        robust_score = robustness['win_rate_mean'] / (robustness['win_rate_cv'] + 1)
        robustness['robust_score'] = robust_score
        
        # Get top robust combinations
        top_robust = robustness.sort_values('robust_score', ascending=False).head(10).reset_index()
        
        # Create bar chart of robust parameter combinations
        plt.figure(figsize=(14, 8))
        
        # Plot bars for win rate
        ax1 = plt.gca()
        bars = ax1.bar(
            top_robust['param_combo'],
            top_robust['win_rate_mean'],
            yerr=top_robust['win_rate_std'],
            capsize=5,
            color='skyblue',
            label='Win Rate (%)'
        )
        
        # Add second axis for coefficient of variation
        ax2 = ax1.twinx()
        line = ax2.plot(
            top_robust['param_combo'],
            top_robust['win_rate_cv'],
            'ro-',
            label='Coefficient of Variation (%)'
        )
        
        # Configure axes
        ax1.set_xlabel('Parameter Combination', fontsize=14)
        ax1.set_ylabel('Win Rate (%)', fontsize=14)
        ax2.set_ylabel('Coefficient of Variation (%)', fontsize=14, color='r')
        ax1.set_xticklabels(top_robust['param_combo'], rotation=45, ha='right')
        
        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.title('Most Robust Parameter Combinations', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        output_file = OUTPUT_DIR / f"parameter_robustness{filename_suffix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parameter robustness analysis to {output_file}")
        
        # Return statistics
        stats_dict = {
            'parameter_impact': param_importance,
            'top_combinations': top_combinations.to_dict('records'),
            'parameter_interactions': interactions,
            'robust_combinations': top_robust.to_dict('records')
        }
        
        return stats_dict

    def analyze_timeframe_impact(self, stock_id=None):
        """Analyze how different timeframes affect trading performance.
        
        Args:
            stock_id: Optional stock ID to filter data
            
        Returns:
            dict: Statistics about timeframe impact
        """
        # Load data - we intentionally don't filter by timeframe here
        df = self.load_cluster_data(stock_id, timeframe_id=None)
        
        if df.empty:
            print("No data available for analysis.")
            return None

        # Ensure required columns are present
        required_columns = ['timeframe_name', 'win_rate', 'profit_factor', 'pattern_count', 'probability_score_stat', 'reward_risk_ratio']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns for analysis: {missing_columns}")
            return None

        # Group by timeframe and calculate metrics
        aggregation_dict = {
            'win_rate': ['mean', 'std', 'count'],
            'profit_factor': ['mean', 'std'],
            'probability_score_stat': 'mean',
            'reward_risk_ratio': 'mean'
        }

        # Add 'pattern_count' to aggregation if it exists
        if 'pattern_count' in df.columns:
            aggregation_dict['pattern_count'] = 'sum'

        timeframe_stats = df.groupby('timeframe_name').agg(aggregation_dict)

        # Flatten multi-level column index
        timeframe_stats.columns = ['_'.join(col).strip('_') for col in timeframe_stats.columns.values]

        # Convert to regular dataframe for easier plotting
        timeframe_stats = timeframe_stats.reset_index()
        
        # Create bar chart comparing win rates across timeframes
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        ax = plt.subplot(111)
        
        # Set width of bars
        bar_width = 0.35
        
        # Set position of bars on x axis
        r1 = np.arange(len(timeframe_stats))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        bars1 = ax.bar(
            r1, 
            timeframe_stats['win_rate_mean'], 
            width=bar_width, 
            yerr=timeframe_stats['win_rate_std'],
            capsize=5,
            label='Win Rate (%)',
            color='royalblue'
        )
        
        bars2 = ax.bar(
            r2, 
            timeframe_stats['profit_factor_mean'], 
            width=bar_width, 
            yerr=timeframe_stats['profit_factor_std'],
            capsize=5,
            label='Profit Factor',
            color='forestgreen'
        )
        
        # Add labels and title
        plt.xlabel('Timeframe', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.title('Win Rate and Profit Factor by Timeframe', fontsize=16)
        plt.xticks([r + bar_width/2 for r in range(len(timeframe_stats))], timeframe_stats['timeframe_name'])
        plt.legend()
        
        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{height:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=11
            )
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=11
            )
        
        plt.tight_layout()
        
        # Save the figure
        output_file = OUTPUT_DIR / "timeframe_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved timeframe performance comparison to {output_file}")
        
        # Create a radar chart comparing different metrics across timeframes
        metrics = ['win_rate_mean', 'profit_factor_mean', 'reward_risk_ratio_mean', 'probability_score_stat_mean']
        
        # Normalize metrics for radar chart
        radar_data = timeframe_stats[metrics].copy()
        for metric in metrics:
            max_val = radar_data[metric].max()
            if max_val > 0:
                radar_data[metric] = radar_data[metric] / max_val
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set up the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each timeframe
        for i, row in timeframe_stats.iterrows():
            values = radar_data.loc[i, metrics].tolist()
            values += values[:1]  # Close the loop
            
            # Plot radar path
            ax.plot(angles, values, linewidth=2, label=row['timeframe_name'])
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').replace('mean', '').title() for m in metrics], size=12)
        ax.set_title('Timeframe Performance Comparison', size=16)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save radar chart
        radar_file = OUTPUT_DIR / "timeframe_radar_comparison.png"
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved timeframe radar comparison to {radar_file}")
        
       
        
        # Return statistics
        stats_dict = {
            'timeframe_statistics': timeframe_stats.to_dict('records')
        }
        
        return stats_dict

    def analyze_pattern_timeframe_relationship(self, stock_id=None):
        """Analyze relationship between patterns and timeframes.
        
        Args:
            stock_id: Optional stock ID to filter data
            
        Returns:
            dict: Statistics about pattern-timeframe relationships
        """
        # Load data - we intentionally don't filter by timeframe here
        df = self.load_cluster_data(stock_id, timeframe_id=None)
        
        if df.empty:
            print("No data available for analysis.")
            return None
        
        # Get unique parameters and timeframes
        param_combos = df['param_combo'].unique()
        timeframes = df['timeframe_name'].unique()
        
        # Create a pivot table showing parameter effectiveness by timeframe
        # Group by parameter combination and timeframe
        grouped = df.groupby(['param_combo', 'timeframe_name']).agg({
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'pattern_count': 'sum',
            'n_pips': 'first',
            'lookback': 'first',
            'hold_period': 'first'
        }).reset_index()
        
        # Create pivot table
        pivot_table = grouped.pivot(
            index='param_combo', 
            columns='timeframe_name', 
            values='win_rate'
        )
        
        # Fill NaN values with 0
        pivot_table = pivot_table.fillna(0)
        
        # Create effectiveness heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pivot_table,
            annot=pivot_table.round(1),  # Show actual win rates
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={'label': 'Relative Effectiveness (%)'}
        )
        
        plt.title('Parameter Combination Effectiveness by Timeframe', fontsize=16)
        plt.ylabel('Parameter Combination', fontsize=14)
        plt.xlabel('Timeframe', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        output_file = OUTPUT_DIR / "parameter_timeframe_effectiveness_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parameter-timeframe effectiveness heatmap to {output_file}")
        
        # 2. Analyze parameter sensitivity across timeframes
        # For each parameter, we want to know how its impact varies by timeframe
        param_sensitivity = {}
        
        for param in ['n_pips', 'lookback', 'hold_period']:
            # Group by timeframe and parameter value
            param_impact = df.groupby(['timeframe_name', param]).agg({
                'win_rate': 'mean',
                'profit_factor': 'mean'
            }).reset_index()
            
            # Create a multi-line chart showing how this parameter affects metrics across timeframes
            plt.figure(figsize=(14, 8))
            
            for timeframe in timeframes:
                subset = param_impact[param_impact['timeframe_name'] == timeframe]
                if not subset.empty:
                    plt.plot(
                        subset[param],
                        subset['win_rate'],
                        marker='o',
                        linewidth=2,
                        label=timeframe
                    )
            
            plt.title(f'Effect of {param} on Win Rate Across Timeframes', fontsize=16)
            plt.xlabel(param, fontsize=14)
            plt.ylabel('Win Rate (%)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Timeframe')
            
            # Save figure
            param_output_file = OUTPUT_DIR / f"{param}_sensitivity_by_timeframe.png"
            plt.savefig(param_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {param} sensitivity analysis by timeframe to {param_output_file}")
            
            # Calculate and store parameter sensitivity for each timeframe
            sensitivity_by_tf = {}
            
            for timeframe in timeframes:
                subset = param_impact[param_impact['timeframe_name'] == timeframe]
                if len(subset) > 1:
                    # Calculate the range of impact this parameter has on win rate for this timeframe
                    win_rate_range = subset['win_rate'].max() - subset['win_rate'].min()
                    profit_factor_range = subset['profit_factor'].max() - subset['profit_factor'].min()
                    
                    # Find the optimal value for this parameter in this timeframe
                    best_value = subset.loc[subset['win_rate'].idxmax()][param]
                    
                    sensitivity_by_tf[timeframe] = {
                        'win_rate_range': win_rate_range,
                        'profit_factor_range': profit_factor_range,
                        'best_value': best_value
                    }
            
            param_sensitivity[param] = sensitivity_by_tf
        
        # 3. Create an interactive 3D visualization showing parameter effectiveness across timeframes
        # Prepare the data
        scatter_data = []
        
        for tf_name in timeframes:
            for param_combo in param_combos:
                subset = grouped[(grouped['timeframe_name'] == tf_name) & 
                                 (grouped['param_combo'] == param_combo)]
                
                if not subset.empty:
                    row = subset.iloc[0]
                    scatter_data.append({
                        'timeframe': tf_name,
                        'param_combo': param_combo,
                        'n_pips': row['n_pips'],
                        'lookback': row['lookback'],
                        'hold_period': row['hold_period'],
                        'win_rate': row['win_rate'],
                        'profit_factor': row['profit_factor'],
                        'pattern_count': row['pattern_count']
                    })
        
        # Convert to DataFrame
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            scatter_df,
            x='n_pips',
            y='lookback',
            z='hold_period',
            color='timeframe',
            size='win_rate',
            symbol='timeframe',
            hover_data=['param_combo', 'win_rate', 'profit_factor'],
            title='Parameter Effectiveness Across Timeframes',
            labels={
                'n_pips': 'N Pips',
                'lookback': 'Lookback Period',
                'hold_period': 'Hold Period',
                'win_rate': 'Win Rate (%)',
                'profit_factor': 'Profit Factor',
                'param_combo': 'Parameter Combination',
                'timeframe': 'Timeframe'
            }
        )
        
        # Update layout
        fig.update_layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis_title='N Pips',
                yaxis_title='Lookback Period',
                zaxis_title='Hold Period'
            )
        )
        
        # Save interactive visualization
        interactive_file = OUTPUT_DIR / "parameter_timeframe_3d.html"
        fig.write_html(str(interactive_file))
        
        print(f"Saved interactive parameter-timeframe visualization to {interactive_file}")
        
        # Return statistics
        stats_dict = {
            'parameter_timeframe_effectiveness': pivot_table.to_dict(),
            'parameter_sensitivity': param_sensitivity
        }
        
        return stats_dict

    def run_all_analyses(self, stock_id=None, timeframe_id=None):
        """Run all analysis methods in sequence.
        
        Args:
            stock_id: Optional stock ID to filter data
            timeframe_id: Optional timeframe ID to filter data
        """
        print("\n===== Running All Analyses =====\n")
        
        # Basic analyses
        print("\n----- Pattern Count vs Profit Factor Analysis -----")
        self.analyze_pattern_count_vs_profit_factor(stock_id, timeframe_id)
        
        print("\n----- Win Rate Analysis by Parameters -----")
        self.analyze_win_rate_by_parameters(stock_id, timeframe_id)
        
        print("\n----- Probability Score Statistical Relationships -----")
        self.analyze_probability_score_stat_relationships(stock_id, timeframe_id)
        
        print("\n----- Parameter Effectiveness Analysis -----")
        self.create_parameter_effectiveness_analysis(stock_id, timeframe_id)
        
        print("\n----- Label Distribution Analysis -----")
        self.analyze_label_distributions(stock_id, timeframe_id)
        
        # Timeframe-specific analyses (only run when no specific timeframe is selected)
        if timeframe_id is None:
            print("\n----- Timeframe Impact Analysis -----")
            self.analyze_timeframe_impact(stock_id)
            
            print("\n----- Pattern-Timeframe Relationship Analysis -----")
            self.analyze_pattern_timeframe_relationship(stock_id)
        
        print("\n===== All Analyses Completed =====\n")

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Parameter Testing Analysis')
    parser.add_argument('--stock', help='Stock ID or symbol to analyze')
    parser.add_argument('--timeframe', help='Timeframe ID or name to analyze')
    parser.add_argument('--analysis', choices=['pattern', 'winrate', 'probability', 'effectiveness', 'label', 'timeframe', 'all'],
                        default='all', help='Type of analysis to run (default: all)')
    
    args = parser.parse_args()
    print(f"Starting advanced analysis with args: {args}")
    
    try:
        print("Initializing AdvancedAnalyzer...")
        analyzer = AdvancedAnalyzer()
        print(f"Database connected at: {DB_PATH}")
        
        # Convert stock name to ID if provided
        stock_id = None
        if args.stock:
            print(f"Processing stock argument: {args.stock}")
            try:
                if args.stock.isdigit():
                    stock_id = int(args.stock)
                    print(f"Using stock ID: {stock_id}")
                else:
                    # Try to look up by symbol
                    cursor = analyzer.conn.cursor()
                    print(f"Looking up stock by symbol: {args.stock}")
                    result = cursor.execute(
                        "SELECT stock_id FROM stocks WHERE symbol LIKE ?", 
                        (f"%{args.stock}%",)
                    ).fetchone()
                    if result:
                        stock_id = result[0]
                        print(f"Found stock ID: {stock_id}")
                    else:
                        print(f"Stock with symbol '{args.stock}' not found")
                        return 1
            except Exception as e:
                print(f"Error looking up stock: {e}")
                traceback.print_exc()
                return 1
        
        # Convert timeframe name to ID if provided
        timeframe_id = None
        if args.timeframe:
            try:
                if args.timeframe.isdigit():
                    timeframe_id = int(args.timeframe)
                else:
                    # Try to look up by name
                    cursor = analyzer.conn.cursor()
                    result = cursor.execute(
                        "SELECT timeframe_id FROM timeframes WHERE name LIKE ?", 
                        (f"%{args.timeframe}%",)
                    ).fetchone()
                    if result:
                        timeframe_id = result[0]
                    else:
                        print(f"Timeframe with name '{args.timeframe}' not found")
                        return 1
            except Exception as e:
                print(f"Error looking up timeframe: {e}")
                return 1
        
        # Run the selected analysis
        if args.analysis == 'all':
            analyzer.run_all_analyses(stock_id, timeframe_id)
        elif args.analysis == 'pattern':
            analyzer.analyze_pattern_count_vs_profit_factor(stock_id, timeframe_id)
        elif args.analysis == 'winrate':
            analyzer.analyze_win_rate_by_parameters(stock_id, timeframe_id)
        elif args.analysis == 'probability':
            analyzer.analyze_probability_score_stat_relationships(stock_id, timeframe_id)
        elif args.analysis == 'effectiveness':
            analyzer.create_parameter_effectiveness_analysis(stock_id, timeframe_id)
        elif args.analysis == 'label':
            analyzer.analyze_label_distributions(stock_id, timeframe_id)
        elif args.analysis == 'timeframe':
            analyzer.analyze_timeframe_impact(stock_id)
        
    except Exception as e:
        import traceback
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return 1
    finally:
        if 'analyzer' in locals():
            analyzer.close()
    
    return 0


if __name__ == "__main__":
    # try:
    #     sys.exit(main())
    # except Exception as e:
    #     print(f"Unhandled exception: {e}")
    #     traceback.print_exc()
    advanced_testing = AdvancedAnalyzer()
    advanced_testing.run_all_analyses(stock_id=1)
    advanced_testing.close()