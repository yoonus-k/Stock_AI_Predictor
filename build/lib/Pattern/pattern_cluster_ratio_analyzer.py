#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern to Cluster Ratio Analyzer

This script analyzes the relationship between pattern count and optimal cluster count
to find if there's a consistent ratio that produces high-quality clustering across
different financial market data and timeframes.

The analysis evaluates cluster quality using silhouette scores across varying
percentages of patterns (from 10% to 90% with a 10% step size) to determine if there's an optimal
percentage of patterns to use as cluster centers for financial time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import sqlite3
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the benchmarking and pattern miner classes
from Pattern.pip_pattern_miner import Pattern_Miner


class PatternClusterRatioAnalyzer:
    """
    Analyzes the relationship between pattern count and optimal cluster count
    to find if there's a consistent ratio that produces high quality clustering.
    """
    
    def __init__(self, n_pips=5, lookback=24):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        n_pips : int
            Number of PIPs to use in pattern miner
        lookback : int
            Lookback window size to use in pattern miner
        """
        self.n_pips = n_pips
        self.lookback = lookback
        self.pattern_miner = None
        self.results = {}
        
    def setup_pattern_miner(self, data):
        """
        Setup the pattern miner with the provided data
        
        Parameters:
        -----------
        data : np.array
            Price data to use for generating patterns
        """
        self.pattern_miner = Pattern_Miner(
            n_pips=self.n_pips, 
            lookback=self.lookback,
            hold_period=6, 
            returns_hold_period=6, 
            distance_measure=2
        )
        
        # Feed data into pattern miner
        self.pattern_miner.train(data)
    def analyze_cluster_ratios(self, data, symbol, timeframe, percentage_range=(10, 90), step_size=10, n_jobs=None):
        """
        Analyze the relationship between pattern count and cluster count
        
        Parameters:
        -----------
        data : np.array
            Price data to analyze
        symbol : str
            Symbol name for labeling
        timeframe : str
            Timeframe name for labeling
        percentage_range : tuple
            Range of percentages to test (min_percentage, max_percentage)
        step_size : int
            Step size for percentages to test (e.g., 10 means 10%, 20%, 30%, etc.)
        n_jobs : int, optional
            Number of parallel jobs to use
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with analysis results
        """
        print(f"Analyzing pattern-to-cluster percentage for {symbol} ({timeframe})")
        
        # Setup pattern miner
        self.setup_pattern_miner(data)
        
        # Get patterns
        patterns = np.array(self.pattern_miner._unique_pip_patterns)
        n_patterns = len(patterns)
        
        print(f"Found {n_patterns} unique patterns")
        
        if n_patterns < 10:
            print(f"Not enough patterns for analysis. Found only {n_patterns} patterns.")
            return None
        
        # Generate percentages to test
        min_percentage, max_percentage = percentage_range
        percentages = np.arange(min_percentage, max_percentage + 1, step_size)
        ratios = percentages / 100.0
        
        # Determine CPU cores to use
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        
        print(f"Using {n_jobs} CPU cores for parallel processing")
        
        # Prepare arguments for parallel processing
        ratio_batches = [(ratio, patterns, n_patterns) for ratio in ratios]
        
        results = {
            'ratio': [],
            'n_clusters': [],
            'silhouette_score': [],
            'calinski_harabasz_score': [],
            'davies_bouldin_score': [],
            'within_cluster_variance': [],
            'between_cluster_distance': [],
            'execution_time': []
        }
        
        # Process ratios in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(self._evaluate_ratio, *batch): batch[0] 
                for batch in ratio_batches
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Testing cluster ratios"):
                ratio = futures[future]
                try:
                    ratio_results = future.result()
                    
                    # Store results
                    results['n_patterns'] = n_patterns
                    results['ratio'].append(ratio)
                    results['n_clusters'].append(ratio_results['n_clusters'])
                    results['silhouette_score'].append(ratio_results['silhouette_score'])
                    results['calinski_harabasz_score'].append(ratio_results['calinski_harabasz_score'])
                    results['davies_bouldin_score'].append(ratio_results['davies_bouldin_score'])
                    results['within_cluster_variance'].append(ratio_results['within_cluster_variance'])
                    results['between_cluster_distance'].append(ratio_results['between_cluster_distance'])
                    results['execution_time'].append(ratio_results['execution_time'])
                    
                    # Print interim results
                    print(f"Ratio: {ratio:.3f} - Clusters: {ratio_results['n_clusters']} - " +
                          f"Silhouette: {ratio_results['silhouette_score']:.4f}")
                    
                except Exception as e:
                    print(f"Error with ratio {ratio}: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate percentage of patterns
        results_df['percentage'] = results_df['ratio'] * 100
        
        # Store results for this symbol and timeframe
        self.results[(symbol, timeframe)] = results_df
        
        # Create visualizations
        self._plot_ratio_analysis(results_df, symbol, timeframe)
        
        return results_df
    def _evaluate_ratio(self, ratio, patterns, n_patterns):
        """
        Evaluate a specific pattern-to-cluster ratio
        
        Parameters:
        -----------
        ratio : float
            Ratio (percentage/100) of patterns to use as clusters
        patterns : np.array
            Pattern data to cluster
        n_patterns : int
            Number of patterns
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Calculate number of clusters based on percentage
        percentage = ratio * 100
        n_clusters = max(2, min(int(n_patterns * ratio), n_patterns - 1))
        
        # Measure execution time
        start_time = time.time()
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patterns)
        
        execution_time = time.time() - start_time
        
        # Calculate quality metrics
        sil_score = silhouette_score(patterns, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
        ch_score = calinski_harabasz_score(patterns, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
        db_score = davies_bouldin_score(patterns, cluster_labels) if len(np.unique(cluster_labels)) > 1 else float('inf')
        
        # Calculate within-cluster variance (inertia)
        within_var = kmeans.inertia_
        
        # Calculate between-cluster distance
        cluster_centers = kmeans.cluster_centers_
        between_dist = 0
        
        if len(cluster_centers) > 1:
            # Calculate pairwise distances between cluster centers
            from scipy.spatial.distance import pdist
            center_distances = pdist(cluster_centers)
            between_dist = np.mean(center_distances)
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'within_cluster_variance': within_var,
            'between_cluster_distance': between_dist,
            'execution_time': execution_time
        }
    
    def _plot_ratio_analysis(self, results_df, symbol, timeframe):
        """
        Create visualizations for ratio analysis results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with ratio analysis results
        symbol : str
            Symbol name for labeling
        timeframe : str
            Timeframe name for labeling
        """
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Silhouette Score vs. Ratio
        plt.subplot(2, 2, 1)
        plt.plot(results_df['percentage'], results_df['silhouette_score'], 'o-', color='blue', linewidth=2)
        
        # Add best ratio annotation
        best_idx = results_df['silhouette_score'].idxmax()
        best_ratio = results_df.iloc[best_idx]['ratio']
        best_percentage = results_df.iloc[best_idx]['percentage']
        best_score = results_df.iloc[best_idx]['silhouette_score']
        
        plt.scatter([best_percentage], [best_score], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_percentage:.1f}%',
                    xy=(best_percentage, best_score),
                    xytext=(best_percentage, best_score * 0.9),
                    arrowprops=dict(arrowstyle="->", color='red'),
                    color='red', fontsize=12)
        
        plt.title('Silhouette Score vs. Pattern-to-Cluster Percentage')
        plt.xlabel('Percentage of Patterns as Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Davies-Bouldin Score vs. Ratio (lower is better)
        plt.subplot(2, 2, 2)
        plt.plot(results_df['percentage'], results_df['davies_bouldin_score'], 'o-', color='green', linewidth=2)
        
        # Add best ratio annotation
        best_db_idx = results_df['davies_bouldin_score'].idxmin()
        best_db_ratio = results_df.iloc[best_db_idx]['ratio']
        best_db_percentage = results_df.iloc[best_db_idx]['percentage']
        best_db_score = results_df.iloc[best_db_idx]['davies_bouldin_score']
        
        plt.scatter([best_db_percentage], [best_db_score], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_db_percentage:.1f}%',
                    xy=(best_db_percentage, best_db_score),
                    xytext=(best_db_percentage, best_db_score * 1.1),
                    arrowprops=dict(arrowstyle="->", color='red'),
                    color='red', fontsize=12)
        plt.title('Davies-Bouldin Score vs. Pattern-to-Cluster Percentage (lower is better)')
        plt.xlabel('Percentage of Patterns as Clusters')
        plt.ylabel('Davies-Bouldin Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cluster Quality Metrics
        plt.subplot(2, 2, 3)
        
        # Normalize the metrics to fit on the same scale
        norm_within = (results_df['within_cluster_variance'] - results_df['within_cluster_variance'].min()) / \
                     (results_df['within_cluster_variance'].max() - results_df['within_cluster_variance'].min() + 1e-10)
        norm_between = (results_df['between_cluster_distance'] - results_df['between_cluster_distance'].min()) / \
                      (results_df['between_cluster_distance'].max() - results_df['between_cluster_distance'].min() + 1e-10)
        
        # Calculate quality ratio (higher is better)
        quality_ratio = norm_between / (norm_within + 1e-10)
        
        plt.plot(results_df['percentage'], norm_within, 'o--', color='purple', label='Within-Cluster Variance (norm)')
        plt.plot(results_df['percentage'], norm_between, 's--', color='brown', label='Between-Cluster Distance (norm)')
        plt.plot(results_df['percentage'], quality_ratio, '^-', color='red', linewidth=2, label='Quality Ratio (Between/Within)')
        plt.title('Cluster Quality Metrics vs. Pattern-to-Cluster Percentage')
        plt.xlabel('Percentage of Patterns as Clusters')
        plt.ylabel('Normalized Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Optimal Ratio Analysis
        plt.subplot(2, 2, 4)
        
        # Create combined score (normalized, higher is better)
        results_df['combined_score'] = (
            (results_df['silhouette_score'] - results_df['silhouette_score'].min()) / 
            (results_df['silhouette_score'].max() - results_df['silhouette_score'].min() + 1e-10)
        ) - (
            (results_df['davies_bouldin_score'] - results_df['davies_bouldin_score'].min()) / 
            (results_df['davies_bouldin_score'].max() - results_df['davies_bouldin_score'].min() + 1e-10)
        ) + (
            quality_ratio
        )
        
        # Find best combined score
        best_combined_idx = results_df['combined_score'].idxmax()
        best_combined_ratio = results_df.iloc[best_combined_idx]['ratio']
        best_combined_percentage = results_df.iloc[best_combined_idx]['percentage']
        best_combined_score = results_df.iloc[best_combined_idx]['combined_score']
        
        plt.bar(results_df['percentage'], results_df['combined_score'], color='skyblue')
        
        plt.scatter([best_combined_percentage], [best_combined_score], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_combined_percentage:.1f}%',
                    xy=(best_combined_percentage, best_combined_score),
                    xytext=(best_combined_percentage, best_combined_score * 0.8),
                    arrowprops=dict(arrowstyle="->", color='red'),
                    color='red', fontsize=12)
        plt.title('Combined Cluster Quality Score vs. Pattern-to-Cluster Percentage')
        plt.xlabel('Percentage of Patterns as Clusters')
        plt.ylabel('Combined Score (higher is better)')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Pattern-to-Cluster Percentage Analysis for {symbol} ({timeframe})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
          # Save figure
        filename = f'pattern_cluster_percentage_{symbol}_{timeframe.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        # Create a second figure focused on financial interpretation
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot: Silhouette score vs. Number of clusters with ratio as color
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(
            results_df['n_clusters'], 
            results_df['silhouette_score'],
            c=results_df['ratio'],
            cmap='viridis',
            s=100,
            alpha=0.8
        )
        
        # Connect points in order of increasing ratio
        plt.plot(
            results_df.sort_values('ratio')['n_clusters'],
            results_df.sort_values('ratio')['silhouette_score'],
            'k--',
            alpha=0.3
        )
        
        # Highlight optimal point
        plt.scatter(
            [results_df.iloc[best_idx]['n_clusters']], 
            [results_df.iloc[best_idx]['silhouette_score']],
            marker='*',
            s=300,
            c='red',
            edgecolors='black',
            zorder=5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Pattern-to-Cluster Ratio')
        
        # Add labels for key points
        key_idxs = [0, len(results_df)//4, len(results_df)//2, 3*len(results_df)//4, -1]
        for i in key_idxs:
            if i < len(results_df):
                plt.annotate(
                    f"{results_df.iloc[i]['percentage']:.1f}%",
                    (results_df.iloc[i]['n_clusters'], results_df.iloc[i]['silhouette_score']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
        
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs. Pattern-to-Cluster Ratio
        plt.subplot(2, 1, 2)
        
        # Execution time and combined score on dual y-axis
        fig, ax1 = plt.subplots(figsize=(14, 5))
        
        # Execution time on primary y-axis
        ax1.set_xlabel('Percentage of Patterns as Clusters')
        ax1.set_ylabel('Execution Time (seconds)', color='tab:blue')
        ax1.plot(results_df['percentage'], results_df['execution_time'], 'o-', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)
        
        # Combined score on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Combined Score', color='tab:red')
        ax2.plot(results_df['percentage'], results_df['combined_score'], 's-', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Execution Time vs. Pattern-to-Cluster Ratio')
        
        # Add financial interpretation summary text
        plt.figtext(0.5, 0.01, 
                   f"Financial Pattern Summary: Optimal cluster ratio is {best_percentage:.1f}% of patterns\n"
                   f"({results_df.iloc[best_idx]['n_clusters']} clusters for {results_df.iloc[best_idx]['n_patterns']} patterns)",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2})
        
        plt.suptitle(f'Financial Pattern Analysis for {symbol} ({timeframe})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save figure
        filename = f'financial_pattern_ratio_{symbol}_{timeframe.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        # Print summary information
        print("\n" + "="*80)
        print(f"PATTERN-TO-CLUSTER RATIO ANALYSIS FOR {symbol} ({timeframe})")
        print("="*80)
        print(f"Total patterns: {results_df['n_patterns'].iloc[0]}")
        print(f"Best silhouette score: {best_score:.4f} at {best_percentage:.1f}% ({results_df.iloc[best_idx]['n_clusters']} clusters)")
        print(f"Best Davies-Bouldin score: {best_db_score:.4f} at {best_db_percentage:.1f}% ({results_df.iloc[best_db_idx]['n_clusters']} clusters)")
        print(f"Best combined score: {best_combined_score:.4f} at {best_combined_percentage:.1f}% ({results_df.iloc[best_combined_idx]['n_clusters']} clusters)")
        print("="*80)
    
    def analyze_all_timeframes(self, symbol='XAUUSD', max_data_points=50000):
        """
        Analyze pattern-to-cluster ratio across all available timeframes for a symbol
        
        Parameters:
        -----------
        symbol : str
            Symbol to analyze
        max_data_points : int
            Maximum number of data points to use per timeframe
            
        Returns:
        --------
        dict
            Dictionary with summary results for each timeframe
        """
        # Connect to database
        conn = sqlite3.connect('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Storage/data.db')
        cursor = conn.cursor()
        
        # Get all timeframes for the symbol
        cursor.execute('''
            SELECT t.timeframe_id, t.name, t.minutes 
            FROM timeframes t
            JOIN stock_data sd ON sd.timeframe_id = t.timeframe_id
            JOIN stocks s ON sd.stock_id = s.stock_id
            WHERE s.symbol = ?
            GROUP BY t.timeframe_id, t.name, t.minutes
            ORDER BY t.minutes
        ''', (symbol,))
        
        timeframes = cursor.fetchall()
        
        if not timeframes:
            print(f"No timeframe data found for {symbol}")
            return None
        
        summary_results = {}
        all_timeframe_results = []
        
        # Process each timeframe
        for timeframe_id, timeframe_name, minutes in timeframes:
            print(f"\nAnalyzing timeframe: {timeframe_name} ({minutes} minutes)")
            
            # Load data for this timeframe
            query = f'''
                SELECT timestamp, close_price 
                FROM stock_data 
                WHERE stock_id = (SELECT stock_id FROM stocks WHERE symbol = ?)
                AND timeframe_id = ?
                ORDER BY timestamp DESC
                LIMIT {max_data_points}
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe_id))
            
            if len(df) < 1000:
                print(f"Not enough data points for timeframe {timeframe_name}, skipping. Found {len(df)} points, needed at least 1000.")
                continue
                
            # Convert to numpy array for processing
            data = df['close_price'].to_numpy()
            
            print(f"Loaded {len(data)} data points for {timeframe_name}")
              # Analyze this timeframe
            results_df = self.analyze_cluster_ratios(
                data, symbol, timeframe_name, 
                percentage_range=(10, 90), step_size=10
            )
            
            if results_df is not None:
                # Extract best ratios
                best_sil_idx = results_df['silhouette_score'].idxmax()
                best_sil_ratio = results_df.iloc[best_sil_idx]['ratio']
                best_sil_percent = results_df.iloc[best_sil_idx]['percentage']
                
                best_db_idx = results_df['davies_bouldin_score'].idxmin()
                best_db_ratio = results_df.iloc[best_db_idx]['ratio']
                best_db_percent = results_df.iloc[best_db_idx]['percentage']
                
                # Combine scores
                results_df['combined_score'] = (
                    (results_df['silhouette_score'] - results_df['silhouette_score'].min()) / 
                    (results_df['silhouette_score'].max() - results_df['silhouette_score'].min() + 1e-10)
                ) - (
                    (results_df['davies_bouldin_score'] - results_df['davies_bouldin_score'].min()) / 
                    (results_df['davies_bouldin_score'].max() - results_df['davies_bouldin_score'].min() + 1e-10)
                )
                
                best_combined_idx = results_df['combined_score'].idxmax()
                best_combined_ratio = results_df.iloc[best_combined_idx]['ratio']
                best_combined_percent = results_df.iloc[best_combined_idx]['percentage']
                
                # Store summary results
                summary_results[timeframe_name] = {
                    'minutes': minutes,
                    'n_patterns': len(self.pattern_miner._unique_pip_patterns),
                    'best_silhouette_ratio': best_sil_ratio,
                    'best_silhouette_percent': best_sil_percent,
                    'best_db_ratio': best_db_ratio,
                    'best_db_percent': best_db_percent,
                    'best_combined_ratio': best_combined_ratio,
                    'best_combined_percent': best_combined_percent
                }
                
                # Add to all results for cross-timeframe analysis
                timeframe_row = {
                    'timeframe': timeframe_name,
                    'minutes': minutes,
                    'n_patterns': len(self.pattern_miner._unique_pip_patterns),
                    'best_silhouette_ratio': best_sil_ratio,
                    'best_silhouette_percent': best_sil_percent,
                    'best_db_ratio': best_db_ratio,
                    'best_db_percent': best_db_percent,
                    'best_combined_ratio': best_combined_ratio,
                    'best_combined_percent': best_combined_percent
                }
                all_timeframe_results.append(timeframe_row)
        
        conn.close()
        
        # Create cross-timeframe analysis if we have multiple timeframes
        if len(all_timeframe_results) > 1:
            self._create_cross_timeframe_analysis(all_timeframe_results, symbol)
        
        return summary_results
    
    def _create_cross_timeframe_analysis(self, all_results, symbol):
        """
        Create visualization comparing optimal ratios across timeframes
        
        Parameters:
        -----------
        all_results : list
            List of dictionaries with summary results for each timeframe
        symbol : str
            Symbol name for labeling
        """
        all_df = pd.DataFrame(all_results)
        
        # Sort by timeframe minutes
        all_df = all_df.sort_values('minutes')
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot optimal percentages by timeframe
        plt.subplot(2, 1, 1)
        
        width = 0.25
        x = np.arange(len(all_df))
        
        plt.bar(x - width, all_df['best_silhouette_percent'], width, label='Silhouette Score', color='blue')
        plt.bar(x, all_df['best_db_percent'], width, label='Davies-Bouldin', color='green')
        plt.bar(x + width, all_df['best_combined_percent'], width, label='Combined Score', color='red')
        
        plt.xlabel('Timeframe')        
        plt.ylabel('Optimal Percentage of Patterns as Clusters')
        plt.title('Optimal Pattern-to-Cluster Percentage by Timeframe')
        plt.xticks(x, all_df['timeframe'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot relationship between timeframe minutes and optimal ratio
        plt.subplot(2, 1, 2)
        
        # Plot points for each metric
        plt.scatter(all_df['minutes'], all_df['best_silhouette_percent'], label='Silhouette Score', 
                   color='blue', s=100, alpha=0.7)
        plt.scatter(all_df['minutes'], all_df['best_db_percent'], label='Davies-Bouldin', 
                   color='green', s=100, alpha=0.7)
        plt.scatter(all_df['minutes'], all_df['best_combined_percent'], label='Combined Score', 
                   color='red', s=100, alpha=0.7)
        
        # Add trendlines
        for metric, color, label in [
            ('best_silhouette_percent', 'blue', 'Silhouette Trendline'),
            ('best_db_percent', 'green', 'Davies-Bouldin Trendline'),
            ('best_combined_percent', 'red', 'Combined Trendline')
        ]:
            if len(all_df) >= 3:  # Need at least 3 points for a meaningful trendline
                z = np.polyfit(all_df['minutes'], all_df[metric], 1)
                p = np.poly1d(z)
                x_trend = sorted(all_df['minutes'])
                plt.plot(x_trend, p(x_trend), f'--', color=color, linewidth=2, 
                        label=f'{label}: y={z[0]:.6f}x+{z[1]:.2f}')
        plt.xlabel('Timeframe Duration (minutes)')
        plt.ylabel('Optimal Percentage of Patterns as Clusters')
        plt.title('Relationship Between Timeframe Duration and Optimal Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations for each timeframe
        for i, row in all_df.iterrows():
            plt.annotate(row['timeframe'], 
                        (row['minutes'], row['best_combined_percent']),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha='center',
                        fontsize=8)
        
        plt.suptitle(f'Cross-Timeframe Pattern-to-Cluster Percentage Analysis for {symbol}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
          # Save figure
        filename = f'cross_timeframe_pattern_percentage_{symbol}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        # Create a table with the summary statistics
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        # Calculate average optimal ratio across timeframes
        avg_sil_percent = all_df['best_silhouette_percent'].mean()
        avg_db_percent = all_df['best_db_percent'].mean()
        avg_combined_percent = all_df['best_combined_percent'].mean()
        
        # Add a row for averages
        summary_row = {
            'timeframe': 'AVERAGE',
            'minutes': all_df['minutes'].mean(),
            'n_patterns': all_df['n_patterns'].mean(),
            'best_silhouette_percent': avg_sil_percent,
            'best_db_percent': avg_db_percent,
            'best_combined_percent': avg_combined_percent
        }
       
   
        # Use pd.concat instead of the deprecated append method
        all_df_summary = pd.concat([all_df, pd.DataFrame([summary_row])], ignore_index=True)
        # Format for table display
        display_df = all_df_summary[['timeframe', 'minutes', 'n_patterns', 
                                   'best_silhouette_percent', 'best_db_percent', 'best_combined_percent']]
        display_df.columns = ['Timeframe', 'Minutes', 'Patterns', 
                             'Silhouette Optimal %', 'Davies-Bouldin Optimal %', 'Combined Optimal %']
          # Prepare data for display - convert to string with proper formatting
        formatted_values = []
        for row in display_df.values:
            formatted_row = []
            for val in row:
                if isinstance(val, (int, float)):
                    formatted_row.append(f"{val:.2f}")
                else:
                    formatted_row.append(str(val))
            formatted_values.append(formatted_row)
            
        # Create the table
        table = plt.table(
            cellText=formatted_values,
            colLabels=display_df.columns,
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2']*len(display_df.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight the average row
        for j in range(len(display_df.columns)):
            table._cells[(len(display_df), j)].set_facecolor('#e6f2ff')
            table._cells[(len(display_df), j)].set_text_props(weight='bold')
        
        plt.title(f'Optimal Pattern-to-Cluster Ratio Summary for {symbol}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        filename = f'pattern_ratio_summary_{symbol}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        # Generate numerical findings to guide the enhanced formula
        numerical_findings = {
            'average_optimal_ratio': {
                'silhouette': float(avg_sil_percent) / 100,  # Convert to ratio
                'davies_bouldin': float(avg_db_percent) / 100,
                'combined': float(avg_combined_percent) / 100
            },
            'ratio_by_timeframe': {
                tf: {
                    'minutes': int(row['minutes']),
                    'silhouette_ratio': float(row['best_silhouette_percent']) / 100,
                    'db_ratio': float(row['best_db_percent']) / 100,
                    'combined_ratio': float(row['best_combined_percent']) / 100
                } for tf, row in all_df.set_index('timeframe').iterrows()
            }
        }
        
        # Check if there's a correlation with timeframe minutes
        if len(all_df) >= 3:
            # Correlation between minutes and optimal ratio
            corr_sil = np.corrcoef(all_df['minutes'], all_df['best_silhouette_percent'])[0, 1]
            corr_db = np.corrcoef(all_df['minutes'], all_df['best_db_percent'])[0, 1]
            corr_combined = np.corrcoef(all_df['minutes'], all_df['best_combined_percent'])[0, 1]
            
            numerical_findings['correlations'] = {
                'minutes_to_silhouette_ratio': float(corr_sil),
                'minutes_to_db_ratio': float(corr_db),
                'minutes_to_combined_ratio': float(corr_combined)
            }
            
            # Linear regression formulae
            z_sil = np.polyfit(all_df['minutes'], all_df['best_silhouette_percent'] / 100, 1)
            z_db = np.polyfit(all_df['minutes'], all_df['best_db_percent'] / 100, 1)
            z_combined = np.polyfit(all_df['minutes'], all_df['best_combined_percent'] / 100, 1)
            
            numerical_findings['ratio_formula'] = {
                'silhouette': {'slope': float(z_sil[0]), 'intercept': float(z_sil[1])},
                'davies_bouldin': {'slope': float(z_db[0]), 'intercept': float(z_db[1])},
                'combined': {'slope': float(z_combined[0]), 'intercept': float(z_combined[1])}
            }
        
        # Save numerical findings to a JSON file for use in the enhanced formula
        with open(f'pattern_ratio_findings_{symbol}.json', 'w') as f:
            json.dump(numerical_findings, f, indent=4)
        
        # Print summary information
        print("\n" + "="*80)
        print(f"CROSS-TIMEFRAME PATTERN-TO-CLUSTER RATIO SUMMARY FOR {symbol}")
        print("="*80)
        print(f"Average optimal ratio (silhouette): {avg_sil_percent:.2f}% of patterns")
        print(f"Average optimal ratio (davies-bouldin): {avg_db_percent:.2f}% of patterns")
        print(f"Average optimal ratio (combined): {avg_combined_percent:.2f}% of patterns")
        
        if 'correlations' in numerical_findings:
            print(f"\nCorrelation with timeframe duration (minutes):")
            print(f"  Silhouette optimal ratio: {numerical_findings['correlations']['minutes_to_silhouette_ratio']:.4f}")
            print(f"  Davies-Bouldin optimal ratio: {numerical_findings['correlations']['minutes_to_db_ratio']:.4f}")
            print(f"  Combined optimal ratio: {numerical_findings['correlations']['minutes_to_combined_ratio']:.4f}")
        
        if 'ratio_formula' in numerical_findings:
            print(f"\nRatio formula (r = m*minutes + b):")
            print(f"  Silhouette: r = {numerical_findings['ratio_formula']['silhouette']['slope']:.6f} × minutes + {numerical_findings['ratio_formula']['silhouette']['intercept']:.4f}")
            print(f"  Combined: r = {numerical_findings['ratio_formula']['combined']['slope']:.6f} × minutes + {numerical_findings['ratio_formula']['combined']['intercept']:.4f}")
        
        print("="*80)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Pattern-to-Cluster Ratio Analyzer")
    parser.add_argument('--symbol', default='XAUUSD', help='Symbol to analyze')
    parser.add_argument('--timeframe', default='5 Minutes', help='Specific timeframe to analyze (if not provided, all timeframes will be analyzed)')
    parser.add_argument('--n_pips', type=int, default=7, help='Number of PIPs to use')
    parser.add_argument('--lookback', type=int, default=24, help='Lookback window size')
    parser.add_argument('--max_data_points', type=int, default=10000, help='Maximum number of data points to use per timeframe')    
    parser.add_argument('--min_percentage', type=int, default=10, help='Minimum percentage of patterns to use as clusters')
    parser.add_argument('--max_percentage', type=int, default=90, help='Maximum percentage of patterns to use as clusters')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for percentages to test')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel jobs to use')
    args = parser.parse_args()

    # Create analyzer
    analyzer = PatternClusterRatioAnalyzer(n_pips=args.n_pips, lookback=args.lookback)

    if args.timeframe:
        # Analyze specific timeframe
        print(f"Analyzing pattern-to-cluster ratio for {args.symbol} ({args.timeframe})")
        
        # Connect to database
        conn = sqlite3.connect('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Storage/data.db')
        
        # Get timeframe ID
        cursor = conn.cursor()
        cursor.execute('SELECT timeframe_id FROM timeframes WHERE name = ?', (args.timeframe,))
        result = cursor.fetchone()
        
        if not result:
            print(f"Timeframe {args.timeframe} not found")
            return
        
        timeframe_id = result[0]
        
        # Load data
        query = f'''
            SELECT timestamp, close_price 
            FROM stock_data 
            WHERE stock_id = (SELECT stock_id FROM stocks WHERE symbol = ?)
            AND timeframe_id = ?
            ORDER BY timestamp DESC
            LIMIT {args.max_data_points}
        '''
        
        df = pd.read_sql_query(query, conn, params=(args.symbol, timeframe_id))
        conn.close()
        
        if len(df) < 1000:
            print(f"Not enough data points, found only {len(df)}")
            return
        
        data = df['close_price'].to_numpy()
        print(f"Loaded {len(data)} data points")
          # Analyze
        analyzer.analyze_cluster_ratios(
            data, args.symbol, args.timeframe, 
            percentage_range=(args.min_percentage, args.max_percentage),
            step_size=args.step_size,
            n_jobs=args.n_jobs
        )
    else:
        # Analyze all timeframes
        analyzer.analyze_all_timeframes(
            symbol=args.symbol,
            max_data_points=args.max_data_points
        )

if __name__ == "__main__":
    main()
