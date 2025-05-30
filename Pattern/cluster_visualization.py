#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster Visualization Comparison Tool

This script provides visual comparisons of different clustering methods applied to the same dataset.
It helps in understanding how each method affects the resulting cluster patterns.

This version works with the simplified cluster benchmarking which focuses on
comparing the Silhouette Method and Enhanced Direct Formula approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
import time
from functools import partial
import argparse
import sys
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil  # For CPU core detection

# Add the parent directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the benchmarking and pattern miner classes
from Pattern.cluster_benchmarking import ClusteringBenchmark
from pip_pattern_miner import Pattern_Miner

class ClusteringVisualizer:
    """Class to visualize different clustering methods"""
    
    def __init__(self, data=None, pattern_miner=None):
        """
        Initialize the visualization tool
        
        Parameters:
        -----------
        data : np.array, optional
            Price data to use for clustering
        pattern_miner : Pattern_Miner, optional
            Pre-configured pattern miner instance
        """
        self.data = data
        if pattern_miner is None:
            self.pattern_miner = Pattern_Miner(n_pips=5, lookback=24, 
                                              hold_period=6, returns_hold_period=6, 
                                              distance_measure=2)
        else:
            self.pattern_miner = pattern_miner
        
        self.benchmark = ClusteringBenchmark(data=data, pattern_miner=self.pattern_miner)
        self.cluster_results = {}
    def prepare_data(self, symbol='SPY', period='2y'):
        """
        Load and prepare data
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to download
        period : str
            Period to download (e.g., '2y' for 2 years)
        """
        self.data = self.benchmark.load_data(symbol=symbol, period=period)
        # Setup pattern miner for visualization
        self.benchmark._setup_pattern_miner()
        
        # Get patterns for visualization
        self.patterns = np.array(self.pattern_miner._unique_pip_patterns)
        print(f"Found {len(self.patterns)} unique patterns")
        
    def apply_all_methods(self):
        """Apply all clustering methods and collect results"""
        print("Applying all clustering methods...")
        
        # Dictionary to store cluster assignments
        self.cluster_results = {}
        
        # Run each method and collect number of clusters and labels
        print("1. Running Silhouette Method...")
        n_silhouette = self.benchmark.run_silhouette_method()
        self.cluster_results['Silhouette Method'] = self._get_cluster_labels()
        
        print("2. Running Enhanced Direct Formula Method...")
        n_enhanced = self.benchmark.run_enhanced_direct_formula()
        self.cluster_results['Enhanced Direct Formula'] = self._get_cluster_labels()
       
        print("All methods applied successfully!")
        return self.cluster_results
    
    def _get_cluster_labels(self):
        """
        Extract cluster labels from current pattern miner state
        
        Returns:
        --------
        tuple
            (n_clusters, labels)
        """
        n_clusters = len(self.pattern_miner._pip_clusters)
        # Convert clusters to labels array
        labels = np.zeros(len(self.patterns), dtype=int)
        for i, cluster in enumerate(self.pattern_miner._pip_clusters):
            for idx in cluster:
                labels[idx] = i
        
        return n_clusters, labels
    
    def generate_2d_projection(self):
        """
        Generate 2D projection of patterns using PCA
        
        Returns:
        --------
        np.array
            2D projection of patterns
        """
        if len(self.patterns) < 2:
            print("Not enough patterns for PCA")
            return None
        
        # Apply PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        projected_patterns = pca.fit_transform(self.patterns)
        
        # Variance explained by components
        variance_explained = pca.explained_variance_ratio_
        print(f"Variance explained by 2 components: {sum(variance_explained)*100:.2f}%")
        
        return projected_patterns
    
    def visualize_cluster_comparison(self):
        """Visualize and compare clustering results from different methods"""
        if not self.cluster_results:
            print("No clustering results to visualize. Run apply_all_methods() first.")
            return
        
        # Generate 2D projection of patterns
        projected_patterns = self.generate_2d_projection()
        if projected_patterns is None:
            return
        
        # Create a colormap that can handle up to 20 clusters
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Set up the plot
        methods = list(self.cluster_results.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:  # Handle case with only one method
            axes = [axes]
        
        for i, method in enumerate(methods):
            n_clusters, labels = self.cluster_results[method]
            
            # Plot clusters
            scatter = axes[i].scatter(
                projected_patterns[:, 0], 
                projected_patterns[:, 1], 
                c=labels, 
                cmap=ListedColormap(colors[:n_clusters]),
                s=30, alpha=0.7
            )
            
            # Add titles and labels
            axes[i].set_title(f"{method}\n({n_clusters} clusters)")
            axes[i].set_xlabel('PCA Component 1')
            if i == 0:  # Only add y-label to the first plot
                axes[i].set_ylabel('PCA Component 2')
            
            # Add legend for clusters
            handles, _ = scatter.legend_elements()
            legend_labels = [f'Cluster {j}' for j in range(n_clusters)]
            if n_clusters <= 10:  # Only show legend if not too many clusters
                axes[i].legend(handles, legend_labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('cluster_method_comparison.png', dpi=300)
        plt.show()
    
    def visualize_pattern_examples(self, n_samples=3):
        """
        Visualize example patterns from each cluster for a selected method
        
        Parameters:
        -----------
        n_samples : int
            Number of example patterns to show per cluster
        """
        if not self.cluster_results:
            print("No clustering results to visualize. Run apply_all_methods() first.")
            return
        
        # Let user select a method
        methods = list(self.cluster_results.keys())
        print("Available methods:")
        for i, method in enumerate(methods):
            print(f"{i+1}. {method}")
        
        method_idx = 0  # Default to first method
        try:
            method_idx = int(input(f"Select method (1-{len(methods)}) [1]: ") or 1) - 1
        except:
            print("Invalid selection, using default")
        
        method = methods[method_idx]
        n_clusters, labels = self.cluster_results[method]
        
        # Set up the plot
        n_cols = min(5, n_clusters)  # At most 5 clusters per row
        n_rows = (n_clusters // n_cols) + (1 if n_clusters % n_cols > 0 else 0)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        if n_rows * n_cols == 1:  # Handle case with only one cluster
            axes = np.array([[axes]])
        elif n_rows == 1:  # Handle case with only one row
            axes = axes.reshape(1, -1)
        elif n_cols == 1:  # Handle case with only one column
            axes = axes.reshape(-1, 1)
        
        # For each cluster
        for i in range(n_clusters):
            # Find patterns in this cluster
            cluster_indices = np.where(labels == i)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Select sample patterns
            sample_indices = np.random.choice(
                cluster_indices, 
                size=min(n_samples, len(cluster_indices)), 
                replace=False
            )
            
            # Get row and column indices
            row_idx = i // n_cols
            col_idx = i % n_cols
            
            # Plot sample patterns
            for j, sample_idx in enumerate(sample_indices):
                pattern = self.patterns[sample_idx]
                
                # Normalize pattern for visualization
                pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
                
                # Plot each pattern with slight offset for visibility
                offset = j * 0.1
                axes[row_idx, col_idx].plot(
                    range(len(pattern)), 
                    pattern + offset, 
                    marker='o', 
                    label=f'Pattern {sample_idx}'
                )
            
            axes[row_idx, col_idx].set_title(f'Cluster {i} (n={len(cluster_indices)})')
            axes[row_idx, col_idx].set_yticks([])  # Hide y-axis as patterns are normalized
            
            # Only show legend if few samples
            if n_samples <= 3:
                axes[row_idx, col_idx].legend()
        
        # Hide unused subplots
        for i in range(n_clusters, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            fig.delaxes(axes[row_idx, col_idx])
        
        plt.tight_layout()
        plt.suptitle(f"Sample Patterns per Cluster using {method} Method", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'pattern_examples_{method}.png', dpi=300)
        plt.show()
        
    def compare_silhouette_scores_by_sample_size(self, symbol='SPY', max_period='5y', sample_sizes=None):
        """
        Compare silhouette scores for different methods across various dataset sizes
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to download
        max_period : str
            Maximum period to download (e.g., '5y' for 5 years)
        sample_sizes : list
            List of sample sizes to test (in days). If None, defaults will be used.
        """
        if sample_sizes is None:
            # Default sample sizes: from 1000 to 10000 in increments of 500
            sample_sizes = set(range(1000, 10001, 500))
            
        print(f"Comparing silhouette scores across different sample sizes for {symbol}")
        
        # Download full dataset
        data = pd.read_csv('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Raw/Stocks/Intraday/60M/BTCUSD60.csv')
        data['Date'] = data['Date'].astype('datetime64[s]')
        data = data.set_index('Date')
        # trim the data to only include the first 1000 data points
        data = data.head(10000)
        full_data = data['Close'].to_numpy()
       
        
        # Check if we have enough data
        if len(full_data) < max(sample_sizes):
            print(f"Warning: Requested max sample size {max(sample_sizes)} exceeds available data length {len(full_data)}")
            sample_sizes = [s for s in sample_sizes if s <= len(full_data)]
            
        print(f"Total data points: {len(full_data)}")
        
        # Dictionary to store results
        results = {
            'sample_size': [],
            'method': [],
            'n_clusters': [],
            'silhouette_score': [],
            'execution_time': []
        }
        
        # Test each sample size
        for sample_size in sample_sizes:
            print(f"\nTesting with sample size: {sample_size}")
            
            # Use the most recent data points
            data_sample = full_data[-sample_size:]
            
            # Set the data for the benchmark
            self.benchmark.data = data_sample
            
            # Test the silhouette method
            print("Testing Silhouette Method...")
            start_time = time.time()
            n_clusters_silhouette = self.benchmark.run_silhouette_method()
            end_time = time.time()
            execution_time_silhouette = end_time - start_time
            
            # Get the silhouette score from the results
            silhouette_score_silhouette = self.benchmark.results['silhouette_score'][-1]
            
            # Store results
            results['sample_size'].append(sample_size)
            results['method'].append('Silhouette Method')
            results['n_clusters'].append(n_clusters_silhouette)
            results['silhouette_score'].append(silhouette_score_silhouette)
            results['execution_time'].append(execution_time_silhouette)
            
            # Test the enhanced direct formula approach
            print("Testing Enhanced Direct Formula Approach...")
            start_time = time.time()
            n_clusters_formula = self.benchmark.run_enhanced_direct_formula()
            end_time = time.time()
            execution_time_formula = end_time - start_time
            
            # Get the silhouette score from the results
            silhouette_score_formula = self.benchmark.results['silhouette_score'][-1]
            
            # Store results
            results['sample_size'].append(sample_size)
            results['method'].append('Enhanced Direct Formula')
            results['n_clusters'].append(n_clusters_formula)
            results['silhouette_score'].append(silhouette_score_formula)
            results['execution_time'].append(execution_time_formula)
            
            print(f"Sample size: {sample_size}")
            print(f"  Silhouette Method: {n_clusters_silhouette} clusters, score: {silhouette_score_silhouette:.4f}, time: {execution_time_silhouette:.4f}s")
            print(f"  Enhanced Direct Formula: {n_clusters_formula} clusters, score: {silhouette_score_formula:.4f}, time: {execution_time_formula:.4f}s")
        
        # Convert to DataFrame
        
        results_df = pd.DataFrame(results)
        
        # Create visualizations
        self._plot_silhouette_comparison(results_df)
        
        return results_df
    def _plot_silhouette_comparison(self, results_df):
        """
        Plot the comparison of silhouette scores and execution times
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with the comparison results
        """
        plt.figure(figsize=(16, 10))
        
        # Set custom styles for methods
        method_styles = {
            'Silhouette Method': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},
            'Enhanced Direct Formula': {'color': 'green', 'marker': 's', 'linestyle': '--', 'linewidth': 2}
        }
        
        # Plot 1: Silhouette Scores Comparison
        plt.subplot(2, 2, 1)
        methods = results_df['method'].unique()
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            style = method_styles.get(method, {})
            plt.plot(method_data['sample_size'], method_data['silhouette_score'], 
                    label=method, **style)
        plt.title('Silhouette Score vs. Sample Size')
        plt.xlabel('Sample Size (days)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Execution Time Comparison
        plt.subplot(2, 2, 2)
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            style = method_styles.get(method, {})
            plt.plot(method_data['sample_size'], method_data['execution_time'], 
                    label=method, **style)
        plt.title('Execution Time vs. Sample Size')
        plt.xlabel('Sample Size (days)')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Number of Clusters Comparison
        plt.subplot(2, 2, 3)
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            style = method_styles.get(method, {})
            plt.plot(method_data['sample_size'], method_data['n_clusters'], 
                    label=method, **style)
        plt.title('Number of Clusters vs. Sample Size')
        plt.xlabel('Sample Size (days)')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Score-to-Time Ratio (Efficiency)
        plt.subplot(2, 2, 4)
        for method in methods:
            method_data = results_df[results_df['method'] == method]
            efficiency = method_data['silhouette_score'] / method_data['execution_time']
            style = method_styles.get(method, {})
            plt.plot(method_data['sample_size'], efficiency, 
                    label=method, **style)
        plt.title('Efficiency (Score/Time) vs. Sample Size')
        plt.xlabel('Sample Size (days)')
        plt.ylabel('Efficiency (Score/Time)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('silhouette_comparison_by_sample_size.png', dpi=300)
        plt.show()
    def analyze_batch_size_impact(self, symbol='BTCUSD', max_data_points=10000, method='enhanced', batch_sizes=None, n_jobs=None):
        """
        Analyze the relationship between batch size and silhouette score using parallel processing
        
        This method helps determine the optimal batch size for processing large datasets
        by measuring clustering quality across different batch sizes.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to download
        max_data_points : int
            Maximum number of data points to use
        method : str
            Method to use for determining number of clusters ('silhouette' or 'enhanced')
        batch_sizes : list
            List of batch sizes to test. If None, will use reasonable defaults based on data size
        n_jobs : int, optional
            Number of processes to use. If None, uses all available cores.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with batch size analysis results
        """
        # Determine number of CPU cores to use
        if n_jobs is None:
            n_jobs = os.cpu_count()  # Use all logical cores
        
        print(f"Analyzing batch size impact for {symbol} using {method} method with {n_jobs} CPU cores")
        
        # Load the data
        data = pd.read_csv(f'D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Raw/Stocks/Intraday/60M/{symbol}60.csv')
        data['Date'] = data['Date'].astype('datetime64[s]')
        data = data.set_index('Date')
        
        # Limit to specified max data points
        full_data = data['Close'].head(max_data_points).to_numpy()
        total_size = len(full_data)
        
        print(f"Total data points: {total_size}")
        
        # Generate reasonable batch sizes if not provided
        if batch_sizes is None:
            # Generate batch sizes from 1% to 25% of total data
            min_size = max(10000, int(total_size * 0.01))  # At least 1000 points
            max_size = max(20000, int(total_size * 0.25))  # At most 25% of data
            
            # Generate sets with steps of 100
            batch_sizes = list(range(min_size, max_size + 1, 2000))
            
            # Remove duplicates and sort
            batch_sizes = sorted(set(batch_sizes))
            
        print(f"Testing batch sizes: {batch_sizes}")
        
        # Dictionary to store results
        results = {
            'batch_size': [],
            'n_clusters': [],
            'silhouette_score': [],
            'execution_time': [],
            'batch_proportion': [],  # Batch size as % of total data
            'within_cluster_variance': [],  # Measure of cluster tightness
            'between_cluster_distance': []   # Measure of cluster separation
        }
        
        # Process each batch size in parallel
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Prepare batch data for parallel processing
            batch_data_list = []
            n_batches = total_size // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_size)
                batch_data = full_data[start_idx:end_idx]
                
                # Skip if too small
                if len(batch_data) < 10:
                    continue
                
                # Add to processing list
                batch_data_list.append((batch_data, batch_size))
            
            # Process all batches for this batch size in parallel
            if batch_data_list:
                batch_results = []
                
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # Submit all tasks
                    future_to_batch = {
                        executor.submit(_process_batch, batch_data, method): 
                        idx for idx, batch_data in enumerate(batch_data_list)
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_batch):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as exc:
                            print(f"Batch processing generated an exception: {exc}")
                
                # Calculate averages across all batches for this batch size
                if batch_results:
                    # Extract metrics
                    batch_clusters = [r['n_clusters'] for r in batch_results]
                    batch_silhouettes = [r['silhouette_score'] for r in batch_results]
                    batch_times = [r['execution_time'] for r in batch_results]
                    batch_within_vars = [r['within_cluster_variance'] for r in batch_results]
                    batch_between_dists = [r['between_cluster_distance'] for r in batch_results]
                    
                    # Calculate averages
                    avg_clusters = np.mean(batch_clusters)
                    avg_silhouette = np.mean(batch_silhouettes)
                    avg_execution_time = np.mean(batch_times)
                    avg_within_var = np.mean(batch_within_vars)
                    avg_between_dist = np.mean(batch_between_dists)
                    
                    # Store results
                    results['batch_size'].append(batch_size)
                    results['n_clusters'].append(avg_clusters)
                    results['silhouette_score'].append(avg_silhouette)
                    results['execution_time'].append(avg_execution_time)
                    results['batch_proportion'].append(batch_size / total_size)
                    results['within_cluster_variance'].append(avg_within_var)
                    results['between_cluster_distance'].append(avg_between_dist)
                    
                    print(f"Batch size {batch_size}: avg clusters = {avg_clusters:.2f}, "
                          f"avg silhouette = {avg_silhouette:.4f}, "
                          f"time = {avg_execution_time:.4f}s")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create visualizations
        self._plot_batch_size_analysis(results_df, method)
        
        return results_df
    def _plot_batch_size_analysis(self, results_df, method_name):
        """
        Create visualizations for batch size analysis
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with batch size analysis results
        method_name : str
            The clustering method used for the analysis
        """
        # Set up the figure
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Silhouette Score vs. Batch Size
        plt.subplot(2, 2, 1)
        plt.plot(results_df['batch_size'], results_df['silhouette_score'], 'o-', color='blue', linewidth=2)
        plt.title('Silhouette Score vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Add best batch size annotation
        best_idx = results_df['silhouette_score'].idxmax()
        best_batch = results_df.iloc[best_idx]['batch_size']
        best_score = results_df.iloc[best_idx]['silhouette_score']
        plt.annotate(f'Best: {int(best_batch)}',
                    xy=(best_batch, best_score),
                    xytext=(best_batch, best_score * 0.9),
                    arrowprops=dict(arrowstyle="->", color='red'),
                    color='red', fontsize=12)
        
        # Plot 2: Number of Clusters vs. Batch Size
        plt.subplot(2, 2, 2)
        plt.plot(results_df['batch_size'], results_df['n_clusters'], 'o-', color='green', linewidth=2)
        plt.title('Number of Clusters vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Execution Time vs. Batch Size
        plt.subplot(2, 2, 3)
        plt.plot(results_df['batch_size'], results_df['execution_time'], 'o-', color='orange', linewidth=2)
        plt.title('Execution Time vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Cluster Quality Metrics
        plt.subplot(2, 2, 4)
        
        # Normalize the metrics to fit on same scale
        norm_within = (results_df['within_cluster_variance'] - results_df['within_cluster_variance'].min()) / \
                     (results_df['within_cluster_variance'].max() - results_df['within_cluster_variance'].min() + 1e-10)
        norm_between = (results_df['between_cluster_distance'] - results_df['between_cluster_distance'].min()) / \
                      (results_df['between_cluster_distance'].max() - results_df['between_cluster_distance'].min() + 1e-10)
                      
        # Calculate quality ratio (higher is better)
        if np.all(norm_within == 0):
            quality_ratio = np.ones_like(norm_between)
        else:
            quality_ratio = norm_between / (norm_within + 1e-10)
        
        plt.plot(results_df['batch_size'], norm_within, 'o--', color='purple', label='Within-Cluster Variance (norm)')
        plt.plot(results_df['batch_size'], norm_between, 's--', color='brown', label='Between-Cluster Distance (norm)')
        #plt.plot(results_df['batch_size'], quality_ratio, '^-', color='red', linewidth=2, label='Quality Ratio (Between/Within)')
        plt.title('Cluster Quality Metrics vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Normalized Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add overall title
        plt.suptitle(f'Batch Size Analysis for {method_name.capitalize()} Method', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        plt.savefig(f'batch_size_analysis_{method_name}.png', dpi=300)
        plt.show()
        
        # Create an additional plot with log scale for batch size
        plt.figure(figsize=(12, 6))
        
        # Primary y-axis: Silhouette Score
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Batch Size (log scale)')
        ax1.set_ylabel('Silhouette Score', color='blue')
        ax1.plot(results_df['batch_size'], results_df['silhouette_score'], 'o-', color='blue', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Highlight the optimal batch size
        ax1.axvline(x=best_batch, linestyle='--', color='red', alpha=0.5)
        ax1.text(best_batch * 1.1, ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                f'Optimal Size: {int(best_batch)}', color='red')
        
        # Secondary y-axis: Number of Clusters
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Clusters', color='green')
        ax2.plot(results_df['batch_size'], results_df['n_clusters'], 's-', color='green', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Title
        plt.title(f'Optimal Batch Size Analysis ({method_name.capitalize()} Method)')
        plt.tight_layout()
        plt.savefig(f'optimal_batch_size_{method_name}.png', dpi=300)
        plt.show()
        
    def analyze_batch_size_by_timeframe(self, symbol='XAUUSD', max_data_points=100000, method='enhanced', n_jobs=None):
        """
        Analyze the relationship between batch size and silhouette score across different timeframes
        
        This method helps determine if the optimal batch size varies based on the timeframe
        of the financial data being analyzed.
        
        Parameters:
        -----------
        symbol : str
            Stock or forex symbol to analyze
        max_data_points : int
            Maximum number of data points to use for each timeframe
        method : str
            Method to use for determining number of clusters ('silhouette' or 'enhanced')
        n_jobs : int, optional
            Number of processes to use. If None, uses all available cores.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with batch size analysis results across timeframes
        """
        import sqlite3
        
        # Define batch sizes to test from 10000 to 20000 with 2000 step increments
        batch_sizes = list(range(10000, 22000, 2000))
        
        # Determine number of CPU cores to use
        if n_jobs is None:
            n_jobs = os.cpu_count()  # Use all logical cores
        
        print(f"Analyzing batch size impact across timeframes for {symbol} using {method} method with {n_jobs} CPU cores")
        
        # Connect to the database to get timeframe data
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
        
        # Dictionary to store results for all timeframes
        all_results = {
            'timeframe_id': [],
            'timeframe_name': [],
            'timeframe_minutes': [],
            'batch_size': [],
            'n_clusters': [],
            'silhouette_score': [],
            'execution_time': [],
            'within_cluster_variance': [],
            'between_cluster_distance': []
        }
        
        # Dictionary to store optimal batch sizes for each timeframe
        optimal_batch_sizes = {}
        
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
            
            
            if len(df) < 10000:
                print(f"Not enough data points for timeframe {timeframe_name}, skipping. Found {len(df)} points, needed at least 10000.")
                continue
                
            # Convert to numpy array for processing
            data = df['close_price'].to_numpy()
            
            print(f"Loaded {len(data)} data points for {timeframe_name}")
            
            # Dictionary to store results for this timeframe
            timeframe_results = {
                'batch_size': [],
                'n_clusters': [],
                'silhouette_score': [],
                'execution_time': [],
                'within_cluster_variance': [],
                'between_cluster_distance': []
            }
            
            # Process each batch size for this timeframe
            for batch_size in batch_sizes:
                print(f"Testing batch size: {batch_size} for {timeframe_name}")
                
                # Prepare batch data for parallel processing
                batch_data_list = []
                total_size = len(data)
                n_batches = total_size // batch_size
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_size)
                    batch_data = data[start_idx:end_idx]
                    
                    # Skip if too small
                    if len(batch_data) < 10:
                        continue
                    
                    # Add to processing list
                    batch_data_list.append((batch_data, batch_size))
                
                # Process all batches for this batch size in parallel
                if batch_data_list:
                    batch_results = []
                    
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        # Submit all tasks
                        future_to_batch = {
                            executor.submit(_process_batch, batch_data, method): 
                            idx for idx, batch_data in enumerate(batch_data_list)
                        }
                        
                        # Process results as they complete
                        for future in as_completed(future_to_batch):
                            try:
                                result = future.result()
                                batch_results.append(result)
                            except Exception as exc:
                                print(f"Batch processing generated an exception: {exc}")
                    
                    # Calculate averages across all batches for this batch size
                    if batch_results:
                        # Extract metrics
                        batch_clusters = [r['n_clusters'] for r in batch_results]
                        batch_silhouettes = [r['silhouette_score'] for r in batch_results]
                        batch_times = [r['execution_time'] for r in batch_results]
                        batch_within_vars = [r['within_cluster_variance'] for r in batch_results]
                        batch_between_dists = [r['between_cluster_distance'] for r in batch_results]
                        
                        # Calculate averages
                        avg_clusters = np.mean(batch_clusters)
                        avg_silhouette = np.mean(batch_silhouettes)
                        avg_execution_time = np.mean(batch_times)
                        avg_within_var = np.mean(batch_within_vars)
                        avg_between_dist = np.mean(batch_between_dists)
                        
                        # Store results for this timeframe
                        timeframe_results['batch_size'].append(batch_size)
                        timeframe_results['n_clusters'].append(avg_clusters)
                        timeframe_results['silhouette_score'].append(avg_silhouette)
                        timeframe_results['execution_time'].append(avg_execution_time)
                        timeframe_results['within_cluster_variance'].append(avg_within_var)
                        timeframe_results['between_cluster_distance'].append(avg_between_dist)
                        
                        # Also store in the overall results
                        all_results['timeframe_id'].append(timeframe_id)
                        all_results['timeframe_name'].append(timeframe_name)
                        all_results['timeframe_minutes'].append(minutes)
                        all_results['batch_size'].append(batch_size)
                        all_results['n_clusters'].append(avg_clusters)
                        all_results['silhouette_score'].append(avg_silhouette)
                        all_results['execution_time'].append(avg_execution_time)
                        all_results['within_cluster_variance'].append(avg_within_var)
                        all_results['between_cluster_distance'].append(avg_between_dist)
                        
                        print(f"Batch size {batch_size} for {timeframe_name}: avg clusters = {avg_clusters:.2f}, "
                              f"avg silhouette = {avg_silhouette:.4f}, time = {avg_execution_time:.4f}s")
            
            # Convert to DataFrame for this timeframe
            if timeframe_results['batch_size']:
                timeframe_df = pd.DataFrame(timeframe_results)
                
                # Find optimal batch size for this timeframe
                best_idx = timeframe_df['silhouette_score'].idxmax()
                best_batch = int(timeframe_df.iloc[best_idx]['batch_size'])
                best_score = timeframe_df.iloc[best_idx]['silhouette_score']
                
                optimal_batch_sizes[timeframe_name] = {
                    'optimal_batch_size': best_batch,
                    'silhouette_score': best_score,
                    'minutes': minutes
                }
                
                print(f"Optimal batch size for {timeframe_name}: {best_batch} (score: {best_score:.4f})")
                
                # Create visualization for this timeframe
                self._plot_batch_size_analysis(timeframe_df, f"{method}_{timeframe_name}")
        
        # Convert all results to DataFrame
        all_results_df = pd.DataFrame(all_results)
        conn.close()
        # Create comparison visualization across timeframes
        self._plot_timeframe_batch_size_comparison(all_results_df, optimal_batch_sizes, method)
        
        return all_results_df, optimal_batch_sizes
    def _plot_timeframe_batch_size_comparison(self, results_df, optimal_batch_sizes, method_name):
        """
        Create visualizations comparing batch size impact across different timeframes
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with batch size analysis results for all timeframes
        optimal_batch_sizes : dict
            Dictionary containing optimal batch sizes for each timeframe
        method_name : str
            The clustering method used for the analysis
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get unique timeframes
        timeframes = results_df['timeframe_name'].unique()
        
        # Set up color map for timeframes
        colors = plt.cm.tab10(np.linspace(0, 1, len(timeframes)))
        timeframe_colors = {tf: colors[i] for i, tf in enumerate(timeframes)}
        
        # Plot 1: Silhouette Score vs. Batch Size across Timeframes
        plt.figure(figsize=(16, 10))
        
        # Silhouette Score vs. Batch Size for all timeframes
        plt.subplot(2, 2, 1)
        for timeframe in timeframes:
            tf_data = results_df[results_df['timeframe_name'] == timeframe]
            plt.plot(tf_data['batch_size'], tf_data['silhouette_score'], 'o-', 
                    label=timeframe, color=timeframe_colors[timeframe])
            
            # Mark optimal batch size
            if timeframe in optimal_batch_sizes:
                optimal = optimal_batch_sizes[timeframe]
                plt.scatter([optimal['optimal_batch_size']], [optimal['silhouette_score']], 
                          marker='*', s=200, color=timeframe_colors[timeframe], 
                          edgecolors='black', linewidths=1.5)
        
        plt.title('Silhouette Score vs. Batch Size across Timeframes')
        plt.xlabel('Batch Size')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Timeframe')
        
        # Plot 2: Optimal Batch Size vs. Timeframe Minutes
        plt.subplot(2, 2, 2)
        tf_minutes = []
        optimal_sizes = []
        tf_names = []
        scores = []
        
        for tf, data in optimal_batch_sizes.items():
            tf_minutes.append(data['minutes'])
            optimal_sizes.append(data['optimal_batch_size'])
            tf_names.append(tf)
            scores.append(data['silhouette_score'])
        
        # Create scatter plot
        scatter = plt.scatter(tf_minutes, optimal_sizes, c=scores, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Silhouette Score')
        
        # Add timeframe labels
        for i, txt in enumerate(tf_names):
            plt.annotate(txt, (tf_minutes[i], optimal_sizes[i]), 
                       textcoords="offset points", xytext=(0, 10), 
                       ha='center', fontsize=9)
        
        # Add trendline if at least 2 timeframes
        if len(tf_minutes) > 1:
            z = np.polyfit(tf_minutes, optimal_sizes, 1)
            p = np.poly1d(z)
            plt.plot(sorted(tf_minutes), p(sorted(tf_minutes)), 'r--', 
                   linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.1f}')
            plt.legend()
        
        plt.title('Optimal Batch Size vs. Timeframe Duration')
        plt.xlabel('Timeframe Duration (minutes)')
        plt.ylabel('Optimal Batch Size')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Number of Clusters vs. Batch Size across Timeframes
        plt.subplot(2, 2, 3)
        for timeframe in timeframes:
            tf_data = results_df[results_df['timeframe_name'] == timeframe]
            plt.plot(tf_data['batch_size'], tf_data['n_clusters'], 'o-', 
                    label=timeframe, color=timeframe_colors[timeframe])
        
        plt.title('Number of Clusters vs. Batch Size across Timeframes')
        plt.xlabel('Batch Size')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Timeframe')
        
        # Plot 4: Execution Time vs. Batch Size across Timeframes
        plt.subplot(2, 2, 4)
        for timeframe in timeframes:
            tf_data = results_df[results_df['timeframe_name'] == timeframe]
            plt.plot(tf_data['batch_size'], tf_data['execution_time'], 'o-', 
                    label=timeframe, color=timeframe_colors[timeframe])
        
        plt.title('Execution Time vs. Batch Size across Timeframes')
        plt.xlabel('Batch Size')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Timeframe')
        
        plt.suptitle(f'Batch Size Analysis Across Timeframes ({method_name.capitalize()} Method)', 
                   fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'timeframe_batch_size_analysis_{method_name}.png', dpi=300)
        plt.show()
        
        # Create a second figure showing the relationship between timeframe and optimal parameters
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Optimal batch size ratio (as % of total data points) vs timeframe
        plt.subplot(2, 1, 1)
        tf_names_sorted = []
        optimal_ratios = []
        
        # Calculate ratios
        for tf, data in sorted(optimal_batch_sizes.items(), key=lambda x: x[1]['minutes']):
            # Assuming 100k max data points
            ratio = (data['optimal_batch_size'] / 100000) * 100
            optimal_ratios.append(ratio)
            tf_names_sorted.append(tf)
        
        # Bar chart
        bars = plt.bar(tf_names_sorted, optimal_ratios, color=[timeframe_colors[tf] for tf in tf_names_sorted])
        
        # Add batch size labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(optimal_batch_sizes[tf_names_sorted[i]]["optimal_batch_size"]):,}',
                    ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title('Optimal Batch Size as Percentage of Total Data by Timeframe')
        plt.xlabel('Timeframe')
        plt.ylabel('Optimal Batch Size (% of total data)')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Silhouette score at optimal batch size vs timeframe
        plt.subplot(2, 1, 2)
        optimal_scores = [optimal_batch_sizes[tf]['silhouette_score'] for tf in tf_names_sorted]
        
        bars = plt.bar(tf_names_sorted, optimal_scores, color=[timeframe_colors[tf] for tf in tf_names_sorted])
        
        # Add score labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{optimal_scores[i]:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title('Silhouette Score at Optimal Batch Size by Timeframe')
        plt.xlabel('Timeframe')
        plt.ylabel('Silhouette Score')
        plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Optimal Parameters by Timeframe ({method_name.capitalize()} Method)', 
                   fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'timeframe_optimal_parameters_{method_name}.png', dpi=300)
        plt.show()
        
    def _process_batch_internal(self, batch_data, method):
        """
        Process a single batch of data (internal instance method)
        
        Parameters:
        -----------
        batch_data : tuple
            Tuple containing (batch_data_array, batch_size)
        method : str
            Method to use ('silhouette' or 'enhanced')
            
        Returns:
        --------
        dict
            Results of the batch processing
        """
        data, batch_size = batch_data
        
        # Create a fresh benchmark and pattern miner to avoid any shared state issues
        pattern_miner = Pattern_Miner(n_pips=5, lookback=24, 
                                     hold_period=6, returns_hold_period=6, 
                                     distance_measure=2)
        benchmark = ClusteringBenchmark(data=data, pattern_miner=pattern_miner)
        
        # Measure execution time
        start_time = time.time()
        
        # Run the specified method
        if method.lower() == 'silhouette':
            n_clusters = benchmark.run_silhouette_method()
        else:
            n_clusters = benchmark.run_enhanced_direct_formula()
            
        execution_time = time.time() - start_time
        
        # Get clustering quality metrics
        silhouette = benchmark.results['silhouette_score'][-1]
        
        # Calculate within-cluster variance and between-cluster distance
        patterns = np.array(pattern_miner._unique_pip_patterns)
        cluster_labels = []
        
        # Convert clusters to labels array
        for idx, cluster in enumerate(pattern_miner._pip_clusters):
            for pattern_idx in cluster:
                cluster_labels.append(idx)
        
        # Convert to numpy array
        if cluster_labels:
            cluster_labels = np.array(cluster_labels)
            
            # Calculate within-cluster variance (average)
            within_var = 0
            cluster_centers = []
            
            # Calculate cluster centers and within-cluster variance
            for cluster_idx in range(n_clusters):
                cluster_patterns = patterns[cluster_labels == cluster_idx]
                if len(cluster_patterns) > 0:
                    cluster_center = np.mean(cluster_patterns, axis=0)
                    cluster_centers.append(cluster_center)
                    # Calculate variance
                    within_var += np.mean(np.var(cluster_patterns, axis=0))
            
            # Average within-cluster variance
            if n_clusters > 0:
                within_var /= n_clusters
            else:
                within_var = 0
            
            # Calculate between-cluster distance (average pairwise distance between centers)
            between_dist = 0
            if len(cluster_centers) > 1:
                cluster_centers = np.array(cluster_centers)
                # Calculate pairwise distances between cluster centers
                from scipy.spatial.distance import pdist
                center_distances = pdist(cluster_centers)
                between_dist = np.mean(center_distances)
        else:
            within_var = 0
            between_dist = 0
        
        return {
            'batch_size': batch_size,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'execution_time': execution_time,
            'within_cluster_variance': within_var,
            'between_cluster_distance': between_dist
        }
def _process_batch(batch_data, method):
    """
    Process a single batch of data (for parallel execution)
    
    Parameters:
    -----------
    batch_data : tuple
        Tuple containing (batch_data_array, batch_size)
    method : str
        Method to use ('silhouette' or 'enhanced')
        
    Returns:
    --------
    dict
        Results of the batch processing
    """
    data, batch_size = batch_data
    
    # Create a fresh benchmark and pattern miner to avoid any shared state issues
    pattern_miner = Pattern_Miner(n_pips=5, lookback=24, 
                                 hold_period=6, returns_hold_period=6, 
                                 distance_measure=2)
    benchmark = ClusteringBenchmark(data=data, pattern_miner=pattern_miner)
    
    # Measure execution time
    start_time = time.time()
    
    # Run the specified method
    if method.lower() == 'silhouette':
        n_clusters = benchmark.run_silhouette_method()
    else:
        n_clusters = benchmark.run_enhanced_direct_formula()
        
    execution_time = time.time() - start_time
    
    # Get clustering quality metrics
    silhouette = benchmark.results['silhouette_score'][-1]
    
    # Calculate within-cluster variance and between-cluster distance
    patterns = np.array(pattern_miner._unique_pip_patterns)
    cluster_labels = []
    
    # Convert clusters to labels array
    for idx, cluster in enumerate(pattern_miner._pip_clusters):
        for pattern_idx in cluster:
            cluster_labels.append(idx)
    
    # Convert to numpy array
    if cluster_labels:
        cluster_labels = np.array(cluster_labels)
        
        # Calculate within-cluster variance (average)
        within_var = 0
        cluster_centers = []
        
        # Calculate cluster centers and within-cluster variance
        for cluster_idx in range(n_clusters):
            cluster_patterns = patterns[cluster_labels == cluster_idx]
            if len(cluster_patterns) > 0:
                cluster_center = np.mean(cluster_patterns, axis=0)
                cluster_centers.append(cluster_center)
                # Calculate variance
                within_var += np.mean(np.var(cluster_patterns, axis=0))
        
        # Average within-cluster variance
        if n_clusters > 0:
            within_var /= n_clusters
        else:
            within_var = 0
        
        # Calculate between-cluster distance (average pairwise distance between centers)
        between_dist = 0
        if len(cluster_centers) > 1:
            cluster_centers = np.array(cluster_centers)
            # Calculate pairwise distances between cluster centers
            from scipy.spatial.distance import pdist
            center_distances = pdist(cluster_centers)
            between_dist = np.mean(center_distances)
    else:
        within_var = 0
        between_dist = 0
    
    return {
        'batch_size': batch_size,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'execution_time': execution_time,
        'within_cluster_variance': within_var,
        'between_cluster_distance': between_dist
    }

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Cluster visualization comparison tool")
    parser.add_argument('--symbol', default='XAUUSD', help='Stock or crypto symbol to use')
    parser.add_argument('--period', default='2y', help='Period to download (e.g., 2y for 2 years)')
    parser.add_argument('--mode', default='timeframe_analysis', 
                        choices=['visualize', 'compare', 'batch_analysis', 'timeframe_analysis'],
                        help='Mode: visualize for cluster visualization, compare for silhouette score comparison, ' +
                             'batch_analysis for batch size analysis, timeframe_analysis for analysis across timeframes')
    parser.add_argument('--method', default='enhanced', choices=['silhouette', 'enhanced'],
                        help='Method to use for determining clusters (silhouette or enhanced)')
    parser.add_argument('--max_data_points', type=int, default=100000,
                        help='Maximum number of data points to use for analysis')
    parser.add_argument('--max_period', default='5y', help='Maximum period for comparison (used only in compare mode)')
    parser.add_argument('--n_jobs', type=int, default=None, 
                        help='Number of CPU cores to use (default: all available physical cores)')
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ClusteringVisualizer()
    
    if args.mode == 'visualize':
        print(f"Starting visualization with {args.symbol} for period {args.period}")
        
        # Prepare data
        visualizer.prepare_data(symbol=args.symbol, period=args.period)
        
        # Apply all methods
        visualizer.apply_all_methods()
        
        # Generate comparison visualization
        visualizer.visualize_cluster_comparison()
        
        # Show example patterns from clusters
        visualizer.visualize_pattern_examples()
    
    elif args.mode == 'batch_analysis':
        print(f"Starting batch size analysis for {args.symbol} using {args.method} method")
        
        # Run batch size analysis
        results_df = visualizer.analyze_batch_size_impact(
            symbol=args.symbol,
            max_data_points=args.max_data_points, 
            method=args.method,
            n_jobs=args.n_jobs
        )
        
        # Print summary of optimal batch size
        best_idx = results_df['silhouette_score'].idxmax()
        best_batch = int(results_df.iloc[best_idx]['batch_size'])
        best_score = results_df.iloc[best_idx]['silhouette_score']
        best_clusters = results_df.iloc[best_idx]['n_clusters']
        
        print("\nOptimal Batch Size Analysis Results:")
        print(f"  Optimal batch size: {best_batch} data points")
        print(f"  Silhouette score: {best_score:.4f}")
        print(f"  Number of clusters: {best_clusters:.1f}")
        print(f"  Proportion of total data: {(best_batch / args.max_data_points) * 100:.2f}%")
        
        # Print batch size recommendations for larger datasets
        print("\nRecommended Batch Sizes for Larger Datasets:")
        for size in [50000, 100000, 500000, 1000000]:
            if size > args.max_data_points:
                recommended = int((best_batch / args.max_data_points) * size)
                print(f"  For {size} data points: {recommended}")
    
    elif args.mode == 'timeframe_analysis':
        print(f"Starting batch size analysis across timeframes for {args.symbol} using {args.method} method")
        
        # Run batch size analysis across timeframes
        results_df, optimal_batch_sizes = visualizer.analyze_batch_size_by_timeframe(
            symbol=args.symbol,
            max_data_points=args.max_data_points, 
            method=args.method,
            n_jobs=args.n_jobs
        )
        
        # Print summary of optimal batch sizes across timeframes
        print("\nOptimal Batch Size Analysis Results by Timeframe:")
        
        for timeframe, data in sorted(optimal_batch_sizes.items(), key=lambda x: x[1]['minutes']):
            optimal_batch = data['optimal_batch_size']
            score = data['silhouette_score']
            minutes = data['minutes']
            
            print(f"\n  Timeframe: {timeframe} ({minutes} minutes)")
            print(f"    Optimal batch size: {optimal_batch} data points")
            print(f"    Silhouette score: {score:.4f}")
            print(f"    Proportion of total data: {(optimal_batch / args.max_data_points) * 100:.2f}%")
        
        # Check if there's a correlation between timeframe duration and optimal batch size
        timeframe_minutes = [data['minutes'] for data in optimal_batch_sizes.values()]
        optimal_sizes = [data['optimal_batch_size'] for data in optimal_batch_sizes.values()]
        
        if len(timeframe_minutes) > 1:
            import numpy as np
            from scipy.stats import pearsonr
            
            correlation, p_value = pearsonr(timeframe_minutes, optimal_sizes)
            
            print("\nCorrelation Analysis:")
            print(f"  Correlation between timeframe duration and optimal batch size: {correlation:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            # Calculate trend line
            z = np.polyfit(timeframe_minutes, optimal_sizes, 1)
            print(f"  Trend line equation: batch_size = {z[0]:.2f} * timeframe_minutes + {z[1]:.2f}")
    
    else:  # compare mode
        print(f"Starting silhouette score comparison for {args.symbol}")
        
        # Define sample sizes to test (in trading days)
        sample_sizes = set(range(1000, 3000, 500))
        
        # Run the comparison
        results_df = visualizer.compare_silhouette_scores_by_sample_size(
            symbol=args.symbol,
            max_period=args.max_period,
            sample_sizes=sample_sizes
        )
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nAverage Silhouette Scores:")
        avg_scores = results_df.groupby('method')['silhouette_score'].mean()
        for method, score in avg_scores.items():
            print(f"  {method}: {score:.4f}")
        
        print("\nAverage Execution Times:")
        avg_times = results_df.groupby('method')['execution_time'].mean()
        for method, time_val in avg_times.items():
            print(f"  {method}: {time_val:.4f} seconds")
        
        # Print improvement percentage of Enhanced Direct Formula over Silhouette Method
        edf_score = avg_scores.get('Enhanced Direct Formula', 0)
        silhouette_score = avg_scores.get('Silhouette Method', 0)
        if silhouette_score > 0:
            improvement = ((edf_score - silhouette_score) / silhouette_score) * 100
            print(f"\nEnhanced Direct Formula improves silhouette score by: {improvement:.2f}% on average")


if __name__ == "__main__":
    main()
