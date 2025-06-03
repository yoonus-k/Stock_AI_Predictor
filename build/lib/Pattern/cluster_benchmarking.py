#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Cluster Number Selection Benchmark

This script focuses on benchmarking two key methods for determining the optimal number of clusters 
in the Pattern_Miner class:
1. Silhouette Method (current implementation)
2. Enhanced Direct Formula Approach (improved approach for better silhouette scores)

The script evaluates each method on:
- Execution time
- Memory usage
- Silhouette score (clustering quality)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc
import os
from math import sqrt
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

# Add parent directory to path to import Pattern_Miner
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import the Pattern_Miner class
from Pattern.pip_pattern_miner import Pattern_Miner

# Import the enhanced direct formula
from Pattern.enhanced_direct_formula import EnhancedDirectFormula, count_optimal_clusters


class ClusteringBenchmark:
    """Class to benchmark different clustering methods on Pattern_Miner"""
    
    def __init__(self, data=None, pattern_miner=None):
        """
        Initialize the benchmark suite
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Price data to use for benchmarking
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
            
        self.results = {
            'method': [],
            'n_clusters': [],
            'execution_time': [],
            'memory_usage': [],
            'silhouette_score': []
        }
        
    def load_data(self, symbol='BTCUSD', period='2y'):
        """
        Load financial data for benchmarking
        
        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        period : str
            Period of data to load
            
        Returns:
        --------
        numpy.ndarray
            Price data as numpy array
        """
        try:
            data = pd.read_csv('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Raw/Stocks/Intraday/60M/BTCUSD60.csv')
            data['Date'] = data['Date'].astype('datetime64[s]')
            data = data.set_index('Date')
            # trim the data to only include the first 1000 data points
            data = data.head(3000)
            arr = data['Close'].to_numpy()
            self.data = arr
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def _setup_pattern_miner(self):
        """
        Prepare the pattern miner for clustering
        
        Returns:
        --------
        int
            Number of unique patterns found
        """
        # We need to extract patterns first before clustering
        self.pattern_miner._data = self.data
        self.pattern_miner._find_unique_patterns()
        
        # Calculate returns needed for cluster evaluation
        self.pattern_miner._returns_next_candle = pd.Series(self.data).diff().shift(-1)
        self.pattern_miner._returns_fixed_hold = self.pattern_miner.calculate_returns_fixed_hold(
            self.data, self.pattern_miner._unique_pip_indices, self.pattern_miner._returns_hold_period)
        self.pattern_miner._returns_mfe, self.pattern_miner._returns_mae = self.pattern_miner.calculate_mfe_mae(
            self.data, self.pattern_miner._unique_pip_indices, self.pattern_miner._returns_hold_period)
        
        return len(self.pattern_miner._unique_pip_patterns)
        
    def _cluster_and_evaluate(self, n_clusters):
        """
        Apply K-means clustering with specific number of clusters and evaluate
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to use
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Perform clustering with the specified number of clusters
        self.pattern_miner._kmeans_cluster_patterns(n_clusters)
        self.pattern_miner._categorize_clusters_by_mean_return()
        
        # Get data in numpy array format for scikit-learn metrics
        patterns = np.array(self.pattern_miner._unique_pip_patterns)
        
        # Convert clusters to labels array for sklearn metrics
        labels = np.zeros(len(patterns), dtype=int)
        for i, cluster in enumerate(self.pattern_miner._pip_clusters):
            for idx in cluster:
                labels[idx] = i

        # Calculate silhouette score
        if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters for evaluation
            silhouette = metrics.silhouette_score(patterns, labels)
        else:
            # Default poor value if only one cluster
            silhouette = -1
            
        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette
        }
        
    def run_silhouette_method(self):
        """
        Run the current silhouette method and measure performance
        
        Returns:
        --------
        int
            Optimal number of clusters according to silhouette method
        """
        print("Running Silhouette Method...")
        
        # Setup pattern miner
        pattern_count = self._setup_pattern_miner()
        
        # Calculate sqrt(n) as a statistical rule of thumb for initial clustering
        sqrt_n = int(np.sqrt(pattern_count))
        
        # Adaptive min/max clusters as in original implementation
        min_clusters = max(3, int(sqrt_n/2))
        max_clusters = min(int(sqrt_n)*2, pattern_count-1)
        
        # Ensure min < max and both are within valid range
        min_clusters = min(min_clusters, pattern_count - 1)
        max_clusters = min(max_clusters, pattern_count - 1)
        max_clusters = max(max_clusters, min_clusters + 2)  # Ensure reasonable range
        
        # Start memory monitoring
        tracemalloc.start()
        start_time = time.time()
        
        # Run silhouette method
        from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
        search_instance = silhouette_ksearch(self.pattern_miner._unique_pip_patterns, 
                                            min_clusters, max_clusters, 
                                            algorithm=silhouette_ksearch_type.KMEANS).process()
        
        # Get optimal number of clusters
        n_clusters = search_instance.get_amount()
        
        # Stop timing and memory monitoring
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Evaluate clustering with the chosen number of clusters
        metrics = self._cluster_and_evaluate(n_clusters)
        
        # Record results
        self.results['method'].append('Silhouette Method')
        self.results['n_clusters'].append(n_clusters)
        self.results['execution_time'].append(end_time - start_time)
        self.results['memory_usage'].append(peak / 10**6)  # Convert to MB
        self.results['silhouette_score'].append(metrics['silhouette_score'])
        
        print(f"Silhouette Method selected {n_clusters} clusters in {end_time - start_time:.4f} seconds")
        
        return n_clusters
    
    def run_enhanced_direct_formula(self):
        """
        Run the enhanced direct formula approach and measure performance
        
        Returns:
        --------
        int
            Optimal number of clusters according to enhanced direct formula
        """
        print("Running Enhanced Direct Formula Approach...")
        
        # Setup pattern miner
        pattern_count = self._setup_pattern_miner()
        
        # Start memory monitoring
        tracemalloc.start()
        start_time = time.time()
        
        # Use the enhanced direct formula
        patterns = np.array(self.pattern_miner._unique_pip_patterns)
        n_clusters, info = count_optimal_clusters(patterns, method='enhanced')
        
        # Stop timing and memory monitoring
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Evaluate clustering with the calculated number of clusters
        metrics = self._cluster_and_evaluate(n_clusters)
        
        # Record results
        self.results['method'].append('Enhanced Direct Formula')
        self.results['n_clusters'].append(n_clusters)
        self.results['execution_time'].append(end_time - start_time)
        self.results['memory_usage'].append(peak / 10**6)  # Convert to MB
        self.results['silhouette_score'].append(metrics['silhouette_score'])
        
        print(f"Enhanced Direct Formula selected {n_clusters} clusters in {end_time - start_time:.4f} seconds")
        #print(f"Formula components: {info['formula_components']}")
        
        return n_clusters
    
    def run_all_benchmarks(self):
        """
        Run all benchmark methods
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with benchmark results
        """
        # Run silhouette method
        silhouette_clusters = self.run_silhouette_method()
        
        # Run enhanced direct formula
        enhanced_clusters = self.run_enhanced_direct_formula()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Print summary
        print("\n=== Benchmark Summary ===")
        print(f"Silhouette Method: {silhouette_clusters} clusters")
        print(f"Enhanced Direct Formula: {enhanced_clusters} clusters")
        print("\n")
        print(results_df)
        
        return results_df
    
    def plot_results(self, results_df):
        """
        Plot benchmark results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with benchmark results
        """
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot number of clusters
        sns.barplot(x='method', y='n_clusters', data=results_df, ax=axes[0, 0])
        axes[0, 0].set_title('Number of Clusters Selected')
        axes[0, 0].set_ylabel('Clusters')
        
        # Plot execution time
        sns.barplot(x='method', y='execution_time', data=results_df, ax=axes[0, 1])
        axes[0, 1].set_title('Execution Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        
        # Plot memory usage
        sns.barplot(x='method', y='memory_usage', data=results_df, ax=axes[1, 0])
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # Plot silhouette scores
        sns.barplot(x='method', y='silhouette_score', data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title('Clustering Quality (Silhouette Score)')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('cluster_methods_comparison.png', dpi=300)
        plt.close()
        
        # Create a spider plot for overall comparison
        self._create_spider_plot(results_df, 'cluster_methods_spider.png')
        
        print("Results visualized and saved")

    def _create_spider_plot(self, plot_df, filename):
        """
        Create a spider plot comparing methods
        
        Parameters:
        -----------
        plot_df : pd.DataFrame
            DataFrame with results
        filename : str
            Filename to save the plot
        """
        # Normalize the metrics for spider plot
        # For silhouette score, higher is better
        min_sil = plot_df['silhouette_score'].min()
        max_sil = plot_df['silhouette_score'].max()
        if max_sil > min_sil:
            plot_df['silhouette_score_norm'] = (plot_df['silhouette_score'] - min_sil) / (max_sil - min_sil)
        else:
            plot_df['silhouette_score_norm'] = 1.0
        
        # For execution time, lower is better, so invert the normalization
        min_time = plot_df['execution_time'].min()
        max_time = plot_df['execution_time'].max()
        if max_time > min_time:
            plot_df['execution_time_norm'] = 1 - (plot_df['execution_time'] - min_time) / (max_time - min_time)
        else:
            plot_df['execution_time_norm'] = 1.0
            
        # For memory usage, lower is better, so invert the normalization
        min_mem = plot_df['memory_usage'].min()
        max_mem = plot_df['memory_usage'].max()
        if max_mem > min_mem:
            plot_df['memory_usage_norm'] = 1 - (plot_df['memory_usage'] - min_mem) / (max_mem - min_mem)
        else:
            plot_df['memory_usage_norm'] = 1.0
            
        # For n_clusters, normalize based on domain knowledge that too many or too few is bad
        # Assume the average is closest to optimal
        mean_clusters = plot_df['n_clusters'].mean()
        max_diff = max(abs(plot_df['n_clusters'] - mean_clusters))
        if max_diff > 0:
            plot_df['n_clusters_norm'] = 1 - abs(plot_df['n_clusters'] - mean_clusters) / max_diff
        else:
            plot_df['n_clusters_norm'] = 1.0
            
        # Setup the spider plot
        categories = ['Silhouette\nScore', 'Speed', 'Memory\nEfficiency', 'Cluster\nCount']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the spider plot for each method
        for i, method in enumerate(plot_df['method']):
            values = [
                plot_df.loc[plot_df['method'] == method, 'silhouette_score_norm'].iloc[0],
                plot_df.loc[plot_df['method'] == method, 'execution_time_norm'].iloc[0],
                plot_df.loc[plot_df['method'] == method, 'memory_usage_norm'].iloc[0],
                plot_df.loc[plot_df['method'] == method, 'n_clusters_norm'].iloc[0]
            ]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Draw y-axis grid lines
        ax.set_rlabel_position(0)
        ax.set_rticks([0.25, 0.5, 0.75, 1])
        ax.set_rmax(1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Method Comparison Across Metrics', size=15, color='blue', y=1.1)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

def main():
    """Main function to run the benchmark"""
    print("Starting Simplified Cluster Selection Methods Benchmark")
    
    # Create the benchmark suite
    benchmark = ClusteringBenchmark()
    
    # Load data
    benchmark.load_data()
    
    # Run all benchmarks
    results_df = benchmark.run_all_benchmarks()
    
    # Plot results
    benchmark.plot_results(results_df)
    
    print("\nBenchmark completed! Results saved as cluster_methods_comparison.png and cluster_methods_spider.png")


if __name__ == "__main__":
    main()
