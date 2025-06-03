#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PIP Pattern Miner V2

Enhanced Pattern Miner with batch processing and parallel computing capabilities.
This version handles large datasets more efficiently by processing data in batches,
enabling analysis of virtually unlimited financial time series data.

Key improvements over v1:
1. Batch processing with configurable overlap
2. Parallel computing for pattern identification and clustering
3. Robust cluster merging using Dynamic Time Warping (DTW)
4. Memory-efficient processing for large datasets
5. Cross-batch normalization for consistent pattern comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import warnings
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import the enhanced direct formula
try:
    from Pattern.enhanced_direct_formula import EnhancedDirectFormula, count_optimal_clusters
except ImportError:
    logger.warning("Enhanced Direct Formula module not found. Using basic ratio method.")

class Pattern_Miner_V2:
    """
    Pattern_Miner_V2: An enhanced class for identifying, clustering, and analyzing 
    price patterns in financial data using batch processing and parallel computing.
    
    This class implements the Perceptually Important Points (PIP) algorithm to identify key points
    in time series data, processes data in manageable batches, and uses parallel computing
    to improve performance on large datasets.
    """
    
    def __init__(self, 
                 n_pips=5, 
                 lookback=24, 
                 hold_period=6, 
                 returns_hold_period=6, 
                 distance_measure=2,
                 batch_size=10000,
                 batch_overlap=1000,
                 cluster_percentage=0.3,
                 similarity_threshold=0.85,
                 n_jobs=None):
        """
        Initialize the Pattern_Miner_V2 with configuration parameters.
        
        Parameters:
        -----------
        n_pips : int
            Number of perceptually important points to identify in each pattern
        lookback : int
            Number of candles to look back when identifying patterns
        hold_period : int
            Number of candles to hold a position after pattern identification
        returns_hold_period : int
            Number of candles to measure returns over after pattern identification
        distance_measure : int
            Distance measure to use for PIP identification:
            1 = Euclidean Distance
            2 = Perpendicular Distance (default)
            3 = Vertical Distance
        batch_size : int
            Size of each batch when processing large datasets
        batch_overlap : int
            Overlap between consecutive batches to ensure pattern continuity
        cluster_percentage : float
            Percentage of patterns to use as clusters (0.1 to 0.9)
        similarity_threshold : float
            Threshold for considering clusters similar enough to merge (0.0 to 1.0)
        n_jobs : int or None
            Number of parallel processes to use (None = use all available cores)
        """        # Configuration parameters
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._returns_hold_period = returns_hold_period
        self._distance_measure = distance_measure
        
        # Validate batch size and overlap to prevent errors
        if batch_size <= 0:
            logger.warning(f"Invalid batch_size ({batch_size}), setting to default 10000")
            batch_size = 10000
        if batch_overlap >= batch_size:
            logger.warning(f"Batch overlap ({batch_overlap}) must be less than batch size ({batch_size}), adjusting to batch_size - 1")
            batch_overlap = batch_size - 1
            
        self._batch_size = batch_size
        self._batch_overlap = batch_overlap
        self._cluster_percentage = cluster_percentage
        self._similarity_threshold = similarity_threshold
        self._n_jobs = n_jobs if n_jobs else max(1, multiprocessing.cpu_count() - 1)
        
        # Pattern storage
        self._unique_pip_patterns = []  # List of unique patterns, each pattern is a list of normalized price points
        self._unique_pip_indices = []   # List of the last indices of the unique patterns in the original data
        self._global_pip_indices = []   # List of each pattern's global indices in the source data
        
        # Clustering results
        self._cluster_centers = []      # List of cluster centers (centroids)
        self._pip_clusters = []         # List of clusters, each cluster is a list of indices into unique_pip_patterns
        
        # Batch processing data
        self._batch_results = []        # Results from each batch
        self._batch_patterns = []       # Patterns from each batch
        self._batch_indices = []        # Pattern indices from each batch
        self._batch_cluster_centers = [] # Cluster centers from each batch
        self._batch_clusters = []       # Clusters from each batch
        self._merged_clusters = {}      # Merged clusters across all batches
        
        # Statistics
        self._max_patterns_count = None # Maximum number of patterns in any cluster
        self._avg_patterns_count = None # Average number of patterns in all clusters
        self._cluster_stats = {}        # Statistics about clusters

        # Signal generation
        self._cluster_signals = []
        self._long_signal = None
        self._short_signal = None
        self._selected_long = []        # Indices of clusters with positive expected returns
        self._selected_short = []       # Indices of clusters with negative expected returns
        self._selected_neutral = []     # Indices of clusters with zero expected returns

        # Performance analysis
        self._cluster_returns = []      # Mean returns for each cluster
        self._cluster_mfe = []          # Mean MFE for each cluster
        self._cluster_mae = []          # Mean MAE for each cluster
        
        # Data and returns
        self._data = None               # Source price data array to mine patterns from
        self._returns_next_candle = None    # Array of next-candle returns
        self._returns_fixed_hold = None     # Array of fixed holding period returns
        self._returns_mfe = None            # Maximum favorable excursion returns
        self._returns_mae = None            # Maximum adverse excursion returns

    #----------------------------------------------------------------------------------------
    # Core Pattern Identification Functions
    #----------------------------------------------------------------------------------------

    def find_pips(self, data: np.array, n_pips: int, dist_measure: int):
        """
        Find Perceptually Important Points (PIPs) in a time series.
        
        Parameters:
        -----------
        data : np.array
            The time series data to analyze
        n_pips : int
            Number of PIPs to identify
        dist_measure : int
            Distance measure to use:
            1 = Euclidean Distance
            2 = Perpendicular Distance
            3 = Vertical Distance
            
        Returns:
        --------
        tuple
            (pips_x, pips_y) where pips_x are the indices and pips_y are the values
        """
        # Initialize with first and last point
        pips_x = [0, len(data) - 1]  # Index
        pips_y = [data[0], data[-1]] # Price

        # Add remaining PIPs one by one
        for curr_point in range(2, n_pips):
            md = 0.0       # Max distance
            md_i = -1      # Max distance index
            insert_index = -1

            # Check distance between each adjacent pair of existing PIPs
            for k in range(0, curr_point - 1):
                # Left adjacent, right adjacent indices
                left_adj = k
                right_adj = k + 1

                # Calculate the line between adjacent PIPs
                time_diff = pips_x[right_adj] - pips_x[left_adj]   # Time difference
                price_diff = pips_y[right_adj] - pips_y[left_adj]  # Price difference
                slope = price_diff / time_diff                     # Slope
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope  # y = mx + c

                # Find point with maximum distance between the line and all points between adjacent PIPs
                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                    # Distance calculations
                    d = 0.0  # Distance
                    if dist_measure == 1:  # Euclidean distance
                        d =  ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                        d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                    elif dist_measure == 2:  # Perpendicular distance
                        d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                    else:  # Vertical distance    
                        d = abs((slope * i + intercept) - data[i])

                    # Keep track of the point with maximum distance
                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            # Insert the point with max distance into PIPs
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])

        return pips_x, pips_y
    
    def _find_batch_patterns(self, batch_data, global_offset=0):
        """
        Find unique patterns in a batch of data.
        
        Parameters:
        -----------
        batch_data : np.array
            Batch of price data to process
        global_offset : int
            Offset for converting local indices to global indices
            
        Returns:
        --------
        tuple
            (patterns, indices, global_indices) where:
            - patterns are normalized PIP patterns
            - indices are the last indices of patterns in the batch
            - global_indices are absolute indices in the original data
        """
        unique_patterns = []
        unique_indices = []
        global_indices = []
        
        last_pips_x = [0] * self._n_pips
        
        # Ensure the batch is at least as large as lookback + hold_period
        if len(batch_data) < self._lookback + self._hold_period:
            logger.warning(f"Batch size ({len(batch_data)}) is smaller than lookback + hold_period ({self._lookback + self._hold_period})")
            return unique_patterns, unique_indices, global_indices
        
        # Slide window through the batch data
        for i in range(self._lookback - 1, len(batch_data) - self._hold_period):
            start_i = i - self._lookback + 1
            window = batch_data[start_i: i + 1]
            
            # Find PIPs in this window
            pips_x, pips_y = self.find_pips(window, self._n_pips, self._distance_measure)
            
            # Convert to global indices (within batch)
            local_pips_x = [j + start_i for j in pips_x]
            
            # Convert to global indices (in original data)
            global_pips_x = [j + global_offset for j in local_pips_x]
            
            # Check if this pattern is the same as the last one
            same = True
            for j in range(1, self._n_pips - 1):
                if local_pips_x[j] != last_pips_x[j]:
                    same = False
                    break
            
            # If this is a new pattern, store it
            if not same:
                data = np.array(pips_y).reshape(-1, 1)
                # Create scalers
                minmax_scaler = MinMaxScaler()
                
                # Normalize the pattern using min-max scaling
                normalized = minmax_scaler.fit_transform(data).flatten()
                
                # Store the normalized pattern and its indices
                unique_patterns.append(normalized.tolist())
                unique_indices.append(i)  # Index of the last point in the pattern within the batch
                global_indices.append(global_pips_x)  # All global PIP indices

            last_pips_x = local_pips_x
            
        return unique_patterns, unique_indices, global_indices
    
    def _process_batch(self, batch_idx, batch_data, global_offset):
        """
        Process a single batch of data.
        
        Parameters:
        -----------
        batch_idx : int
            Index of the current batch
        batch_data : np.array
            Batch of price data to process
        global_offset : int
            Offset for converting local indices to global indices
            
        Returns:
        --------
        dict
            Dictionary containing batch processing results
        """
        batch_start_time = time.time()
        
        try:
            # Find patterns in this batch
            patterns, indices, global_indices = self._find_batch_patterns(batch_data, global_offset)
            
            # Skip if not enough patterns found
            if len(patterns) < 3:
                logger.warning(f"Batch {batch_idx} has less than 3 patterns ({len(patterns)}). Skipping.")
                return {
                    'batch_idx': batch_idx,
                    'patterns': [],
                    'indices': [],
                    'global_indices': [],
                    'cluster_centers': [],
                    'clusters': [],
                    'execution_time': time.time() - batch_start_time,
                    'status': 'skipped'
                }
            
            # Determine cluster count (percentage of patterns)
            cluster_count = max(2, min(int(len(patterns) * self._cluster_percentage), len(patterns) - 1))
            
            # Use KMEANS clustering
            # Initialize cluster centers using k-means++
            initial_centers = kmeans_plusplus_initializer(patterns, cluster_count, random_state=batch_idx).initialize()
            kmeans_instance = kmeans(patterns, initial_centers)
            kmeans_instance.process()
            
            # Extract clustering results: clusters and their centers
            clusters = kmeans_instance.get_clusters()
            centers = kmeans_instance.get_centers()
            
            # Calculate performance metrics for this batch
            if len(patterns) >= 3 and len(clusters) >= 2:
                labels = np.zeros(len(patterns), dtype=int)
                for cluster_idx, cluster_members in enumerate(clusters):
                    for member_idx in cluster_members:
                        labels[member_idx] = cluster_idx
                
                silhouette = silhouette_score(patterns, labels) if len(np.unique(labels)) > 1 else 0
                db_score = davies_bouldin_score(patterns, labels) if len(np.unique(labels)) > 1 else float('inf')
            else:
                silhouette = 0
                db_score = float('inf')
            
            batch_results = {
                'batch_idx': batch_idx,
                'patterns': patterns,
                'indices': indices,
                'global_indices': global_indices,
                'cluster_centers': centers,
                'clusters': clusters,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'execution_time': time.time() - batch_start_time,
                'n_patterns': len(patterns),
                'n_clusters': len(clusters),
                'status': 'success'
            }
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            return {
                'batch_idx': batch_idx,
                'status': 'error',
                'error': str(e),
                'execution_time': time.time() - batch_start_time
            }
    
    #----------------------------------------------------------------------------------------
    # Batch Processing and Clustering Functions
    #----------------------------------------------------------------------------------------
    
    def _dynamic_time_warping(self, series1, series2):
        """
        Calculate similarity between two time series using Dynamic Time Warping (DTW).
        
        Parameters:
        -----------
        series1, series2 : np.array
            Time series to compare
            
        Returns:
        --------
        float
            DTW distance (lower = more similar)
        """
        # Convert to numpy arrays if they aren't already
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        n, m = len(s1), len(s2)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, :] = np.inf
        dtw_matrix[:, 0] = np.inf
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                            dtw_matrix[i, j-1],     # deletion
                                            dtw_matrix[i-1, j-1])   # match
        
        return dtw_matrix[n, m]
    
    def _merge_similar_clusters(self, all_batch_results):
        """
        Merge similar clusters across different batches.
        
        Parameters:
        -----------
        all_batch_results : list
            List of dictionaries containing batch processing results
            
        Returns:
        --------
        dict
            Dictionary with merged clusters and their metadata
        """
        # Flatten all centers into a single list with batch metadata
        all_centers = []
        
        for batch_result in all_batch_results:
            if batch_result['status'] != 'success':
                continue
                
            batch_idx = batch_result['batch_idx']
            
            for center_idx, center in enumerate(batch_result['cluster_centers']):
                # Get patterns belonging to this cluster
                cluster_patterns = [batch_result['patterns'][i] for i in batch_result['clusters'][center_idx]]
                
                # Get global indices of patterns in this cluster
                cluster_indices = [batch_result['global_indices'][i] for i in batch_result['clusters'][center_idx]]
                
                all_centers.append({
                    'center': center,
                    'batch_idx': batch_idx,
                    'center_idx': center_idx,
                    'patterns': cluster_patterns,
                    'global_indices': cluster_indices
                })
        
        # Calculate pairwise distances between all cluster centers using DTW
        n_centers = len(all_centers)
        if n_centers <= 1:
            logger.warning("Not enough centers to perform merging")
            return {}
            
        logger.info(f"Calculating distances between {n_centers} cluster centers")
        distance_matrix = np.zeros((n_centers, n_centers))
        
        # Fill the upper triangle of the distance matrix
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                distance = self._dynamic_time_warping(
                    all_centers[i]['center'], 
                    all_centers[j]['center']
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Apply hierarchical clustering on the distance matrix
        Z = linkage(squareform(distance_matrix), method='ward')
        
        # Cut the dendrogram at similarity threshold
        # The distance used here is 1.0 - similarity_threshold because higher similarity means lower distance
        merged_cluster_labels = fcluster(Z, t=1.0 - self._similarity_threshold, criterion='distance')
        
        # Organize results by merged cluster
        merged_clusters = {}
        for i, label in enumerate(merged_cluster_labels):
            if label not in merged_clusters:
                merged_clusters[label] = {
                    'centers': [],
                    'patterns': [],
                    'global_indices': [],
                    'batches': set(),
                    'member_centers': []
                }
            
            merged_clusters[label]['centers'].append(all_centers[i]['center'])
            merged_clusters[label]['patterns'].extend(all_centers[i]['patterns'])
            merged_clusters[label]['global_indices'].extend(all_centers[i]['global_indices'])
            merged_clusters[label]['batches'].add(all_centers[i]['batch_idx'])
            merged_clusters[label]['member_centers'].append({
                'batch_idx': all_centers[i]['batch_idx'],
                'center_idx': all_centers[i]['center_idx'],
                'center': all_centers[i]['center']
            })
        
        # Calculate representative center for each merged cluster
        for label, cluster in merged_clusters.items():
            # Use the average of all centers as the representative center
            cluster['representative_center'] = np.mean(cluster['centers'], axis=0)
            
            # Store statistics about this merged cluster
            cluster['n_patterns'] = len(cluster['patterns'])
            cluster['n_batches'] = len(cluster['batches'])
            cluster['n_centers'] = len(cluster['centers'])
        
        return merged_clusters
    
    #----------------------------------------------------------------------------------------
    # Return Calculation Functions
    #----------------------------------------------------------------------------------------
    
    def calculate_returns_fixed_hold(self, data: np.array, pattern_indices: list, hold_period: int) -> np.array:
        """
        Calculate returns for a fixed holding period after each pattern.
        
        Parameters:
        -----------
        data : np.array
            Price data array
        pattern_indices : list
            List of indices where patterns end
        hold_period : int
            Number of candles to hold after pattern identification
            
        Returns:
        --------
        np.array
            Array of returns for each pattern
        """
        returns = []
        for idx in pattern_indices:
            if idx + hold_period < len(data):
                returns.append((data[idx + hold_period] - data[idx]) / data[idx])
            else:
                returns.append(0)  # Not enough future data
        return np.array(returns)
    
    def calculate_mfe_mae(self, data: np.array, pattern_indices: list, hold_period: int) -> tuple:
        """
        Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
        
        MFE: Best unrealized gain during the trade.
        MAE: Worst unrealized loss during the trade.
        
        Parameters:
        -----------
        data : np.array
            Price data array
        pattern_indices : list
            List of indices where patterns end
        hold_period : int
            Number of candles to hold after pattern identification
            
        Returns:
        --------
        tuple
            (mfe, mae) arrays for each pattern
        """
        mfe, mae = [], []
        for idx in pattern_indices:
            if idx + hold_period < len(data):
                window = data[idx: idx + hold_period]
                max_price = np.max(window)
                min_price = np.min(window)
                mfe.append((max_price - data[idx]) / data[idx])  # Maximum favorable excursion
                mae.append((min_price - data[idx]) / data[idx])  # Maximum adverse excursion
            else:
                mfe.append(0)
                mae.append(0)
        return np.array(mfe), np.array(mae)
    
    def _categorize_merged_clusters(self, merged_clusters, full_data):
        """
        Categorize merged clusters into long, short, and neutral based on mean returns.
        
        Parameters:
        -----------
        merged_clusters : dict
            Dictionary with merged clusters
        full_data : np.array
            Complete price data array
            
        Returns:
        --------
        dict
            Dictionary with cluster categories and statistics
        """
        categorized_clusters = {}
        
        for label, cluster in merged_clusters.items():
            # Generate flattened global indices from all patterns in this cluster
            global_indices = []
            for pattern_indices in cluster['global_indices']:
                # The last index in the pattern is where the pattern completes
                if pattern_indices:  # Check if not empty
                    global_indices.append(pattern_indices[-1])
            
            if not global_indices:  # Skip if no valid indices
                continue
                
            # Calculate returns for this cluster
            fixed_returns = self.calculate_returns_fixed_hold(
                full_data, global_indices, self._returns_hold_period
            )
            
            mfe_returns, mae_returns = self.calculate_mfe_mae(
                full_data, global_indices, self._returns_hold_period
            )
            
            # Calculate statistics
            mean_return = np.mean(fixed_returns)
            mean_mfe = np.mean(mfe_returns)
            mean_mae = np.mean(mae_returns)
            
            # Categorize based on mean return direction
            category = 'neutral'
            if mean_return > 0:
                category = 'long'
            elif mean_return < 0:
                category = 'short'
            
            # Store cluster with its statistics
            categorized_clusters[label] = {
                'category': category,
                'mean_return': mean_return,
                'mean_mfe': mean_mfe,
                'mean_mae': mean_mae,
                'n_patterns': len(global_indices),
                'representative_center': cluster['representative_center'].tolist(),
                'stats': {
                    'return_std': np.std(fixed_returns),
                    'return_sharpe': mean_return / np.std(fixed_returns) if np.std(fixed_returns) > 0 else 0,
                    'win_rate': np.sum(fixed_returns > 0) / len(fixed_returns) if len(fixed_returns) > 0 else 0,
                    'max_return': np.max(fixed_returns),
                    'min_return': np.min(fixed_returns)
                }
            }
            
        # Store categorized results
        self._selected_long = [label for label, cluster in categorized_clusters.items() 
                              if cluster['category'] == 'long']
        self._selected_short = [label for label, cluster in categorized_clusters.items() 
                               if cluster['category'] == 'short']
        self._selected_neutral = [label for label, cluster in categorized_clusters.items() 
                                 if cluster['category'] == 'neutral']
        
        return categorized_clusters
    
    #----------------------------------------------------------------------------------------
    # Training and Prediction Functions
    #----------------------------------------------------------------------------------------
    
    def train(self, data: np.array, save_intermediate=False, intermediate_dir="./temp"):
        """
        Train the pattern miner on a price data array using batch processing.
        
        Parameters:
        -----------
        data : np.array
            Price data array
        save_intermediate : bool
            Whether to save intermediate batch results to disk
        intermediate_dir : str
            Directory to save intermediate results
            
        Returns:
        --------
        dict
            Dictionary with training results summary
        """
        start_time = time.time()
        self._data = data
        
        logger.info(f"Training with {len(data)} data points, batch size: {self._batch_size}, "
                    f"overlap: {self._batch_overlap}, n_jobs: {self._n_jobs}")
        
        # Create intermediate directory if needed
        if save_intermediate and not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
              # Generate batches with overlap
        batches = []
        # Ensure step size is at least 1 to avoid "range() arg 3 must not be zero" error
        step_size = max(1, self._batch_size - self._batch_overlap)
        for start_idx in range(0, len(data), step_size):
            end_idx = min(start_idx + self._batch_size, len(data))
            if end_idx - start_idx < self._lookback + self._hold_period:
                # Skip if batch is too small
                continue
            batches.append((len(batches), data[start_idx:end_idx], start_idx))
            
            # Stop if we've reached the end of the data
            if end_idx == len(data):
                break
                
        logger.info(f"Created {len(batches)} batches")
        
        # Process batches in parallel
        all_batch_results = []
        
        with ProcessPoolExecutor(max_workers=self._n_jobs) as executor:
            # Submit all batch processing tasks
            futures = {executor.submit(self._process_batch, *batch): batch[0] for batch in batches}
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_idx = futures[future]
                try:
                    batch_result = future.result()
                    all_batch_results.append(batch_result)
                    
                    # Save intermediate results if requested
                    if save_intermediate:
                        batch_file = os.path.join(intermediate_dir, f"batch_{batch_idx}.json")
                        with open(batch_file, 'w') as f:
                            # Convert numpy arrays to lists for JSON serialization
                            save_result = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                                         for k, v in batch_result.items()}
                            json.dump(save_result, f)
                    
                    logger.info(f"Batch {batch_idx} processed: "
                                f"{batch_result.get('n_patterns', 0)} patterns, "
                                f"{batch_result.get('n_clusters', 0)} clusters, "
                                f"status: {batch_result['status']}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
        
        # Store successful batch results
        self._batch_results = [r for r in all_batch_results if r['status'] == 'success']
        
        # Skip pattern merging if no successful batches
        if not self._batch_results:
            logger.error("No successful batch results to merge")
            return {
                'status': 'error',
                'message': 'No successful batch results to merge',
                'execution_time': time.time() - start_time
            }
            
        # Store pattern information for successful batches
        for batch_result in self._batch_results:
            self._batch_patterns.append(batch_result['patterns'])
            self._batch_indices.append(batch_result['indices'])
            self._batch_cluster_centers.append(batch_result['cluster_centers'])
            self._batch_clusters.append(batch_result['clusters'])
            
        # Merge similar clusters across batches
        logger.info("Merging clusters across batches")
        self._merged_clusters = self._merge_similar_clusters(all_batch_results)
        
        # Calculate the returns and categorize merged clusters
        logger.info("Categorizing merged clusters")
        self._cluster_stats = self._categorize_merged_clusters(self._merged_clusters, data)
        
        # Calculate pattern statistics
        total_patterns = sum(len(batch['patterns']) for batch in self._batch_results)
        total_clusters = sum(len(batch['clusters']) for batch in self._batch_results)
        merged_clusters_count = len(self._merged_clusters)
        
        # Update class attributes with the collective results
        self._unique_pip_patterns = [p for batch in self._batch_patterns for p in batch]
        
        # Flatten all cluster centers into one list
        self._cluster_centers = [cluster['representative_center'] 
                              for cluster in self._merged_clusters.values()]
        
        # Create flat cluster-to-pattern mappings
        self._pip_clusters = []
        pattern_index = 0
        for label, cluster in self._merged_clusters.items():
            cluster_patterns = []
            for _ in range(len(cluster['patterns'])):
                cluster_patterns.append(pattern_index)
                pattern_index += 1
            self._pip_clusters.append(cluster_patterns)
            
        # Store execution time
        execution_time = time.time() - start_time
        
        # Print summary
        logger.info(f"Training completed in {execution_time:.2f} seconds")
        logger.info(f"Total patterns: {total_patterns}")
        logger.info(f"Total batch clusters: {total_clusters}")
        logger.info(f"Merged clusters: {merged_clusters_count}")
        logger.info(f"Long clusters: {len(self._selected_long)}")
        logger.info(f"Short clusters: {len(self._selected_short)}")
        logger.info(f"Neutral clusters: {len(self._selected_neutral)}")
        
        return {
            'status': 'success',
            'execution_time': execution_time,
            'total_patterns': total_patterns,
            'total_batch_clusters': total_clusters,
            'merged_clusters': merged_clusters_count,
            'long_clusters': len(self._selected_long),
            'short_clusters': len(self._selected_short),
            'neutral_clusters': len(self._selected_neutral)
        }
    
    def predict(self, new_data: np.array, last_n: int = None):
        """
        Predict patterns in new data by comparing to existing clusters.
        
        Parameters:
        -----------
        new_data : np.array
            New price data to analyze
        last_n : int or None
            If provided, use only the last n data points
            
        Returns:
        --------
        list
            List of detected patterns with their cluster assignments and metadata
        """
        if not self._merged_clusters:
            logger.error("No merged clusters available. Please train the model first.")
            return []
            
        # Use only the last N points if specified
        if last_n is not None and last_n < len(new_data):
            data_to_analyze = new_data[-last_n:]
        else:
            data_to_analyze = new_data
            
        # Find patterns in the new data
        patterns, indices, global_indices = self._find_batch_patterns(data_to_analyze)
        
        if not patterns:
            logger.warning("No patterns found in the new data")
            return []
            
        # Match each pattern to the closest cluster
        matches = []
        
        for i, pattern in enumerate(patterns):
            min_distance = float('inf')
            best_match = None
            
            # Compare with each merged cluster's representative center
            for label, cluster in self._merged_clusters.items():
                distance = self._dynamic_time_warping(pattern, cluster['representative_center'])
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = {
                        'cluster_label': label,
                        'distance': distance,
                        'similarity': 1.0 / (1.0 + distance),  # Convert distance to similarity
                        'category': self._cluster_stats[label]['category'] if label in self._cluster_stats else 'unknown',
                        'expected_return': self._cluster_stats[label]['mean_return'] if label in self._cluster_stats else 0,
                        'expected_mfe': self._cluster_stats[label]['mean_mfe'] if label in self._cluster_stats else 0,
                        'expected_mae': self._cluster_stats[label]['mean_mae'] if label in self._cluster_stats else 0
                    }
                    
            if best_match:
                match_info = {
                    'pattern_idx': i,
                    'data_idx': indices[i],
                    'global_idx': global_indices[i][-1] if global_indices[i] else None,
                    'pattern': pattern,
                    'match': best_match
                }
                matches.append(match_info)
                
        return matches
   
    #----------------------------------------------------------------------------------------
    # Visualization Functions
    #----------------------------------------------------------------------------------------
    
    def plot_merged_clusters(self, n_clusters=9, figsize=(15, 10)):
        """
        Plot representative centers of merged clusters.
        
        Parameters:
        -----------
        n_clusters : int
            Number of top clusters to display (by pattern count)
        figsize : tuple
            Figure size
        """
        if not self._merged_clusters:
            logger.error("No merged clusters available. Please train the model first.")
            return
        
        # Get top clusters by pattern count
        top_clusters = sorted(
            self._merged_clusters.items(),
            key=lambda x: len(x[1]['patterns']),
            reverse=True
        )[:n_clusters]
        
        # Create a grid for plotting
        grid_size = int(np.ceil(np.sqrt(len(top_clusters))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()
        
        for i, (label, cluster) in enumerate(top_clusters):
            if i >= len(axes):
                break
                
            # Plot representative center
            axes[i].plot(cluster['representative_center'], 'b-', linewidth=2)
            
            # Add category and stats if available
            if label in self._cluster_stats:
                category = self._cluster_stats[label]['category']
                mean_return = self._cluster_stats[label]['mean_return']
                color = 'green' if category == 'long' else 'red' if category == 'short' else 'gray'
                
                # Set title with color indicating pattern type
                axes[i].set_title(f"Cluster {label}: {category.upper()}\n"
                                 f"Return: {mean_return:.2%}", 
                                 color=color)
            else:
                axes[i].set_title(f"Cluster {label}")
                
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
            
        # Hide unused axes
        for i in range(len(top_clusters), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.suptitle("Top Pattern Clusters", fontsize=16, y=1.02)
        plt.show()
    
    def plot_batch_stats(self):
        """
        Plot statistics about batch processing.
        """
        if not self._batch_results:
            logger.error("No batch results available. Please train the model first.")
            return
            
        # Extract statistics
        batch_indices = [r['batch_idx'] for r in self._batch_results]
        n_patterns = [r.get('n_patterns', 0) for r in self._batch_results]
        n_clusters = [r.get('n_clusters', 0) for r in self._batch_results]
        silhouette = [r.get('silhouette', 0) for r in self._batch_results]
        execution_time = [r.get('execution_time', 0) for r in self._batch_results]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Patterns and clusters by batch
        axes[0, 0].bar(batch_indices, n_patterns, alpha=0.7, label='Patterns')
        axes[0, 0].bar(batch_indices, n_clusters, alpha=0.7, label='Clusters')
        axes[0, 0].set_title('Patterns and Clusters by Batch')
        axes[0, 0].set_xlabel('Batch Index')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Silhouette score by batch
        axes[0, 1].plot(batch_indices, silhouette, 'o-', color='blue')
        axes[0, 1].set_title('Silhouette Score by Batch')
        axes[0, 1].set_xlabel('Batch Index')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Execution time by batch
        axes[1, 0].plot(batch_indices, execution_time, 'o-', color='green')
        axes[1, 0].set_title('Execution Time by Batch')
        axes[1, 0].set_xlabel('Batch Index')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Pattern-to-cluster ratio by batch
        ratio = [p/c if c > 0 else 0 for p, c in zip(n_patterns, n_clusters)]
        axes[1, 1].plot(batch_indices, ratio, 'o-', color='purple')
        axes[1, 1].set_title('Pattern-to-Cluster Ratio by Batch')
        axes[1, 1].set_xlabel('Batch Index')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle("Batch Processing Statistics", fontsize=16, y=1.02)
        plt.show()
    
    def plot_merged_cluster_examples(self, candle_data: pd.DataFrame, cluster_label: int, grid_size: int = 5, n_examples: int = 25):
        """
        Plot examples of a merged cluster using candlestick charts.
        
        Parameters:
        -----------
        candle_data : pd.DataFrame
            Candlestick data with OHLC values
        cluster_label : int
            Label of the merged cluster to plot examples for
        grid_size : int
            Size of the grid for subplot layout
        n_examples : int
            Number of examples to plot
        """
        if not self._merged_clusters or cluster_label not in self._merged_clusters:
            logger.error(f"Cluster {cluster_label} not found.")
            return
            
        cluster = self._merged_clusters[cluster_label]
        global_indices = []
        
        # Get global indices where patterns end
        for pattern_indices in cluster['global_indices']:
            if pattern_indices:  # Check if not empty
                global_indices.append(pattern_indices[-1])
                
        # Limit to requested number of examples
        global_indices = global_indices[:n_examples]
        
        if not global_indices:
            logger.error("No valid global indices found for this cluster.")
            return
            
        # Setup plot
        plt.style.use('dark_background')
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        flat_axs = axs.flatten()
        
        # Plot each example
        for i, idx in enumerate(global_indices):
            if i >= len(flat_axs):
                break
                
            # Check if index is valid for the provided candle data
            if idx - self._lookback + 1 < 0 or idx >= len(candle_data):
                continue
                
            # Get data slice for this pattern
            data_slice = candle_data.iloc[idx - self._lookback + 1: idx + 1]
            
            if len(data_slice) < self._lookback:
                continue
                
            # Get PIPs for this pattern
            try:
                plot_pip_x, plot_pip_y = self.find_pips(
                    data_slice['Close'].to_numpy(), self._n_pips, self._distance_measure
                )
                
                # Convert to DataFrame indices
                idx_vals = data_slice.index.values
                
                # Create lines connecting PIPs
                pip_lines = []
                colors = []
                for line_i in range(self._n_pips - 1):
                    l0 = [(idx_vals[plot_pip_x[line_i]], plot_pip_y[line_i]), 
                          (idx_vals[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
                    pip_lines.append(l0)
                    colors.append('w')
                    
                # Plot candlestick chart with PIP lines
                mpf.plot(
                    data_slice, type='candle', 
                    alines=dict(alines=pip_lines, colors=colors), 
                    ax=flat_axs[i], style='charles',
                    update_width_config=dict(candle_linewidth=1.5)
                )
                
                # Remove tick labels for cleaner display
                flat_axs[i].set_yticklabels([])
                flat_axs[i].set_xticklabels([])
                flat_axs[i].set_xticks([])
                flat_axs[i].set_yticks([])
                flat_axs[i].set_ylabel("")
                
            except Exception as e:
                logger.error(f"Error plotting example {i}: {str(e)}")
        
        # Hide unused axes
        for i in range(len(global_indices), len(flat_axs)):
            flat_axs[i].set_visible(False)
        
        # Set title with category and stats if available
        title = f"Cluster {cluster_label} Examples"
        if cluster_label in self._cluster_stats:
            category = self._cluster_stats[cluster_label]['category']
            mean_return = self._cluster_stats[cluster_label]['mean_return']
            title = f"Cluster {cluster_label}: {category.upper()} - Return: {mean_return:.2%}"
            
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    #----------------------------------------------------------------------------------------
    # Getter Methods
    #----------------------------------------------------------------------------------------
    
    def get_cluster_stats(self):
        """Get statistics for all clusters."""
        return self._cluster_stats
    
    def get_merged_clusters(self):
        """Get merged clusters."""
        return self._merged_clusters
    
    def get_batch_results(self):
        """Get batch processing results."""
        return self._batch_results
    
    def get_long_clusters(self):
        """Get indices of long clusters."""
        return self._selected_long
    
    def get_short_clusters(self):
        """Get indices of short clusters."""
        return self._selected_short
    
    def get_cluster_centers(self):
        """Get cluster centers."""
        return self._cluster_centers


if __name__ == "__main__":
    # Example usage
    import cProfile
    import pstats
    from memory_profiler import profile
    
    # Load sample data
    try:
        data = pd.read_csv('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Raw/Stocks/Intraday/15M/BTCUSD15.csv')
        data['Date'] = data['Date'].astype('datetime64[s]')
        data = data.set_index('Date')
        # Trim the data for testing
        data = data.head(10000)
        arr = data['Close'].to_numpy()
        
        # Initialize the pattern miner with batch processing
        miner = Pattern_Miner_V2(
            n_pips=5,
            lookback=24,
            hold_period=6,
            batch_size=1000,
            batch_overlap=100,
            cluster_percentage=0.3,
            n_jobs=4
        )
        
        # Profile the training process
        with cProfile.Profile() as pr:
            miner.train(arr)
            
        # Save profiling stats
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats('pip_miner_v2_profile.prof')
        
        # Print results
        print("\nTraining completed")
        merged_clusters = miner.get_merged_clusters()
        print(f"Total merged clusters: {len(merged_clusters)}")
        
        # Plot merged clusters
        miner.plot_merged_clusters(n_clusters=9)
        
        # Plot batch statistics
        miner.plot_batch_stats()
        
        # Example of predicting on new data
        new_data = arr[-5000:]
        matches = miner.predict(new_data, last_n=1000)
        print(f"Found {len(matches)} pattern matches in new data")
        
        # Print instructions to view profiling results
        print("\nTo visualize performance profile, run in terminal:")
        print("snakeviz .\\pip_miner_v2_profile.prof")
        
    except Exception as e:
        print(f"Error running example: {str(e)}")
