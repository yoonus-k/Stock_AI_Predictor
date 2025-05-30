#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory-Efficient Enhanced Direct Formula for Optimal Cluster Determination

This module provides a low-memory implementation of the Enhanced Direct Formula approach
for determining the optimal number of clusters in the Pattern_Miner class, using:

- Streaming calculations to avoid storing full distance matrices
- Approximate nearest neighbors for large datasets
- Mini-batch processing for scalable operations
- Dimensionality reduction preprocessing option
- Incremental distance calculations
- Memory monitoring and strict limits
- Sparse representations where appropriate
"""
from memory_profiler import profile
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KDTree
import warnings
import time
import gc  # For explicit garbage collection
import psutil  # For memory monitoring
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform, euclidean
import math
from itertools import islice

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class EnhancedDirectFormula:
    """Memory-efficient implementation of Enhanced Direct Formula for optimal clustering"""
    
    def __init__(self):
        """Initialize the enhanced formula components with memory constraints"""
        self.data = None
        self.data_sample = None
        self.data_characteristics = {}
        self.meta_parameters = {
            'alpha': 0.15,  # Balance between density and dimensionality
            'beta': 0.35,   # Weight for variance consideration
            'gamma': 0.5,   # Weight for distribution shape
            'min_clusters': 2,
            'max_clusters': 50,
            'max_memory_gb': 2.0,    # Maximum memory usage in GB - strict limit
            'sample_threshold': 1000, # Sample data when more than this many points
            'nn_algorithm': 'auto',   # Algorithm for nearest neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'
            'use_pca': True,          # Whether to use PCA for dimensionality reduction
            'pca_components': 0.95,   # PCA variance to retain or number of components
            'use_mini_batch': True,   # Use MiniBatchKMeans for large datasets
            'batch_size': 100,        # Batch size for mini-batch operations
        }
        # Memory tracking
        self._memory_usage_start = 0
        self._track_memory(reset_peak=True)
        
    def _track_memory(self, reset_peak=False):
        """Track current memory usage"""
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 * 1024 * 1024)  # GB
        
        if not hasattr(self, '_peak_memory') or reset_peak:
            self._peak_memory = mem_gb
        else:
            self._peak_memory = max(self._peak_memory, mem_gb)
            
        return mem_gb
    
    def _check_memory_limit(self):
        """Check if memory usage exceeds limit and raise error if needed"""
        current_mem = self._track_memory()
        if current_mem > self.meta_parameters['max_memory_gb']:
            # Force immediate garbage collection
            gc.collect()
            current_mem = self._track_memory()
            if current_mem > self.meta_parameters['max_memory_gb']:
                # Still exceeding - raise error or reduce sample size drastically
                raise MemoryError(f"Memory usage ({current_mem:.2f}GB) exceeds limit " 
                                f"({self.meta_parameters['max_memory_gb']:.2f}GB)")
        return current_mem
    
    def _adaptive_sampling(self, data):
        """
        Adapt sample size based on available memory and dataset size
        
        Returns:
        --------
        ndarray: Sampled data
        bool: Whether data was sampled
        """
        n_samples = data.shape[0]
        if n_samples <= self.meta_parameters['sample_threshold']:
            return data, False
            
        # Calculate memory per sample point as an estimate
        used_mem = self._track_memory()
        mem_per_sample = used_mem / max(1, n_samples) 
        
        # Calculate how many samples we can afford with 50% of our memory budget
        safe_budget = self.meta_parameters['max_memory_gb'] * 0.5
        available_budget = max(0.1, safe_budget - used_mem)
        affordable_samples = int(available_budget / mem_per_sample)
        
        # Ensure a reasonable sample size between minimum and maximum
        sample_size = max(
            min(affordable_samples, self.meta_parameters['sample_threshold']),
            min(100, n_samples)
        )
        
        # If we're sampling less than 20% of the data, warn about it
        sample_ratio = sample_size / n_samples
        if sample_ratio < 0.2:
            warnings.warn(f"Using only {sample_ratio:.1%} of data ({sample_size}/{n_samples}) due to memory constraints")
            
        # Sample the data
        idx = np.random.choice(n_samples, size=sample_size, replace=False)
        return data[idx], True
        
    def _apply_dimensionality_reduction(self, data):
        """
        Apply PCA for dimensionality reduction if enabled and beneficial
        
        Returns:
        --------
        ndarray: Reduced data
        """
        if not self.meta_parameters['use_pca']:
            return data
            
        n_samples, n_features = data.shape
        
        # Only apply PCA if we have more than 3 features
        if n_features <= 3:
            return data
            
        # Use IncrementalPCA for large datasets
        if n_samples > 5000:
            # Batch size that fits in memory
            batch_size = min(self.meta_parameters['batch_size'], n_samples)
            ipca = IncrementalPCA(n_components=self.meta_parameters['pca_components'])
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                batch = data[i:min(i+batch_size, n_samples)]
                ipca.partial_fit(batch)
                
            # Transform the data in batches
            reduced_data = np.empty((n_samples, ipca.n_components_))
            for i in range(0, n_samples, batch_size):
                batch = data[i:min(i+batch_size, n_samples)]
                reduced_data[i:min(i+batch_size, n_samples)] = ipca.transform(batch)
                
            variance_retained = sum(ipca.explained_variance_ratio_)
            n_components = ipca.n_components_
        else:
            # Use regular PCA for smaller datasets
            pca = PCA(n_components=self.meta_parameters['pca_components'])
            reduced_data = pca.fit_transform(data)
            variance_retained = sum(pca.explained_variance_ratio_)
            n_components = pca.n_components_
            
        #print(f"PCA: Reduced dimensions from {n_features} to {n_components} retaining {variance_retained:.2%} variance")
        return reduced_data
        
    
    def analyze_data_characteristics(self, data):
        """
        Memory-efficient analysis of key data characteristics
        
        Parameters:
        -----------
        data : array-like
            Pattern data to analyze
        """
        self._memory_usage_start = self._track_memory(reset_peak=True)
        #print(f"Starting memory: {self._memory_usage_start:.2f}GB")
        
        # Store reference to original data
        self.data = data
        n_samples, n_features = data.shape
        
        # For large datasets, work with a sample to estimate characteristics
        self.data_sample, data_sampled = self._adaptive_sampling(data)
        
        # Apply dimensionality reduction if enabled
        if self.meta_parameters['use_pca']:
            self.data_sample_reduced = self._apply_dimensionality_reduction(self.data_sample)
        else:
            self.data_sample_reduced = self.data_sample
            
        # Track memory after preprocessing
        preprocess_mem = self._track_memory()
        #print(f"Memory after preprocessing: {preprocess_mem:.2f}GB")
        
        # Basic statistics that don't require distances
        sample_variance = np.var(self.data_sample, axis=0).mean()
        sample_range = np.ptp(self.data_sample, axis=0).mean()
        
        # Store basic characteristics
        self.data_characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'variance': sample_variance,
            'range': sample_range,
            'data_sampled': data_sampled,
            'sample_size': len(self.data_sample),
            'sample_ratio': len(self.data_sample) / len(data) if data_sampled else 1.0,
        }
        
        # Efficient nearest neighbor statistics using KD-Tree or Ball-Tree
        nn_stats = self._compute_nn_statistics_efficient()
        
        # Distribution shape analysis without storing full distance matrix
        dist_shape = self._analyze_distribution_shape_efficient()
        
        # Estimate density and intrinsic dimension using nearest neighbors
        density = 1.0 / max(nn_stats['mean'], 1e-10)
        intrinsic_dim = self._estimate_intrinsic_dim_efficient()
        
        # Update data characteristics
        self.data_characteristics.update({
            'density': density,
            'distribution_shape': dist_shape,
            'intrinsic_dimensionality': intrinsic_dim,
            'nearest_neighbor_stats': nn_stats,
        })
        
        # Check final memory usage
        final_mem = self._track_memory()
        #print(f"Memory after analysis: {final_mem:.2f}GB (Peak: {self._peak_memory:.2f}GB)")
        
        return self.data_characteristics
    
    def _compute_nn_statistics_efficient(self, k=5):
        """
        Compute nearest neighbor statistics efficiently using sklearn's NearestNeighbors
        
        Much more memory efficient than computing the full distance matrix
        """
        # Use the reduced sample data if available
        data = self.data_sample_reduced if hasattr(self, 'data_sample_reduced') else self.data_sample
        
        # Limit k to avoid excessive memory usage
        k = min(k, data.shape[0]-1, 10)
        
        # Initialize nearest neighbors model
        nn = NearestNeighbors(
            n_neighbors=k+1,  # +1 because the first neighbor is the point itself
            algorithm=self.meta_parameters['nn_algorithm'],
            n_jobs=-1  # Use all cores
        )
        nn.fit(data)
        
        # Query distances in batches to limit memory usage
        batch_size = min(100, data.shape[0])
        all_distances = []
        
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:min(i+batch_size, data.shape[0])]
            # Get distances to k+1 nearest neighbors (first is self with distance=0)
            distances, _ = nn.kneighbors(batch)
            # Skip the first column (distance to self)
            all_distances.append(distances[:, 1:])
            
            # Check memory after each batch
            self._check_memory_limit()
        
        # Flatten all distances
        distances_flat = np.concatenate([d.flatten() for d in all_distances])
        
        # Calculate statistics
        stats = {
            'mean': np.mean(distances_flat),
            'std': np.std(distances_flat),
            'min': np.min(distances_flat),
            'max': np.max(distances_flat),
        }
        stats['ratio_max_min'] = stats['max'] / (stats['min'] + 1e-10)
        
        # Clean up
        del all_distances, distances_flat
        gc.collect()
        
        return stats
        
    def _analyze_distribution_shape_efficient(self, max_samples=1000):
        """
        Analyze shape of distance distribution without storing full distance matrix
        
        Uses random sampling of pairs to estimate the distribution
        """
        # Use reduced sample data if available
        data = self.data_sample_reduced if hasattr(self, 'data_sample_reduced') else self.data_sample
        n_samples = data.shape[0]
        
        # For very small datasets, compute all pairwise distances
        if n_samples <= 100:
            distances = pdist(data)
        else:
            # For larger datasets, sample random pairs
            num_pairs = min(int(n_samples * math.log(n_samples)), max_samples)
            
            # Sample pairs of points randomly
            idx1 = np.random.choice(n_samples, num_pairs)
            idx2 = np.random.choice(n_samples, num_pairs)
            
            # Calculate distances between pairs
            distances = np.array([
                euclidean(data[i], data[j]) 
                for i, j in zip(idx1, idx2)
            ])
        
        # Calculate histogram with adaptive bins
        hist, bin_edges = np.histogram(distances, bins=min(30, int(np.sqrt(len(distances)))), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks in the histogram
        peaks, _ = find_peaks(hist)
        n_peaks = len(peaks)
        
        # Calculate distribution statistics
        mean_val = np.mean(distances)
        std_val = np.std(distances)
        
        if std_val > 0:
            normalized_vals = (distances - mean_val) / std_val
            skewness = np.mean(normalized_vals ** 3)
            kurtosis = np.mean(normalized_vals ** 4) - 3
            del normalized_vals
        else:
            skewness, kurtosis = 0, 0
            
        # Clean up
        del distances
        gc.collect()
        
        return {
            'n_peaks': n_peaks,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'peak_positions': bin_centers[peaks].tolist() if len(peaks) > 0 else []
        }
        
    def _estimate_intrinsic_dim_efficient(self, k=10):
        """
        Estimate intrinsic dimensionality efficiently using nearest neighbors
        
        Uses the Maximum Likelihood Estimation approach on k nearest neighbors
        """
        # Use reduced sample data if available
        data = self.data_sample_reduced if hasattr(self, 'data_sample_reduced') else self.data_sample
        n_samples, n_features = data.shape
        
        # Early return for small datasets
        if n_samples <= k:
            return 1.0
            
        # Adjust k if needed
        k = min(k, n_samples-1)
        
        # Use NearestNeighbors for memory efficiency
        nn = NearestNeighbors(n_neighbors=k+1, algorithm=self.meta_parameters['nn_algorithm'])
        nn.fit(data)
        
        # Process in batches to limit memory usage
        batch_size = min(100, n_samples)
        distance_ratios = []
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:min(i+batch_size, n_samples)]
            # Get distances to k+1 nearest neighbors
            distances, _ = nn.kneighbors(batch)
            
            # Skip the first neighbor (self) and calculate log ratios
            k_distances = distances[:, 1:k+1]
            ratios = np.log(k_distances[:, 1:] / np.clip(k_distances[:, :-1], 1e-10, None))
            distance_ratios.extend(ratios.mean(axis=1))
            
            # Check memory
            self._check_memory_limit()
        
        # Calculate MLE estimate of intrinsic dimension
        dim_estimate = 1.0 / np.mean(distance_ratios)
        
        # Clean up
        del distance_ratios
        gc.collect()
        
        # Clip to reasonable range
        return np.clip(dim_estimate, 1.0, n_features)
    
    def compute_optimal_clusters(self, data=None, constraints=None):
        """
        Compute the optimal number of clusters using memory-efficient approach
        
        Parameters:
        -----------
        data : array-like, optional
            Pattern data if not already set
        constraints : dict, optional
            Additional constraints to consider
            
        Returns:
        --------
        int
            Optimal number of clusters
        dict
            Additional metrics and information
        """
        start_time = time.time()
        
        # Initialize memory tracking
        self._memory_usage_start = self._track_memory(reset_peak=True)
        #print(f"Starting memory usage: {self._memory_usage_start:.2f}GB")
        
        # Set data if provided
        if data is not None:
            self.data = np.array(data)
        
        if self.data is None:
            raise ValueError("No data provided for cluster calculation")
            
        # Apply constraints if provided
        if constraints:
            for key, value in constraints.items():
                if key in self.meta_parameters:
                    self.meta_parameters[key] = value
                    
        # Analyze data if not already done
        if not hasattr(self, 'data_characteristics') or self.data_characteristics.get('n_samples') != self.data.shape[0]:
            self.analyze_data_characteristics(self.data)
            
        # Memory check after analysis
        post_analysis_mem = self._track_memory()
        #print(f"Memory after data analysis: {post_analysis_mem:.2f}GB")
        
        # Extract key characteristics
        n_samples = self.data_characteristics['n_samples']
        n_features = self.data_characteristics['n_features']
        density = self.data_characteristics['density']
        dist_shape = self.data_characteristics['distribution_shape']
        intrinsic_dim = self.data_characteristics['intrinsic_dimensionality']
        
        # Get cluster constraints
        min_clusters = self.meta_parameters['min_clusters']
        max_clusters = self.meta_parameters['max_clusters']
        
        # Formula components
        log_factor = np.log10(max(n_samples, 10))
        
        # Base formula with complexity adjustment
        base_k = np.sqrt(n_samples) / log_factor
        
        # Corrections
        density_correction = self.meta_parameters['alpha'] * np.log1p(density)
        dim_correction = np.sqrt(intrinsic_dim / max(n_features, 1))
        shape_factor = 1 + self.meta_parameters['gamma'] * (dist_shape['kurtosis'] / 4.0)
        
        # Adaptive optimal k calculation
        adaptive_k = base_k * shape_factor * (1 + density_correction) * dim_correction
        
        # Adjust for multi-modal distributions
        if dist_shape['n_peaks'] > 1:
            peak_adjustment = 0.5 * dist_shape['n_peaks']
            adaptive_k = 0.7 * adaptive_k + 0.3 * peak_adjustment
        
        # Bound k within constraints
        k = max(min(int(round(adaptive_k)), max_clusters), min_clusters)
        
        # Validate with sampling for large datasets
        validation_score = None
        if n_samples > 1000:
            # Create validation sample - much smaller than before
            validation_size = min(500, n_samples)
            if n_samples > validation_size:
                indices = np.random.choice(n_samples, validation_size, replace=False)
                validation_data = self.data[indices]
            else:
                validation_data = self.data
            
            # Check memory and apply PCA if needed
            if validation_data.shape[1] > 10 and self.meta_parameters['use_pca']:
                validation_data = self._apply_dimensionality_reduction(validation_data)
            
            # Use MiniBatchKMeans for large datasets
            if self.meta_parameters['use_mini_batch'] and validation_size > 1000:
                # Test a small range around k
                best_score = -1
                best_k = k
                
                # Narrow range when memory constrained
                k_range = [max(k-1, min_clusters), k, min(k+1, max_clusters)]
                
                for k_test in k_range:
                    if k_test < 2:
                        continue
                        
                    kmeans = MiniBatchKMeans(
                        n_clusters=k_test, 
                        batch_size=min(100, validation_size),
                        random_state=42
                    )
                    labels = kmeans.fit_predict(validation_data)
                    
                    if len(np.unique(labels)) > 1:
                        # Calculate silhouette on a subsample if needed
                        if validation_size > 500:
                            subsample_idx = np.random.choice(validation_size, 500, replace=False)
                            score = silhouette_score(
                                validation_data[subsample_idx], 
                                labels[subsample_idx]
                            )
                        else:
                            score = silhouette_score(validation_data, labels)
                            
                        if score > best_score:
                            best_score = score
                            best_k = k_test
                    
                    # Clean up after each iteration
                    self._check_memory_limit()
                
                # Update k based on validation
                if best_score > -1:  # If we found a valid silhouette score
                    validation_score = best_score
                    k = best_k
            else:
                # Use regular KMeans for smaller datasets
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = kmeans.fit_predict(validation_data)
                
                if len(np.unique(labels)) > 1:
                    validation_score = silhouette_score(validation_data, labels)
            
            # Clean up
            del validation_data
            gc.collect()
            
        # Final memory check
        gc.collect()
        end_mem = self._track_memory()
        #print(f"Final memory usage: {end_mem:.2f}GB (Peak: {self._peak_memory:.2f}GB)")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return k, {
            'execution_time': execution_time,
            'data_characteristics': self.data_characteristics,
            'formula_components': {
                'base_k': base_k,
                'density_correction': density_correction,
                'dim_correction': dim_correction,
                'shape_factor': shape_factor
            },
            'validation_score': validation_score,
            'memory_usage': {
                'start_gb': self._memory_usage_start,
                'peak_gb': self._peak_memory,
                'final_gb': end_mem,
                'change_gb': end_mem - self._memory_usage_start
            }
        }
   
def count_optimal_clusters(patterns, method='enhanced', constraints=None):
    """
    Memory-efficient implementation to determine the optimal number of clusters
    
    Parameters:
    -----------
    patterns : array-like
        The patterns to cluster
    method : str, optional
        Clustering method to use (currently only 'enhanced' is supported)
    constraints : dict, optional
        Additional constraints for cluster calculation:
        - 'max_memory_gb': Maximum memory usage in GB (default: 2)
        - 'sample_threshold': Sample data when more than this many points (default: 1000)
        - 'use_pca': Whether to use dimensionality reduction (default: True)
        - 'use_mini_batch': Whether to use mini-batch processing (default: True)
        
    Returns:
    --------
    int
        Optimal number of clusters
    dict
        Additional metrics and information
    """
    # Initialize the enhanced formula
    ef = EnhancedDirectFormula()
    
    # Apply memory constraints
    if constraints is None:
        constraints = {}
    
    # Use the enhanced formula
    n_clusters, info = ef.compute_optimal_clusters(patterns, constraints)
    
    # Clean up
    gc.collect()
    
    return n_clusters, info


def main():
    """
    Main function to demonstrate the memory-efficient optimal clustering
    """
   
    
    print("\n=== Testing with large dataset (memory-optimized) ===")
    large_patterns = np.random.rand(100000, 20)
    n_clusters_large, info_large = count_optimal_clusters(
        large_patterns, 
        constraints={
            'max_memory_gb': 4.0,
            'sample_threshold': 1000,
            'use_pca': True,
            'use_mini_batch': True
        }
    )
    print(f"Optimal clusters for large dataset (n={large_patterns.shape[0]}): {n_clusters_large}")
    print(f"Execution time: {info_large['execution_time']:.4f} seconds")
    print(f"Memory peak: {info_large['memory_usage']['peak_gb']:.2f}GB")
    

if __name__ == "__main__":
    main()