import pandas as pd
import numpy as np
import warnings

# Add a warnings attribute to numpy if it doesn't exist
if not hasattr(np, 'warnings'):
    np.warnings = warnings

print("NumPy warnings patch applied successfully")
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Try to import TensorFlow for GPU acceleration
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, using CPU calculations only")


class Pattern_Miner:
    """
    Pattern_Miner: A class for identifying, clustering, and analyzing price patterns in financial data.
    
    This class implements the Perceptually Important Points (PIP) algorithm to identify key points
    in time series data, clusters similar patterns, and analyzes their predictive performance.
    "
    """
    def __init__(self, n_pips: int=5, lookback: int=24, hold_period: int=6, returns_hold_period: int=6, distance_measure: int=2):
        """
        Initialize the Pattern_Miner with configuration parameters.
        
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
        """
        # Configuration parameters
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._returns_hold_period = returns_hold_period
        self._distance_measure = distance_measure
        
        # GPU support flag - will be set by the parameter tester if GPU is available
        self.use_gpu = False
        
        # Pattern storage
        self._unique_pip_patterns = []  # List of unique patterns, each pattern is a list of normalized price points
        self._unique_pip_indices = []   # List of the last indices of the unique patterns in the original data
        self._global_pip_indices = []   # List of each pattern's global indices in the source data
        
        # Clustering results
        self._cluster_centers = []      # List of cluster centers (centroids)
        self._pip_clusters = []         # List of clusters, each cluster is a list of indices into unique_pip_patterns
        
        # some statistics
        self._max_patterns_count = None # Maximum number of patterns in any cluster
        self._avg_patterns_count = None # Average number of patterns in all clusters

        # Signal generation
        self._cluster_signals = []
        self._cluster_objs = []
        self._long_signal = None
        self._short_signal = None
        self._selected_long = []        # Indices of clusters with positive expected returns
        self._selected_short = []       # Indices of clusters with negative expected returns
        self._selected_neutral = []     # Indices of clusters with zero expected returns

        # Performance analysis
        self._fit_martin = None
        self._perm_martins = []
        
        # Data and returns
        self._data = None               # Source price data array to mine patterns from
        self._returns_next_candle = None    # Array of next-candle returns
        self._returns_fixed_hold = None     # Array of fixed holding period returns
        self._returns_mfe = None            # Maximum favorable excursion returns
        self._returns_mae = None            # Maximum adverse excursion returns
        self._cluster_returns = []          # Mean returns for each cluster
        self._cluster_mfe = []              # Mean MFE for each cluster
        self._cluster_mae = []              # Mean MAE for each cluster

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
                    # Euclidean Distance:
                    # Use when you need the shortest path between two points.
                    # Example: Calculating the distance between two cities on a map.
                    
                    # Perpendicular Distance:
                    # Use when you need the shortest distance from a point to a line.
                    # Example: Finding the closest point on a road to a given location.
                    
                    # Vertical Distance:
                    # Use when you care only about the vertical difference between a point and a line.
                    # Example: Measuring the error between observed and predicted values in regression analysis.
                    
                    d = 0.0  # Distance
                    if dist_measure == 1:  # Euclidean distance
                        d =  ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5  # Left distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                        d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5  # Right distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                    elif dist_measure == 2:  # Perpendicular distance
                        d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5  # Perpendicular distance formula : |Ax + By + C| / (A^2 + B^2)^0.5
                    else:  # Vertical distance    
                        d = abs((slope * i + intercept) - data[i])  # Vertical distance formula : |Ax + By + C| 

                    # Keep track of the point with maximum distance
                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            # Insert the point with max distance into PIPs
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])

        return pips_x, pips_y

    def _find_unique_patterns(self):
        """
        Find unique patterns in the data by identifying PIPs and normalizing them.
        Stores patterns in internal arrays for later processing.
        """
        self._unique_pip_indices.clear()
        self._unique_pip_patterns.clear()
        self._global_pip_indices = []     # Store ALL global indices for each pattern
        
        last_pips_x = [0] * self._n_pips
        # Slide window through the data
        for i in range(self._lookback - 1, len(self._data) - self._hold_period):
            start_i = i - self._lookback + 1
            window = self._data[start_i: i + 1]
            
            # Find PIPs in this window
            pips_x, pips_y = self.find_pips(window, self._n_pips, 3)
            # Convert to global indices
            global_pips_x = [j + start_i for j in pips_x]  # Convert to global index

            # Check if this pattern is the same as the last one
            same = True
            for j in range(1, self._n_pips - 1):
                if global_pips_x[j] != last_pips_x[j]:
                    same = False
                    break
            
            # If this is a new pattern, store it
            if not same:
                data = np.array(pips_y).reshape(-1, 1)
                # Create scalers
                minmax_scaler = MinMaxScaler()
                std_scaler = StandardScaler()
                
                # Normalize the pattern using min-max scaling
                normalized = minmax_scaler.fit_transform(data).flatten()
                standardized = std_scaler.fit_transform(data).flatten()
                
                # Store the normalized pattern and its indices
                self._unique_pip_patterns.append(normalized.tolist())
                self._unique_pip_indices.append(i)  # Index of the last point in the pattern
                self._global_pip_indices.append(global_pips_x)  # All global PIP indices

            last_pips_x = global_pips_x

    #----------------------------------------------------------------------------------------
    # Clustering Functions
    #----------------------------------------------------------------------------------------    
    def _kmeans_cluster_patterns(self, amount_clusters):
        """
        Cluster the patterns using K-means algorithm.
        
        Parameters:
        -----------
        amount_clusters : int
            Number of clusters to create
        """
        # Initialize cluster centers using k-means++
        initial_centers = kmeans_plusplus_initializer(self._unique_pip_patterns, amount_clusters).initialize()
        
        # Use GPU acceleration if available
        if TF_AVAILABLE and hasattr(self, 'use_gpu') and self.use_gpu:
            try:
                print("Using GPU acceleration for clustering...")
                
                # Convert data to TensorFlow tensors
                data_points = np.array(self._unique_pip_patterns)
                centers = np.array(initial_centers)
                
                # Perform K-means clustering with GPU acceleration
                max_iterations = 100
                tolerance = 1e-4
                
                for _ in range(max_iterations):
                    # Calculate distances using GPU
                    data_points_tf = tf.convert_to_tensor(data_points, dtype=tf.float32)
                    centers_tf = tf.convert_to_tensor(centers, dtype=tf.float32)
                    
                    # Expand dimensions for broadcasting
                    expanded_data = tf.expand_dims(data_points_tf, 1)  # Shape [n_samples, 1, n_features]
                    expanded_centers = tf.expand_dims(centers_tf, 0)  # Shape [1, n_clusters, n_features]
                    
                    # Calculate squared distances
                    distances = tf.reduce_sum(tf.square(expanded_data - expanded_centers), axis=2)
                    
                    # Assign points to nearest cluster
                    cluster_assignments = tf.argmin(distances, axis=1).numpy()
                    
                    # Update centers
                    new_centers = []
                    for i in range(amount_clusters):
                        # Get points in this cluster
                        cluster_points = data_points[cluster_assignments == i]
                        if len(cluster_points) > 0:
                            # Calculate new center
                            new_centers.append(np.mean(cluster_points, axis=0))
                        else:
                            # Keep old center if no points assigned
                            new_centers.append(centers[i])
                    
                    # Check for convergence
                    new_centers = np.array(new_centers)
                    center_shift = np.sum(np.square(centers - new_centers))
                    centers = new_centers
                    
                    if center_shift < tolerance:
                        break
                
                # Store cluster information
                self._cluster_centers = centers
                self._pip_clusters = [[] for _ in range(amount_clusters)]
                
                # Assign patterns to clusters
                for i, pattern_idx in enumerate(range(len(self._unique_pip_patterns))):
                    cluster_idx = cluster_assignments[i]
                    self._pip_clusters[cluster_idx].append(pattern_idx)
                
               
                
                print(f"GPU-accelerated clustering completed with {amount_clusters} clusters")
                return
            except Exception as e:
                print(f"GPU clustering failed: {e}, falling back to CPU")
          # Fall back to CPU clustering if GPU fails or is unavailable
        kmeans_instance = kmeans(self._unique_pip_patterns, initial_centers)
        kmeans_instance.process()
        
        # Get clustering information
        clusters = kmeans_instance.get_clusters()
        centers = kmeans_instance.get_centers()
        
        # Set up class variables for later use
        self._cluster_centers = centers
        self._pip_clusters = clusters
        
        # Calculate silhouette score
        

        # Extract clustering results: clusters and their centers
        self._pip_clusters = kmeans_instance.get_clusters()
        self._cluster_centers = kmeans_instance.get_centers()

    def _categorize_clusters_by_mean_return(self):
        """
        Categorize clusters into long, short, and neutral based on mean returns.
        
        Long clusters: mean return > 0
        Short clusters: mean return < 0
        Neutral clusters: mean return == 0
        """
        self._cluster_returns.clear()  # Clear previous returns
        self._selected_long = []  # Clear previous selections
        self._selected_short = []
        self._selected_neutral = []  # Stores neutral clusters
        
        for clust_i, clust in enumerate(self._pip_clusters):
            # Get returns for all patterns in this cluster
            returns = self._returns_fixed_hold[clust]
            mfe = self._returns_mfe[clust]
            mae = self._returns_mae[clust]
            mean_return = np.mean(returns)
            mean_mfe = np.mean(mfe)
            mean_mae = np.mean(mae)
            
            # Store the cluster returns
            self._cluster_returns.append(mean_return)
            self._cluster_mfe.append(mean_mfe)
            self._cluster_mae.append(mean_mae)
            
            # Categorize based on mean return direction
            if mean_return > 0:
                self._selected_long.append(clust_i)
            elif mean_return < 0:
                self._selected_short.append(clust_i)
            else:
                self._selected_neutral.append(clust_i)

    #----------------------------------------------------------------------------------------
    # Return Calculation Functions
    #----------------------------------------------------------------------------------------
    
    # function to get the maximum patterns count 
    def get_max_patterns_count(self) -> int:
        """
        Get the maximum number of patterns identified.
        
        Returns:
        --------
        int
            Maximum number of patterns
        """
        #loop through the unique patterns and get the maximum count
        max_count = 0
        for i in range(len(self._pip_clusters)):
            if len(self._pip_clusters[i]) > max_count:
                max_count = len(self._pip_clusters[i])
        return max_count    
    
    # funtion to get the average patterns count
    def get_avg_patterns_count(self) -> int:
        """
        Get the average number of patterns identified.
        
        Returns:
        --------
        int
            Average number of patterns
        """
        #loop through the unique patterns and get the average count
        avg_count = 0
        for i in range(len(self._pip_clusters)):
            avg_count += len(self._pip_clusters[i])
        return avg_count / len(self._pip_clusters)

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

    def _get_martin(self, rets: np.array):
        """
        Calculate the Martin ratio (a risk-adjusted return measure).
        
        Parameters:
        -----------
        rets : np.array
            Array of returns
            
        Returns:
        --------
        float
            Martin ratio
        """
        rsum = np.sum(rets)
        short = False
        if rsum < 0.0:
            rets *= -1
            rsum *= -1
            short = True

        csum = np.cumsum(rets)
        eq = pd.Series(np.exp(csum))
        sumsq = np.sum(((eq / eq.cummax()) - 1) ** 2.0)
        ulcer_index = (sumsq / len(rets)) ** 0.5
        martin = rsum / ulcer_index
        if short:
            martin = -martin

        return martin

    def calculate_distances_gpu(self, data_points, cluster_centers):
        """Calculate distances between data points and cluster centers using GPU acceleration.
        
        Parameters:
        -----------
        data_points : numpy.ndarray
            Array of data points with shape (n_samples, n_features)
        cluster_centers : numpy.ndarray
            Array of cluster centers with shape (n_clusters, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix with shape (n_samples, n_clusters)
        """
        if not TF_AVAILABLE or not self.use_gpu:
            return None
            
        try:
            # Convert to TensorFlow tensors
            data_points_tf = tf.convert_to_tensor(data_points, dtype=tf.float32)
            centers_tf = tf.convert_to_tensor(cluster_centers, dtype=tf.float32)
            
            # Compute distances (squared Euclidean distance)
            expanded_data = tf.expand_dims(data_points_tf, 1)  # Shape [n_samples, 1, n_features]
            expanded_centers = tf.expand_dims(centers_tf, 0)  # Shape [1, n_clusters, n_features]
            
            # Calculate squared distances
            distances = tf.reduce_sum(tf.square(expanded_data - expanded_centers), axis=2)
            
            # Convert back to NumPy
            return distances.numpy()
        except Exception as e:
            print(f"GPU distance calculation error: {e}")
            return None

    #----------------------------------------------------------------------------------------
    # Training and Prediction Functions
    #----------------------------------------------------------------------------------------

    def train(self, arr: np.array):
        """
        Train the pattern miner on a price data array.
        
        Parameters:
        -----------
        arr : np.array
            Price data array
        """
        self._data = arr
        
        # Find patterns
        self._find_unique_patterns()
        
        # Calculate the returns of the data
        self._returns_next_candle = pd.Series(arr).diff().shift(-1)  # Calculate the returns of the data
        self._returns_fixed_hold = self.calculate_returns_fixed_hold(arr, self._unique_pip_indices, self._returns_hold_period)  # Calculate the fixed holding period returns
        self._returns_mfe, self._returns_mae = self.calculate_mfe_mae(arr, self._unique_pip_indices, self._returns_hold_period)  # Calculate the MFE and MAE returns
       
        # Fully dynamic cluster range calculation
        pattern_count = len(self._unique_pip_patterns)
       
        # Calculate sqrt(n) as a statistical rule of thumb for initial clustering
        sqrt_n = int(np.sqrt(pattern_count))
        
        # Adaptive min clusters: sqrt(n)/2
        min_clusters = max(3, int(sqrt_n/2))
        
        # Adaptive max clusters: Between sqrt(n)*2 and 20% of patterns
        max_clusters = min(int(sqrt_n)*2,pattern_count-1)
        
        # Ensure min < max and both are within valid range
        min_clusters = min(min_clusters, pattern_count - 1)
        max_clusters = min(max_clusters, pattern_count - 1)
        max_clusters = max(max_clusters, min_clusters + 2)  # Ensure reasonable range
        
        # Use silhouette method to find optimal number of clusters
        search_instance = silhouette_ksearch(
                self._unique_pip_patterns, min_clusters, max_clusters, algorithm=silhouette_ksearch_type.KMEANS).process()
        
        amount = search_instance.get_amount()
        self._kmeans_cluster_patterns(amount)
        self._categorize_clusters_by_mean_return()
        
        self._max_patterns_count = self.get_max_patterns_count()  # Get the maximum patterns count
        self._avg_patterns_count = self.get_avg_patterns_count()  # Get the average patterns count

    def predict(self, pips_y: list, current_price: float):
        """
        Predict future price based on a pattern.
        
        Parameters:
        -----------
        pips_y : list
            List of price points in the pattern
        current_price : float
            Current price
            
        Returns:
        --------
        float
            Predicted price
        """
        norm_y = (np.array(pips_y) - np.mean(pips_y)) / np.std(pips_y)

        # Find the closest cluster
        best_dist = 1.e30
        best_clust = -1
        for clust_i in range(len(self._cluster_centers)):
            center = np.array(self._cluster_centers[clust_i])
            dist = np.linalg.norm(norm_y-center)
            if dist < best_dist:
                best_dist = dist
                best_clust = clust_i
        
        # Get the return of the best cluster
        best_cluster_return = self._returns[best_clust]
        # Calculate the predicted price
        predicted_price = current_price + (best_cluster_return * current_price)
        return predicted_price

    #----------------------------------------------------------------------------------------
    # Evaluation and Analysis Functions
    #----------------------------------------------------------------------------------------

    def filter_clusters(self, buy_threshold=0.03, sell_threshold=-0.03):
        """
        Filter clusters based on return thresholds.
        
        Parameters:
        -----------
        buy_threshold : float
            Minimum return threshold for buy clusters
        sell_threshold : float
            Maximum return threshold for sell clusters
            
        Returns:
        --------
        tuple
            (buy_clusters, sell_clusters) lists of cluster indices
        """
        buy_clusters = []
        sell_clusters = []
        
        # Filter the long clusters
        for clust_i in self._selected_long:
            mean_return = np.mean(self._returns_fixed_hold[self._pip_clusters[clust_i]])
            if mean_return > buy_threshold:
                buy_clusters.append(clust_i)
                
        # Filter the short clusters
        for clust_i in self._selected_short:
            mean_return = np.mean(self._returns_fixed_hold[self._pip_clusters[clust_i]])
            if mean_return < sell_threshold:
                sell_clusters.append(clust_i)
        
        return buy_clusters, sell_clusters

    def evaluate_clusters(self):
        """
        Evaluate the performance of each cluster.
        
        Returns:
        --------
        list
            List of dictionaries containing performance metrics for each cluster
        """
        cluster_metrics = []
        for cluster_i in range(len(self._pip_clusters)):
            cluster_indices = self._pip_clusters[cluster_i]
            cluster_returns = []
            
            for idx in cluster_indices:
                pattern_end = self._unique_pip_indices[idx]
                future_return = self._returns_next_candle[pattern_end: pattern_end + self._hold_period].sum()
                cluster_returns.append(future_return)
            
            # Calculate metrics
            avg_return = np.mean(cluster_returns)
            win_rate = np.mean(np.array(cluster_returns) > 0) * 100
            sharpe_ratio = np.mean(cluster_returns) / np.std(cluster_returns) if np.std(cluster_returns) != 0 else 0
            max_drawdown = np.min(cluster_returns) - np.max(cluster_returns)
            
            cluster_metrics.append({
                'cluster': cluster_i,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })
        
        return cluster_metrics
    
    def backtest(self, buy_clusters, sell_clusters):
        """
        Backtest a strategy based on selected clusters.
        
        Parameters:
        -----------
        buy_clusters : list
            List of cluster indices to generate buy signals
        sell_clusters : list
            List of cluster indices to generate sell signals
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        signals = np.zeros(len(self._data))
        for cluster_i in buy_clusters:
            for idx in self._pip_clusters[cluster_i]:
                pattern_end = self._unique_pip_indices[idx]
                signals[pattern_end: pattern_end + self._hold_period] = 1  # Buy signal
        
        for cluster_i in sell_clusters:
            for idx in self._pip_clusters[cluster_i]:
                pattern_end = self._unique_pip_indices[idx]
                signals[pattern_end: pattern_end + self._hold_period] = -1  # Sell signal
        
        # Calculate returns
        strategy_returns = signals * self._returns_next_candle
        cumulative_returns = np.cumsum(strategy_returns)
        
        # Calculate performance metrics
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) != 0 else 0
        max_drawdown = np.min(cumulative_returns) - np.max(cumulative_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }

    #----------------------------------------------------------------------------------------
    # Getter Methods
    #----------------------------------------------------------------------------------------
        
    def get_fit_martin(self):
        """Get the Martin ratio for the fitted model."""
        return self._fit_martin

    def get_permutation_martins(self):
        """Get the Martin ratios for permutation tests."""
        return self._perm_martins

    #----------------------------------------------------------------------------------------
    # Visualization Functions
    #----------------------------------------------------------------------------------------

    def plot_backtest_results(self, cumulative_returns):
        """
        Plot the cumulative returns from a backtest.
        
        Parameters:
        -----------
        cumulative_returns : np.array
            Array of cumulative returns
        """
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns')
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_clusters(self):
        """Plot all cluster centers."""
        plt.figure(figsize=(10, 6))
        for i, center in enumerate(self._cluster_centers):
            plt.plot(center, label=f'Cluster {i+1}', marker='o' if i == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Cluster Centers')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_cluster_by_index(self, index, type):
        """
        Plot a specific cluster by its index.
        
        Parameters:
        -----------
        index : int
            Index of the cluster to plot
        type : str
            'buy' or 'sell'
        """
        plt.plot(self._cluster_centers[index], label=f'Pattern {index+1}', marker='o' if index == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        
        if type == 'buy':
            plt.title(f'Plot of Cluster {index} Buy')
        else:
            plt.title(f'Plot of Cluster {index} Sell')
            
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_cluster_members(self, cluster_i):
        """
        Plot all members of a specific cluster.
        
        Parameters:
        -----------
        cluster_i : int
            Index of the cluster to plot members for
        """
        for i in self._pip_clusters[cluster_i]:
            plt.plot(self._unique_pip_patterns[i], label=f'Pattern {i+1}', marker='o' if i == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Cluster Members')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_cluster_examples(self, candle_data: pd.DataFrame, cluster_i: int, grid_size: int = 5):
        """
        Plot examples of a cluster using candlestick charts.
        
        Parameters:
        -----------
        candle_data : pd.DataFrame
            Candlestick data with OHLC values
        cluster_i : int
            Index of the cluster to plot examples for
        grid_size : int
            Size of the grid for subplot layout
        """
        plt.style.use('dark_background')
        fig, axs = plt.subplots(grid_size, grid_size)
        flat_axs = axs.flatten()
        for i in range(len(flat_axs)):
            if i >= len(self._pip_clusters[cluster_i]):
                break
            
            pat_i = self._unique_pip_indices[self._pip_clusters[cluster_i][i]]
            data_slice = candle_data.iloc[pat_i - self._lookback + 1: pat_i + 1]
            idx = data_slice.index
            plot_pip_x, plot_pip_y = self.find_pips(data_slice['Close'].to_numpy(), self._n_pips, 3)
            
            pip_lines = []
            colors = []
            for line_i in range(self._n_pips - 1):
                l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]), (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
                pip_lines.append(l0)
                colors.append('w')

            mpf.plot(data_slice, type='candle',alines=dict(alines=pip_lines, colors=colors), ax=flat_axs[i], style='charles', update_width_config=dict(candle_linewidth=1.75) )
            flat_axs[i].set_yticklabels([])
            flat_axs[i].set_xticklabels([])
            flat_axs[i].set_xticks([])
            flat_axs[i].set_yticks([])
            flat_axs[i].set_ylabel("")

        fig.suptitle(f"Cluster {cluster_i}", fontsize=32)
        plt.show()
    
    def plot_cluster_by_center(self, center, type):
        """
        Plot a pattern based on a provided center.
        
        Parameters:
        -----------
        center : list
            Center coordinates to plot
        type : str
            'buy' or 'sell'
        """
        # Plot informational labels
        plt.plot(0.4, label='Confidence Score: 0.7%', color='green', marker='.')
        plt.plot(0.4, label='Type: Bullish', color='green', marker='.')
        plt.plot(0.4, label='Max Expected Drawdown: - 0.3%', color='red', marker='.')
        
        # Plot the pattern
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.plot(center, label=f'Matched Pattern ID: #1543', marker='o' if 0 == 0 else 'x')
        
        if type == 'buy':
            plt.title(f'Plot of Cluster Buy')
        else:
            plt.title(f'Plot of Cluster Sell')
            
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    query = """
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM stock_data
        WHERE stock_id = ? AND timeframe_id = ?
    """
    params = [1, 5]
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    
    # Add date filters if provided
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    import sqlite3
    query += " ORDER BY timestamp"
    conn = sqlite3.connect("./Data/Storage/data.db")
    # Execute query and fetch data
    cursor =conn.cursor()
    cursor.execute(query, params)
    data = cursor.fetchall()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
     # trim the data to only include the first 50 data points
    data = df.head(1000)
  
    #data = data[data.index < '01-01-2020']
    arr = data['close'].to_numpy()

    pip_miner = Pattern_Miner(n_pips=5, lookback=24, hold_period=6)
    pip_miner.train(arr)
    
    # print max patterns count
    print(f"Max Patterns Count: {pip_miner._max_patterns_count}")
    conn.close()
    # plot the clusters
    pip_miner.plot_clusters()














