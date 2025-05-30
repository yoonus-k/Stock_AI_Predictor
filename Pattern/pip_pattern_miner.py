import cProfile
import os
import sys
from memory_profiler import profile
import pstats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the enhanced direct formula
from Pattern.enhanced_direct_formula import EnhancedDirectFormula, count_optimal_clusters


class Pattern_Miner:
    """
    Pattern_Miner: A class for identifying, clustering, and analyzing price patterns in financial data.
    
    This class implements the Perceptually Important Points (PIP) algorithm to identify key points
    in time series data, clusters similar patterns, and analyzes their predictive performance.
    
    The class supports two methods for determining the optimal number of clusters:
    1. Silhouette Method: Traditional approach using silhouette scores
    2. Enhanced Direct Formula: An advanced approach that analyzes data characteristics 
       to determine optimal cluster count with better silhouette scores
    """
    def __init__(self,cluster_method= 'enhanced', n_pips: int=5, lookback: int=24, hold_period: int=6, returns_hold_period: int=6, distance_measure: int=2):
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
        
        Notes:
        ------
        When using the train() method, you can specify the clustering method with cluster_method parameter:
        - 'silhouette': Traditional silhouette score method (default)
        - 'enhanced': Enhanced direct formula method for better silhouette scores and computational efficiency
        
        Example:
        --------
        ```python
        pip_miner = Pattern_Miner(n_pips=5, lookback=24)
        pip_miner.train(price_data, cluster_method='enhanced')  # Use enhanced direct formula
        ```
        """
        # Configuration parameters
        self._cluster_method = cluster_method  # Clustering method to use: 'silhouette' or 'enhanced'
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._returns_hold_period = returns_hold_period
        self._distance_measure = distance_measure
          
        
        # Configuration parameters
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._returns_hold_period = returns_hold_period
        
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
                #std_scaler = StandardScaler()
                
                # Normalize the pattern using min-max scaling
                normalized = minmax_scaler.fit_transform(data).flatten()
                #standardized = std_scaler.fit_transform(data).flatten()
                
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
        initial_centers = kmeans_plusplus_initializer(self._unique_pip_patterns, amount_clusters,random_state=1).initialize()
        kmeans_instance = kmeans(self._unique_pip_patterns, initial_centers)
        kmeans_instance.process()

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
        cluster_method : str, optional
            Method to use for determining optimal number of clusters:
            - 'silhouette': Use silhouette score method (default)
            - 'enhanced': Use enhanced direct formula method
        """
        self._data = arr
       
        # Find patterns
        self._find_unique_patterns()
       
        # Calculate the returns of the data
        self._returns_next_candle = pd.Series(arr).diff().shift(-1)  # Calculate the returns of the data
        self._returns_fixed_hold = self.calculate_returns_fixed_hold(arr, self._unique_pip_indices, self._returns_hold_period)  # Calculate the fixed holding period returns
        self._returns_mfe, self._returns_mae = self.calculate_mfe_mae(arr, self._unique_pip_indices, self._returns_hold_period)  # Calculate the MFE and MAE returns
       
        # Get the unique patterns as numpy array for clustering
        patterns = np.array(self._unique_pip_patterns)
        pattern_count = len(patterns)
        
        # Determine optimal number of clusters based on the selected method
        if self._cluster_method == 'enhanced':
           
            # Use enhanced direct formula method
            print("Using Enhanced Direct Formula method for cluster count selection")
            #amount, info = count_optimal_clusters(patterns, method='enhanced')
            amount = int(pattern_count*0.3);
            
            print(f"Enhanced Direct Formula selected {amount} clusters")
        else:
            # Use the default silhouette method
            print("Using Silhouette method for cluster count selection")
            
            # Fully dynamic cluster range calculation
            # Calculate sqrt(n) as a statistical rule of thumb for initial clustering
            sqrt_n = int(np.sqrt(pattern_count))
            
            # Adaptive min clusters: sqrt(n)/2
            min_clusters = max(3, int(sqrt_n/2))
            
            # Adaptive max clusters: Between sqrt(n)*2 and 20% of patterns
            max_clusters = min(int(sqrt_n)*2, pattern_count-1)
            
            # Ensure min < max and both are within valid range
            min_clusters = min(min_clusters, pattern_count - 1)
            max_clusters = min(max_clusters, pattern_count - 1)
            max_clusters = max(max_clusters, min_clusters + 2)  # Ensure reasonable range
            
            # Use silhouette method to find optimal number of clusters
            search_instance = silhouette_ksearch(patterns, min_clusters, max_clusters, 
                                                algorithm=silhouette_ksearch_type.KMEANS).process()
            
            amount = search_instance.get_amount()
            print(f"Silhouette method selected {amount} clusters")
        
        self._kmeans_cluster_patterns(amount)
        self._categorize_clusters_by_mean_return()
        
        self._max_patterns_count = self.get_max_patterns_count()  # Get the maximum patterns count
        self._avg_patterns_count = self.get_avg_patterns_count()  # Get the average patterns count

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
    data = pd.read_csv('D:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Raw/Stocks/Intraday/15M/BTCUSD15.csv')
    data['Date'] = data['Date'].astype('datetime64[s]')
    data = data.set_index('Date')
    # trim the data to only include the first 5000 data points
    data = data.head(1000000)
   
    #data = data[data.index < '01-01-2020']
    arr = data['Close'].to_numpy()
    
   
    # Now with enhanced direct formula
    pip_miner_enhanced = Pattern_Miner(n_pips=3, lookback=24, hold_period=6 , cluster_method='enhanced')
  
    
    print("\n===== Using Enhanced Direct Formula =====")
    with cProfile.Profile() as pr_enhanced:
        print("Training the PIP Miner with Enhanced Direct Formula...")
        pip_miner_enhanced.train(arr)
        print("Training completed.")
        
    stats_enhanced = pstats.Stats(pr_enhanced)
    stats_enhanced.sort_stats(pstats.SortKey.TIME)
    stats_enhanced.dump_stats('pip_miner_enhanced_profile.prof')
    # print number of patterns found
    print(f"Unique patterns found: {len(pip_miner_enhanced._unique_pip_patterns)}")
    print(f"Enhanced Direct Formula cluster count: {len(pip_miner_enhanced._pip_clusters)}")
    pip_miner_enhanced.plot_cluster_members(0)  # Plot members of the first cluster as an example
    

    
    # Run in terminal to visualize performance:
    # snakeviz .\pip_miner_silhouette_profile.prof
    # snakeviz .\pip_miner_enhanced_profile.prof
    
 












