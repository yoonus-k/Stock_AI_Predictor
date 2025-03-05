# **1. Data Handling and Computation**
import pandas as pd  # Used for handling and manipulating stock market data
import numpy as np  # Provides support for numerical operations and array handling
import math  # Includes mathematical functions for pattern calculations

# **2. Data Visualization**
import matplotlib.pyplot as plt  # Core visualization library for plotting graphs
import mplfinance as mpf  # Specialized library for visualizing stock price movements (candlestick charts)

# **3. Clustering and Pattern Recognition**
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch  
# Used to determine the optimal number of clusters in K-Means clustering

from pyclustering.cluster.kmeans import kmeans  
# K-Means clustering algorithm for grouping similar stock price patterns

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer  
# Initializes cluster centers optimally for K-Means algorithm to improve convergence

# **4. Data Preprocessing and Scaling**
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
# MinMaxScaler: Scales data to a given range (e.g., 0 to 1)
# StandardScaler: Standardizes data by removing mean and scaling to unit variance

# **5. Perceptually Important Points (PIP)**
import perceptually_important  
# Implements the Perceptually Important Points (PIP) algorithm for pattern extraction

# **6. Additional Visualization**
import matplotlib.pyplot as plt  # (Redundant but included again for visualization tasks)



class PIPPatternMiner:

    def __init__(self, n_pips: int, lookback: int, hold_period: int):
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        
        self._unique_pip_patterns = [] # Unique patterns
        self._unique_pip_indices = [] # Indices of unique patterns
        self._cluster_centers = [] # Cluster centers
        self._pip_clusters = [] # Cluster patterns

        self._cluster_signals = []
        self._cluster_objs = []

        self._long_signal = None
        self._short_signal = None

        self._selected_long = []
        self._selected_short = []

        self._fit_martin = None
        self._perm_martins = []
        
        self._data = None # Array of log closing prices to mine patterns
        self._returns = None # Array of next log returns, concurrent with _data
        self._max_drawdown = None # Array of max drawdowns for each pattern
        self._n_ahead = 1 # number of periods (legs) ahead to calculate return
        
        self.minmax_scaler = MinMaxScaler()
        self.std_scaler = StandardScaler()

    @staticmethod
    def find_pips(data: np.array, n_pips: int, dist_measure: int):
        # dist_measure
        # 1 = Euclidean Distance
        # 2 = Perpindicular Distance
        # 3 = Vertical Distance

        pips_x = [0, len(data) - 1]  # Index
        pips_y = [data[0], data[-1]] # Price

        for curr_point in range(2, n_pips):
            md = 0.0 # Max distance
            md_i = -1 # Max distance index
            insert_index = -1
            for k in range(0, curr_point - 1):
                # Left adjacent, right adjacent indices
                left_adj = k
                right_adj = k + 1

                time_diff = pips_x[right_adj] - pips_x[left_adj] # Time difference
                price_diff = pips_y[right_adj] - pips_y[left_adj] # Price difference
                slope = price_diff / time_diff # Slope of the line
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope; # Intercept of the line

                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]): # Iterate through points
                    
                    d = 0.0 # Distance
                    if dist_measure == 1: # Euclidean distance
                        d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                        d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                    elif dist_measure == 2: # Perpindicular distance
                        d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                    else: # Vertical distance    
                        d = abs( (slope * i + intercept) - data[i] )

                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])
        return pips_x, pips_y

    def get_fit_martin(self):
        return self._fit_martin

    def get_permutation_martins(self):
        return self._perm_martins

    def plot_cluster_examples(self, candle_data: pd.DataFrame, cluster_i: int, grid_size: int = 5):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(grid_size, grid_size)
        flat_axs = axs.flatten()
        for i in range(len(flat_axs)):
            if i >= len(self._pip_clusters[cluster_i]):
                break
            
            pat_i = self._unique_pip_indices[self._pip_clusters[cluster_i][i]]
            data_slice = candle_data.iloc[pat_i - self._lookback + 1: pat_i + 1]
            idx = data_slice.index
            plot_pip_x, plot_pip_y = perceptually_important.find_pips(data_slice['close'].to_numpy(), self._n_pips, 3)
            
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


    def predict(self, pips_y: list):
        norm_y = (np.array(pips_y) - np.mean(pips_y)) / np.std(pips_y)

        # Find cluster
        best_dist = 1.e30
        best_clust = -1
        for clust_i in range(len(self._cluster_centers)):
            center = np.array(self._cluster_centers[clust_i])
            dist = np.linalg.norm(norm_y-center)
            if dist < best_dist:
                best_dist = dist
                best_clust = clust_i

        if best_clust in self._selected_long:
            return 1.0
        elif best_clust in self._selected_short:
            return -1.0
        else:
            return 0.0
    def calculate_returns_after_n_candles(self,data: np.array, pattern_indices: list, n_candles: int) -> np.array:
        """
        Calculate the percentage return after `n_candles` for each pattern.

        Parameters:
            data (np.array): Array of price data.
            pattern_indices (list): List of indices where patterns occur.
            n_candles (int): Number of candles after the pattern to calculate the return.

        Returns:
            np.array: Array of returns for each pattern.
        """
        returns = []
        for idx in pattern_indices:
            if idx+self._lookback - 1 + n_candles - self._hold_period < len(data):
                price_at_pattern = data[idx+self._lookback - 1- self._hold_period]
                price_after_n_candles = data[idx + self._lookback - 1 - self._hold_period+ n_candles]
                return_percent = ((price_after_n_candles - price_at_pattern) / price_at_pattern) * 100
                returns.append(return_percent)
            else:
                returns.append(0)  # If there are not enough candles after the pattern, return 0
        return np.array(returns)
    
    # function to calculate the returns for the last leg of the pattern
    def calculate_returns_for_last_leg(self) -> np.array:
        returns = []
        for pattern in self._unique_pip_patterns:
            # get the last n-1 elements of the pattern
            before_last_element = pattern[len(pattern)-2]
            # get the last element of the pattern
            last_element = pattern[len(pattern)-1]
           
            # calculate the return
            return_change = last_element - before_last_element
            returns.append(return_change)
        return np.array(returns)
    def calculate_max_drawdown_for_patterns(self)-> np.array:
        """
        Calculate the maximum drawdown for each pattern before hitting the target return.
        
        
        Returns:
            np.array: Array of returns for each pattern.the maximum drawdowns.
        """
        max_drawdowns = []

        for pattern_idx in range(len(self._unique_pip_patterns)):
            pattern_start = self._unique_pip_indices[pattern_idx]
            pattern_end = pattern_start + self._lookback - 1 - self._hold_period
            
            # get the return for the pattern
            
            pattern_return = self._returns[pattern_idx]

            # Initialize variables to track drawdown
            max_drawdown = 0.0
            if pattern_end  < len(data):
                peak = self._data[pattern_end]
            end_price = peak

            # Iterate through the data after the pattern ends
            for i in range(pattern_end + 1, len(self._data)):
                current_price = self._data[i]
                
                # check if the target return is positive or negative
                if pattern_return >= 0: # positive target return
                    if current_price < peak:
                        peak = current_price
                           
                else: # negative target return
                    if current_price > peak:
                        peak = current_price
                        
                # Calculate drawdown
                if pattern_return > 0:
                    drawdown = (peak - end_price) / peak
                else:
                    drawdown = (end_price- peak) / peak
                
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
                # get the absolute value of the drawdown
                
                # Check if target return is reached
                if abs(((current_price - self._data[pattern_end]) / self._data[pattern_end]) * 100) >=abs(pattern_return):
                    break

            max_drawdowns.append(max_drawdown)

        return max_drawdowns
    def plot_pattern_with_dots(self,data: np.array, pattern_start_idx: int, n_candles: int):
        """
        Plot the original price data with a red dot at the end of the pattern and a blue dot at the next n-th candle.

        Parameters:
            data (np.array): Original price data.
            pattern_end_idx (int): Index of the end of the pattern.
            n_candles (int): Number of candles after the pattern to calculate the return.
        """
        # Step 1: Plot the original price data
        plt.figure(figsize=(14, 7))
        plt.plot(data[0:100], label='Original Price', color='blue')

        # Step 2: Plot a red dot at the start
        plt.scatter(pattern_start_idx, data[pattern_start_idx], color='red', label='Pattern start', marker='o', s=100)
        
        pattern_end_idx = pattern_start_idx + self._lookback - 1 - self._hold_period
        
        # Step 3: Plot a blue dot at the next n-th candle
        plt.scatter(pattern_end_idx, data[pattern_end_idx], color='red', label='Pattern end', marker='o', s=100)
        # Step 3: Plot a blue dot at the next n-th candle
        if pattern_end_idx + n_candles < len(data):
            plt.scatter(
                pattern_end_idx+ n_candles,
                data[pattern_end_idx + n_candles],
                color='blue',
                label=f'After {n_candles} Candles',
                marker='o',
                s=100
            )

            # Step 4: Calculate and annotate the percentage change
            price_at_pattern_end = data[pattern_end_idx]
            price_after_n_candles = data[pattern_end_idx + n_candles]
            return_percent = ((price_after_n_candles - price_at_pattern_end) / price_at_pattern_end) * 100

            # Annotate the percentage change
            plt.annotate(
                f'{return_percent:.2f}%',
                xy=(pattern_start_idx + n_candles, price_after_n_candles),
                xytext=(pattern_start_idx + n_candles, price_after_n_candles + 10),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12,
                color='green'
            )

        # Add labels, title, and legend
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.title('Pattern End and Next N-th Candle on Original Chart')
        plt.legend()
        plt.grid()
        plt.show()
        
        # plot the first pattern in the first pattern
        plt.figure(figsize=(10, 6))
        plt.plot(self._unique_pip_patterns[2], label=f'Pattern 1', marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Unique Pip Patterns')
        plt.legend()
        plt.grid()
        plt.show()
    def train(self, arr: np.array, n_reps=-1):
        self._data = arr
        
        self._find_unique_patterns()
        
        # print the _unique_pip_patterns
        # print(self._unique_pip_patterns)
        
        # Convert to a NumPy array
        data_array = np.array(self._unique_pip_patterns)

        
        # Plot each pattern on the same chart
        # plt.figure(figsize=(10, 6))
        # for i, pattern in enumerate(data_array):
        # plt.plot(pattern, label=f'Pattern {i+1}', marker='o' if i == 0 else 'x')  # Use different markers for each pattern

        # Add labels, title, and legend
        # plt.xlabel('Time Steps')
        # plt.ylabel('Value')
        # plt.title('Plot of Unique Pip Patterns')
        # plt.legend()
        # plt.grid()
        # plt.show()
        

        search_instance = silhouette_ksearch(
                self._unique_pip_patterns, 5, 40, algorithm=silhouette_ksearch_type.KMEANS).process()
        
        amount = search_instance.get_amount()
        self._kmeans_cluster_patterns(amount)

        #self._get_cluster_signals()
        #self._assign_clusters()
        
        # calculate the returns after n candles
        #self._returns = self.calculate_returns_after_n_candles(arr, self._unique_pip_indices, 6)
        # calculate the returns for the last leg of the pattern
        self._returns = self.calculate_returns_after_n_candles(arr, self._unique_pip_indices, 6)
        
        print(self._returns[2])
        
        #self._fit_martin = self._get_total_performance()
        
        # calculate the max drawdown for each pattern
        self._max_drawdown = self.calculate_max_drawdown_for_patterns()
        print(self._max_drawdown[2])
        
        # plot the first pattern in the first pattern 
        
        pattern_start_idx = self._unique_pip_indices[2]  # End index of the first pattern
        n_candles = 6  # Number of candles after the pattern to calculate the return
        self.plot_pattern_with_dots(arr, pattern_start_idx, n_candles)
       

        


    def _find_unique_patterns(self):
        self._unique_pip_indices.clear()
        self._unique_pip_patterns.clear()
        
        last_pips_x = [0] * self._n_pips
        for i in range(self._lookback - 1, len(self._data) - self._hold_period):
            start_i = i - self._lookback + 1
            window = self._data[start_i: i + 1]
            pips_x, pips_y = perceptually_important.find_pips(window, self._n_pips, 3)
            pips_x = [j + start_i for j in pips_x]  # Convert to global index

            # Check internal pips to see if it is the same as last
            same = True
            for j in range(1, self._n_pips - 1):
                if pips_x[j] != last_pips_x[j]:
                    same = False
                    break
            
            if not same:
                data = np.array(pips_y).reshape(-1, 1)

                # Fit and transform
                normalized = self.minmax_scaler.fit_transform(data).flatten()
                standardized = self.minmax_scaler.fit_transform(data).flatten()
                
                self._unique_pip_patterns.append(normalized.tolist())
                self._unique_pip_indices.append(i)

            last_pips_x = pips_x


    def _kmeans_cluster_patterns(self, amount_clusters):
        # Cluster Patterns
        initial_centers = kmeans_plusplus_initializer(self._unique_pip_patterns, amount_clusters).initialize()
        kmeans_instance = kmeans(self._unique_pip_patterns, initial_centers)
        kmeans_instance.process()

        # Extract clustering results: clusters and their centers
        self._pip_clusters = kmeans_instance.get_clusters()
        self._cluster_centers = kmeans_instance.get_centers()
        
        
    def plot_clusters(self):
        # plot the cluster centers
        plt.figure(figsize=(10, 6))
        for i, center in enumerate(self._cluster_centers):
            plt.plot(center, label=f'Cluster {i+1}', marker='o' if i == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Cluster Centers')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_cluster_members(self, cluster_i):
        # plot each member of the cluster number 1
        print("Cluster Members")
        print(self._pip_clusters)
        for i in self._pip_clusters[cluster_i]:
            
            plt.plot(self._unique_pip_patterns[i], label=f'Pattern {i+1}', marker='o' if i == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Cluster 1 Members')
        plt.legend()
        plt.grid()
        plt.show()

    def _get_martin(self, rets: np.array):
        rsum = np.sum(rets)
        short = False
        if rsum < 0.0:
            rets *= -1
            rsum *= -1
            short = True

        csum = np.cumsum(rets)
        eq = pd.Series(np.exp(csum))
        sumsq = np.sum( ((eq / eq.cummax()) - 1) ** 2.0 )
        ulcer_index = (sumsq / len(rets)) ** 0.5
        martin = rsum / ulcer_index
        if short:
            martin = -martin

        return martin

    def _get_cluster_signals(self):
        self._cluster_signals.clear()

        for clust in self._pip_clusters: # Loop through each cluster
            signal = np.zeros(len(self._data))
            for mem in clust: # Loop through each member in cluster
                arr_i = self._unique_pip_indices[mem]
                
                # Fill signal with 1s following pattern identification
                # for hold period specified
                signal[arr_i: arr_i + self._hold_period] = 1. 
            
            self._cluster_signals.append(signal)

    def _assign_clusters(self):
        self._selected_long.clear()
        self._selected_short.clear()
        
        # Assign clusters to long/short/neutral
        cluster_martins = []
        for clust_i in range(len(self._pip_clusters)): # Loop through each cluster
            sig = self._cluster_signals[clust_i]
            sig_ret = self._returns * sig
            martin = self._get_martin(sig_ret)
            cluster_martins.append(martin)

        best_long = np.argmax(cluster_martins)
        best_short = np.argmin(cluster_martins)
        self._selected_long.append(best_long)
        self._selected_short.append(best_short)

    def _get_total_performance(self):

        long_signal = np.zeros(len(self._data))
        short_signal = np.zeros(len(self._data))

        for clust_i in range(len(self._pip_clusters)):
            if clust_i in self._selected_long:
                long_signal += self._cluster_signals[clust_i]
            elif clust_i in self._selected_short:
                short_signal += self._cluster_signals[clust_i]
        
        long_signal /= len(self._selected_long)
        short_signal /= len(self._selected_short)
        short_signal *= -1

        self._long_signal = long_signal
        self._short_signal = short_signal
        rets = (long_signal + short_signal) * self._returns

        martin = self._get_martin(rets)
        return martin
    

    
if __name__ == '__main__':
    data = pd.read_csv('C:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    # trim the data to only include the first 50 data points
    data = data.head(100)

    #data = data[data.index < '01-01-2020']
    arr = data['close'].to_numpy()
    
   

    pip_miner = PIPPatternMiner(n_pips=5, lookback=24, hold_period=6)
    pip_miner.train(arr, n_reps=-1)
    

    
 
    
  

    
  

    
    





    

