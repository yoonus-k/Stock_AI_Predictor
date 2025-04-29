import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Pattern_Miner:

    def __init__(self, n_pips: int=5, lookback: int=24, hold_period: int=6 , returns_hold_period: int=6):
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._returns_hold_period = returns_hold_period
        
        self._unique_pip_patterns = [] # List of unique patterns, each pattern is a list of the normalized price points
        self._unique_pip_indices = [] # List of the last indices of the unique patterns in the original data
        self._global_pip_indices = [] # List of each pattern's global indices, each pattern is a list of the global indices of the price points in the pattern like [0, 1, 2, 3, 4] for the first pattern and [5, 6, 7, 8, 9] for the second pattern
        self._cluster_centers = [] # List of cluster centers, each center is a list of the unique price points of the patterns centered around the center
        self._pip_clusters = [] # List of clusters, each cluster is a list of indices of the unique patterns array

        self._cluster_signals = []
        self._cluster_objs = []

        self._long_signal = None
        self._short_signal = None

        self._selected_long = []
        self._selected_short = []

        self._fit_martin = None
        self._perm_martins = []
        
        self._data = None # Array of log closing prices to mine patterns
        
        self._returns_next_candle = None # Array of next log returns, concurrent with _data
        self._returns_fixed_hold = None # Array of fixed hold returns, concurrent with _data
        self._returns_mfe = None # Array of maximum favorable excursion returns, concurrent with _data
        self._returns_mae = None # Array of maximum adverse excursion returns, concurrent with _data
        self._cluster_returns = [] # Array of cluster returns
        self._cluster_mfe = [] # Array of cluster maximum favorable excursion returns
        self._cluster_mae = [] # Array of cluster maximum adverse excursion returns

    # find pips funtion
    def find_pips(self,data: np.array, n_pips: int, dist_measure: int):
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
                slope = price_diff / time_diff # Slope
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope; # y = mx + c

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
                    
                    d = 0.0 # Distance
                    if dist_measure == 1: # Euclidean distance
                        d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5 # Left distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                        d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5 # Right distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                    elif dist_measure == 2: # Perpindicular distance
                        d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5 # Perpindicular distance formula : |Ax + By + C| / (A^2 + B^2)^0.5
                    else: # Vertical distance    
                        d = abs( (slope * i + intercept) - data[i] ) # Vertical distance formula : |Ax + By + C| 

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


    def predict(self, pips_y: list, current_price: float):
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
        
        # get the return of the best cluster
        best_cluster_return = self._returns[best_clust]
        # calculate the predicted price
        predicted_price = current_price + (best_cluster_return * current_price)
        return predicted_price
    
    ##-------------- Functions to calculate the returns of each candle stick in the data --------------##
    ## ---------------------------------------------------------------------------------- ##
    # Fixed Holding Period Returns
    def calculate_returns_fixed_hold(self,data: np.array, pattern_indices: list, hold_period: int) -> np.array:
        returns = []
        for idx in pattern_indices:
            if idx + hold_period < len(data):
                returns.append((data[idx + hold_period] - data[idx]) / data[idx])
            else:
                returns.append(0)  # Not enough future data
        return np.array(returns)
    
    # Maximum Favorable Excursion (MFE) / Maximum Adverse Excursion (MAE)
    # Definition:
    # MFE: Best unrealized gain during the trade.
    # MAE: Worst unrealized loss during the trade.
    def calculate_mfe_mae(self,data: np.array, pattern_indices: list, hold_period: int) -> tuple:
        mfe, mae = [], []
        for idx in pattern_indices:
            if idx + hold_period < len(data):
                window = data[idx: idx + hold_period]
                max_price = np.max(window)
                min_price = np.min(window)
                mfe.append((max_price - data[idx]) / data[idx]) # This will return the maximum favorable excursion
                mae.append((min_price - data[idx]) / data[idx]) # This will return the maximum adverse excursion
            else:
                mfe.append(0)
                mae.append(0)
        return np.array(mfe), np.array(mae)
    
    def train(self, arr: np.array):
        self._data = arr
        
        self._find_unique_patterns()
        
        # Calculate the returns of the data
        self._returns_next_candle = pd.Series(arr).diff().shift(-1) # Calculate the returns of the data
        self._returns_fixed_hold = self.calculate_returns_fixed_hold(arr, self._unique_pip_indices, self._returns_hold_period) # Calculate the fixed holding period returns
        self._returns_mfe, self._returns_mae = self.calculate_mfe_mae(arr, self._unique_pip_indices, self._returns_hold_period) # Calculate the MFE and MAE returns
       
    
        search_instance = silhouette_ksearch(
                self._unique_pip_patterns, 30, 100, algorithm=silhouette_ksearch_type.KMEANS).process()
        
        amount = search_instance.get_amount()
        self._kmeans_cluster_patterns(amount)
        self._categorize_clusters_by_mean_return()
        

    def _find_unique_patterns(self):
        self._unique_pip_indices.clear()
        self._unique_pip_patterns.clear()
        self._global_pip_indices = []     # New: Store ALL global indices for each pattern
        
        last_pips_x = [0] * self._n_pips
        for i in range(self._lookback - 1, len(self._data) - self._hold_period):
            start_i = i - self._lookback + 1
            window = self._data[start_i: i + 1]
            pips_x, pips_y = self.find_pips(window, self._n_pips, 3)
            global_pips_x = [j + start_i for j in  pips_x]  # Convert to global index by adding start_i to each index inglobal_pips_x list 

            # Check internal pips to see if it is the same as last
            same = True
            for j in range(1, self._n_pips - 1):
                if global_pips_x[j] != last_pips_x[j]:
                    same = False
                    break
            
            if not same:
                data = np.array(pips_y).reshape(-1, 1)
                # Create scalers
                minmax_scaler = MinMaxScaler()
                std_scaler = StandardScaler()
                # Min-Max normalization to [0, 1]
                # Fit and transform
                normalized = minmax_scaler.fit_transform(data).flatten()
                standardized = std_scaler.fit_transform(data).flatten()
                
                self._unique_pip_patterns.append(normalized.tolist())
                self._unique_pip_indices.append(i) # Append the index of the pattern , this is the index of the last pip in the pattern
                self._global_pip_indices.append(global_pips_x)  # New: All global PIP indices

            last_pips_x =global_pips_x


    def _kmeans_cluster_patterns(self, amount_clusters):
        # Cluster Patterns
        initial_centers = kmeans_plusplus_initializer(self._unique_pip_patterns, amount_clusters).initialize()
        kmeans_instance = kmeans(self._unique_pip_patterns, initial_centers)
        kmeans_instance.process()

        # Extract clustering results: clusters and their centers
        self._pip_clusters = kmeans_instance.get_clusters()
        self._cluster_centers = kmeans_instance.get_centers()
        

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

    def _categorize_clusters_by_mean_return(self):
        self._cluster_returns.clear()  # Clear previous returns
        """Categorizes clusters into long, short, and neutral based on mean returns.
        
        Long clusters: mean return > 0
        Short clusters: mean return < 0
        Neutral clusters: mean return == 0
        """
        self._selected_long = []  # Clear previous selections
        self._selected_short = []
        self._selected_neutral = []  # New: Stores neutral clusters
        
        for clust_i, clust in enumerate(self._pip_clusters):
            # Get returns for all patterns in this cluster
            returns = self._returns_fixed_hold[clust]
            mfe = self._returns_mfe[clust]
            mae = self._returns_mae[clust]
            mean_return = np.mean(returns)
            mean_mfe = np.mean(mfe)
            mean_mae = np.mean(mae)
            
            # store the cluster returns
            self._cluster_returns.append(mean_return)
            self._cluster_mfe.append(mean_mfe)
            self._cluster_mae.append(mean_mae)
            
            if mean_return > 0:
                self._selected_long.append(clust_i)
            elif mean_return < 0:
                self._selected_short.append(clust_i)
            else:
                self._selected_neutral.append(clust_i)
        
        
    def filter_clusters(self,buy_threshold=0.03, sell_threshold=-0.03):
        buy_clusters = []
        sell_clusters = []
        
        # filter the long clusters
        for clust_i in self._selected_long:
            mean_return = np.mean(self._returns_fixed_hold[self._pip_clusters[clust_i]])
            if mean_return > buy_threshold:
                buy_clusters.append(clust_i)
                
        # filter the short clusters
        for clust_i in self._selected_short:
            mean_return = np.mean(self._returns_fixed_hold[self._pip_clusters[clust_i]])
            if mean_return < sell_threshold:
                sell_clusters.append(clust_i)
        
        return buy_clusters, sell_clusters

    
    def evaluate_clusters(self):
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
        #total_return = cumulative_returns[-1]
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) != 0 else 0
        max_drawdown = np.min(cumulative_returns) - np.max(cumulative_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
            }
        
    ### ------------- Plotting Functions ------------- ###
    def plot_backtest_results(self, cumulative_returns):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns')
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()
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
        
    # funtion to print a cluster by its index
    def plot_cluster_by_index(self, index, type):
        
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
        # plot each member of the cluster number 1
        #print("Cluster Members")
        #print(self._pip_clusters)
        for i in self._pip_clusters[cluster_i]:
            
            plt.plot(self._unique_pip_patterns[i], label=f'Pattern {i+1}', marker='o' if i == 0 else 'x')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title('Plot of Cluster 1 Members')
        plt.legend()
        plt.grid()
        plt.show()
        
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
        
    # funtion to plot the cluster by the cluster centers as input
    def plot_cluster_by_center(self, center, type):
       
        # plot the confidence score label of the cluster of 0.7%
        plt.plot(0.4, label='Confidence Score: 0.7%', color='green', marker='.')
        # plot the type label of the cluster ( bullish or bearish )
        plt.plot(0.4, label='Type: Bullish', color='green', marker='.')
        
        # plot the maximum expected drawdown of the cluster of 0.7%
        plt.plot(0.4, label='Max Expected Drawdown: - 0.3%', color='red', marker='.')
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
    data = pd.read_csv('C:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Stocks/BTCUSD60.csv')
    data['Date'] = data['Date'].astype('datetime64[s]')
    data = data.set_index('Date')
    # trim the data to only include the first 50 data points
    data = data.head(1000)
   
    #data = data[data.index < '01-01-2020']
    arr = data['Close'].to_numpy()
    
    pip_miner = Pattern_Miner(n_pips=5, lookback=24, hold_period=6)
    #pip_miner.train(arr)
    
    


    pip_miner.plot_cluster_by_center([0.4,0.9,0.3,0.6,0.2], 'buy')
    
    
  


    
    





    

