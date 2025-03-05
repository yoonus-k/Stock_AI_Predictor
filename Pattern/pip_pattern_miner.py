import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from perceptually_important import find_pips
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Pattern_Miner:

    def __init__(self, n_pips: int, lookback: int, hold_period: int):
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        
        self._unique_pip_patterns = []
        self._unique_pip_indices = []
        self._cluster_centers = []
        self._pip_clusters = []

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
            plot_pip_x, plot_pip_y = find_pips(data_slice['Close'].to_numpy(), self._n_pips, 3)
            
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
    
    
    def train(self, arr: np.array, n_reps=-1):
        self._data = arr
        self._returns = pd.Series(arr).diff().shift(-1)
        self._find_unique_patterns()
        

        search_instance = silhouette_ksearch(
                self._unique_pip_patterns, 5, 40, algorithm=silhouette_ksearch_type.KMEANS).process()
        
        amount = search_instance.get_amount()
        self._kmeans_cluster_patterns(amount)

        self._get_cluster_signals()
        self._assign_clusters()
        self._fit_martin = self._get_total_performance()
        
        print(self._fit_martin)

        if n_reps <= 1:
            return

        # Start monte carlo permutation test
        data_copy = self._data.copy()
        returns_copy = self._returns.copy()
        
        for rep in range(1, n_reps):
            x = np.diff(data_copy).copy()
            np.random.shuffle(x)
            x = np.concatenate([np.array([data_copy[0]]), x])
            self._data = np.cumsum(x)
            self._returns = pd.Series(self._data).diff().shift(-1)
            print("rep", rep) 
            self._find_unique_patterns()
            search_instance = silhouette_ksearch(
                    self._unique_pip_patterns, 5, 40, algorithm=silhouette_ksearch_type.KMEANS).process()
            amount = search_instance.get_amount()
            self._kmeans_cluster_patterns(amount)
            self._get_cluster_signals()
            self._assign_clusters()
            perm_martin = self._get_total_performance()
            self._perm_martins.append(perm_martin)


    def _find_unique_patterns(self):
        self._unique_pip_indices.clear()
        self._unique_pip_patterns.clear()
        
        last_pips_x = [0] * self._n_pips
        for i in range(self._lookback - 1, len(self._data) - self._hold_period):
            start_i = i - self._lookback + 1
            window = self._data[start_i: i + 1]
            pips_x, pips_y = find_pips(window, self._n_pips, 3)
            pips_x = [j + start_i for j in pips_x]  # Convert to global index

            # Check internal pips to see if it is the same as last
            same = True
            for j in range(1, self._n_pips - 1):
                if pips_x[j] != last_pips_x[j]:
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
        
        # print("Cluster Centers")
        # print(self._cluster_centers)
        # print("Cluster Patterns")
        # print(self._pip_clusters)
        
        #self.plot_clusters()
        #self.plot_cluster_members(0)
        
        
        
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
    
    def evaluate_clusters(self):
        cluster_metrics = []
        for cluster_i in range(len(self._pip_clusters)):
            cluster_indices = self._pip_clusters[cluster_i]
            cluster_returns = []
            
            for idx in cluster_indices:
                pattern_end = self._unique_pip_indices[idx]
                future_return = self._returns[pattern_end: pattern_end + self._hold_period].sum()
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
    
    def filter_clusters(self, cluster_metrics, buy_threshold=0.03, sell_threshold=-0.03):
        buy_clusters = []
        sell_clusters = []
        
        for metrics in cluster_metrics:
            # if the average return is positive and the win rate is above 50% then buy
            if metrics['avg_return'] > buy_threshold and metrics['win_rate'] > 50:
                buy_clusters.append(metrics['cluster'])
            # if the average return is negative and the win rate is above 50% then sell
            elif metrics['avg_return'] < sell_threshold and metrics['win_rate'] > 50:
                sell_clusters.append(metrics['cluster'])
        
        return buy_clusters, sell_clusters
    
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
        strategy_returns = signals * self._returns
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
        
    def plot_backtest_results(self, cumulative_returns):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Cumulative Returns')
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()
    

    
if __name__ == '__main__':
    data = pd.read_csv('C:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Mining_ML_Lib/BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    # trim the data to only include the first 50 data points
    data = data.head(1000)
    #data = np.log(data)
    
    # plot the price chart
    #plt.plot(data['close'], label='Close Price')
    
    #data = data[data.index < '01-01-2020']
    arr = data['close'].to_numpy()
    
    # rolling windows of size 24 candles and plot using ploting function
  
    start_i = 20
    window = arr[start_i: start_i + 24]
    pips_x, pips_y = find_pips(window, 5, 3)
    pips_x = [j + start_i for j in pips_x] # Convert to global index

    # get the indecies of the dataframe corresponding to the pips_x
    indeces = data.index[pips_x]
  
    
    pip_miner = Pattern_Miner(n_pips=5, lookback=24, hold_period=6)
    pip_miner.train(arr, n_reps=-1)
    
    
    # Evaluate clusters
    cluster_metrics = pip_miner.evaluate_clusters()
   
    # print the cluster metrics in formatteed form
    # for cluster in cluster_metrics:
    #     print(f"Cluster {cluster['cluster']}:")
    #     print(f"Average Return: {cluster['avg_return']:.4f}")
    #     print(f"Win Rate: {cluster['win_rate']:.2f}%")
    #     print(f"Sharpe Ratio: {cluster['sharpe_ratio']:.4f}")
    #     print(f"Max Drawdown: {cluster['max_drawdown']:.4f}")
    #     print()

    # Filter clusters
    buy_clusters, sell_clusters = pip_miner.filter_clusters(cluster_metrics)
    print("Buy Clusters:", buy_clusters)
    print("Sell Clusters:", sell_clusters)
    
    # plot the buy and sell clusters
    # for cluster in buy_clusters:
    #     pip_miner.plot_cluster_by_index(cluster, 'buy')
    # for cluster in sell_clusters:
    #     pip_miner.plot_cluster_by_index(cluster, 'sell')

    # Backtest
    backtest_results = pip_miner.backtest(buy_clusters, sell_clusters)

    # Plot results
    pip_miner.plot_backtest_results(backtest_results['cumulative_returns'])
   

    
    # Monte Carlo test, takes about an hour..
    #pip_miner.train(arr, n_reps=1)
    
    # plt.style.use('dark_background')
    # actual_martin = pip_miner.get_fit_martin()
    # perm_martins = pip_miner.get_permutation_martins()
    # ax = pd.Series(perm_martins).hist()
    # ax.set_ylabel("# Of Permutations")
    # ax.set_xlabel("Martin Ratio")
    # ax.set_title("Permutation's Martin Ratio BTC-USDT 1H 2018-2020")
    # ax.axvline(actual_martin, color='red')
    

    
    





    

