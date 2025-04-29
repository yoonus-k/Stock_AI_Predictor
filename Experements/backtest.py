import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from Data.db_cloud import Database


def backtest_with_cluster_matching(
    db, stock_id, train_start, train_end, test_start, test_end,
    lookback=24, hold_period=6, n_pips=5
):
    # === Load and split data ===
    df = db.get_stock_data(stock_id)
    
    df = df[(df.index >= train_start) & (df.index <= test_end)]
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]
    prices = test_df['ClosePrice'].values
    scaler = MinMaxScaler()

    # === Load cluster data from DB ===
    clusters = db.get_clusters_by_stock_id(stock_id)
    features = clusters['AVGPricePoints'].values
    features = np.array([np.array(x.split(','), dtype=float) for x in features])
    labels = np.array([i for i in range(len(features))])
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(features, labels)

    # === Backtesting Loop ===
    returns = np.zeros(len(test_df))
    equity = [1.0]  # Start with 1 unit
    props =[]
    total_buys = 0
    total_sells = 0
    total_trades = 0
    total_tp_hits = 0
    total_sl_hits = 0
    total_hold_exits = 0
    
    for i in range(lookback, len(test_df) - hold_period):
        # Extract window
        window = prices[i - lookback:i + 1]
        pips_y = extract_pips(window, n_pips=n_pips, dist_type=3)  # Normalized

        if pips_y is None:
            continue
        
        # Predict cluster
        cluster_id = svm.predict(pips_y.reshape(1, -1))[0]
        # check the prbability of the cluster
        prob = svm.predict_proba(pips_y.reshape(1, -1))[0]
      
        # Create a mapping from SVM class IDs to DataFrame indices
        cluster_id_to_index = {id: idx for id, idx in enumerate(clusters.index)}
        actual_index = cluster_id_to_index[cluster_id]
        cluster = clusters.loc[actual_index]
        
        # print cluster info
        stock_id = cluster['StockID']


        # Determine action
        label = cluster['Label']
        outcome = cluster['Outcome']
        max_gain = cluster['MaxGain']
        max_drawdown = cluster['MaxDrawdown']
        reward_risk = abs(cluster['MaxGain']) / (abs(max_drawdown) + 1e-6)
        
        # take trade only if the reward_risk is greater than 1.5
        if reward_risk < 1:
            continue

        entry_price = prices[i]
        
        # simulate trade
        if label == 'Buy':
            exit_price, profit_loss, reason = simulate_trade(
                prices[i:i + hold_period + 1], entry_price, max_gain, max_drawdown, hold_period, trade_type="BUY"
            )
            total_buys += 1
            
        elif label == 'Sell':
            exit_price, profit_loss, reason = simulate_trade(
                prices[i:i + hold_period + 1], entry_price, max_gain, max_drawdown, hold_period, trade_type="SELL"
            )
            total_sells += 1
            
        else:
            continue
        if reason == "TP Hit":
            total_tp_hits += 1
        elif reason == "SL Hit":
            total_sl_hits += 1
        elif reason == "Hold Exit":
            total_hold_exits += 1
     
        trade_return = (exit_price - entry_price) / entry_price

        # BUY or SELL based on cluster label
        if label == 'Buy':
            returns[i + hold_period] = trade_return
        elif label == 'Sell':
            returns[i + hold_period] = -trade_return
        else:
            continue
        
       

        equity.append(equity[-1] * (1 + returns[i + hold_period]))
    
    # print the total trades
    total_trades = total_buys + total_sells
    print(f"Total Buys: {total_buys}, Total Sells: {total_sells}, Total Trades: {total_trades}")
    print(f"Total TP Hits: {total_tp_hits}, Total SL Hits: {total_sl_hits}, Total Hold Exits: {total_hold_exits}")
    # === Final Metrics ===
    equity = equity[:len(test_df)]  # Match length
    returns = returns[:len(test_df)]

    total_return = equity[-1] - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
    cum_returns = np.cumsum(returns)
    max_dd = np.min(cum_returns) - np.max(cum_returns)

    # === Visualization ===
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, np.cumprod(1 + returns[:len(test_df)]), label='Equity Curve')
    plt.title(f'Backtest Performance (Stock ID {stock_id})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Results ===
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'cumulative_returns': cum_returns,
        'returns': returns,
    }


def extract_pips(window, n_pips=5, dist_type=3):
    """
    Extract and normalize PIPs from a price window.
    Returns normalized Y points or None if fails.
    """
    try:
        from Pattern.perceptually_important import find_pips
        x, y = find_pips(window, n_pips, dist_type)
        scaler = MinMaxScaler()
        norm_y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        return norm_y
    except:
        return None

def simulate_trade(prices, entry_price, max_gain, max_drawdown, hold_period, trade_type="BUY"):
        """
        Simulates a trade and returns exit price, PnL, and outcome.

        Args:
            prices (list): future price window starting from entry index
            entry_price (float)
            max_gain (float): e.g. 0.03 = +3%
            max_drawdown (float): e.g. 0.01 = -1%
            hold_period (int)
            trade_type (str): 'BUY' or 'SELL'

        Returns:
            exit_price (float), profit_loss (float), reason (str)
        """
        if len(prices) < hold_period:
            return entry_price, 0, "Insufficient data"

        for j in range(1, hold_period + 1):
            current_price = prices[j]

            # === BUY Logic ===
            if trade_type == "BUY":
                # Take Profit
                if current_price >= entry_price * (1 + max_gain):
                    return current_price, current_price - entry_price, "TP Hit"
                # Stop Loss
                elif current_price <= entry_price * (1 + max_drawdown):
                    return current_price, current_price - entry_price, "SL Hit"

            # === SELL Logic ===
            elif trade_type == "SELL":
                # Take Profit (price drops)
                if current_price <= entry_price * (1 - max_gain):
                    return current_price, entry_price - current_price, "TP Hit"
                # Stop Loss (price rises)
                elif current_price >= entry_price * (1 + max_drawdown):
                    return current_price, entry_price - current_price, "SL Hit"

        # Hold period end
        final_price = prices[hold_period]
        if trade_type == "BUY":
            return final_price, final_price - entry_price, "Hold Exit"
        else:
            return final_price, entry_price - final_price, "Hold Exit"



db = Database()
result = backtest_with_cluster_matching(
    db=db,
    stock_id=1,
    # train_start=pd.Timestamp("2019-01-02 01:00:00"),
    # train_end=pd.Timestamp("2024-01-08 01:00:00"),
    # test_start=pd.Timestamp("2024-01-09 01:00:00"),
    # test_end=pd.Timestamp("2025-04-10 23:00:00"),
    train_start=pd.Timestamp("2019-01-02 01:00:00"),
    train_end=pd.Timestamp("2024-01-08 01:00:00"),
    test_start=pd.Timestamp("2024-01-09 01:00:00"),
    test_end=pd.Timestamp("2025-01-15 23:00:00"),
    lookback=24,
    hold_period=12,
    n_pips=5
)

# print the result in formatted string , remove the float64

print(f"Total Return: {result['total_return']:.2f}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result['max_drawdown']:.2f}")
print(f"Cumulative Returns: {result['cumulative_returns'][-1]:.2f}")
print(f"Returns: {result['returns'][-1]:.2f}")


