import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
    
    # train the SVM model
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(features, labels)
    # train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)

    # === Backtesting Loop ===
    returns = np.zeros(len(test_df))
    equity = [1.0]  # Start with 1 unit
    trade_results = []  # To store individual trade results
    
    # Variables for tracking unique patterns
    last_pips_x = [0] * n_pips
    unique_patterns_seen = []
    
    for i in range(lookback, len(test_df) - hold_period):
        # Extract window
        window = prices[i - lookback:i + 1]
        pips_x, pips_y = extract_pips(window, n_pips=n_pips, dist_type=3)
        
        if pips_y is None:
            continue
        
        # Convert to global indices
        start_i = i - lookback
        global_pips_x = [x + start_i for x in pips_x]
        
        # Check if this is a unique pattern
        is_unique = True
        for j in range(1, n_pips - 1):
            if global_pips_x[j] == last_pips_x[j]:
                is_unique = False
                break
                
        if not is_unique:
            continue
            
        # Normalize the pattern
        scaler = MinMaxScaler()
        pips_y_normalized = scaler.fit_transform(np.array(pips_y).reshape(-1, 1)).flatten()
        
        # Store this pattern for future comparison
        last_pips_x = global_pips_x
        unique_patterns_seen.append(pips_y_normalized.tolist())
        
        # Predict cluster
        cluster_id = svm.predict(pips_y_normalized.reshape(1, -1))[0]
        prob = svm.predict_proba(pips_y_normalized.reshape(1, -1))[0]
      
        cluster_id_to_index = {id: idx for id, idx in enumerate(clusters.index)}
        actual_index = cluster_id_to_index[cluster_id]
        cluster = clusters.loc[actual_index]
        
        # Determine action
        label = cluster['Label']
        outcome = cluster['Outcome']
        max_gain = cluster['MaxGain']
        max_drawdown = cluster['MaxDrawdown']
        reward_risk = abs(cluster['MaxGain']) / (abs(max_drawdown) + 1e-6)
        
        if reward_risk < 1:
            continue

        entry_price = prices[i]
        entry_time = test_df.index[i]
        
        # simulate trade
        if label == 'Buy':
            exit_price, profit_loss, reason = simulate_trade(
                prices[i:i + hold_period + 1], entry_price, max_gain, max_drawdown, hold_period, trade_type="BUY"
            )
            trade_type = "BUY"
        elif label == 'Sell':
            exit_price, profit_loss, reason = simulate_trade(
                prices[i:i + hold_period + 1], entry_price, max_gain, max_drawdown, hold_period, trade_type="SELL"
            )
            trade_type = "SELL"
        else:
            continue
            
        # Calculate trade metrics
        trade_return_pct = (profit_loss / entry_price) * 100
        trade_duration = min(hold_period, len(prices[i:i + hold_period + 1]))
        
        # Store trade details
        trade_results.append({
            'entry_time': entry_time,
            'exit_time': test_df.index[i + trade_duration],
            'type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': trade_return_pct,
            'profit_loss': profit_loss,
            'outcome': 'win' if profit_loss > 0 else 'loss',
            'reason': reason,
            'max_gain': max_gain,
            'max_drawdown': max_drawdown,
            'reward_risk': reward_risk,
            'duration': trade_duration
        })
        
        # Update returns array
        if label == 'Buy':
            returns[i + trade_duration] = profit_loss / entry_price
        elif label == 'Sell':
            returns[i + trade_duration] = -profit_loss / entry_price
        
        equity.append(equity[-1] * (1 + returns[i + trade_duration]))
    
    # === Calculate Performance Metrics ===
    if not trade_results:
        print("No trades were executed during the backtest period.")
        return None
    
    trades_df = pd.DataFrame(trade_results)
    
    # Basic metrics
    total_return_pct = (equity[-1] - 1) * 100
    num_trades = len(trades_df)
    num_wins = len(trades_df[trades_df['outcome'] == 'win'])
    win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
    
    # Risk/Reward metrics
    avg_win = trades_df[trades_df['outcome'] == 'win']['return_pct'].mean()
    avg_loss = trades_df[trades_df['outcome'] == 'loss']['return_pct'].mean()
    avg_rrr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Profit factor
    gross_profit = trades_df[trades_df['outcome'] == 'win']['profit_loss'].sum()
    gross_loss = abs(trades_df[trades_df['outcome'] == 'loss']['profit_loss'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown calculations
    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    daily_returns = pd.Series(returns).dropna()
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
    
    # === Print Comprehensive Results ===
    print("\n=== Backtest Results ===")
    print(f"Time Period: {test_start.date()} to {test_end.date()}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: {avg_win:.2f}% | Average Loss: {avg_loss:.2f}%")
    print(f"Average Risk/Reward: {avg_rrr:.2f}:1")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Unique Patterns Used: {len(unique_patterns_seen)}")
    
    # Breakdown by trade type
    if 'type' in trades_df.columns:
        print("\n=== By Trade Type ===")
        for trade_type in trades_df['type'].unique():
            type_trades = trades_df[trades_df['type'] == trade_type]
            type_win_rate = (len(type_trades[type_trades['outcome'] == 'win']) / len(type_trades)) * 100
            print(f"{trade_type} Trades: {len(type_trades)} | Win Rate: {type_win_rate:.2f}%")
    
    # Breakdown by exit reason
    print("\n=== Exit Reasons ===")
    for reason in trades_df['reason'].unique():
        reason_trades = trades_df[trades_df['reason'] == reason]
        reason_win_rate = (len(reason_trades[reason_trades['outcome'] == 'win']) / len(reason_trades)) * 100
        print(f"{reason}: {len(reason_trades)} trades | Win Rate: {reason_win_rate:.2f}%")
    
    # === Visualization ===
    plt.figure(figsize=(14, 8))
    
    # Equity curve
    
    plt.plot(test_df.index[:len(equity)], equity, label='Equity Curve')
    plt.title(f'Backtest Performance (Stock ID {stock_id}) - Total Return: {total_return_pct:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    # # Drawdown
    # plt.subplot(2, 1, 2)
    # plt.plot(test_df.index[:len(drawdown)], drawdown * 100, label='Drawdown', color='red')
    # plt.title(f'Drawdown (Max: {max_drawdown_pct:.2f}%)')
    # plt.xlabel('Date')
    # plt.ylabel('Drawdown (%)')
    # plt.grid(True)
    # plt.legend()
    
    plt.tight_layout()
    plt.show()

    # === Return Detailed Results ===
    return {
        'total_return_pct': total_return_pct,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'avg_rrr': avg_rrr,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'trades_df': trades_df,
        'equity_curve': equity,
        'unique_patterns_used': len(unique_patterns_seen),
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
        return x,norm_y
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
    stock_id=3,
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



