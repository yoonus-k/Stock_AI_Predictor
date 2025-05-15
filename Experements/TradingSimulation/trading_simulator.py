"""
Trading Strategy Simulator

This module simulates trading strategy performance based on statistical parameters
like win rate, reward-to-risk ratio, and position sizing. It generates equity curves
and performance metrics to evaluate strategy profitability and risk.

The simulator uses Monte Carlo methods to create realistic trading outcomes based on
the probabilistic nature of trading strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import style
import pandas as pd
from scipy.stats import bernoulli


# === Strategy Configuration Parameters ===
PROFIT_FACTOR = 1.54          # Ratio of gross profits to gross losses
WIN_RATE = 0.5337             # Percentage of winning trades (53.37%)
REWARD_RISK_RATIO = 1.33      # Average winner size / average loser size
RISK_PER_TRADE = 0.01         # Risk 1% of capital per trade
NUM_TRADES = 1000             # Number of trades to simulate
INITIAL_BALANCE = 10000       # Starting account balance

# === Derived Parameters ===
LOSS_RATE = 1 - WIN_RATE
RISK_REWARD_RATIO = 1 / REWARD_RISK_RATIO  # Converts to risk:reward format

# Verify the profit factor calculation matches input (sanity check)
calculated_pf = (WIN_RATE * REWARD_RISK_RATIO) / LOSS_RATE
print(f"Calculated Profit Factor: {calculated_pf:.2f} (Should match {PROFIT_FACTOR})")

# === Matplotlib Configuration for Professional Visualization ===
style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def simulate_trading():
    """
    Simulates a trading strategy based on statistical parameters.
    
    Generates a sequence of trade outcomes (win/loss) based on the win rate probability
    and calculates the resulting equity curve.
    
    Returns:
        tuple: (balance_history, trade_results)
            - balance_history: List of account balances after each trade
            - trade_results: List of P&L for each individual trade
    """
    balance = INITIAL_BALANCE
    balance_history = [balance]
    trade_results = []
    
    for _ in range(NUM_TRADES):
        # Generate trade outcome (win/loss) based on win rate probability
        is_win = bernoulli.rvs(WIN_RATE)
        
        # Calculate P&L based on outcome and position sizing
        if is_win:
            pnl = RISK_PER_TRADE * balance * REWARD_RISK_RATIO  # Win scenario
        else:
            pnl = -RISK_PER_TRADE * balance  # Loss scenario
        
        # Update account balance
        balance += pnl
        balance_history.append(balance)
        trade_results.append(pnl)
    
    return balance_history, trade_results


def plot_balance(balance_history):
    """
    Creates a visualization of the trading equity curve with key performance metrics.
    
    Args:
        balance_history: List of account balances after each trade
    """
    fig, ax = plt.subplots()
    
    # Convert to pandas for better data handling and analysis
    df = pd.DataFrame({
        'Balance': balance_history,
        'Trade': range(len(balance_history))
    })
    
    # Plot equity curve
    ax.plot(df['Trade'], df['Balance'], 
            color='#1f77b4', linewidth=2, label='Account Balance')
    
    # Add horizontal reference line at initial balance
    ax.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting chart elements
    ax.set_title('Trading Strategy Simulation\n'
                f'Win Rate: {WIN_RATE*100:.1f}% | Reward/Risk: {REWARD_RISK_RATIO:.2f} | Profit Factor: {PROFIT_FACTOR:.2f}',
                pad=20)
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Account Balance ($)')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Calculate and display performance metrics
    final_balance = balance_history[-1]
    total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    # Calculate maximum drawdown (peak-to-trough decline)
    balance_series = pd.Series(balance_history)
    rolling_max = balance_series.cummax()
    drawdown = (rolling_max - balance_series) / rolling_max * 100
    max_drawdown = drawdown.max()
    
    # Format metrics for display
    metrics_text = (f"Initial Balance: ${INITIAL_BALANCE:,.0f}\n"
                    f"Final Balance: ${final_balance:,.0f}\n"
                    f"Total Return: {total_return:.1f}%\n"
                    f"Max Drawdown: {max_drawdown:.1f}%")
    
    # Add metrics textbox to chart
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# === Main Execution Block ===
if __name__ == "__main__":
    print(f"Simulating {NUM_TRADES} trades with:")
    print(f"- Win Rate: {WIN_RATE*100:.2f}%")
    print(f"- Reward/Risk Ratio: {REWARD_RISK_RATIO:.2f}")
    print(f"- Risk per Trade: {RISK_PER_TRADE*100:.0f}% of account")
    print(f"- Initial Balance: ${INITIAL_BALANCE:,.0f}")
    
    balance_history, trade_results = simulate_trading()
    plot_balance(balance_history)