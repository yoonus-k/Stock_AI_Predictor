import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import style
import pandas as pd
from scipy.stats import bernoulli

# Trading strategy parameters
PROFIT_FACTOR = 1.54
WIN_RATE = 0.5337  # 53.37%
REWARD_RISK_RATIO = 1.33
RISK_PER_TRADE = 0.01  # Risk 1% of capital per trade
NUM_TRADES = 1000
INITIAL_BALANCE = 10000

# Calculate derived parameters
LOSS_RATE = 1 - WIN_RATE
RISK_REWARD_RATIO = 1 / REWARD_RISK_RATIO  # Converts to risk:reward

# Verify the profit factor calculation matches input
calculated_pf = (WIN_RATE * REWARD_RISK_RATIO) / LOSS_RATE
print(f"Calculated Profit Factor: {calculated_pf:.2f} (Should match {PROFIT_FACTOR})")

# Set up professional matplotlib style
style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def simulate_trading():
    balance = INITIAL_BALANCE
    balance_history = [balance]
    trade_results = []
    
    for _ in range(NUM_TRADES):
        # Generate trade outcome (win/loss)
        is_win = bernoulli.rvs(WIN_RATE)
        
        # Calculate P&L based on outcome
        if is_win:
            pnl = RISK_PER_TRADE * balance * REWARD_RISK_RATIO
        else:
            pnl = -RISK_PER_TRADE * balance
        
        # Update balance
        balance += pnl
        balance_history.append(balance)
        trade_results.append(pnl)
    
    return balance_history, trade_results

def plot_balance(balance_history):
    fig, ax = plt.subplots()
    
    # Convert to pandas for nice datetime handling
    df = pd.DataFrame({
        'Balance': balance_history,
        'Trade': range(len(balance_history))
    })
    
    # Plot equity curve
    ax.plot(df['Trade'], df['Balance'], 
            color='#1f77b4', linewidth=2, label='Account Balance')
    
    # Add horizontal line at initial balance
    ax.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    ax.set_title('Trading Strategy Simulation\n'
                f'Win Rate: {WIN_RATE*100:.1f}% | Reward/Risk: {REWARD_RISK_RATIO:.2f} | Profit Factor: {PROFIT_FACTOR:.2f}',
                pad=20)
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Account Balance ($)')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Add performance metrics to plot
    final_balance = balance_history[-1]
    total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    max_drawdown = ((pd.Series(balance_history).cummax() - pd.Series(balance_history)).max() / pd.Series(balance_history).cummax().max() * 100)
    
    metrics_text = (f"Initial Balance: ${INITIAL_BALANCE:,.0f}\n"
                    f"Final Balance: ${final_balance:,.0f}\n"
                    f"Total Return: {total_return:.1f}%\n"
                    f"Max Drawdown: {max_drawdown:.1f}%")
    
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Simulating {NUM_TRADES} trades with:")
    print(f"- Win Rate: {WIN_RATE*100:.2f}%")
    print(f"- Reward/Risk Ratio: {REWARD_RISK_RATIO:.2f}")
    print(f"- Risk per Trade: {RISK_PER_TRADE*100:.0f}% of account")
    print(f"- Initial Balance: ${INITIAL_BALANCE:,.0f}")
    
    balance_history, trade_results = simulate_trading()
    plot_balance(balance_history)