import matplotlib.pyplot as plt

# Metrics
metrics = {
    'Total Return': 68.57,
    'Win Rate': 53.51,
    'Profit Factor': 1.31,
    'Sharpe Ratio': 0.78,
    'Max Drawdown': -9.27
}

colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#F44336']

plt.figure(figsize=(10, 5))
bars = plt.barh(list(metrics.keys()), list(metrics.values()), color=colors)
plt.title("📊 Model Forward Test Metrics Summary (2024)", fontsize=14)
plt.xlabel("Metric Value")
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add values to bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.2f}', va='center', ha='left')

plt.tight_layout()
plt.show()


labels = ['TP Hit', 'SL Hit', 'Hold Exit']
sizes = [553, 289, 225]
colors = ['#00C49A', '#FF6F61', '#FFA500']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, startangle=140, wedgeprops={'width':0.4}, autopct='%1.1f%%')
plt.title("🎯 Trade Exit Reason Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()



trade_types = ['BUY', 'SELL']
trades = [869, 198]
win_rates = [54.89, 47.47]

fig, ax1 = plt.subplots(figsize=(8, 5))

bar = ax1.bar(trade_types, trades, color="#4DB6AC", label="Number of Trades")
ax2 = ax1.twinx()
line = ax2.plot(trade_types, win_rates, color="#FF7043", marker='o', label="Win Rate (%)")

ax1.set_ylabel("Trades")
ax2.set_ylabel("Win Rate (%)")
ax1.set_title("📈 Buy vs Sell Trade Analysis")

fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.show()




