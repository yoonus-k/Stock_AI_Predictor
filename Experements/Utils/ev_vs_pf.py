import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Define a range of win rates (p) and reward-to-risk ratios (RR)
p_values = np.linspace(0.01, 0.99, 100)
RR_values = np.linspace(0.1, 5, 100)

# Create a meshgrid for plotting
P, RR = np.meshgrid(p_values, RR_values)

# Calculate Profit Factor and Expected Value
PF = (P * RR) / (1 - P)
EV = (P * RR) - (1 - P)

# Flatten data for DataFrame
data = pd.DataFrame({
    "Win Rate (p)": P.ravel(),
    "RR": RR.ravel(),
    "Profit Factor": PF.ravel(),
    "Expected Value": EV.ravel()
})

# Determine profitability zones
data["Profitable (EV > 0)"] = data["Expected Value"] > 0
data["High PF (PF > 1)"] = data["Profit Factor"] > 1

# Plot 1: Heatmap of Expected Value
plt.figure(figsize=(12, 6))
pivot_ev = data.pivot_table(index="RR", columns="Win Rate (p)", values="Expected Value")
sns.heatmap(pivot_ev, cmap="coolwarm", center=0, cbar_kws={'label': 'Expected Value'})
plt.title("Expected Value Heatmap")
plt.xlabel("Win Rate (p)")
plt.ylabel("Reward-to-Risk Ratio (RR)")
plt.tight_layout()
plt.show()

# Plot 2: Heatmap of Profit Factor
plt.figure(figsize=(12, 6))
pivot_pf = data.pivot_table(index="RR", columns="Win Rate (p)", values="Profit Factor")
sns.heatmap(pivot_pf, cmap="viridis", cbar_kws={'label': 'Profit Factor'})
plt.title("Profit Factor Heatmap")
plt.xlabel("Win Rate (p)")
plt.ylabel("Reward-to-Risk Ratio (RR)")
plt.tight_layout()
plt.show()

# Plot 3: EV vs PF scatter plot colored by profitability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Profit Factor", y="Expected Value", hue="Profitable (EV > 0)", alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.axvline(1, color='blue', linestyle='--')
plt.title("Expected Value vs Profit Factor")
plt.xlabel("Profit Factor")
plt.ylabel("Expected Value")
plt.legend(title="Profitable (EV > 0)")
plt.tight_layout()
plt.show()
