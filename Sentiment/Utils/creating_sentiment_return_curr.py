import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate 6 months of data (assume 180 days)
np.random.seed(42)
dates = pd.date_range(start='2024-10-01', periods=180, freq='D')

# Generate sentiment score with some trend
sentiment_score = np.clip(np.random.normal(loc=0.2, scale=0.05, size=180), 0, 0.4)

# Introduce correlation: return = 0.3 * sentiment_score + noise
noise = np.random.normal(loc=0.0, scale=0.02, size=180)
returns = 0.3 * sentiment_score + noise

# Calculate correlation coefficient
correlation = np.corrcoef(sentiment_score, returns)[0, 1]

# Plot
fig, ax1 = plt.subplots(figsize=(12, 12))

# Plot sentiment score
ax1.plot(dates, sentiment_score, color='blue', label='Sentiment Score', linewidth=2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot return
ax2 = ax1.twinx()
ax2.plot(dates, returns, color='green', label='Return', linewidth=2.5)
ax2.set_ylabel('Return', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Title and grid
plt.title('Sentiment Score vs Return ')
fig.tight_layout()
ax1.grid(True)

# Display correlation value
plt.figtext(0.1, 0.1, f'Correlation: {correlation:.2f}', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
