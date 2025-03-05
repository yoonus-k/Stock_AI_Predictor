import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def plot_pattern_comparison(patterns):
    plt.figure(figsize=(12, 6))
    
    for i, pattern in enumerate(patterns):
        # Convert to 2D array for sklearn (samples, features)
        data = np.array(pattern).reshape(-1, 1)
        
        # Create scalers
        minmax_scaler = MinMaxScaler()
        std_scaler = StandardScaler()
        
        # Fit and transform
        normalized = minmax_scaler.fit_transform(data).flatten()
        standardized = std_scaler.fit_transform(data).flatten()
        
        # Print results
        print(f"\nPattern {i+1} ({pattern}):")
        print(f"Original:    {np.round(pattern, 2)}")
        print(f"Normalized:  {np.round(normalized, 2)}")
        print(f"Standardized: {np.round(standardized, 2)}")
        
        # Plotting
        plt.subplot(len(patterns), 3, i*3 + 1)
        plt.plot(pattern, marker='o', color='blue')
        plt.title(f'Pattern {i+1}\nOriginal')
        if i == 0: plt.ylabel('Price')
        
        plt.subplot(len(patterns), 3, i*3 + 2)
        plt.plot(normalized, marker='o', color='green')
        plt.title('Min-Max Normalized')
        if i == 0: plt.ylabel('Normalized [0-1]')
        
        plt.subplot(len(patterns), 3, i*3 + 3)
        plt.plot(standardized, marker='o', color='red')
        plt.title('Z-Score Standardized')
        if i == 0: plt.ylabel('Standardized')

    plt.tight_layout()
    plt.show()

# Test patterns
patterns = [
    [100, 120, 110, 130,100],  # Large scale pattern
    [10, 20, 15, 25,10],       # Small scale pattern with similar shape
    [100, 110 , 105 , 115,100]         # Different shape pattern
]

plot_pattern_comparison(patterns)