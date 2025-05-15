"""
Data Scaling Utilities for Pattern Comparison

This module provides utilities for comparing different scaling methods for price patterns.
It helps visualize how MinMaxScaler and StandardScaler transform the same pattern differently,
which is useful for pattern recognition and machine learning model preparation.

The module includes visualization functions to compare original, normalized, and standardized patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def plot_pattern_comparison(patterns):
    """
    Visualizes and compares different scaling methods for price patterns.
    
    This function takes a list of patterns (price sequences) and creates visualizations
    showing the original pattern alongside its normalized and standardized versions.
    It also prints the numerical values for each transformation.
    
    Args:
        patterns (list): List of price patterns (each pattern is a list or array of values)
    """
    plt.figure(figsize=(12, 6))
    
    for i, pattern in enumerate(patterns):
        # Convert to 2D array for sklearn (samples, features)
        data = np.array(pattern).reshape(-1, 1)
        
        # Create scalers
        minmax_scaler = MinMaxScaler()  # Scales to [0,1] range
        std_scaler = StandardScaler()   # Transforms to zero mean, unit variance
        
        # Fit and transform data
        normalized = minmax_scaler.fit_transform(data).flatten()
        standardized = std_scaler.fit_transform(data).flatten()
        
        # Print results for comparison
        print(f"\nPattern {i+1} ({pattern}):")
        print(f"Original:     {np.round(pattern, 2)}")
        print(f"Normalized:   {np.round(normalized, 2)}")
        print(f"Standardized: {np.round(standardized, 2)}")
        
        # Create three subplot panels for each pattern
        # Panel 1: Original pattern
        plt.subplot(len(patterns), 3, i*3 + 1)
        plt.plot(pattern, marker='o', color='blue')
        plt.title(f'Pattern {i+1}\nOriginal')
        if i == 0: plt.ylabel('Price')
        
        # Panel 2: MinMax normalized pattern
        plt.subplot(len(patterns), 3, i*3 + 2)
        plt.plot(normalized, marker='o', color='green')
        plt.title('Min-Max Normalized')
        if i == 0: plt.ylabel('Normalized [0-1]')
        
        # Panel 3: Z-score standardized pattern
        plt.subplot(len(patterns), 3, i*3 + 3)
        plt.plot(standardized, marker='o', color='red')
        plt.title('Z-Score Standardized')
        if i == 0: plt.ylabel('Standardized')

    plt.tight_layout()
    plt.show()


# Example usage with test patterns
if __name__ == "__main__":
    # Define test patterns with different characteristics
    patterns = [
        [100, 120, 110, 130, 100],  # Large scale pattern
        [10, 20, 15, 25, 10],       # Small scale pattern with similar shape
        [100, 110, 105, 115, 100]   # Different shape pattern
    ]
    
    # Visualize and compare the patterns
    plot_pattern_comparison(patterns)