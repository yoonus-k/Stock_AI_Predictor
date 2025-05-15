"""
Statistical Analysis Utilities for Trading Systems

This module provides utilities for computing and analyzing statistical metrics
of trading systems, including cluster performance, pattern distributions, and
overall system effectiveness.

The functions can be used to evaluate trading strategy performance and identify
the most effective patterns and market conditions.
"""


def get_system_stats(self, stock_id):
    """
    Returns a comprehensive statistics report for a given stock ID.
    
    This function analyzes clusters and patterns for a specific stock and generates
    a detailed statistical report including performance metrics, distribution analysis,
    and top/bottom performing clusters.
    
    Args:
        stock_id (int): The ID of the stock to analyze
        
    Returns:
        dict: A dictionary containing structured statistics about the trading system
              organized into categories like "Basic Metrics", "Performance Statistics", etc.
    """
    # Get all clusters and patterns for the stock
    clusters = self.get_clusters(stock_id)
    patterns = self.get_patterns(stock_id)
    
    if len(clusters) == 0:
        return {"Error": "No clusters found for this stock"}
    
    # === Basic Metrics ===
    # Count clusters and patterns
    total_clusters = len(clusters)
    total_patterns = len(patterns)
    
    # Analyze cluster types
    total_buy_clusters = len(clusters[clusters['Label'] == 'Buy'])
    total_sell_clusters = len(clusters[clusters['Label'] == 'Sell'])
    total_neutral_clusters = len(clusters[clusters['Label'] == 'Neutral'])
    avg_patterns_per_cluster = total_patterns / total_clusters if total_clusters > 0 else 0
    
    # === Performance Metrics ===
    # Calculate average performance statistics
    avg_win_rate = clusters['ProbabilityScore'].mean() * 100  # Convert to percentage
    avg_max_gain = clusters['MaxGain'].mean() * 100
    avg_max_drawdown = clusters['MaxDrawdown'].mean() * 100
    
    # Calculate reward/risk ratios (with small epsilon to prevent division by zero)
    clusters['RewardRiskRatio'] = clusters['MaxGain'] / (abs(clusters['MaxDrawdown']) + 1e-10)
    avg_reward_risk = clusters['RewardRiskRatio'].mean()
    
    # Identify best performing cluster
    best_cluster_idx = clusters['Outcome'].idxmax()
    best_cluster_return = clusters.loc[best_cluster_idx, 'Outcome'] * 100
    
    # === Distribution Analysis ===
    # Market condition distribution (as percentages)
    market_cond_dist = clusters['MarketCondition'].value_counts(normalize=True) * 100
    
    # Pattern label distribution (as percentages)
    label_dist = clusters['Label'].value_counts(normalize=True) * 100
    
    # === Prepare Structured Statistics Report ===
    stats = {
        "Basic Metrics": {
            "Total Clusters": total_clusters,
            "Total Patterns": total_patterns,
            "Avg Patterns/Cluster": round(avg_patterns_per_cluster, 1),
            "Total Buy Clusters": total_buy_clusters,
            "Total Sell Clusters": total_sell_clusters,
            "Total Neutral Clusters": total_neutral_clusters
        },
        "Performance Statistics": {
            "Avg Win Rate": f"{round(avg_win_rate, 1)}%",
            "Avg Max Gain": f"{round(avg_max_gain, 1)}%",
            "Avg Max Drawdown": f"{round(avg_max_drawdown, 1)}%",
            "Avg Reward/Risk": round(avg_reward_risk, 2),
            "Best Cluster Return": f"+{round(best_cluster_return, 1)}%",
        },
        "Distribution": {
            "Market Conditions": {
                "Bullish": f"{round(market_cond_dist.get('Bullish', 0), 1)}%",
                "Bearish": f"{round(market_cond_dist.get('Bearish', 0), 1)}%",
                "Neutral": f"{round(market_cond_dist.get('Neutral', 0), 1)}%",
            },
            "Pattern Labels": {
                "Buy": f"{round(label_dist.get('Buy', 0), 1)}%",
                "Sell": f"{round(label_dist.get('Sell', 0), 1)}%",
                "Neutral": f"{round(label_dist.get('Neutral', 0), 1)}%",
            }
        },
        "Top Performing Clusters": self._get_top_clusters(clusters, 3),
        "Worst Performing Clusters": self._get_worst_clusters(clusters, 3)
    }
    
    return stats


def _get_top_clusters(self, clusters, n=3):
    """
    Returns information about the top N performing clusters.
    
    Args:
        clusters (DataFrame): DataFrame containing cluster data
        n (int): Number of top clusters to return
        
    Returns:
        list: List of dictionaries with details about top clusters
    """
    # Sort clusters by outcome (descending)
    top_clusters = clusters.sort_values('Outcome', ascending=False).head(n)
    
    # Format the results
    result = []
    for idx, cluster in top_clusters.iterrows():
        result.append({
            "Cluster ID": int(idx),
            "Label": cluster['Label'],
            "Return": f"{round(cluster['Outcome'] * 100, 2)}%",
            "Win Rate": f"{round(cluster['ProbabilityScore'] * 100, 1)}%",
            "Market Condition": cluster['MarketCondition']
        })
    
    return result


def _get_worst_clusters(self, clusters, n=3):
    """
    Returns information about the worst N performing clusters.
    
    Args:
        clusters (DataFrame): DataFrame containing cluster data
        n (int): Number of worst clusters to return
        
    Returns:
        list: List of dictionaries with details about worst clusters
    """
    # Sort clusters by outcome (ascending to get worst performers)
    worst_clusters = clusters.sort_values('Outcome', ascending=True).head(n)
    
    # Format the results
    result = []
    for idx, cluster in worst_clusters.iterrows():
        result.append({
            "Cluster ID": int(idx),
            "Label": cluster['Label'],
            "Return": f"{round(cluster['Outcome'] * 100, 2)}%",
            "Win Rate": f"{round(cluster['ProbabilityScore'] * 100, 1)}%",
            "Market Condition": cluster['MarketCondition']
        })
    
    return result


# Example SQL queries for reference:
# Get count of buy clusters: SELECT COUNT(*) FROM clusters WHERE stock_id = ? AND label = 'Buy'
# Get count of sell clusters: SELECT COUNT(*) FROM clusters WHERE stock_id = ? AND label = 'Sell'
# Get count of neutral clusters: SELECT COUNT(*) FROM clusters WHERE stock_id = ? AND label = 'Neutral'


