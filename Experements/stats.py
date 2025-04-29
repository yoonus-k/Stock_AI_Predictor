
def get_system_stats(self, stock_id):
    """
    Returns a comprehensive statistics report for a given stock ID.
    Includes cluster metrics, performance stats, and pattern analysis.
    """
    # Get all clusters for the stock
    clusters = self.get_clusters(stock_id)
    patterns = self.get_patterns(stock_id)
    
    if len(clusters) == 0:
        return {"Error": "No clusters found for this stock"}
    
    # Basic counts
    total_clusters = len(clusters)
    total_patterns = len(patterns)
    
    # total cluster types
    total_buy_clusters = len(clusters[clusters['Label'] == 'Buy'])
    total_sell_clusters = len(clusters[clusters['Label'] == 'Sell'])
    total_neutral_clusters = len(clusters[clusters['Label'] == 'Neutral'])
    avg_patterns_per_cluster = total_patterns / total_clusters if total_clusters > 0 else 0
    
    # Performance metrics
    avg_win_rate = clusters['ProbabilityScore'].mean() * 100  # Convert to percentage
    avg_max_gain = clusters['MaxGain'].mean() * 100
    avg_max_drawdown = clusters['MaxDrawdown'].mean() * 100
    
    # Calculate reward/risk ratios
    clusters['RewardRiskRatio'] = clusters['MaxGain'] / (abs(clusters['MaxDrawdown']) + 1e-10)
    avg_reward_risk = clusters['RewardRiskRatio'].mean()
    
    # Best performing cluster
    best_cluster_idx = clusters['Outcome'].idxmax()
    best_cluster_return = clusters.loc[best_cluster_idx, 'Outcome'] * 100
    
    # Market condition distribution
    market_cond_dist = clusters['MarketCondition'].value_counts(normalize=True) * 100
    
    # Pattern label distribution
    label_dist = clusters['Label'].value_counts(normalize=True) * 100
    
    # Prepare the statistics report
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


# sql query to get the number of buy and sell and neutral clusters
# SELECT COUNT(*) FROM clusters WHERE stock_id = ? AND label = 'Buy'


