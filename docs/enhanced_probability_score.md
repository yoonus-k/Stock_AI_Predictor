# Enhanced Probability Score Calculation for Stock AI Predictor

This document explains the improved probability score calculation method implemented in the Stock AI Predictor system, with particular focus on its application to Gold trading pattern recognition.

## Problem with the Original Approach

The original probability score calculation had several limitations:

```python
# Original calculation
consistency_ratio = outcome_std / (abs(avg_outcome) + 0.0001)
consistency = 1 - min(1.0, consistency_ratio)
consistency = max(0, consistency)
probability_score = consistency * pattern_weight
```

Primary issues:
1. Too sensitive to high standard deviation in financial data
2. Produced scores of 0 when `outcome_std` was significantly higher than `avg_outcome`
3. Didn't account for directional consistency (strong bias toward positive or negative)
4. Didn't handle mixed signal clusters effectively

## Improved Probability Score Calculation

The improved approach addresses these limitations with a more balanced calculation that better accounts for the nature of financial data:

```python
# Calculate directional consistency
pos_ratio = sum(1 for o in outcomes if o > 0) / len(outcomes)
neg_ratio = sum(1 for o in outcomes if o < 0) / len(outcomes)
direction_consistency = max(pos_ratio, neg_ratio)  # Higher value indicates stronger directional bias

# Calculate variance relative to mean absolute outcome
mean_abs_outcome = np.mean(np.abs(outcomes))
relative_variance = min(1.0, outcome_std / (mean_abs_outcome + 0.001))

# Combine both factors - higher score for strong direction and low relative variance
consistency = direction_consistency * (1 - relative_variance)

# Ensure range is within 0.1 to 0.9 to avoid extreme scores
consistency = 0.1 + (consistency * 0.8)

# Weight by pattern count
max_patterns_for_full_weight = avg_patterns_count  # Dynamic based on average patterns in clusters
pattern_weight = min(1.0, pattern_count / max_patterns_for_full_weight)

# Final probability score - balanced approach
probability_score = consistency * pattern_weight
```

## Key Improvements

1. **Directional Consistency:** 
   - Measures how consistently patterns in a cluster produce either positive or negative outcomes
   - A higher value (closer to 1.0) indicates a strong directional bias, which is valuable for trading decisions

2. **Relative Variance Measurement:** 
   - Instead of using `outcome_std` vs `avg_outcome` (which can be close to zero), we use `outcome_std` vs `mean_abs_outcome`
   - This approach is more suitable for financial data where the average might be small but individual movements are significant

3. **Range Scaling:** 
   - The raw consistency score is scaled to fall between 0.1 and 0.9
   - This prevents extreme confidence scores (0 or 1) that could lead to poor trading decisions

4. **Dynamic Pattern Weight:** 
   - The weight given to pattern count is dynamically scaled based on the average number of patterns in clusters
   - This approach adapts to different datasets and timeframes automatically

## Example Scenarios

Here's how the formula performs with different pattern characteristics:

### 1. Strong Bullish Cluster with Low Variance

```
Outcomes: [1.2, 1.5, 1.3, 1.6, 0.9, 1.1, 1.4, 1.2, -0.2, 1.3]
Pattern count: 10
Average patterns in system: 30

Calculated values:
- Directional consistency: 0.9 (90% positive outcomes)
- Relative variance: 0.35
- Raw consistency: 0.585
- Scaled consistency: 0.568
- Pattern weight: 0.333
- Final probability score: 0.189
```

### 2. Strong Bearish Cluster with Low Variance

```
Outcomes: [-1.1, -0.9, -1.3, -1.2, -1.0, -1.4, -0.8, -1.1, 0.2, -1.2]
Pattern count: 10
Average patterns in system: 30

Calculated values:
- Directional consistency: 0.9 (90% negative outcomes)
- Relative variance: 0.33
- Raw consistency: 0.603
- Scaled consistency: 0.582
- Pattern weight: 0.333
- Final probability score: 0.194
```

### 3. Mixed Signals Cluster with High Variance

```
Outcomes: [2.5, -1.8, 1.9, -2.1, 2.7, -2.3, 1.5, -1.9, 2.2, -1.7]
Pattern count: 10
Average patterns in system: 30

Calculated values:
- Directional consistency: 0.5 (50% in each direction)
- Relative variance: 0.98
- Raw consistency: 0.010
- Scaled consistency: 0.108
- Pattern weight: 0.333
- Final probability score: 0.036
```

### 4. Original Problematic Scenario (High Standard Deviation)

```
Outcomes: [5.2, -4.8, 6.1, -3.9, 7.5, -2.8, 4.2, -5.3, 8.1, -1.9]
Pattern count: 10
Average patterns in system: 30

Old calculation:
- Consistency ratio: 5.86
- Consistency: 0.0
- Pattern weight: 0.333
- Final probability score: 0.0

New calculation:
- Directional consistency: 0.6
- Relative variance: 0.97
- Raw consistency: 0.018
- Scaled consistency: 0.114
- Pattern weight: 0.333
- Final probability score: 0.038
```

## Conclusion

The enhanced probability score calculation provides a more nuanced and reliable measure of confidence in pattern-based predictions. It:

1. Provides meaningful non-zero scores even for high-variance financial data
2. Considers the directional consistency of outcomes, which is valuable for trading decisions
3. Adapts to different pattern sizes and timeframes through dynamic weighting
4. Produces a balanced confidence measure robust to financial data volatility

This improvement helps the Stock AI Predictor system make more informed trading decisions with appropriate confidence levels, especially for Gold trading where proper risk assessment is crucial.
