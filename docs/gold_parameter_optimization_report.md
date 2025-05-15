# Gold Parameter Optimization Report

## Overview

This report outlines the strategy and analysis for o2. **Timeframe vs. Hold Period**:
   - Do higher timeframes benefit more from proportionally longer or shorter hold periods?
   - Is there an optimal ratio between lookback window and hold period (e.g., 4:1, 3:1)?
   - How sensitive is pattern performance to hold period changes across different timeframes?mizing pattern mining parameters for Gold across multiple timeframes. By systematically testing combinations of timeframes, lookback periods, and PIP (Perceptually Important Points) settings, we aim to identify optimal configurations for predictive accuracy and trading performance.

## Parameter Space

Testing will evaluate the following parameters:

### Timeframes
- 1 minute (1min)
- 5 minutes (5min)
- 15 minutes (15min)
- 30 minutes (30min)
- 1 hour (1h)
- 4 hours (4h)
- Daily (1d)

### Lookback Windows
- 12 bars
- 24 bars
- 36 bars
- 48 bars

### Number of PIPs
- 3 points
- 4 points
- 5 points
- 6 points
- 7 points
- 8 points

### Distance Measure
- Perpendicular distance (2) - Selected as the standard measure for all tests

### Hold Period Optimization
- Set `hold_period = returns_hold_period` for experimental consistency
- **Timeframe-Scaled Approach**:
  - Lower timeframes (1min, 5min): [3, 6]
  - Medium timeframes (15min, 30min, 1h): [6, 12]
  - Higher timeframes (4h, Daily): [12, 24]
- **Formula-Based Option**: `hold_period = max(3, lookback/4)` to scale proportionally with lookback period

This approach reduces the parameter space while ensuring appropriate holding periods for each timeframe. The total combinations become 7×4×6×1×2 = 336 (with 2 hold_period values per timeframe) or 7×4×6×1 = 168 combinations (using the formula-based approach), making the experiment manageable while still covering the essential parameter space.

## Testing Methodology

The parameter testing framework follows this process:

1. **Data Preparation**:
   - Split Gold price data for each timeframe into 80% training, 20% testing sets
   - Ensure sufficient data quantity for statistical significance
   - Extract close prices for pattern mining

2. **Pattern Mining**:
   - Apply the `Pattern_Miner` algorithm with each parameter combination
   - Train on the training dataset
   - Identify patterns and make predictions on the test dataset
   - For each timeframe, test appropriate hold_period values:
     * Either use timeframe-appropriate values (two per timeframe)
     * Or calculate dynamically as a percentage of lookback period

3. **Performance Evaluation**:
  Cluster-Based Evaluation:
  o evaluate parameter combinations:

   outcome: Average outcome of patterns in the cluster
   probability_score: Confidence level of predictions
   pattern_count: Number of patterns in the cluster
   max_gain: Maximum gain achieved
   max_drawdown: Maximum drawdown experienced
   reward_risk_ratio: Ratio of reward to risk
   profit_factor: Profit factor of the cluster

   Instead of testing against price data, use the cluster attributes directly for evaluation:
   Profit Factor
   Reward-Risk Ratio
   Pattern Count per Cluster
   Probability Score
   - Record training time for computational efficiency analysis
   Weighted Scoring System:

   Profit Factor (35%)
   Reward-Risk Ratio (25%)
   Pattern Count per Cluster (20%)
   Probability Score (20%)

4. **Result Storage**:
   - Save performance metrics to the database
   - Generate visual plots for pattern comparison
   - Create detailed reports for each timeframe

## Expected Insights

### Timeframe-Specific Patterns

Different timeframes are expected to have distinct optimal parameter settings:

- **Lower Timeframes (1min, 5min)**
  - Likely benefit from smaller lookback windows (12-24 bars)
  - May perform better with fewer PIPs (3-5) to reduce noise
  - Could be more sensitive to distance measure selection
  - Expected to generate more trade signals but potentially lower profit factor

- **Medium Timeframes (15min, 30min, 1h)**
  - Potentially optimal with moderate lookback (24-36 bars)
  - May perform best with 4-6 PIPs for pattern identification
  - Expected to balance signal frequency and quality

- **Higher Timeframes (4h, Daily)**
  - Likely require larger lookback windows (36-48 bars)
  - May benefit from more PIPs (6-8) to capture complex patterns
  - Expected to generate fewer but higher-quality signals
  - Potentially higher profit factor but fewer trade opportunities

### Parameter Interactions

Key interactions to analyze:

1. **PIP Count vs. Lookback Window**
   - How does the optimal PIP count change as lookback window increases?
   - Is there a consistent ratio of PIPs to lookback period across timeframes?

2. **Timeframe vs. Hold Period**
   - Do higher timeframes require proportionally shorter hold periods?
   - How does the optimal hold period affect directional accuracy?

3. **Distance Measure vs. Gold Price Volatility**
   - How does perpendicular distance perform across different volatility regimes?
   - Is there a correlation between Gold's volatility and optimal PIP/lookback settings?

## Analysis Dimensions

The analysis will focus on these key dimensions:

### Pattern Recognition Effectiveness
- **Cluster Formation**: Optimal number of clusters for pattern categorization
- **Pattern Diversity**: Number of unique patterns identified
- **Pattern Quality**: How consistently patterns predict future price movements

### Trading Performance
- **Directional Accuracy**: Percentage of correct price movement predictions
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Trade Frequency**: Number of signals generated per time period

### Computational Efficiency
- **Training Time**: Processing time required for pattern mining
- **Prediction Speed**: Time to identify patterns and generate predictions
- **Scalability**: How performance scales with increased data volume

## Data Volume Considerations

A critical aspect of this analysis is the varying data volume across timeframes:

- Daily charts may have only a few thousand data points
- 1-minute charts could have millions of data points

The testing framework will:

1. Ensure minimum thresholds for data quantity (minimum 100 points for training, 50 for testing)
2. Normalize metrics where appropriate to account for data volume differences
3. Consider statistical significance of results for each timeframe

## Visualization Strategy

To effectively communicate results, the testing framework generates:

1. **Heatmaps** showing:
   - Directional accuracy by n_pips and lookback
   - Win rate by hold period and distance measure

2. **Bar charts** displaying:
   - Profit factor across parameter combinations
   - Trade count vs. accuracy for different parameter sets

3. **Detailed reports** including:
   - Top 5 parameter combinations by directional accuracy
   - Top 5 parameter combinations by profit factor
   - Parameter impact analysis for n_pips and lookback window

## Implementation Recommendations

### Testing Approach
- Begin with a targeted subset of combinations to establish baselines
- Test both hold period strategies (timeframe-scaled and formula-based) to determine the optimal approach
- Expand testing to all combinations for Gold specifically
- Set `test_all_params=True` in the `run_all_tests()` method to evaluate all combinations

### Parameter Validation
- Cross-validate results using different time periods
- Test robustness against different market conditions
- Verify that parameter performance is consistent over time

### Operational Considerations
- Higher timeframes should be tested first due to lower computational requirements
- Lower timeframes may require batch processing or distributed computing
- Consider monitoring RAM usage during testing of high-volume timeframes

## Evaluating Results

The optimal parameter combination for each timeframe will be selected based on a weighted scoring system that evaluates the clusters directly:

- **Primary Metrics** (Based on Cluster Attributes):
  - Profit Factor (35%)
  - Reward-Risk Ratio (25%)
  - Pattern Count per Cluster (20%)
  - Probability Score (20%)

- **Secondary Considerations**:
  - Number of distinct pattern clusters formed
  - Distribution of positive vs. negative outcome clusters
  - Maximum gain vs. maximum drawdown ratios
  - Market condition adaptability

## Conclusion

This systematic approach to parameter optimization will identify the most effective combinations of timeframe, lookback window, and PIP count for pattern mining in Gold price data. The results will enable the development of more accurate predictive models and potentially more profitable trading strategies.

By analyzing the relationships between these parameters and their impact on pattern recognition effectiveness, we can gain deeper insights into the temporal structure of Gold price movements across different timeframes.

The framework is designed to be extended to other assets in the future, providing a robust methodology for pattern parameter optimization across the entire market.
