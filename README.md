# Stock_AI_Predictor

An advanced AI-powered p4. **Parameter Optimization**:
   - The system uses advanced statistical analysis to test and optimize pattern recognition parameters
   - Probability scores are calculated using a directional consistency approach
     * Measures both the directional strength of outcomes (positive vs negative bias)
     * Accounts for the relative variance to handle high-volatility financial data
     * Bounded between 0.1-0.9 to avoid extreme confidence scores
   - Pattern counts are weighted using a dynamic capping mechanism to balance quantity vs. quality
   - Hold periods are selected based on timeframe category or lookback ratiorm for predicting stock market movements using pattern recognition, sentiment analysis, and reinforcement learning techniques.

## Overview

Stock_AI_Predictor is a sophisticated trading prediction system that combines technical analysis, sentiment analysis from news and social media, and reinforcement learning to provide actionable trading recommendations. The system identifies historical price patterns, analyzes current market sentiment, and uses machine learning models to predict price movements with confidence scores and risk assessments.

## Key Features

- **Multiple Analysis Methods**:
  - Pattern Recognition: Identifies perceptually important points in price data and matches them to historical patterns
  - Sentiment Analysis: Processes news articles and Twitter data to gauge market sentiment
  - Reinforcement Learning: Uses a PPO (Proximal Policy Optimization) model to make trading decisions
  - Parameter Optimization: Systematically tests and optimizes pattern mining parameters for each asset and timeframe, with special focus on Gold trading

- **Comprehensive Predictions**:
  - Price targets with confidence levels
  - Trading action recommendations (Buy, Sell, Hold)
  - Risk/reward metrics and position sizing recommendations
  - Detailed reports and visualizations

- **User Interfaces**:
  - Command Line Interface (CLI) for quick predictions
  - Graphical User Interface (GUI) built with Streamlit for visual analysis
  - Email notifications with prediction reports

- **Supported Assets**:
  - GOLD (XAUUSD)
  - Bitcoin (BTCUSD)
  - Apple (AAPL)
  - Amazon (AMZN)
  - NVIDIA (NVDA)

## System Workflow

1. **Data Collection**:
   - Stock price data is retrieved from the database or fetched from Yahoo Finance
   - News sentiment data is collected from Alpha Vantage API
   - Twitter sentiment is analyzed from relevant tweets

2. **Pattern Analysis**:
   - Price data is processed to identify perceptually important points
   - These points are matched against historical patterns using machine learning
   - Pattern matching determines the likelihood of specific price movements

3. **Parameter Optimization**:
   - Gold-specific testing identifies optimal parameters for each timeframe
   - Probability scores are calculated using statistical consistency approach
   - Pattern counts are weighted using a dynamic capping mechanism to handle large pattern counts
   - Hold periods are selected based on timeframe category or lookback ratio

4. **Sentiment Integration**:
   - News and Twitter sentiment scores are calculated
   - Sentiment data is combined with pattern analysis to refine predictions

4. **Reinforcement Learning Decision**:
   - A pre-trained RL model (PPO) makes the final trading decision
   - The model considers pattern analysis and sentiment data

## Parameter Optimization

The system includes a comprehensive framework for optimizing pattern mining parameters across different assets and timeframes, with special emphasis on Gold trading:

1. **Gold Parameter Optimization**:
   - Systematically tests combinations of timeframes, lookback periods, and PIP settings
   - Evaluates performance using cluster-based metrics: profit factor, reward-risk ratio, etc.
   - Implements two hold period strategies: timeframe-scaled and formula-based
   - Features enhanced probability score calculation for statistical consistency

2. **Hold Period Strategies**:
   - Timeframe-scaled: Applies timeframe-appropriate hold periods (3-6 bars for lower timeframes, 6-12 for medium, 12-24 for higher)
   - Formula-based: Dynamically scales hold period as ratio of lookback (e.g., hold_period = max(3, lookback/4))

3. **Risk Assessment**:
   - Maximum gain potential is calculated
   - Maximum drawdown risk is assessed
   - Reward-to-risk ratio is determined
   - Position sizing is recommended based on confidence and risk metrics

4. **Parameter Optimization**:
   - Systematic testing of pattern mining parameters across timeframes
   - Optimized settings for Gold across multiple timeframes
   - Clustering analysis for pattern effectiveness evaluation
   - Performance metrics based on cluster attributes (profit factor, reward-risk ratio)

5. **Report Generation**:
   - Comprehensive reports with all metrics are generated
   - Visual charts show price predictions with confidence intervals
   - Reports can be viewed in the app, emailed, or downloaded as PDFs

## System Workflow

### Prerequisites

- Python 3.9+
- Required Python packages (install using `pip install -r requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - stable-baselines3
  - streamlit
  - matplotlib
  - reportlab

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Stock_AI_Predictor.git
   cd Stock_AI_Predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the database (SQLite is used by default):
   ```
   python -m Data.db
   ```

### Usage

#### Command Line Interface

Run the CLI for quick predictions:

```
python -m __main__ --mode cli
```

Follow the prompts to:
1. Enter your username and password
2. Select a stock to analyze
3. Enter a date for the prediction
4. View the prediction report and receive it via email

#### Graphical User Interface

Run the GUI for a more interactive experience:

```
python -m __main__ --mode gui
```

Or directly with:

```
streamlit run gui.py
```

In the GUI, you can:
1. Log in with your credentials
2. Select a stock and date from the sidebar
3. Click "Predict" to generate forecasts
4. View interactive charts and detailed prediction reports
5. Send the report via email or download as PDF

## System Architecture

- **Core Engine (`engine_v2.py`)**: The central prediction system that combines all analysis techniques
- **Database (`Data/db.py`)**: Stores historical data, user info, and prediction records
- **Stock Data (`Data/Stocks/`)**: Contains historical stock data and data retrieval functions
- **Pattern Recognition (`Pattern/`)**: Implements pattern mining and matching algorithms
- **Parameter Testing (`Data/Utils/ParamTesting/`)**: Tools for optimizing pattern mining parameters
- **Sentiment Analysis (`Sentiment/`)**: Processes news and social media sentiment
- **Reinforcement Learning (`RL/`)**: Implements and trains the RL model for decision making
- **User Interfaces**: CLI (`cli.py`) and GUI (`gui.py`) for user interaction
- **Documentation (`docs/`)**: Contains detailed reports and implementation guides

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Gold Parameter Optimization System

The recently developed Gold parameter optimization system provides a comprehensive framework for finding the most effective pattern mining settings for Gold trading across multiple timeframes:

### Key Components

1. **Parameter Testing Framework**:
   - Tests combinations of n_pips (3-8), lookback (12-48), and hold periods
   - Integrates with database for storing configurations, patterns, and performance metrics
   - Evaluates results using cluster-based analysis instead of direct price testing

2. **Enhanced Probability Score Calculation**:
   - Improved directional consistency approach balancing win-rate and variance:
     * Directional consistency: `max(pos_ratio, neg_ratio)` measures strength of directional bias
     * Relative variance: `min(1.0, outcome_std / (mean_abs_outcome + 0.001))` handles high-volatility data
     * Combined consistency: `direction_consistency * (1 - relative_variance)` with range scaling (0.1-0.9)
   - Handles large pattern counts appropriately through dynamic weight capping based on average pattern count
   - Final score: `consistency * pattern_weight` provides balanced confidence measure robust to financial data volatility

3. **Dual Hold Period Strategies**:
   - Timeframe-scaled: Different hold periods based on timeframe category
     * Lower (1min, 5min): 3-6 bars
     * Medium (15min, 30min, 1h): 6-12 bars
     * Higher (4h, Daily): 12-24 bars
   - Formula-based: `hold_period = max(3, int(lookback / 4))`

4. **Comprehensive Visualization**:
   - Heatmaps showing parameter relationships
   - Bar charts for cluster distribution analysis
   - Comparative plots for hold period strategies

5. **Documentation**:
   - Detailed reports for each timeframe
   - Implementation guides for different strategies
   - Comparative analysis of hold period approaches

### Probability Score Examples

The improved probability score calculation handles different scenarios effectively:

| Scenario | Description | Score |
|----------|-------------|-------|
| Strong Bullish | High directional consistency (90% positive), low variance | 0.60-0.80 |
| Strong Bearish | High directional consistency (90% negative), low variance | 0.60-0.80 |
| Mixed Signals | Low directional consistency (60% bias), high variance | 0.20-0.40 |
| High Volatility | Medium directional bias with very high variance | 0.25-0.45 |
| Large Pattern Count | Strong consistency with many patterns (>30) | 0.60-0.90 |
| Small Pattern Count | Strong consistency but few patterns (<5) | 0.20-0.40 |

The system uses this probability score to determine confidence in predictions and suitable position sizes.

### Running Gold Parameter Tests

```
python -m Data.Utils.ParamTesting.run_gold_optimization --timeframe 1h --hold-strategy formula
```

For detailed information, refer to the documentation in the `/docs` folder.

## License

This project is licensed under the terms of the license included in the repository.

## Acknowledgments

- This project uses multiple AI techniques, including pattern recognition, sentiment analysis, and reinforcement learning
- Libraries used include stable-baselines3, scikit-learn, pandas, and streamlit
- Special thanks to all contributors who have helped develop this system

