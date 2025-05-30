# Reinforcement Learning Trading System

This module implements an advanced reinforcement learning (RL) system designed for algorithmic trading based on technical patterns, sentiment analysis, and market conditions. The system integrates with the broader Stock_AI_Predictor project, leveraging pattern recognition and sentiment analysis while incorporating risk-adjusted reward optimizations.

## Key Features

- **Risk-Adjusted Rewards**: Multiple reward functions optimized for different trading objectives
- **Adaptive Position Sizing**: Dynamic adjustment based on market conditions and performance
- **Drawdown Protection**: Special mechanisms to protect capital during adverse market conditions
- **Performance Metrics**: Comprehensive tracking of trading performance beyond simple P&L

## System Components

### Environment (`Envs/trading_env.py`)

The trading environment simulates a financial market where the RL agent can:
- Observe market conditions through various features
- Take actions (buy, sell, or hold with varying position sizes and risk parameters)
- Receive risk-adjusted rewards based on trade outcomes and overall portfolio performance

### Risk-Adjusted Reward Functions

The system implements several reward functions to optimize for different trading objectives:

1. **Sharpe Ratio Reward**: Optimizes for risk-adjusted returns
2. **Sortino Ratio Reward**: Focuses on minimizing downside risk
3. **Drawdown Protection**: Heavily penalizes drawdowns, especially those exceeding key thresholds
4. **Win Rate Optimization**: Rewards consistent winning trades and profit factor
5. **Calmar Ratio Reward**: Optimizes for return relative to maximum drawdown
6. **Combined Risk-Reward**: A balanced approach incorporating multiple risk metrics

### Adaptive Position Sizing

Position sizes are dynamically adjusted based on:
1. Current drawdown level (reduces exposure during drawdowns)
2. Win/loss streaks (increases size after wins, decreases after losses)
3. Market volatility (reduces size in high volatility environments)

### Agent (`Agents/policy.py`)

Uses the Proximal Policy Optimization (PPO) algorithm to learn optimal trading policies:
- Leverages stable-baselines3 implementation
- Maintains policy and value networks
- Maps market observations to trading decisions

### Data Pipeline (`Data/loader.py`)

Provides processed market data for training:
- Loads historical data from the project database
- Integrates pattern recognition and sentiment analysis
- Prepares feature vectors for the RL environment

### Monitoring Tools (`Scripts/`)

Comprehensive monitoring and analysis suite for tracking model performance:
- **Feature Importance Analysis**: Identifies which features have the greatest impact on decisions
- **Trading Strategy Analysis**: Visualizes portfolio performance and trading patterns
- **Decision Boundary Analysis**: Visualizes how the model makes decisions based on different features
- **TensorBoard Integration**: Tracks metrics and rewards during training
- **Checkpoint Analysis**: Evaluates and compares models saved during training
- **Live Training Dashboard**: Real-time monitoring of training progress

For detailed usage instructions, see [`Scripts/README_MONITORING.md`](Scripts/README_MONITORING.md).

### Observation Space

The agent receives the following market and portfolio information:

#### Market Features
1. `probability`: Probability of a successful trade based on pattern recognition
2. `action`: Pattern action signal (1=bullish, 2=bearish)
3. `reward_risk_ratio`: Ratio of potential reward to risk
4. `max_gain`: Maximum gain potential
5. `max_drawdown`: Maximum potential drawdown
6. `impact_score`: Market impact score
7. `news_score`: News sentiment score 
8. `twitter_score`: Twitter sentiment score

#### Portfolio Features
1. Normalized balance (current balance / initial balance)
2. Position ratio (position value / total portfolio value)
3. Absolute position size

### Action Space

The agent takes actions consisting of:
1. `action_type`: Discrete choice of 0=hold, 1=buy, 2=sell
2. `position_size`: Fraction of portfolio to allocate (0.1-1.0)
3. `risk_reward_multiplier`: Adjusts take-profit and stop-loss levels (0.5-3.0)

## Scripts and Tools

### Training and Evaluation

- **`train_compare_rewards.py`**: Trains models with different reward functions and compares performance
- **`evaluate_model.py`**: Detailed evaluation of a trained model with performance metrics and visualizations
- **`test_risk_management.py`**: Tests models under various stress scenarios to evaluate risk management

## How the RL System Learns

### Learning Process

1. **Observation**: Agent receives market state (patterns, sentiment) and portfolio status
2. **Decision**: Agent selects an action (buy/sell/hold) and position size
3. **Execution**: Trade is executed in the environment with TP/SL based on pattern data
4. **Outcome**: Agent receives reward based on trade profitability
5. **Update**: Neural network weights are updated to improve future decisions

### Adaptation Mechanisms

The system adapts to changing market conditions through:

#### 1. Exploration vs. Exploitation
- The PPO algorithm maintains a balance between trying new strategies (exploration) and using proven ones (exploitation)
- Entropy bonus in the loss function encourages action diversity
- Action probability distributions allow for occasional unexpected actions

#### 2. Continuous Learning
- Agent can be periodically retrained on recent data while preserving previous knowledge
- Learning rate can be adjusted to control adaptation speed
- Experience replay helps retain important lessons from past market conditions

#### 3. Neural Network Architecture
- Multi-layer perceptrons capture complex patterns in market data
- Network updates gradually incorporate new information while preserving critical knowledge
- Value function helps estimate long-term consequences of actions

## Advanced Implementation Strategies

### Multi-Agent Systems
For optimal performance, consider implementing multiple specialized agents:

1. **Technical Agent**: Trained primarily on price patterns and indicators
2. **Sentiment Agent**: Specialized in trading based on news and social media
3. **Time-Pattern Agent**: Specializes in time-of-day and session effects
4. **Meta-Agent**: Determines which specialized agent to trust based on current conditions

### Dynamic TP/SL Placement
Enhance the action space to allow the agent to learn optimal take-profit and stop-loss levels:

1. Expanded action space: [action_type, position_size, tp_multiplier, sl_multiplier]
2. Agent learns to adjust risk management based on market conditions
3. Provides flexibility beyond fixed TP/SL rules

## Recommended Enhancements

### 1. Enhanced Feature Engineering
Add these features to improve the agent's understanding of market conditions:

- **Technical Indicators**
  - RSI (Relative Strength Index): Identify overbought/oversold conditions
  - ATR (Average True Range): Measure volatility for better position sizing
  - Moving Averages: Trend identification across multiple timeframes
  - Volume Indicators: Confirm price moves with volume context

- **Time-Based Features**
  - Trading Session Indicators: NY, London, Asian sessions
  - Hour of Day: Encoded cyclically to capture time patterns (e.g., gold at hour 16)
  - Day of Week: Capture weekly patterns
  - Market Events Calendar: Pre/post major economic announcements

- **Pattern Enhancements**
  - Cluster ID: Leverage existing pattern clustering from main project
  - Pattern Confidence: Weight signals by confidence level
  - Historical Performance: Include success rate of similar past patterns

### 2. Risk Management Improvements

- **Adaptive Position Sizing**
  - Adjust position size based on pattern confidence and volatility
  - Implement Kelly criterion or risk parity approaches
  - Learn to size positions differently across market regimes

- **Dynamic Stop Management**
  - Train agent to move stops to breakeven at appropriate times
  - Implement trailing stops based on market volatility
  - Partial profit-taking strategies

### 3. Market Regime Detection

- **Explicit Regime Identification**
  - Volatility-based regime detection
  - Correlation structure analysis
  - Sentiment influence tracking

- **Regime-Specific Models**
  - Maintain separate models for different regimes
  - Quick-switching mechanism between models
  - Gradual parameter adjustment based on detected regime

## Integration with Stock_AI_Predictor

This RL system complements the existing pattern recognition and sentiment analysis:

1. **Pattern Recognition**: RL learns optimal trading of detected patterns
2. **Sentiment Analysis**: RL learns when sentiment impacts price and when it doesn't
3. **Risk Management**: RL optimizes position sizing and TP/SL placement

## Performance Evaluation

Evaluate the system on:
1. Total return and risk-adjusted metrics (Sharpe, Sortino)
2. Robustness across different market conditions
3. Adaptation speed to regime changes
4. Risk management effectiveness (max drawdown, VaR)

## Implementation Guide

To enhance the current RL system:

1. Expand the observation space to include additional features
2. Implement feature importance tracking to adapt to changing market conditions
3. Add temporal memory for better learning of time-based patterns
4. Create separate training regimes for different market conditions
5. Implement backtesting with proper walk-forward validation
6. Add adaptive hyperparameter optimization

## Usage Guide

### Training Models with Different Reward Functions

```bash
python -m RL.Scripts.train_compare_rewards
```

This will train models using various reward functions and produce comparison charts showing their performance differences.

### Evaluating a Trained Model

```bash
python -m RL.Scripts.evaluate_model --model_path "RL/Models/ppo_trading_combined.zip" --reward_type "combined"
```

### Testing Risk Management in Stress Scenarios

```bash
python -m RL.Scripts.test_risk_management
```

Tests all available models under normal, drawdown, volatility, and choppy market conditions to evaluate robustness.

## Recommended Workflow

1. **Train Models**: Use `train_compare_rewards.py` to train models with different reward functions
2. **Evaluate Performance**: Use `evaluate_model.py` to analyze model behavior and performance
3. **Stress Test**: Use `test_risk_management.py` to test model robustness under adverse conditions
4. **Select Best Model**: Choose the model that best aligns with your trading objectives (return, drawdown control, etc.)
5. **Deploy**: Implement the chosen model in your live trading system

## Performance Considerations

Different reward functions optimize for different trading objectives:
- **Sharpe Reward**: Best for consistent, risk-adjusted returns
- **Drawdown Focus**: Best for capital preservation and prop firm challenges
- **Win Rate**: Best for psychological comfort and consistent small wins
- **Combined**: Best for balanced performance across multiple metrics

Select the reward function that best matches your trading goals and risk tolerance.

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Deng, Y., et al. (2016). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading.
- Fischer, T. G. (2018). Reinforcement learning in financial markets-a survey.
