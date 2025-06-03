# RL Trading Environment Specifications

## Overview
This document provides complete specifications for the Reinforcement Learning trading environment, including action space, observation space, reward functions, and key parameters.

---

## Action Space

### MultiDiscrete Action Space: `[3, 10, 10]`

The agent takes actions consisting of three components:

| Component | Index | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| **Action Type** | 0 | Discrete | 0-2 | Trading decision |
| **Position Size** | 1 | Discrete | 0-9 | Position sizing index |
| **Risk-Reward Ratio** | 2 | Discrete | 0-9 | Risk management multiplier |

### Action Type Details

| Value | Action | Description |
|-------|--------|-------------|
| 0 | HOLD | No trading action, maintain current position |
| 1 | BUY | Open long position or add to existing long |
| 2 | SELL | Open short position or add to existing short |

### Position Size Mapping

| Index | Position Size | Portfolio Allocation |
|-------|--------------|---------------------|
| 0 | 0.1 | 10% of available balance |
| 1 | 0.2 | 20% of available balance |
| 2 | 0.3 | 30% of available balance |
| 3 | 0.4 | 40% of available balance |
| 4 | 0.5 | 50% of available balance |
| 5 | 0.6 | 60% of available balance |
| 6 | 0.7 | 70% of available balance |
| 7 | 0.8 | 80% of available balance |
| 8 | 0.9 | 90% of available balance |
| 9 | 1.0 | 100% of available balance |

### Risk-Reward Ratio Mapping

| Index | Multiplier | Take Profit/Stop Loss Ratio |
|-------|------------|----------------------------|
| 0 | 0.50 | Conservative (1:2 risk-reward) |
| 1 | 0.75 | Cautious |
| 2 | 1.00 | Balanced (1:1 risk-reward) |
| 3 | 1.25 | Slightly aggressive |
| 4 | 1.50 | Moderate aggressive |
| 5 | 1.75 | Aggressive |
| 6 | 2.00 | High risk (2:1 risk-reward) |
| 7 | 2.25 | Very aggressive |
| 8 | 2.50 | Extremely aggressive |
| 9 | 3.00 | Maximum risk (3:1 risk-reward) |

---

## Observation Space

### Total Features: 31
Box space with shape `(31,)`, normalized to range `[-1.0, 1.0]`

### Feature Categories

#### 1. Base Pattern Features (7 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | `probability` | Float | Probability of successful trade from pattern recognition |
| 1 | `action` | Float | Pattern action signal (1=bullish, 2=bearish, 0=neutral) |
| 2 | `reward_risk_ratio` | Float | Expected reward-to-risk ratio from pattern |
| 3 | `max_gain` | Float | Maximum potential gain from pattern |
| 4 | `max_drawdown` | Float | Maximum potential drawdown from pattern |
| 5 | `mse` | Float | Mean squared error of pattern prediction |
| 6 | `expected_value` | Float | Expected value of the trading opportunity |

#### 2. Technical Indicators (3 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 7 | `rsi` | Float | Relative Strength Index (0-100) |
| 8 | `atr` | Float | Average True Range (volatility measure) |
| 9 | `atr_ratio` | Float | ATR relative to price (normalized volatility) |

#### 3. Sentiment Features (2 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 10 | `unified_sentiment` | Float | Combined sentiment score from multiple sources |
| 11 | `sentiment_count` | Float | Number of sentiment data points used |

#### 4. COT (Commitment of Traders) Data (6 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 12 | `net_noncommercial` | Float | Net non-commercial positions |
| 13 | `net_nonreportable` | Float | Net non-reportable positions |
| 14 | `change_nonrept_long` | Float | Change in non-reportable long positions |
| 15 | `change_nonrept_short` | Float | Change in non-reportable short positions |
| 16 | `change_noncommercial_long` | Float | Change in non-commercial long positions |
| 17 | `change_noncommercial_short` | Float | Change in non-commercial short positions |

#### 5. Time-Based Features (7 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 18 | `hour_sin` | Float | Sine of hour (cyclical time encoding) |
| 19 | `hour_cos` | Float | Cosine of hour (cyclical time encoding) |
| 20 | `day_sin` | Float | Sine of day (cyclical time encoding) |
| 21 | `day_cos` | Float | Cosine of day (cyclical time encoding) |
| 22 | `asian_session` | Float | Asian trading session indicator (0 or 1) |
| 23 | `london_session` | Float | London trading session indicator (0 or 1) |
| 24 | `ny_session` | Float | New York trading session indicator (0 or 1) |

#### 6. Portfolio Features (6 features)
| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 25 | `balance_ratio` | Float | Current balance / initial balance |
| 26 | `position_ratio` | Float | Position value / total portfolio value |
| 27 | `position` | Float | Absolute position size (shares/contracts) |
| 28 | `max_drawdown_portfolio` | Float | Maximum drawdown experienced |
| 29 | `win_rate` | Float | Winning trades / total trades |
| 30 | `steps_without_action` | Float | Consecutive HOLD actions (normalized) |

---

## Reward Functions

### Available Reward Types

| Type | Description | Focus |
|------|-------------|-------|
| `standard` | Basic profit/loss based rewards | Raw returns |
| `sharpe` | Sharpe ratio optimization | Risk-adjusted returns |
| `sortino` | Sortino ratio optimization | Downside risk minimization |
| `drawdown_focus` | Heavy drawdown penalties | Capital preservation |
| `calmar` | Calmar ratio optimization | Return/max drawdown ratio |
| `win_rate` | Win rate and profit factor | Consistency |
| `consistency` | Low volatility returns | Stable performance |
| `combined` | **Default** - Balanced approach | Multiple risk metrics |

### Reward Components

#### Base Reward Calculation
- **Profitable trades**: Positive reward proportional to profit
- **Unprofitable trades**: Negative reward proportional to loss
- **HOLD actions**: Small penalty (-0.0001) to encourage trading when appropriate

#### Risk Adjustments
- **Drawdown penalties**: Progressive penalties for increasing drawdowns
- **Volatility adjustment**: Rewards adjusted based on return volatility  
- **Win rate bonus**: Additional rewards for maintaining high win rates
- **Consistency bonus**: Rewards for stable, predictable returns

---

## Environment Parameters

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | Market data with features |
| `initial_balance` | Float | 100,000 | Starting account balance |
| `reward_type` | String | 'combined' | Type of reward function |
| `normalize_observations` | Bool | True | Enable observation normalization |
| `normalization_range` | Tuple | (-1.0, 1.0) | Range for normalized values |
| `enable_adaptive_scaling` | Bool | False | Enable adaptive feature scaling |

### Episode Termination Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **Severe Drawdown** | Balance < 90% of initial | Episode ends with penalty |
| **Data Exhaustion** | End of dataset reached | Episode ends normally |
| **Excessive Holds** | 100+ consecutive HOLD actions | Episode ends with penalty |

---

## Performance Metrics

### Real-time Tracking

| Metric | Type | Description |
|--------|------|-------------|
| `portfolio_balance` | Float | Current total account value |
| `position` | Float | Current position size |
| `entry_price` | Float | Entry price of current position |
| `trade_direction` | Int | 1=long, -1=short, 0=neutral |
| `trade_pnl` | Float | Profit/loss of completed trade |
| `win_rate` | Float | Percentage of winning trades |
| `max_drawdown` | Float | Maximum drawdown experienced |

### Action Statistics

| Metric | Description |
|--------|-------------|
| `action_counts` | Count of each action type taken |
| `position_sizes_counts` | Distribution of position sizes used |
| `risk_reward_ratios_counts` | Distribution of risk-reward ratios used |
| `steps_without_action` | Consecutive HOLD actions counter |

---

## Advanced Features

### Adaptive Position Sizing

Position sizes are automatically adjusted based on:

1. **Drawdown Protection**: Reduces size during drawdowns
   - 2% drawdown: 20% reduction
   - 5% drawdown: 48% reduction (cumulative)
   - 8% drawdown: 76% reduction (cumulative)

2. **Performance Streaks**: 
   - **Winning streaks**: Up to 20% increase in position size
   - **Losing streaks**: Up to 50% decrease in position size

3. **Volatility Adjustment**:
   - High volatility (ATR > 1.5x baseline): 30% reduction
   - Low volatility (ATR < 0.7x baseline): 20% increase

### Dynamic Stop-Loss and Take-Profit

TP/SL levels are calculated based on:
- Pattern-based max gain/drawdown expectations
- Risk-reward multiplier from agent's action
- Agreement/disagreement with pattern signals

**Example for Long Position (BUY)**:
- **Agreeing with bullish pattern**: TP = entry × (1 + max_gain × RR_multiplier)
- **Contradicting bearish pattern**: TP = entry × (1 + abs(max_drawdown) × RR_multiplier)

---

## Usage Examples

### Basic Environment Creation
```python
from RL.Envs.trading_env import TradingEnv

env = TradingEnv(
    data=market_data,
    initial_balance=100000,
    reward_type='combined',
    normalize_observations=True
)
```

### Action Interpretation
```python
# Example action: [1, 5, 7] 
# - Action type: 1 (BUY)
# - Position size: 5 → 60% of balance
# - Risk-reward: 7 → 2.25x multiplier (aggressive)

action = [1, 5, 7]
observation, reward, done, truncated, info = env.step(action)
```

### Monitoring Performance
```python
# Access performance metrics
print(f"Portfolio Value: ${info['portfolio_balance']:,.2f}")
print(f"Win Rate: {info['win_rate']:.1%}")
print(f"Max Drawdown: {info['max_drawdown']:.1%}")
```

---

## Model Compatibility

### Supported Algorithms
- **PPO (Proximal Policy Optimization)** - Recommended
- **A2C (Advantage Actor-Critic)**
- **TRPO (Trust Region Policy Optimization)**
- Any algorithm supporting MultiDiscrete action spaces

### Integration with Stable-Baselines3
```python
from stable_baselines3 import PPO

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    verbose=1
)
```

---

This environment provides a comprehensive framework for training RL agents to make sophisticated trading decisions with proper risk management and performance tracking.
