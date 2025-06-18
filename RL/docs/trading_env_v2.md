# TradingEnvV2: Enhanced Trading Environment

## Overview

`TradingEnvV2` is an enhanced version of the original trading environment that simulates more realistic trading conditions. The key improvements focus on multiple concurrent positions, position management across timesteps, and dynamic equity tracking.

## Key Features

1. **Multiple Concurrent Positions**
   - Supports up to 10 active trades simultaneously
   - Tracks each position independently
   - Allows both long and short positions

2. **Persistent Positions**
   - Positions remain open across multiple timesteps
   - TP/SL/time exit conditions checked on each step
   - Dynamic position management

3. **Realistic Portfolio Tracking**
   - Cash balance and position equity tracked separately
   - Unrealized PnL calculated in real-time
   - Drawdown and peak equity monitoring

4. **Enhanced Reward Function**
   - Portfolio-based reward calculation
   - Includes both realized and unrealized gains
   - Rewards for good trade management

5. **Vectorized Operations**
   - Optimized for performance using NumPy arrays
   - Batch processing of position checks
   - Efficient state management

## Comparison with TradingEnv (V1)

| Feature | TradingEnv (V1) | TradingEnvV2 |
|---------|----------------|-------------|
| Positions | Single position | Multiple concurrent positions (up to 10) |
| Position Lifecycle | Immediate execution & outcome | Persistent across multiple timesteps |
| PnL Tracking | Only realized PnL | Both realized and unrealized PnL |
| Balance | Single balance value | Separate cash + positions equity |
| Trade Exits | Simulated all at once | Checked at each timestep (TP/SL/time) |
| Reward | Based on individual trades | Based on portfolio performance |
| Risk Management | Basic | Advanced with position sizing limits |

## Usage Example

```python
from RL.Envs.trading_env_v2 import TradingEnvV2
from stable_baselines3 import PPO

# Create environment
env = TradingEnvV2(
    features=features_data,
    initial_balance=100000,
    reward_type="combined",
    max_positions=10,
    commission_rate=0.001
)

# Train a model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test the model
obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # Access portfolio information
    print(f"Equity: ${info['equity']:.2f}")
    print(f"Active positions: {info['active_positions']}")
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `features` | DataFrame with market features | None |
| `initial_balance` | Starting account balance | 100000 |
| `reward_type` | Type of reward function to use | "combined" |
| `normalize_observations` | Whether to normalize observations | True |
| `normalization_range` | Range for normalized observations | (-1.0, 1.0) |
| `timeframe_id` | Identifier for the timeframe | 5 |
| `stock_id` | Identifier for the stock | 1 |
| `start_date` | Start date for data range | "2024-01-01" |
| `end_date` | End date for data range | "2025-01-01" |
| `max_positions` | Maximum number of concurrent positions | 10 |
| `commission_rate` | Transaction cost as a percentage | 0.001 |
| `enable_short` | Whether to allow short selling | True |

## Position Management

Each position is tracked as a dictionary with the following information:
- Unique position ID
- Direction (long/short)
- Size (units)
- Entry price and timestamp
- Take-profit and stop-loss levels
- Intended holding time
- Risk parameters

Positions are automatically closed when:
1. Take-profit level is reached
2. Stop-loss level is reached
3. Maximum holding time is reached

## Reward Function

The reward function is portfolio-focused and includes multiple components:

1. **Portfolio Change**: Based on equity growth
2. **Trade Outcomes**: Rewards for successful trade exits
3. **Trade Quality**: Rewards for good entry/exit timing
4. **Risk Management**: Penalties for excessive drawdowns
5. **Capital Efficiency**: Small rewards for efficient capital usage

All these components are combined and adjusted using the `RewardCalculator` from the original environment, ensuring compatibility with existing reward types.

## Implementation Details

The environment uses efficient data structures and vectorized operations wherever possible:
- NumPy arrays for price data
- List of dictionaries for active positions
- OrderedDict for trade history
- Vectorized calculations for unrealized PnL

## Integration

The new environment maintains the same API as the original `TradingEnv`, making it a drop-in replacement for most applications. The observation and action spaces remain identical.
