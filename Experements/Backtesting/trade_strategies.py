"""
Trade Strategy Module

This module implements various trading strategies for backtesting.
It provides standardized trade execution, exit rules, and position sizing.

Usage:
    Import this module to implement different trading strategies in backtests.
"""

from enum import Enum
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable


class TradeType(Enum):
    """Types of trades."""
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class ExitReason(Enum):
    """Reasons for trade exit."""
    TP_HIT = "TP Hit"
    SL_HIT = "SL Hit"
    TRAILING_SL = "Trailing SL"
    TIME_EXIT = "Hold Exit"
    SIGNAL_REVERSAL = "Signal Reversal"
    SESSION_END = "Session End"


class TradeStrategy:
    """Base class for trade strategies."""
    
    def __init__(self, name: str = "Base Strategy"):
        self.name = name
    
    def generate_signal(self, data: pd.DataFrame, i: int) -> TradeType:
        """
        Generate a trade signal at index i.
        
        Args:
            data: DataFrame with price data
            i: Current index
            
        Returns:
            TradeType enum value
        """
        return TradeType.NONE
    
    def calculate_target_and_stop(
        self, 
        entry_price: float, 
        trade_type: TradeType,
        data: pd.DataFrame = None,
        i: int = None
    ) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss levels.
        
        Args:
            entry_price: Entry price of the trade
            trade_type: Type of trade (BUY or SELL)
            data: Optional DataFrame with price data
            i: Optional current index
            
        Returns:
            Tuple of (take profit price, stop loss price)
        """
        # Default implementation with 1:1 risk-reward
        if trade_type == TradeType.BUY:
            take_profit = entry_price * 1.01  # +1%
            stop_loss = entry_price * 0.99    # -1%
        elif trade_type == TradeType.SELL:
            take_profit = entry_price * 0.99  # -1%
            stop_loss = entry_price * 1.01    # +1%
        else:
            take_profit = entry_price
            stop_loss = entry_price
            
        return take_profit, stop_loss
    
    def position_size(self, account_balance: float, risk_per_trade: float, 
                     entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Percentage of account to risk per trade (0-1)
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            
        Returns:
            Position size in units
        """
        risk_amount = account_balance * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        return risk_amount / price_risk


class PatternBasedStrategy(TradeStrategy):
    """Strategy based on pattern recognition."""
    
    def __init__(
        self,
        pattern_matcher: Callable,
        fixed_tp_pct: Optional[float] = None,
        fixed_sl_pct: Optional[float] = None,
        min_reward_risk: float = 1.0,
        name: str = "Pattern Based Strategy"
    ):
        """
        Initialize the strategy.
        
        Args:
            pattern_matcher: Function that matches patterns and returns trade information
            fixed_tp_pct: Optional fixed take profit percentage
            fixed_sl_pct: Optional fixed stop loss percentage
            min_reward_risk: Minimum reward-to-risk ratio to take a trade
            name: Strategy name
        """
        super().__init__(name)
        self.pattern_matcher = pattern_matcher
        self.fixed_tp_pct = fixed_tp_pct
        self.fixed_sl_pct = fixed_sl_pct
        self.min_reward_risk = min_reward_risk
        
    def generate_signal(self, data: pd.DataFrame, i: int) -> Tuple[TradeType, Dict[str, Any]]:
        """
        Generate a trade signal based on pattern matching.
        
        Args:
            data: DataFrame with price data
            i: Current index
            
        Returns:
            Tuple of (TradeType, dict with pattern details)
        """
        # Call the pattern matcher to get pattern information
        pattern_info = self.pattern_matcher(data, i)
        
        if pattern_info is None:
            return TradeType.NONE, {}
            
        # Extract trade type and pattern details
        if pattern_info['label'] == 'Buy':
            trade_type = TradeType.BUY
        elif pattern_info['label'] == 'Sell':
            trade_type = TradeType.SELL
        else:
            trade_type = TradeType.NONE
            
        # Check reward-to-risk ratio if required
        if trade_type != TradeType.NONE and self.min_reward_risk > 0:
            max_gain = abs(pattern_info.get('max_gain', 0))
            max_drawdown = abs(pattern_info.get('max_drawdown', 1e-6))
            reward_risk = max_gain / max_drawdown
            
            if reward_risk < self.min_reward_risk:
                trade_type = TradeType.NONE
                
        return trade_type, pattern_info
    
    def calculate_target_and_stop(
        self, 
        entry_price: float, 
        trade_type: TradeType,
        pattern_info: Dict[str, Any] = None
    ) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss levels based on pattern or fixed percentages.
        
        Args:
            entry_price: Entry price of the trade
            trade_type: Type of trade (BUY or SELL)
            pattern_info: Dictionary with pattern information
            
        Returns:
            Tuple of (take profit price, stop loss price)
        """
        # Use fixed percentages if provided
        if self.fixed_tp_pct is not None and self.fixed_sl_pct is not None:
            if trade_type == TradeType.BUY:
                take_profit = entry_price * (1 + self.fixed_tp_pct)
                stop_loss = entry_price * (1 - self.fixed_sl_pct)
            elif trade_type == TradeType.SELL:
                take_profit = entry_price * (1 - self.fixed_tp_pct)
                stop_loss = entry_price * (1 + self.fixed_sl_pct)
            else:
                take_profit = entry_price
                stop_loss = entry_price
                
            return take_profit, stop_loss
        
        # Otherwise, use pattern-derived values
        if pattern_info is not None:
            max_gain = pattern_info.get('max_gain', 0.02)
            max_drawdown = pattern_info.get('max_drawdown', 0.01)
            
            if trade_type == TradeType.BUY:
                take_profit = entry_price * (1 + max_gain)
                stop_loss = entry_price * (1 + max_drawdown)  # max_drawdown is negative
            elif trade_type == TradeType.SELL:
                take_profit = entry_price * (1 - max_gain)
                stop_loss = entry_price * (1 + max_drawdown)
            else:
                take_profit = entry_price
                stop_loss = entry_price
                
            return take_profit, stop_loss
        
        # Fallback to default
        return super().calculate_target_and_stop(entry_price, trade_type)


class VotingEnsembleStrategy(TradeStrategy):
    """Strategy that uses voting from multiple strategies."""
    
    def __init__(
        self,
        strategies: List[TradeStrategy],
        weights: Optional[List[float]] = None,
        threshold: float = 0.6,
        name: str = "Voting Ensemble Strategy"
    ):
        """
        Initialize the ensemble strategy.
        
        Args:
            strategies: List of strategies to use
            weights: Optional list of weights for each strategy (must sum to 1)
            threshold: Minimum vote fraction to generate a signal
            name: Strategy name
        """
        super().__init__(name)
        self.strategies = strategies
        
        # Normalize weights if provided, otherwise use equal weights
        if weights is None:
            self.weights = [1/len(strategies)] * len(strategies)
        else:
            assert len(weights) == len(strategies), "Weights must match number of strategies"
            weight_sum = sum(weights)
            self.weights = [w/weight_sum for w in weights]
            
        self.threshold = threshold
        
    def generate_signal(self, data: pd.DataFrame, i: int) -> Tuple[TradeType, Dict[str, Any]]:
        """
        Generate a trade signal based on voting from multiple strategies.
        
        Args:
            data: DataFrame with price data
            i: Current index
            
        Returns:
            Tuple of (TradeType, dict with voting results)
        """
        votes = {TradeType.BUY: 0, TradeType.SELL: 0, TradeType.NONE: 0}
        pattern_infos = {}
        
        # Collect votes from each strategy
        for j, strategy in enumerate(self.strategies):
            signal, pattern_info = strategy.generate_signal(data, i)
            votes[signal] += self.weights[j]
            pattern_infos[strategy.name] = pattern_info
            
        # Determine final signal based on threshold
        final_signal = TradeType.NONE
        max_vote = max(votes[TradeType.BUY], votes[TradeType.SELL])
        if max_vote >= self.threshold:
            if votes[TradeType.BUY] > votes[TradeType.SELL]:
                final_signal = TradeType.BUY
            else:
                final_signal = TradeType.SELL
                
        # Prepare voting results
        voting_results = {
            'votes': votes,
            'pattern_infos': pattern_infos,
            'final_signal': final_signal
        }
        
        return final_signal, voting_results
    
    def calculate_target_and_stop(
        self, 
        entry_price: float, 
        trade_type: TradeType,
        voting_results: Dict[str, Any] = None
    ) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss levels based on averaging
        from multiple strategies.
        
        Args:
            entry_price: Entry price of the trade
            trade_type: Type of trade (BUY or SELL)
            voting_results: Dictionary with voting results
            
        Returns:
            Tuple of (take profit price, stop loss price)
        """
        if voting_results is None or trade_type == TradeType.NONE:
            return super().calculate_target_and_stop(entry_price, trade_type)
            
        # Get take profit and stop loss from each strategy
        take_profits = []
        stop_losses = []
        
        for j, strategy in enumerate(self.strategies):
            pattern_info = voting_results['pattern_infos'][strategy.name]
            tp, sl = strategy.calculate_target_and_stop(entry_price, trade_type, pattern_info)
            take_profits.append(tp)
            stop_losses.append(sl)
            
        # Calculate weighted average
        avg_tp = sum(tp * self.weights[j] for j, tp in enumerate(take_profits))
        avg_sl = sum(sl * self.weights[j] for j, sl in enumerate(stop_losses))
        
        return avg_tp, avg_sl


def simulate_trade(
    prices: np.array,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    trade_type: TradeType,
    hold_period: int,
    use_trailing_stop: bool = False,
    trailing_pct: float = 0.01
) -> Tuple[float, float, ExitReason]:
    """
    Simulate a trade over a price series.
    
    Args:
        prices: Array of future prices
        entry_price: Entry price
        take_profit: Take profit price
        stop_loss: Stop loss price
        trade_type: Trade type (BUY or SELL)
        hold_period: Maximum hold period
        use_trailing_stop: Whether to use a trailing stop
        trailing_pct: Trailing stop percentage
        
    Returns:
        Tuple of (exit_price, profit_loss, exit_reason)
    """
    if len(prices) == 0:
        return entry_price, 0, ExitReason.SESSION_END
        
    # Limit hold period to available prices
    periods = min(hold_period, len(prices))
    
    # Track highest/lowest price for trailing stop
    highest_price = entry_price
    lowest_price = entry_price
    
    for i in range(periods):
        current_price = prices[i]
        
        # Update trailing stop if enabled
        if use_trailing_stop:
            if trade_type == TradeType.BUY and current_price > highest_price:
                highest_price = current_price
                # Update stop loss to trail by trailing_pct below highest price
                stop_loss = highest_price * (1 - trailing_pct)
            elif trade_type == TradeType.SELL and current_price < lowest_price:
                lowest_price = current_price
                # Update stop loss to trail by trailing_pct above lowest price
                stop_loss = lowest_price * (1 + trailing_pct)
        
        # Check for take profit or stop loss hits
        if trade_type == TradeType.BUY:
            if current_price >= take_profit:
                return current_price, current_price - entry_price, ExitReason.TP_HIT
            elif current_price <= stop_loss:
                if use_trailing_stop and highest_price > entry_price:
                    return current_price, current_price - entry_price, ExitReason.TRAILING_SL
                else:
                    return current_price, current_price - entry_price, ExitReason.SL_HIT
        elif trade_type == TradeType.SELL:
            if current_price <= take_profit:
                return current_price, entry_price - current_price, ExitReason.TP_HIT
            elif current_price >= stop_loss:
                if use_trailing_stop and lowest_price < entry_price:
                    return current_price, entry_price - current_price, ExitReason.TRAILING_SL
                else:
                    return current_price, entry_price - current_price, ExitReason.SL_HIT
    
    # Exit at the end of hold period with the final price
    final_price = prices[periods - 1]
    if trade_type == TradeType.BUY:
        return final_price, final_price - entry_price, ExitReason.TIME_EXIT
    elif trade_type == TradeType.SELL:
        return final_price, entry_price - final_price, ExitReason.TIME_EXIT
    else:
        return entry_price, 0, ExitReason.TIME_EXIT
