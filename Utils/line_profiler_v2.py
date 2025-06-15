import sys
from pathlib import Path
import line_profiler

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from Backtesting.strategies import RLTradingStrategy
from Backtesting.evaluator import BacktestingPyEvaluator
from Backtesting.data_provider import ObservationManager
from Backtesting.Tests.test_backtrading_py_evaluator import test_portfolio_evaluator

# Setup line profiler
profile = line_profiler.LineProfiler()

# Profile specific methods that may be slow
# profile.add_function(RLTradingStrategy.next)
profile.add_function(RLTradingStrategy._get_rl_action)
# profile.add_function(RLTradingStrategy._execute_action)
# profile.add_function(RLTradingStrategy._calculate_portfolio_features)
# profile.add_function(RLTradingStrategy._manage_risk_limits)
# profile.add_function(BacktestingPyEvaluator.evaluate_portfolio)
# profile.add_function(BacktestingPyEvaluator._prepare_evaluation_data)

profile.add_function(ObservationManager.get_market_observation)

# Run the profiled code
profile.run('test_portfolio_evaluator()')

# Print results
profile.print_stats()

# Save to file
with open('line_profile_results.txt', 'w') as f:
    profile.print_stats(stream=f)