import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time
warnings.filterwarnings('ignore')

class BacktestingEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 risk_free_rate: float = 0.02):
        """
        Initialize the backtesting engine
        
        Args:
            initial_capital: Starting capital for backtest
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            risk_free_rate: Risk-free rate for metrics calculation
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        
        # Initialize tracking variables
        self.reset_backtest()
    
    def reset_backtest(self):
        """Reset backtest state"""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.drawdown_history = []
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
        
    def run_backtest(self, data: pd.DataFrame, strategy) -> Dict:
        """
        Run complete backtest on given data with strategy
        
        Args:
            data: Market data with OHLCV columns
            strategy: Strategy object with calculate_signals method
            
        Returns:
            Dictionary with backtest results
        """
        self.reset_backtest()
        
        # Calculate strategy signals
        signals_data = strategy.calculate_signals(data)
        
        # Process each day
        for i in range(len(signals_data)):
            row = signals_data.iloc[i]
            self._process_day(row, strategy)
            
        # Calculate final metrics
        results = self._calculate_metrics(signals_data)
        results['trades'] = self.trades
        results['portfolio_history'] = self.portfolio_history
        results['signals_data'] = signals_data
        
        return results
    
    def _process_day(self, row: pd.Series, strategy):
        """Process a single day of trading"""
        current_date = row.name
        current_price = row['Close']
        signal = row['signal']
        position = row['position']
        
        # Update portfolio value
        self.portfolio_value = self.cash
        for symbol, pos_info in self.positions.items():
            self.portfolio_value += pos_info['quantity'] * current_price
        
        # Calculate position size
        temp_data = pd.DataFrame([row]).T.T
        temp_data = strategy.calculate_position_size(temp_data, self.portfolio_value)
        desired_position_size = temp_data['position_size'].iloc[0] if 'position_size' in temp_data.columns else 0
        
        # Execute trades
        if signal != 0:
            self._execute_trade(current_date, current_price, desired_position_size, signal)
        
        # Update drawdown
        self._update_drawdown()
        
        # Record portfolio state
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': self.portfolio_value - self.cash,
            'drawdown': self.current_drawdown
        })
    
    def _execute_trade(self, date, price: float, position_size: float, signal: int):
        """Execute a trade"""
        symbol = 'ASSET'  # Generic symbol for single asset backtest
        
        # Calculate trade details
        trade_value = abs(position_size) * price
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate
        total_cost = commission + slippage
        
        # Check if we have enough cash
        if signal > 0 and (trade_value + total_cost) > self.cash:
            # Adjust position size based on available cash
            available_cash = self.cash - total_cost
            position_size = available_cash / price
            trade_value = position_size * price
        
        # Execute trade
        if abs(position_size) > 0:
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
            
            old_quantity = self.positions[symbol]['quantity']
            new_quantity = old_quantity + position_size
            
            # Update cash
            self.cash -= (position_size * price + total_cost)
            
            # Update position
            if new_quantity != 0:
                self.positions[symbol]['quantity'] = new_quantity
                self.positions[symbol]['avg_price'] = price
            else:
                del self.positions[symbol]
            
            # Record trade
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'price': price,
                'quantity': position_size,
                'value': trade_value,
                'commission': commission,
                'slippage': slippage,
                'signal': signal
            })
    
    def _update_drawdown(self):
        """Update drawdown calculations"""
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
            if self.current_drawdown < self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.portfolio_history) == 0:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Basic metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        num_trades = len(self.trades)
        
        # Calculate annualized metrics
        trading_days = len(portfolio_df)
        years = trading_days / 252  # Assuming 252 trading days per year
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1/years) - 1
        else:
            annualized_return = 0
            
        # Risk metrics
        daily_returns = portfolio_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trades if t['quantity'] * t['price'] > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Calculate benchmark (buy and hold)
        benchmark_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'final_portfolio_value': self.portfolio_value,
            'benchmark_return': benchmark_return,
            'excess_return_vs_benchmark': total_return - benchmark_return,
            'portfolio_df': portfolio_df
        }

def _run_single_simulation(args):
    """
    Run a single Monte Carlo simulation - designed for multiprocessing
    """
    data, strategy_params, backtest_params, seed = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create fresh instances for this process
    from mean_reversion_strategy import MeanReversionStrategy
    strategy = MeanReversionStrategy(**strategy_params)
    backtest_engine = BacktestingEngine(**backtest_params)
    
    # Bootstrap sample the data
    sample_data = data.sample(n=len(data), replace=True).sort_index()
    
    # Run backtest on sample
    result = backtest_engine.run_backtest(sample_data, strategy)
    
    # Return only the metrics we need to save memory
    return {
        'total_return': result.get('total_return', 0),
        'annualized_return': result.get('annualized_return', 0),
        'volatility': result.get('volatility', 0),
        'sharpe_ratio': result.get('sharpe_ratio', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'win_rate': result.get('win_rate', 0),
        'num_trades': result.get('num_trades', 0)
    }

class MonteCarloSimulator:
    """
    Optimized Monte Carlo simulation for strategy robustness testing
    """
    
    def __init__(self, num_simulations: int = 100, use_multiprocessing: bool = True, n_jobs: int = None):
        self.num_simulations = num_simulations
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs or min(cpu_count(), 8)  # Limit to 8 cores to avoid overloading
    
    def run_simulation(self, data: pd.DataFrame, strategy, backtest_engine: BacktestingEngine) -> Dict:
        """
        Run optimized Monte Carlo simulation with multiprocessing
        
        Args:
            data: Historical market data
            strategy: Trading strategy
            backtest_engine: Backtesting engine
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running {self.num_simulations} Monte Carlo simulations using {self.n_jobs} cores...")
        start_time = time.time()
        
        # Prepare parameters for multiprocessing
        strategy_params = {
            'lookback_period': strategy.lookback_period,
            'entry_zscore': strategy.entry_zscore,
            'exit_zscore': strategy.exit_zscore,
            'stop_loss_zscore': strategy.stop_loss_zscore,
            'position_sizing': strategy.position_sizing,
            'risk_per_trade': strategy.risk_per_trade
        }
        
        backtest_params = {
            'initial_capital': backtest_engine.initial_capital,
            'commission_rate': backtest_engine.commission_rate,
            'slippage_rate': backtest_engine.slippage_rate,
            'risk_free_rate': backtest_engine.risk_free_rate
        }
        
        # Create arguments for each simulation
        args_list = [
            (data, strategy_params, backtest_params, i) 
            for i in range(self.num_simulations)
        ]
        
        # Run simulations
        if self.use_multiprocessing and self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(_run_single_simulation, args_list)
        else:
            # Fallback to sequential processing
            results = []
            for i, args in enumerate(args_list):
                if i % 10 == 0:
                    print(f"Progress: {i}/{self.num_simulations} simulations completed")
                result = _run_single_simulation(args)
                results.append(result)
        
        elapsed_time = time.time() - start_time
        print(f"Monte Carlo simulation completed in {elapsed_time:.2f} seconds")
        
        # Aggregate results
        metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                  'max_drawdown', 'win_rate', 'num_trades']
        
        simulation_summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
            if values:
                simulation_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentile_5': np.percentile(values, 5),
                    'percentile_95': np.percentile(values, 95)
                }
        
        return {
            'simulation_summary': simulation_summary,
            'all_results': results,
            'execution_time': elapsed_time
        }