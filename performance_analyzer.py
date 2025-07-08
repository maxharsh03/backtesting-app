import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and risk metrics calculator
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PerformanceAnalyzer
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_comprehensive_metrics(self, 
                                      portfolio_history: pd.DataFrame,
                                      benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_history: DataFrame with portfolio values over time
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Dictionary with all performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(portfolio_history))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(portfolio_history))
        
        # Advanced metrics
        metrics.update(self._calculate_advanced_metrics(portfolio_history))
        
        # Benchmark comparison
        if benchmark_data is not None:
            metrics.update(self._calculate_benchmark_metrics(portfolio_history, benchmark_data))
        
        return metrics
    
    def _calculate_return_metrics(self, portfolio_history: pd.DataFrame) -> Dict:
        """Calculate return-based metrics"""
        returns = portfolio_history['returns'].dropna()
        portfolio_values = portfolio_history['portfolio_value']
        
        # Total return
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        
        # Annualized return
        trading_days = len(portfolio_history)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Compound Annual Growth Rate (CAGR)
        cagr = annualized_return
        
        # Monthly and daily returns
        monthly_return = returns.mean() * 21  # Approximate monthly return
        daily_return = returns.mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cagr': cagr,
            'monthly_return': monthly_return,
            'daily_return': daily_return
        }
    
    def _calculate_risk_metrics(self, portfolio_history: pd.DataFrame) -> Dict:
        """Calculate risk-based metrics"""
        returns = portfolio_history['returns'].dropna()
        portfolio_values = portfolio_history['portfolio_value']
        
        # Volatility
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_periods(drawdown)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        return {
            'daily_volatility': daily_volatility,
            'annualized_volatility': annualized_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': drawdown_periods['avg_duration'],
            'max_drawdown_duration': drawdown_periods['max_duration'],
            'downside_deviation': downside_deviation
        }
    
    def _calculate_advanced_metrics(self, portfolio_history: pd.DataFrame) -> Dict:
        """Calculate advanced performance metrics"""
        returns = portfolio_history['returns'].dropna()
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < self.risk_free_rate/252]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Calmar ratio
        annualized_return = returns.mean() * 252
        max_drawdown = abs(self._calculate_risk_metrics(portfolio_history)['max_drawdown'])
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Win/Loss ratios
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        total_days = len(returns)
        
        win_rate = winning_days / total_days if total_days > 0 else 0
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
        
        # Profit factor
        total_gains = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        # Maximum consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(returns > 0)
        max_consecutive_losses = self._calculate_max_consecutive(returns < 0)
        
        # Stability metrics
        rolling_sharpe = self._calculate_rolling_sharpe(returns)
        sharpe_stability = rolling_sharpe.std() if len(rolling_sharpe) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'sharpe_stability': sharpe_stability
        }
    
    def _calculate_benchmark_metrics(self, portfolio_history: pd.DataFrame, 
                                   benchmark_data: pd.DataFrame) -> Dict:
        """Calculate benchmark comparison metrics"""
        portfolio_returns = portfolio_history['returns'].dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Align data
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return {}
        
        # Beta calculation
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        portfolio_annual_return = portfolio_returns.mean() * 252
        benchmark_annual_return = benchmark_returns.mean() * 252
        alpha = portfolio_annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        
        # Tracking error
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        
        # Information ratio
        excess_return = portfolio_returns.mean() - benchmark_returns.mean()
        information_ratio = (excess_return * 252) / tracking_error if tracking_error > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        
        return {
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation_with_benchmark': correlation
        }
    
    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> Dict:
        """Calculate drawdown period statistics"""
        # Find drawdown periods, handling NaN values
        is_drawdown = (drawdown < 0).fillna(False)
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        durations = []
        start_idx = None
        
        for i, (start, end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if start:
                start_idx = i
            if end and start_idx is not None:
                durations.append(i - start_idx)
                start_idx = None
        
        if len(durations) == 0:
            return {'avg_duration': 0, 'max_duration': 0}
        
        return {
            'avg_duration': np.mean(durations),
            'max_duration': np.max(durations)
        }
    
    def _calculate_max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive occurrences"""
        if len(condition) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        return (rolling_mean - self.risk_free_rate/252) / rolling_std * np.sqrt(252)
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """
        Generate a formatted performance report
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Formatted performance report string
        """
        report = "="*60 + "\n"
        report += "PERFORMANCE ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"
        
        # Return Metrics
        report += "RETURN METRICS:\n"
        report += "-"*30 + "\n"
        report += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
        report += f"Annualized Return: {metrics.get('annualized_return', 0):.2%}\n"
        report += f"CAGR: {metrics.get('cagr', 0):.2%}\n"
        report += f"Monthly Return: {metrics.get('monthly_return', 0):.2%}\n"
        report += f"Daily Return: {metrics.get('daily_return', 0):.4%}\n\n"
        
        # Risk Metrics
        report += "RISK METRICS:\n"
        report += "-"*30 + "\n"
        report += f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}\n"
        report += f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"VaR (95%): {metrics.get('var_95', 0):.2%}\n"
        report += f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}\n"
        report += f"Downside Deviation: {metrics.get('downside_deviation', 0):.2%}\n\n"
        
        # Advanced Metrics
        report += "ADVANCED METRICS:\n"
        report += "-"*30 + "\n"
        report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
        report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        report += f"Skewness: {metrics.get('skewness', 0):.2f}\n"
        report += f"Kurtosis: {metrics.get('kurtosis', 0):.2f}\n\n"
        
        # Benchmark Metrics (if available)
        if 'beta' in metrics:
            report += "BENCHMARK COMPARISON:\n"
            report += "-"*30 + "\n"
            report += f"Beta: {metrics.get('beta', 0):.2f}\n"
            report += f"Alpha: {metrics.get('alpha', 0):.2%}\n"
            report += f"Information Ratio: {metrics.get('information_ratio', 0):.2f}\n"
            report += f"Correlation: {metrics.get('correlation_with_benchmark', 0):.2f}\n\n"
        
        report += "="*60 + "\n"
        
        return report
    
    def analyze_trade_performance(self, trades: List[Dict]) -> Dict:
        """
        Analyze individual trade performance
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Trade analysis metrics
        """
        if not trades:
            return {}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate P&L for each trade
        trades_df['pnl'] = trades_df['quantity'] * trades_df['price']
        trades_df['total_cost'] = trades_df['commission'] + trades_df['slippage']
        trades_df['net_pnl'] = trades_df['pnl'] - trades_df['total_cost']
        
        # Trade statistics
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Risk-reward ratio
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Trading frequency
        date_range = (trades_df['date'].max() - trades_df['date'].min()).days
        trades_per_day = num_trades / date_range if date_range > 0 else 0
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'trades_per_day': trades_per_day,
            'total_commission': trades_df['commission'].sum(),
            'total_slippage': trades_df['slippage'].sum(),
            'gross_pnl': trades_df['pnl'].sum(),
            'net_pnl': trades_df['net_pnl'].sum()
        }