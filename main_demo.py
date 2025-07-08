#!/usr/bin/env python3
"""
Comprehensive Mean Reversion Trading Strategy Demo

This script demonstrates the complete workflow of:
1. Data acquisition and preprocessing
2. Mean reversion strategy implementation
3. Backtesting and performance analysis
4. Monte Carlo simulation
5. Model validation and statistical analysis
6. Data visualization and reporting

Author: Trading Strategy Framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from mean_reversion_strategy import MeanReversionStrategy
from backtesting_engine import BacktestingEngine, MonteCarloSimulator
from data_manager import DataManager
from performance_analyzer import PerformanceAnalyzer
from visualization_dashboard import VisualizationDashboard
from model_validation import ModelValidator

def main():
    """
    Main demonstration function
    """
    print("="*80)
    print("MEAN REVERSION TRADING STRATEGY - COMPREHENSIVE DEMO")
    print("="*80)
    
    # Configuration
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000
    
    print(f"\nStrategy Configuration:")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,}")
    
    # Initialize components
    data_manager = DataManager(cache_data=True)
    strategy = MeanReversionStrategy(
        lookback_period=20,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        position_sizing='volatility_adjusted',
        risk_per_trade=0.02
    )
    backtest_engine = BacktestingEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    performance_analyzer = PerformanceAnalyzer()
    visualization_dashboard = VisualizationDashboard()
    model_validator = ModelValidator()
    
    print("\n" + "="*80)
    print("STEP 1: DATA ACQUISITION AND PREPROCESSING")
    print("="*80)
    
    # Fetch market data
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    data = data_manager.fetch_data(symbol, start_date, end_date)
    
    if data.empty:
        print("Failed to fetch data. Using synthetic data for demonstration...")
        data = data_manager.create_synthetic_data(start_date, end_date)
    
    # Add technical indicators
    data = data_manager.add_technical_indicators(data)
    
    # Get data info
    data_info = data_manager.get_data_info(data)
    print(f"Data shape: {data_info['shape']}")
    print(f"Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
    print(f"Price range: ${data_info['price_stats']['min']:.2f} - ${data_info['price_stats']['max']:.2f}")
    
    # Split data for walk-forward analysis
    train_data, test_data = data_manager.split_data(data, train_ratio=0.8)
    print(f"Training data: {len(train_data)} days")
    print(f"Testing data: {len(test_data)} days")
    
    print("\n" + "="*80)
    print("STEP 2: STRATEGY BACKTESTING")
    print("="*80)
    
    # Run backtest on full data
    print("Running backtest on full dataset...")
    backtest_results = backtest_engine.run_backtest(data, strategy)
    
    # Display basic results
    print(f"\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Annualized Return: {backtest_results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Number of Trades: {backtest_results['num_trades']}")
    print(f"Win Rate: {backtest_results['win_rate']:.2%}")
    
    # Compare with benchmark
    benchmark_return = backtest_results['benchmark_return']
    excess_return = backtest_results['excess_return_vs_benchmark']
    print(f"Benchmark Return: {benchmark_return:.2%}")
    print(f"Excess Return: {excess_return:.2%}")
    
    print("\n" + "="*80)
    print("STEP 3: COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Calculate comprehensive metrics
    portfolio_history = backtest_results['portfolio_df']
    comprehensive_metrics = performance_analyzer.calculate_comprehensive_metrics(portfolio_history)
    
    # Generate performance report
    performance_report = performance_analyzer.generate_performance_report(comprehensive_metrics)
    print(performance_report)
    
    # Analyze individual trades
    if backtest_results['trades']:
        trade_analysis = performance_analyzer.analyze_trade_performance(backtest_results['trades'])
        print(f"Trade Analysis:")
        print(f"Average Win: ${trade_analysis['avg_win']:.2f}")
        print(f"Average Loss: ${trade_analysis['avg_loss']:.2f}")
        print(f"Risk-Reward Ratio: {trade_analysis['risk_reward_ratio']:.2f}")
        print(f"Total Commission: ${trade_analysis['total_commission']:.2f}")
    
    print("\n" + "="*80)
    print("STEP 4: MONTE CARLO SIMULATION")
    print("="*80)
    
    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation (100 iterations)...")
    monte_carlo = MonteCarloSimulator(num_simulations=100, use_multiprocessing=True)
    simulation_results = monte_carlo.run_simulation(data, strategy, backtest_engine)
    
    # Display simulation results
    if 'simulation_summary' in simulation_results:
        print(f"\nMonte Carlo Results:")
        summary = simulation_results['simulation_summary']
        
        for metric, stats in summary.items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  95% CI: [{stats['percentile_5']:.4f}, {stats['percentile_95']:.4f}]")
    
    print("\n" + "="*80)
    print("STEP 5: MODEL VALIDATION AND STATISTICAL ANALYSIS")
    print("="*80)
    
    # Validate mean reversion assumptions
    print("Validating statistical assumptions...")
    validation_results = model_validator.validate_mean_reversion_assumptions(data)
    
    # Display validation results
    print(f"\nStatistical Validation:")
    
    if 'stationarity' in validation_results:
        stationarity = validation_results['stationarity']
        print(f"Stationarity: {stationarity['interpretation']}")
    
    if 'mean_reversion' in validation_results:
        mean_reversion = validation_results['mean_reversion']
        print(f"Mean Reversion: {mean_reversion['interpretation']['meaning']}")
        print(f"Hurst Exponent: {mean_reversion['hurst_exponent']:.3f}")
    
    if 'half_life' in validation_results:
        half_life = validation_results['half_life']
        print(f"Half-life: {half_life['interpretation']}")
    
    # Cross-validation
    print("\nRunning cross-validation...")
    cv_results = model_validator.cross_validate_strategy(data, strategy, backtest_engine, n_splits=5)
    
    if 'cv_summary' in cv_results:
        print(f"Cross-Validation Results:")
        for metric, stats in cv_results['cv_summary'].items():
            print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Parameter sensitivity analysis
    print("\nRunning parameter sensitivity analysis...")
    param_ranges = {
        'lookback_period': [10, 15, 20, 25, 30],
        'entry_zscore': [1.5, 2.0, 2.5, 3.0],
        'exit_zscore': [0.2, 0.5, 0.8, 1.0]
    }
    
    sensitivity_results = model_validator.parameter_sensitivity_analysis(
        data, MeanReversionStrategy, backtest_engine, param_ranges
    )
    
    best_params = sensitivity_results['best_parameters']
    print(f"Best Parameters (by Sharpe ratio): {best_params}")
    
    # Robustness tests
    print("\nRunning robustness tests...")
    robustness_results = model_validator.run_robustness_tests(data, strategy, backtest_engine)
    
    if 'subperiod' in robustness_results:
        consistency_score = robustness_results['subperiod']['consistency_score']
        print(f"Consistency Score: {consistency_score:.3f}")
    
    print("\n" + "="*80)
    print("STEP 6: DATA VISUALIZATION AND REPORTING")
    print("="*80)
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Creating visualizations...")
    
    # Portfolio performance chart
    portfolio_fig = visualization_dashboard.create_portfolio_performance_chart(
        portfolio_history,
        save_path=f'{output_dir}/portfolio_performance.html'
    )
    
    # Strategy signals chart
    signals_fig = visualization_dashboard.create_strategy_signals_chart(
        backtest_results['signals_data'],
        save_path=f'{output_dir}/strategy_signals.html'
    )
    
    # Risk metrics chart
    risk_fig = visualization_dashboard.create_risk_metrics_chart(
        comprehensive_metrics,
        save_path=f'{output_dir}/risk_metrics.html'
    )
    
    # Monte Carlo results chart
    mc_fig = visualization_dashboard.create_monte_carlo_results_chart(
        simulation_results,
        save_path=f'{output_dir}/monte_carlo_results.html'
    )
    
    # Summary dashboard
    summary_fig = visualization_dashboard.create_summary_dashboard(
        portfolio_history,
        backtest_results['signals_data'],
        comprehensive_metrics,
        save_path=f'{output_dir}/summary_dashboard.html'
    )
    
    print(f"Charts saved to {output_dir}/ directory")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    
    report_content = f"""
# MEAN REVERSION TRADING STRATEGY - COMPREHENSIVE REPORT

## Executive Summary
- **Strategy**: Mean Reversion with Z-score based entry/exit
- **Asset**: {symbol}
- **Period**: {start_date} to {end_date}
- **Initial Capital**: ${initial_capital:,}

## Performance Summary
- **Total Return**: {backtest_results['total_return']:.2%}
- **Annualized Return**: {backtest_results['annualized_return']:.2%}
- **Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}
- **Maximum Drawdown**: {backtest_results['max_drawdown']:.2%}
- **Number of Trades**: {backtest_results['num_trades']}
- **Win Rate**: {backtest_results['win_rate']:.2%}

## Risk Analysis
- **Volatility**: {comprehensive_metrics.get('annualized_volatility', 0):.2%}
- **VaR (95%)**: {comprehensive_metrics.get('var_95', 0):.2%}
- **CVaR (95%)**: {comprehensive_metrics.get('cvar_95', 0):.2%}
- **Sortino Ratio**: {comprehensive_metrics.get('sortino_ratio', 0):.2f}
- **Calmar Ratio**: {comprehensive_metrics.get('calmar_ratio', 0):.2f}

## Model Validation
- **Stationarity**: {validation_results.get('stationarity', {}).get('interpretation', 'N/A')}
- **Mean Reversion**: {validation_results.get('mean_reversion', {}).get('interpretation', {}).get('meaning', 'N/A')}
- **Hurst Exponent**: {validation_results.get('mean_reversion', {}).get('hurst_exponent', 0):.3f}

## Cross-Validation Results
"""
    
    if 'cv_summary' in cv_results:
        for metric, stats in cv_results['cv_summary'].items():
            report_content += f"- **{metric.replace('_', ' ').title()}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
    
    report_content += f"""
## Monte Carlo Simulation
- **Simulations**: 100
- **Return Range (95% CI)**: [{summary.get('total_return', {}).get('percentile_5', 0):.2%}, {summary.get('total_return', {}).get('percentile_95', 0):.2%}]
- **Sharpe Range (95% CI)**: [{summary.get('sharpe_ratio', {}).get('percentile_5', 0):.2f}, {summary.get('sharpe_ratio', {}).get('percentile_95', 0):.2f}]

## Optimal Parameters
"""
    
    for param, value in best_params.items():
        report_content += f"- **{param}**: {value}\n"
    
    report_content += f"""
## Files Generated
- Portfolio Performance Chart: {output_dir}/portfolio_performance.html
- Strategy Signals Chart: {output_dir}/strategy_signals.html
- Risk Metrics Chart: {output_dir}/risk_metrics.html
- Monte Carlo Results: {output_dir}/monte_carlo_results.html
- Summary Dashboard: {output_dir}/summary_dashboard.html

## Conclusion
This comprehensive analysis demonstrates the {symbol} mean reversion strategy's performance characteristics, risk profile, and statistical validity. The strategy shows {'strong' if backtest_results['sharpe_ratio'] > 1.0 else 'moderate' if backtest_results['sharpe_ratio'] > 0.5 else 'weak'} risk-adjusted returns with a Sharpe ratio of {backtest_results['sharpe_ratio']:.2f}.

Key considerations:
1. **Performance**: The strategy {'outperformed' if excess_return > 0 else 'underperformed'} the benchmark by {excess_return:.2%}
2. **Risk**: Maximum drawdown of {backtest_results['max_drawdown']:.2%} indicates {'moderate' if abs(backtest_results['max_drawdown']) < 0.15 else 'high'} downside risk
3. **Consistency**: Cross-validation shows {'consistent' if cv_results['cv_summary']['sharpe_ratio']['std'] < 0.5 else 'variable'} performance across different periods
4. **Statistical Validity**: Mean reversion assumptions are {'supported' if validation_results.get('mean_reversion', {}).get('is_mean_reverting', False) else 'not fully supported'} by the data

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f'{output_dir}/comprehensive_report.md', 'w') as f:
        f.write(report_content)
    
    print(f"Comprehensive report saved to {output_dir}/comprehensive_report.md")
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nFiles generated:")
    print(f"- {output_dir}/portfolio_performance.html")
    print(f"- {output_dir}/strategy_signals.html")
    print(f"- {output_dir}/risk_metrics.html")
    print(f"- {output_dir}/monte_carlo_results.html")
    print(f"- {output_dir}/summary_dashboard.html")
    print(f"- {output_dir}/comprehensive_report.md")
    
    print(f"\nOpen the HTML files in your browser to view interactive charts.")
    print(f"Read the comprehensive report for detailed analysis.")
    
    return {
        'backtest_results': backtest_results,
        'performance_metrics': comprehensive_metrics,
        'validation_results': validation_results,
        'simulation_results': simulation_results,
        'sensitivity_results': sensitivity_results
    }

if __name__ == "__main__":
    # Run the demo
    try:
        results = main()
        print("\n✓ Demo completed successfully!")
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()