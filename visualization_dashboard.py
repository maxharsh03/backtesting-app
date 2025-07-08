import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationDashboard:
    """
    Comprehensive visualization dashboard for backtesting results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization dashboard
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def create_portfolio_performance_chart(self, 
                                         portfolio_history: pd.DataFrame,
                                         benchmark_data: Optional[pd.DataFrame] = None,
                                         save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive portfolio performance chart
        
        Args:
            portfolio_history: Portfolio performance data
            benchmark_data: Optional benchmark data
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Cumulative Returns', 'Drawdown', 
                          'Rolling Volatility', 'Monthly Returns', 'Distribution of Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=portfolio_history.index, y=portfolio_history['portfolio_value'],
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Cumulative returns
        initial_value = portfolio_history['portfolio_value'].iloc[0]
        cumulative_returns = (portfolio_history['portfolio_value'] / initial_value - 1) * 100
        fig.add_trace(
            go.Scatter(x=portfolio_history.index, y=cumulative_returns,
                      name='Cumulative Returns (%)', line=dict(color='green')),
            row=1, col=2
        )
        
        # Add benchmark if provided
        if benchmark_data is not None:
            benchmark_returns = (benchmark_data['Close'] / benchmark_data['Close'].iloc[0] - 1) * 100
            fig.add_trace(
                go.Scatter(x=benchmark_data.index, y=benchmark_returns,
                          name='Benchmark (%)', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
        
        # Drawdown
        if 'drawdown' in portfolio_history.columns:
            fig.add_trace(
                go.Scatter(x=portfolio_history.index, y=portfolio_history['drawdown'] * 100,
                          name='Drawdown (%)', fill='tonexty', line=dict(color='red')),
                row=2, col=1
            )
        
        # Rolling volatility
        if 'returns' in portfolio_history.columns:
            rolling_vol = portfolio_history['returns'].rolling(window=30).std() * np.sqrt(252) * 100
            fig.add_trace(
                go.Scatter(x=portfolio_history.index, y=rolling_vol,
                          name='Rolling Volatility (30d)', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Monthly returns
        if 'returns' in portfolio_history.columns:
            monthly_returns = portfolio_history['returns'].resample('M').sum() * 100
            fig.add_trace(
                go.Bar(x=monthly_returns.index, y=monthly_returns.values,
                      name='Monthly Returns (%)', marker_color='lightblue'),
                row=3, col=1
            )
        
        # Distribution of returns
        if 'returns' in portfolio_history.columns:
            returns_pct = portfolio_history['returns'].dropna() * 100
            fig.add_trace(
                go.Histogram(x=returns_pct, nbinsx=50, name='Returns Distribution',
                            marker_color='lightgreen'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance Dashboard',
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_strategy_signals_chart(self, 
                                    data: pd.DataFrame,
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create strategy signals visualization
        
        Args:
            data: DataFrame with price data and signals
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Signals', 'Z-Score', 'Position'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'],
                      name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        if 'rolling_mean' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['rolling_mean'],
                          name='Rolling Mean', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Add buy/sell signals
        if 'signal' in data.columns:
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                              mode='markers', name='Buy Signal',
                              marker=dict(color='green', size=10, symbol='triangle-up')),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                              mode='markers', name='Sell Signal',
                              marker=dict(color='red', size=10, symbol='triangle-down')),
                    row=1, col=1
                )
        
        # Z-Score
        if 'zscore' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['zscore'],
                          name='Z-Score', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Position
        if 'position' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['position'],
                          name='Position', line=dict(color='brown'), fill='tonexty'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Mean Reversion Strategy Signals',
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_risk_metrics_chart(self, 
                                metrics: Dict,
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Create risk metrics visualization
        
        Args:
            metrics: Dictionary of risk metrics
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk-Return Scatter', 'Drawdown Analysis', 
                          'Risk Metrics Comparison', 'Return Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Risk-return scatter (placeholder for multiple strategies)
        fig.add_trace(
            go.Scatter(x=[metrics.get('annualized_volatility', 0)],
                      y=[metrics.get('annualized_return', 0)],
                      mode='markers+text',
                      text=['Strategy'],
                      textposition="top center",
                      marker=dict(size=15, color='blue'),
                      name='Strategy'),
            row=1, col=1
        )
        
        # Risk metrics comparison
        risk_metrics = {
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Max Drawdown': abs(metrics.get('max_drawdown', 0)),
            'Volatility': metrics.get('annualized_volatility', 0)
        }
        
        fig.add_trace(
            go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values()),
                   name='Risk Metrics', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Risk Analysis Dashboard',
            showlegend=True,
            height=600,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_monte_carlo_results_chart(self, 
                                       simulation_results: Dict,
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create Monte Carlo simulation results visualization
        
        Args:
            simulation_results: Monte Carlo simulation results
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Total Returns Distribution', 'Sharpe Ratio Distribution',
                          'Max Drawdown Distribution', 'Confidence Intervals',
                          'Risk-Return Scatter', 'Volatility Distribution')
        )
        
        if 'simulation_summary' in simulation_results:
            summary = simulation_results['simulation_summary']
            
            # Total returns distribution
            if 'total_return' in summary:
                returns_data = [r['total_return'] for r in simulation_results['all_results'] 
                               if 'total_return' in r]
                fig.add_trace(
                    go.Histogram(x=returns_data, nbinsx=50, name='Total Returns',
                                marker_color='lightblue'),
                    row=1, col=1
                )
            
            # Sharpe ratio distribution
            if 'sharpe_ratio' in summary:
                sharpe_data = [r['sharpe_ratio'] for r in simulation_results['all_results'] 
                              if 'sharpe_ratio' in r]
                fig.add_trace(
                    go.Histogram(x=sharpe_data, nbinsx=50, name='Sharpe Ratio',
                                marker_color='lightgreen'),
                    row=1, col=2
                )
            
            # Max drawdown distribution
            if 'max_drawdown' in summary:
                dd_data = [r['max_drawdown'] for r in simulation_results['all_results'] 
                          if 'max_drawdown' in r]
                fig.add_trace(
                    go.Histogram(x=dd_data, nbinsx=50, name='Max Drawdown',
                                marker_color='lightcoral'),
                    row=1, col=3
                )
            
            # Confidence intervals
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
            ci_data = []
            for metric in metrics:
                if metric in summary:
                    ci_data.append({
                        'metric': metric,
                        'mean': summary[metric]['mean'],
                        'lower_5': summary[metric]['percentile_5'],
                        'upper_95': summary[metric]['percentile_95']
                    })
            
            if ci_data:
                ci_df = pd.DataFrame(ci_data)
                fig.add_trace(
                    go.Scatter(x=ci_df['metric'], y=ci_df['mean'],
                              error_y=dict(
                                  type='data',
                                  symmetric=False,
                                  array=ci_df['upper_95'] - ci_df['mean'],
                                  arrayminus=ci_df['mean'] - ci_df['lower_5']
                              ),
                              mode='markers',
                              name='95% Confidence Interval',
                              marker=dict(size=10)),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='Monte Carlo Simulation Results',
            showlegend=True,
            height=600,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_correlation_heatmap(self, 
                                 data: pd.DataFrame,
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            data: DataFrame with multiple columns
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_performance_comparison_chart(self, 
                                          results_dict: Dict[str, Dict],
                                          save_path: Optional[str] = None) -> go.Figure:
        """
        Create performance comparison chart for multiple strategies
        
        Args:
            results_dict: Dictionary of strategy results
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns Comparison', 'Risk Metrics Comparison',
                          'Risk-Return Scatter', 'Drawdown Comparison')
        )
        
        strategies = list(results_dict.keys())
        
        # Returns comparison
        returns_data = [results_dict[strategy].get('total_return', 0) * 100 
                       for strategy in strategies]
        fig.add_trace(
            go.Bar(x=strategies, y=returns_data, name='Total Returns (%)',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Risk metrics comparison
        risk_metrics = ['sharpe_ratio', 'max_drawdown', 'volatility']
        for i, metric in enumerate(risk_metrics):
            values = [results_dict[strategy].get(metric, 0) for strategy in strategies]
            fig.add_trace(
                go.Bar(x=strategies, y=values, name=metric.replace('_', ' ').title(),
                       marker_color=self.color_palette[i % len(self.color_palette)]),
                row=1, col=2
            )
        
        # Risk-return scatter
        for strategy in strategies:
            fig.add_trace(
                go.Scatter(x=[results_dict[strategy].get('annualized_volatility', 0)],
                          y=[results_dict[strategy].get('annualized_return', 0)],
                          mode='markers+text',
                          text=[strategy],
                          textposition="top center",
                          marker=dict(size=15),
                          name=strategy),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Strategy Performance Comparison',
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_all_charts(self, 
                       portfolio_history: pd.DataFrame,
                       signals_data: pd.DataFrame,
                       metrics: Dict,
                       simulation_results: Optional[Dict] = None,
                       output_dir: str = 'charts') -> None:
        """
        Save all charts to files
        
        Args:
            portfolio_history: Portfolio performance data
            signals_data: Strategy signals data
            metrics: Performance metrics
            simulation_results: Optional Monte Carlo results
            output_dir: Output directory for charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create and save charts
        self.create_portfolio_performance_chart(
            portfolio_history, 
            save_path=f'{output_dir}/portfolio_performance.html'
        )
        
        self.create_strategy_signals_chart(
            signals_data,
            save_path=f'{output_dir}/strategy_signals.html'
        )
        
        self.create_risk_metrics_chart(
            metrics,
            save_path=f'{output_dir}/risk_metrics.html'
        )
        
        if simulation_results:
            self.create_monte_carlo_results_chart(
                simulation_results,
                save_path=f'{output_dir}/monte_carlo_results.html'
            )
        
        print(f"All charts saved to {output_dir}/")
    
    def create_summary_dashboard(self, 
                               portfolio_history: pd.DataFrame,
                               signals_data: pd.DataFrame,
                               metrics: Dict,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive summary dashboard
        
        Args:
            portfolio_history: Portfolio performance data
            signals_data: Strategy signals data
            metrics: Performance metrics
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Portfolio Value', 'Cumulative Returns', 'Drawdown',
                          'Z-Score Distribution', 'Monthly Returns', 'Risk Metrics',
                          'Trade Signals', 'Rolling Sharpe', 'Performance Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"type": "bar"}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=portfolio_history.index, y=portfolio_history['portfolio_value'],
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Cumulative returns
        initial_value = portfolio_history['portfolio_value'].iloc[0]
        cum_returns = (portfolio_history['portfolio_value'] / initial_value - 1) * 100
        fig.add_trace(
            go.Scatter(x=portfolio_history.index, y=cum_returns,
                      name='Cumulative Returns (%)', line=dict(color='green')),
            row=1, col=2
        )
        
        # Drawdown
        if 'drawdown' in portfolio_history.columns:
            fig.add_trace(
                go.Scatter(x=portfolio_history.index, y=portfolio_history['drawdown'] * 100,
                          name='Drawdown (%)', fill='tonexty', line=dict(color='red')),
                row=1, col=3
            )
        
        # Performance summary table
        summary_data = [
            ['Total Return', f"{metrics.get('total_return', 0):.2%}"],
            ['Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
            ['Volatility', f"{metrics.get('annualized_volatility', 0):.2%}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.2%}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=[[row[0] for row in summary_data],
                                 [row[1] for row in summary_data]],
                          fill_color='white')
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Complete Strategy Dashboard',
            showlegend=True,
            height=1000,
            width=1400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig