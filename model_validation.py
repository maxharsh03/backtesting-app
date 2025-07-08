import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Statistical model validation and analysis for trading strategies
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize ModelValidator
        
        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
    
    def validate_mean_reversion_assumptions(self, data: pd.DataFrame) -> Dict:
        """
        Validate statistical assumptions for mean reversion strategy
        
        Args:
            data: Price data with required columns
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Test for stationarity (ADF test)
        results['stationarity'] = self._test_stationarity(data['Close'])
        
        # Test for normality of returns
        returns = data['Close'].pct_change().dropna()
        results['normality'] = self._test_normality(returns)
        
        # Test for autocorrelation
        results['autocorrelation'] = self._test_autocorrelation(returns)
        
        # Test for heteroscedasticity
        results['heteroscedasticity'] = self._test_heteroscedasticity(returns)
        
        # Test for mean reversion (Hurst exponent)
        results['mean_reversion'] = self._test_mean_reversion(data['Close'])
        
        # Half-life of mean reversion
        results['half_life'] = self._calculate_half_life(data['Close'])
        
        return results
    
    def _test_stationarity(self, series: pd.Series) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(series.dropna())
            
            return {
                'test_name': 'Augmented Dickey-Fuller Test',
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < self.significance_level,
                'interpretation': 'Stationary' if result[1] < self.significance_level else 'Non-stationary'
            }
        except ImportError:
            # Fallback method using simple variance ratio test
            return self._variance_ratio_test(series)
    
    def _variance_ratio_test(self, series: pd.Series) -> Dict:
        """Simple variance ratio test for mean reversion"""
        returns = series.pct_change().dropna()
        
        # Calculate variance ratios for different lags
        var_ratios = []
        for lag in [2, 5, 10]:
            if len(returns) > lag:
                var_k = returns.rolling(window=lag).var().mean()
                var_1 = returns.var()
                var_ratio = var_k / (lag * var_1)
                var_ratios.append(var_ratio)
        
        avg_var_ratio = np.mean(var_ratios)
        
        return {
            'test_name': 'Variance Ratio Test',
            'variance_ratios': var_ratios,
            'average_variance_ratio': avg_var_ratio,
            'is_mean_reverting': avg_var_ratio < 1.0,
            'interpretation': 'Mean Reverting' if avg_var_ratio < 1.0 else 'Trending'
        }
    
    def _test_normality(self, series: pd.Series) -> Dict:
        """Test for normality using multiple tests"""
        series_clean = series.dropna()
        
        # Shapiro-Wilk test (for smaller samples)
        if len(series_clean) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(series_clean)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(series_clean)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(series_clean, 'norm',
                                    args=(series_clean.mean(), series_clean.std()))
        
        return {
            'test_name': 'Normality Tests',
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.significance_level if shapiro_p else None
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > self.significance_level
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > self.significance_level
            },
            'skewness': stats.skew(series_clean),
            'kurtosis': stats.kurtosis(series_clean)
        }
    
    def _test_autocorrelation(self, series: pd.Series) -> Dict:
        """Test for autocorrelation using Ljung-Box test"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Test for autocorrelation up to lag 10
            lb_result = acorr_ljungbox(series.dropna(), lags=10, return_df=True)
            
            return {
                'test_name': 'Ljung-Box Test',
                'results': lb_result.to_dict(),
                'has_autocorrelation': any(lb_result['lb_pvalue'] < self.significance_level),
                'interpretation': 'Autocorrelation detected' if any(lb_result['lb_pvalue'] < self.significance_level) else 'No significant autocorrelation'
            }
        except ImportError:
            # Fallback: simple correlation with lagged values
            correlations = []
            for lag in range(1, 11):
                corr = series.corr(series.shift(lag))
                correlations.append(corr)
            
            return {
                'test_name': 'Simple Autocorrelation',
                'correlations': correlations,
                'max_correlation': max(correlations, key=abs),
                'interpretation': 'Manual autocorrelation analysis'
            }
    
    def _test_heteroscedasticity(self, series: pd.Series) -> Dict:
        """Test for heteroscedasticity using Breusch-Pagan test"""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            
            # Create time trend
            n = len(series.dropna())
            X = sm.add_constant(np.arange(n))
            y = series.dropna().values
            
            # Fit OLS model
            model = OLS(y, X).fit()
            
            # Breusch-Pagan test
            lm, lm_p, fvalue, f_p = het_breuschpagan(model.resid, model.model.exog)
            
            return {
                'test_name': 'Breusch-Pagan Test',
                'lm_statistic': lm,
                'lm_p_value': lm_p,
                'f_statistic': fvalue,
                'f_p_value': f_p,
                'has_heteroscedasticity': lm_p < self.significance_level,
                'interpretation': 'Heteroscedasticity detected' if lm_p < self.significance_level else 'Homoscedasticity'
            }
        except ImportError:
            # Fallback: simple variance test
            mid_point = len(series) // 2
            first_half_var = series[:mid_point].var()
            second_half_var = series[mid_point:].var()
            
            return {
                'test_name': 'Simple Variance Test',
                'first_half_variance': first_half_var,
                'second_half_variance': second_half_var,
                'variance_ratio': second_half_var / first_half_var,
                'interpretation': 'Simple variance comparison'
            }
    
    def _test_mean_reversion(self, series: pd.Series) -> Dict:
        """Test for mean reversion using Hurst exponent"""
        def hurst_exponent(ts):
            """Calculate Hurst exponent"""
            lags = range(2, min(100, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            # Remove any NaN or infinite values
            tau = [t for t in tau if np.isfinite(t) and t > 0]
            lags = lags[:len(tau)]
            
            if len(tau) < 2:
                return 0.5  # Return neutral value if calculation fails
            
            # Linear regression to find Hurst exponent
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        hurst = hurst_exponent(series.dropna().values)
        
        return {
            'test_name': 'Hurst Exponent',
            'hurst_exponent': hurst,
            'interpretation': {
                'value': hurst,
                'meaning': 'Mean reverting' if hurst < 0.5 else 'Trending' if hurst > 0.5 else 'Random walk'
            },
            'is_mean_reverting': hurst < 0.5
        }
    
    def _calculate_half_life(self, series: pd.Series) -> Dict:
        """Calculate half-life of mean reversion"""
        try:
            # Calculate lagged values
            y = series.diff().dropna()
            x = series.shift(1).dropna()
            
            # Align series
            min_len = min(len(y), len(x))
            y = y.iloc[:min_len]
            x = x.iloc[:min_len]
            
            # Fit regression: y_t = a + b*y_{t-1} + e_t
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Half-life calculation
            if beta[1] < 0:
                half_life = -np.log(2) / np.log(1 + beta[1])
            else:
                half_life = np.inf
            
            return {
                'half_life': half_life,
                'regression_coefficient': beta[1],
                'interpretation': f'Half-life: {half_life:.2f} periods' if np.isfinite(half_life) else 'No mean reversion detected'
            }
        except Exception as e:
            return {
                'half_life': None,
                'error': str(e),
                'interpretation': 'Could not calculate half-life'
            }
    
    def cross_validate_strategy(self, data: pd.DataFrame, strategy, 
                              backtest_engine, n_splits: int = 5) -> Dict:
        """
        Cross-validate strategy using time series splits
        
        Args:
            data: Market data
            strategy: Trading strategy
            backtest_engine: Backtesting engine
            n_splits: Number of time series splits
            
        Returns:
            Cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Run backtest on test data
            result = backtest_engine.run_backtest(test_data, strategy)
            results.append(result)
        
        # Aggregate results
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown']
        cv_summary = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                cv_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'cv_score': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                }
        
        return {
            'cv_summary': cv_summary,
            'individual_results': results,
            'n_splits': n_splits
        }
    
    def parameter_sensitivity_analysis(self, data: pd.DataFrame, strategy_class,
                                     backtest_engine, param_ranges: Dict) -> Dict:
        """
        Analyze parameter sensitivity
        
        Args:
            data: Market data
            strategy_class: Strategy class
            backtest_engine: Backtesting engine
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Sensitivity analysis results
        """
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            # Create strategy with parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            result = backtest_engine.run_backtest(data, strategy)
            result['parameters'] = params
            results.append(result)
        
        # Analyze sensitivity
        sensitivity_analysis = self._analyze_parameter_sensitivity(results, param_ranges)
        
        return {
            'all_results': results,
            'sensitivity_analysis': sensitivity_analysis,
            'best_parameters': self._find_best_parameters(results),
            'parameter_ranges': param_ranges
        }
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def _analyze_parameter_sensitivity(self, results: List[Dict], 
                                     param_ranges: Dict) -> Dict:
        """Analyze parameter sensitivity"""
        sensitivity = {}
        
        for param_name in param_ranges.keys():
            # Group results by parameter value
            param_groups = {}
            for result in results:
                param_value = result['parameters'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result)
            
            # Calculate statistics for each parameter value
            param_stats = {}
            for param_value, group_results in param_groups.items():
                returns = [r.get('total_return', 0) for r in group_results]
                sharpe_ratios = [r.get('sharpe_ratio', 0) for r in group_results]
                
                param_stats[param_value] = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'std_sharpe': np.std(sharpe_ratios),
                    'count': len(group_results)
                }
            
            sensitivity[param_name] = param_stats
        
        return sensitivity
    
    def _find_best_parameters(self, results: List[Dict]) -> Dict:
        """Find best parameters based on Sharpe ratio"""
        best_result = max(results, key=lambda x: x.get('sharpe_ratio', -np.inf))
        return best_result['parameters']
    
    def run_robustness_tests(self, data: pd.DataFrame, strategy,
                           backtest_engine) -> Dict:
        """
        Run comprehensive robustness tests
        
        Args:
            data: Market data
            strategy: Trading strategy
            backtest_engine: Backtesting engine
            
        Returns:
            Robustness test results
        """
        results = {}
        
        # Bootstrap test
        results['bootstrap'] = self._bootstrap_test(data, strategy, backtest_engine)
        
        # Subperiod analysis
        results['subperiod'] = self._subperiod_analysis(data, strategy, backtest_engine)
        
        # Regime analysis
        results['regime'] = self._regime_analysis(data, strategy, backtest_engine)
        
        return results
    
    def _bootstrap_test(self, data: pd.DataFrame, strategy, 
                       backtest_engine, n_bootstrap: int = 100) -> Dict:
        """Bootstrap test for strategy robustness"""
        bootstrap_results = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_data = data.sample(n=len(data), replace=True).sort_index()
            
            # Run backtest
            result = backtest_engine.run_backtest(sample_data, strategy)
            bootstrap_results.append(result)
        
        # Calculate bootstrap statistics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        bootstrap_stats = {}
        
        for metric in metrics:
            values = [r[metric] for r in bootstrap_results if metric in r]
            if values:
                bootstrap_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'percentile_5': np.percentile(values, 5),
                    'percentile_95': np.percentile(values, 95)
                }
        
        return bootstrap_stats
    
    def _subperiod_analysis(self, data: pd.DataFrame, strategy, 
                           backtest_engine) -> Dict:
        """Analyze strategy performance in different subperiods"""
        # Split data into subperiods
        n_periods = 4
        period_length = len(data) // n_periods
        
        subperiod_results = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(data)
            
            subperiod_data = data.iloc[start_idx:end_idx]
            result = backtest_engine.run_backtest(subperiod_data, strategy)
            result['period'] = i + 1
            subperiod_results.append(result)
        
        return {
            'subperiod_results': subperiod_results,
            'consistency_score': self._calculate_consistency_score(subperiod_results)
        }
    
    def _regime_analysis(self, data: pd.DataFrame, strategy, 
                        backtest_engine) -> Dict:
        """Analyze strategy performance in different market regimes"""
        # Identify market regimes based on volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        high_vol_threshold = volatility.quantile(0.7)
        low_vol_threshold = volatility.quantile(0.3)
        
        # Classify regimes
        regimes = pd.Series(index=data.index, dtype='object')
        regimes[volatility <= low_vol_threshold] = 'Low Volatility'
        regimes[volatility >= high_vol_threshold] = 'High Volatility'
        regimes[(volatility > low_vol_threshold) & (volatility < high_vol_threshold)] = 'Medium Volatility'
        
        # Analyze performance by regime
        regime_results = {}
        
        for regime in regimes.unique():
            if pd.notna(regime):
                regime_data = data[regimes == regime]
                if len(regime_data) > 20:  # Minimum data points
                    result = backtest_engine.run_backtest(regime_data, strategy)
                    regime_results[regime] = result
        
        return regime_results
    
    def _calculate_consistency_score(self, results: List[Dict]) -> float:
        """Calculate consistency score across subperiods"""
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        
        # Consistency score: lower std deviation of Sharpe ratios is better
        if len(sharpe_ratios) > 1:
            mean_sharpe = np.mean(sharpe_ratios)
            std_sharpe = np.std(sharpe_ratios)
            consistency_score = mean_sharpe / (std_sharpe + 1e-8)  # Add small epsilon to avoid division by zero
        else:
            consistency_score = 0
        
        return consistency_score
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Formatted validation report
        """
        report = "="*60 + "\n"
        report += "MODEL VALIDATION REPORT\n"
        report += "="*60 + "\n\n"
        
        if 'assumptions' in validation_results:
            report += "STATISTICAL ASSUMPTIONS:\n"
            report += "-"*30 + "\n"
            
            assumptions = validation_results['assumptions']
            
            if 'stationarity' in assumptions:
                report += f"Stationarity: {assumptions['stationarity']['interpretation']}\n"
            
            if 'normality' in assumptions:
                report += f"Normality: Normal distribution assumption {'met' if assumptions['normality']['jarque_bera']['is_normal'] else 'violated'}\n"
            
            if 'mean_reversion' in assumptions:
                report += f"Mean Reversion: {assumptions['mean_reversion']['interpretation']['meaning']}\n"
            
            if 'half_life' in assumptions:
                report += f"Half-life: {assumptions['half_life']['interpretation']}\n"
            
            report += "\n"
        
        if 'cross_validation' in validation_results:
            report += "CROSS-VALIDATION RESULTS:\n"
            report += "-"*30 + "\n"
            
            cv = validation_results['cross_validation']['cv_summary']
            
            for metric, stats in cv.items():
                report += f"{metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
            
            report += "\n"
        
        if 'sensitivity' in validation_results:
            report += "PARAMETER SENSITIVITY:\n"
            report += "-"*30 + "\n"
            
            best_params = validation_results['sensitivity']['best_parameters']
            for param, value in best_params.items():
                report += f"Best {param}: {value}\n"
            
            report += "\n"
        
        report += "="*60 + "\n"
        
        return report