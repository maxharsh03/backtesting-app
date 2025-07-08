import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy Implementation
    
    This strategy identifies when asset prices deviate significantly from their 
    historical mean and trades on the assumption they will revert.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5,
                 stop_loss_zscore: float = 3.0,
                 position_sizing: str = 'fixed',
                 risk_per_trade: float = 0.02):
        """
        Initialize the Mean Reversion Strategy
        
        Args:
            lookback_period: Number of periods to calculate rolling mean/std
            entry_zscore: Z-score threshold to enter positions
            exit_zscore: Z-score threshold to exit positions
            stop_loss_zscore: Z-score threshold for stop loss
            position_sizing: Position sizing method ('fixed', 'volatility_adjusted')
            risk_per_trade: Risk per trade as percentage of portfolio
        """
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on mean reversion logic
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and intermediate calculations
        """
        df = data.copy()
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['Close'].rolling(window=self.lookback_period).mean()
        df['rolling_std'] = df['Close'].rolling(window=self.lookback_period).std()
        
        # Calculate Z-score
        df['zscore'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Entry signals
        df.loc[df['zscore'] > self.entry_zscore, 'signal'] = -1  # Short (price too high)
        df.loc[df['zscore'] < -self.entry_zscore, 'signal'] = 1   # Long (price too low)
        
        # Exit signals
        df.loc[abs(df['zscore']) <= self.exit_zscore, 'signal'] = 0
        
        # Stop loss signals
        df.loc[df['zscore'] > self.stop_loss_zscore, 'signal'] = 0
        df.loc[df['zscore'] < -self.stop_loss_zscore, 'signal'] = 0
        
        # Calculate positions using vectorized operations
        # Use forward fill to propagate non-zero signals
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Set position to 0 where signal is explicitly 0 and we should exit
        exit_mask = (df['signal'] == 0) & (abs(df['zscore']) <= self.exit_zscore)
        df.loc[exit_mask, 'position'] = 0
        
        # Forward fill positions after exits
        df['position'] = df['position'].ffill().fillna(0)
        
        return df
    
    def calculate_position_size(self, data: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
        """
        Calculate position sizes based on chosen method
        
        Args:
            data: DataFrame with signals
            portfolio_value: Current portfolio value
            
        Returns:
            DataFrame with position sizes
        """
        df = data.copy()
        
        if self.position_sizing == 'fixed':
            df['position_size'] = df['position'] * (portfolio_value * self.risk_per_trade) / df['Close']
        
        elif self.position_sizing == 'volatility_adjusted':
            # Adjust position size based on volatility
            df['volatility'] = df['Close'].pct_change().rolling(window=self.lookback_period).std()
            
            # Handle NaN values in volatility by using forward fill or a default value
            df['volatility'] = df['volatility'].fillna(method='ffill').fillna(0.02)  # Default 2% volatility
            
            # Ensure volatility is not zero to avoid division by zero
            df['volatility'] = df['volatility'].replace(0, 0.01)
            
            df['vol_adj_factor'] = 1 / (df['volatility'] * 100)  # Inverse volatility scaling
            df['position_size'] = (df['position'] * df['vol_adj_factor'] * 
                                 (portfolio_value * self.risk_per_trade) / df['Close'])
        
        return df
    
    def get_strategy_parameters(self) -> Dict:
        """Return strategy parameters"""
        return {
            'lookback_period': self.lookback_period,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'position_sizing': self.position_sizing,
            'risk_per_trade': self.risk_per_trade
        }