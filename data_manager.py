import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """
    Data acquisition and processing pipeline for backtesting
    """
    
    def __init__(self, cache_data: bool = True):
        """
        Initialize DataManager
        
        Args:
            cache_data: Whether to cache downloaded data
        """
        self.cache_data = cache_data
        self.cached_data = {}
    
    def fetch_data(self, 
                   symbols: Union[str, List[str]], 
                   start_date: str, 
                   end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance
        
        Args:
            symbols: Stock symbol(s) to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"{'-'.join(symbols)}_{start_date}_{end_date}_{interval}"
        
        if self.cache_data and cache_key in self.cached_data:
            return self.cached_data[cache_key].copy()
        
        try:
            # Try using ticker.history() method first (more reliable)
            if len(symbols) == 1:
                ticker = yf.Ticker(symbols[0])
                data = ticker.history(
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                # Remove extra columns that might cause issues
                if 'Dividends' in data.columns:
                    data = data.drop('Dividends', axis=1)
                if 'Stock Splits' in data.columns:
                    data = data.drop('Stock Splits', axis=1)
                    
            else:
                # Fall back to yf.download for multiple symbols
                data = yf.download(symbols, start=start_date, end=end_date, interval=interval, progress=False)
                
                # Handle multi-symbol column names
                if len(symbols) > 1:
                    # Keep multi-level columns as is
                    pass
                else:
                    # Clean single symbol column names
                    data.columns = [col.replace(f' {symbols[0]}', '') if f' {symbols[0]}' in col else col 
                                   for col in data.columns]
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Clean data
            data = self._clean_data(data)
            
            if self.cache_data:
                self.cached_data[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data
        
        Args:
            data: Raw market data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing values
        data = data.dropna()
        
        # Remove rows with zero volume (if volume exists)
        if 'Volume' in data.columns:
            data = data[data['Volume'] > 0]
        
        # Remove obvious errors (negative prices, etc.)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Check for price consistency
        if all(col in data.columns for col in price_columns):
            data = data[(data['Low'] <= data['High']) & 
                       (data['Low'] <= data['Open']) & 
                       (data['Low'] <= data['Close']) &
                       (data['High'] >= data['Open']) & 
                       (data['High'] >= data['Close'])]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_histogram'] = self._calculate_macd(df['Close'])
        
        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(df)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def create_synthetic_data(self, 
                            start_date: str, 
                            end_date: str,
                            initial_price: float = 100,
                            volatility: float = 0.2,
                            drift: float = 0.1,
                            mean_reversion_strength: float = 0.1) -> pd.DataFrame:
        """
        Create synthetic market data for testing
        
        Args:
            start_date: Start date
            end_date: End date
            initial_price: Starting price
            volatility: Price volatility
            drift: Price drift
            mean_reversion_strength: Mean reversion strength
            
        Returns:
            Synthetic market data
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate price series with mean reversion
        prices = [initial_price]
        long_term_mean = initial_price
        
        for i in range(1, n_days):
            # Mean reversion component
            mean_reversion = mean_reversion_strength * (long_term_mean - prices[-1])
            
            # Random walk component
            random_change = np.random.normal(drift/252, volatility/np.sqrt(252))
            
            # Calculate next price
            next_price = prices[-1] * (1 + random_change + mean_reversion/prices[-1])
            prices.append(max(next_price, 0.01))  # Ensure positive prices
        
        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        
        # Generate OHLC from close prices
        daily_volatility = volatility / np.sqrt(252)
        for i in range(len(data)):
            close_price = data['Close'].iloc[i]
            
            # Generate intraday range
            high_low_range = close_price * daily_volatility * np.random.uniform(0.5, 2.0)
            
            data.loc[data.index[i], 'High'] = close_price + high_low_range/2
            data.loc[data.index[i], 'Low'] = close_price - high_low_range/2
            
            # Generate open price
            if i == 0:
                data.loc[data.index[i], 'Open'] = close_price
            else:
                prev_close = data['Close'].iloc[i-1]
                gap = np.random.normal(0, daily_volatility/4)
                data.loc[data.index[i], 'Open'] = prev_close * (1 + gap)
        
        # Generate volume
        data['Volume'] = np.random.lognormal(mean=10, sigma=1, size=len(data)).astype(int)
        
        # Ensure OHLC consistency
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        """
        Split data into training and testing sets
        
        Args:
            data: Market data
            train_ratio: Ratio for training data
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_point = int(len(data) * train_ratio)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        return train_data, test_data
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': data.shape,
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max(),
                'days': len(data)
            },
            'price_stats': {
                'min': data['Close'].min(),
                'max': data['Close'].max(),
                'mean': data['Close'].mean(),
                'std': data['Close'].std()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'data_quality': {
                'complete_rows': len(data.dropna()),
                'completeness_ratio': len(data.dropna()) / len(data)
            }
        }
        
        return info