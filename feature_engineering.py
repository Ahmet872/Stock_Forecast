import pandas as pd
import talib
import logging

logging.basicConfig( # Logging setup
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



class FeatureEngineer: # Class for calculating technical indicators

    
    def __init__(self): # Initialize required columns
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    

    def calculate_sma(self, df: pd.DataFrame, period: int = 14) -> pd.Series: # Calculate Simple Moving Average
        
        return pd.Series(
            talib.SMA(df['Close'].values, timeperiod=period),
            index=df.index,
            name=f'SMA_{period}')
        

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series: # Calculate Relative Strength Index
        
        return pd.Series(
            talib.RSI(df['Close'].values, timeperiod=period),
            index=df.index,
            name=f'RSI_{period}')
        

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame: # Calculate MACD indicator
        
        macd, signal, hist = talib.MACD(
            df['Close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9)
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Hist': hist}, index=df.index)
        
    
    def calculate_bollinger(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame: # Calculate Bollinger Bands
        
        upper, middle, lower = talib.BBANDS(
            df['Close'].values,
            timeperiod=period,
            nbdevup=2, # upper bound
            nbdevdn=2) # lower bound
        
        return pd.DataFrame({
            f'BB_Upper': upper,
            f'BB_Middle': middle,
            f'BB_Lower': lower},index=df.index)
        
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame: # Add all technical indicators to dataframe
        
        try:
            missing_cols = [col for col in self.required_columns if col not in df.columns] # Check required columns
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            df_features = df.copy()# Copy original dataframe
            
            df_features = pd.concat([ # Add SMA 14 and 50
                df_features,
                self.calculate_sma(df, 14),
                self.calculate_sma(df, 50)], axis=1)
            
            df_features = pd.concat([# Add RSI 14
                df_features,
                self.calculate_rsi(df)], axis=1)           
                        
            df_features = pd.concat([ # Add MACD
                df_features,
                self.calculate_macd(df)], axis=1)
            
            df_features = pd.concat([ #Add Bollinger Bands
                df_features,
                self.calculate_bollinger(df)], axis=1)
            
            
            df_features = df_features.fillna(method='bfill').fillna(method='ffill') #Fill NaaN values
            return df_features
            
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")# Log errors
            raise    
        
engineer = FeatureEngineer()# Create global instance


def test_engineer(): # Test feature engineering
    
    try:
        from data_fetcher import fetcher # Import fetcher

        df = fetcher.fetch_stock( 
            "AAPL",
            "2025-08-01",
            "2025-08-15")
        df_features = engineer.add_features(df) 
        print("\nFeature Engineering Results:") 
        print(f"Original columns: {len(df.columns)}")
        print(f"Feature columns: {len(df_features.columns)}")
        print("\nNew features:") 
        new_cols = set(df_features.columns) - set(df.columns)
        for col in sorted(new_cols):
            print(f"- {col}")
        print("\nAll tests passed!")
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":# Run test if script is executed directly
    test_engineer()
