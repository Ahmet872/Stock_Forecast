import pandas as pd 
import numpy as np 
import logging        # Logging for debugging

logging.basicConfig(  # Logging setup
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



class DataCleaner:#Simple data cleaning class for stock market data
    
    
    def __init__(self):  # Standard columns we expect in stock data
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Essential columns
    

    def validate_data(self, df: pd.DataFrame) -> None:  # Basic data validation
        missing_cols = [col for col in self.required_columns if col not in df.columns]  # Check missing columns
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")  # Raise error if columns missing
        if len(df) < 5:  # Check for minimum data points (5 trading days)
            raise ValueError(f"Insufficient data: {len(df)} rows < 5 minimum")
        for col in self.required_columns:  # Check for negative values
            if (df[col] < 0).any():
                raise ValueError(f"Negative values found in {col}")
        invalid_prices = (  # Check logical consistency of prices
            (df['Low'] > df['High']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low']))
        
        if invalid_prices.any():
            raise ValueError("Invalid price relationships detected") # Raise error if inconsistency found
    

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame: # Fill missing values
        if df.isnull().any().any():  
            df = df.fillna(method='ffill', limit=3)
            df = df.fillna(method='bfill', limit=3)
            if df.isnull().any().any():  
                raise ValueError("Unable to fill all missing values") 
        return df
    

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame: #Remove extreme outliers
        df_clean = df.copy()
    
        for col in self.required_columns:  
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std()) 
            df_clean.loc[z_scores > 3, col] = np.nan 
        df_clean = self.handle_missing_values(df_clean) # Refill NaN values
        return df_clean
    

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:# Main cleaning function
        try:
            df_clean = df.copy() 
            self.validate_data(df_clean) 
            df_clean = self.handle_missing_values(df_clean) # Fill missing values
            df_clean = self.remove_outliers(df_clean) # Remove outliers
            return df_clean
        except Exception as e:
            logger.error(f"Cleaning error: {str(e)}")# Log error if cleaning fails
            raise

cleaner = DataCleaner() # Create global instance


def test_cleaner(): # Testing the data cleaner
    try:  
        dates = pd.date_range('2025-08-01', '2025-08-10') 
        data = {  
            'Open':  [100, 101, 102, np.nan, 104, 105, 106, 107, 108, 109],
            'High':  [102, 103, 104, np.nan, 106, 107, 108, 109, 110, 111],
            'Low':   [98,  99,  100, np.nan, 102, 103, 104, 105, 106, 107],
            'Close': [101, 102, 103, np.nan, 105, 106, 107, 108, 109, 110],
            'Volume': [1000] * 10}
        
        df = pd.DataFrame(data, index=dates)
        df_clean = cleaner.clean_data(df)  
        
        print("\nCleaning Test Results:")
        print(f"Original shape: {df.shape}")  
        print(f"Cleaned shape: {df_clean.shape}") 
        print(f"NaN in original: {df.isnull().sum().sum()}") 
        print(f"NaN in cleaned: {df_clean.isnull().sum().sum()}")
        print("\nAll tests passed!")
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}") 
        return False

if __name__ == "__main__":
    test_cleaner() #Run test if script executed directly
