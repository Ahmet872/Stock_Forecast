import pandas as pd  
import yfinance as yf  
import logging  
from pathlib import Path 
from datetime import datetime, timedelta 
import pytz 

logging.basicConfig( # Logging setup configuration
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s') # Log message format
logger = logging.getLogger(__name__)  



class DataFetcher: # Retrieving financial data
    

    def __init__(self, cache_dir: str = "data/cache"): #Initializinh with cache directory
        
        self.cache_dir = Path(cache_dir) # # Define cache directory path
        self.cache_dir.mkdir(parents=True, exist_ok=True) # Create cache folder if missing
    

    def fetch_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d") -> pd.DataFrame:
    
        try:         
            cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"# Create cache filename
            
            if cache_file.exists(): # Try loading from cache first
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True) # Load cached CSV
                logger.info(f"Loaded {ticker} data from cache") # Log cache hit
                return df
                   
            ticker_obj = yf.Ticker(ticker)# If not in cache, download from yfinance
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval) # Fetch stock history
            
            if df.empty: # Validate non-empty dataframe
                raise ValueError(f"No data returned for {ticker}")
            df.columns = [col.title() for col in df.columns] # Capitalize column names    
            
            
            if df.index.tzinfo is None: # Ensure UTC timezone
                df.index = df.index.tz_localize('UTC') 
                        
            df.to_csv(cache_file) # Save data to cache
            logger.info(f"Saved {ticker} data to cache") # Log cache save
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}") #Log error
            raise
    

    def get_current_price(self, ticker: str) -> float: #Fetches current price of the ticker

        try:
            ticker_obj = yf.Ticker(ticker) # Create ticker object
            data = ticker_obj.history(period="1d") 
            
            if data.empty: # Validate data
                raise ValueError(f"No data available for {ticker}")
            return float(data['Close'].iloc[-1]) # Return last closing price    
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {str(e)}") # Log error
            raise       

fetcher = DataFetcher() #Instantiate data fetcher


def test_fetcher():# Test the data fetcher functionality
    
    try:
        #parameters for test function
        ticker = "AAPL" # Test stock ticker
        end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d') # Current date in UTC
        start_date = (datetime.now(pytz.UTC) - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"\nTesting fetch for {ticker}...") # Test log output
        
        df = fetcher.fetch_stock(ticker, start_date, end_date)  
        print(f"Fetched data shape: {df.shape}") 
        print(f"Columns: {df.columns.tolist()}")
        
        price = fetcher.get_current_price(ticker)
        print(f"Current price: ${price:.2f}")
        print("All tests passed!") 
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_fetcher() # Run tests if executed directly
