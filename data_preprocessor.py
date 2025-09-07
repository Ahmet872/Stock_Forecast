import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(# Logging configuration
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor: # Handles data preprocessing, scaling, and preparing data for forecasting models
    
    
    SUPPORTED_SCALERS = {  # Supported scaler types
        "minmax": MinMaxScaler,
        "standard": StandardScaler}
    

    def __init__(
        self,
        scaler_type: str = "minmax",# Default scaler type
        window_size: int = 10, # Number of time steps for each sequence
        test_size: float = 0.2): 

        self.window_size = window_size  # Sliding window size
        self.test_size = test_size  
        self.scaler_type = scaler_type.lower()  
        self.feature_scaler = None 
        self.target_scaler = None  
        
        if self.scaler_type not in self.SUPPORTED_SCALERS:  # Validating scaler type
            raise ValueError(f"Unsupported scaler type: {scaler_type}")


    def fit_scalers(self, features: np.ndarray, target: np.ndarray):  # Fit scalers for features and target
        scaler_cls = self.SUPPORTED_SCALERS[self.scaler_type]  
        self.feature_scaler = scaler_cls()  
        self.target_scaler = scaler_cls() 
        self.feature_scaler.fit(features)  
        self.target_scaler.fit(target)  
    

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray): # Create sliding window sequences # type: ignore
        X, y = [], []
        for i in range(self.window_size, len(features)):
            X.append(features[i - self.window_size:i]) # Collect feature window
            y.append(target[i]) # Collect target value
        return np.array(X), np.array(y)
    
    
    def preprocess(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = 'Close') -> Dict[str, Any]:
    
        if df.empty: # Check if DataFrame is empty
            raise ValueError("Input DataFrame is empty!")
        if feature_cols is None: # Select feature columns automatically
            feature_cols = [col for col in df.columns if col != target_col]
        
        features = df[feature_cols].values # Extract feature values
        target = df[target_col].values.reshape(-1, 1) 
        dates = df.index 
        
        split_idx = int(len(df) * (1 - self.test_size)) # Train/Test split index
        if split_idx <= self.window_size: # Validate split index
            raise ValueError(f"Test size or window size too large! split_idx: {split_idx}")
        
        features_train, features_test = features[:split_idx], features[split_idx:] # features
        target_train, target_test = target[:split_idx], target[split_idx:] # target
        dates_train, dates_test = dates[:split_idx], dates[split_idx:] #dates
        
        self.fit_scalers(features_train, target_train) #Fittin scalers with training data
        features_train_scaled = self.feature_scaler.transform(features_train) 
        features_test_scaled = self.feature_scaler.transform(features_test)  
        target_train_scaled = self.target_scaler.transform(target_train)  
        target_test_scaled = self.target_scaler.transform(target_test)  
        
        X_train, y_train = self.create_sequences(features_train_scaled, target_train_scaled) # Create train sequences
        X_test, y_test = self.create_sequences(features_test_scaled, target_test_scaled) # Create test sequences
        
        X_last_window = features[-self.window_size:] # Extract last input window
        X_last_window_scaled = self.feature_scaler.transform(X_last_window).reshape(1, self.window_size, -1) # Scale last window
        
        logger.info(f"Preprocessing completed: Train shape {X_train.shape}, Test shape {X_test.shape}, Features {len(feature_cols)}")# Log preprocessing summary
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_last_window": X_last_window_scaled,
            "feature_columns": feature_cols,
            "dates_train": dates_train[self.window_size:], #Adjust train dates
            "dates_test": dates_test[self.window_size:]} # Adjust test dates
        
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray: # Convert scaled target back to original scale
        if self.target_scaler is None: # Ensure scaler is fitted
            raise RuntimeError("Target scaler not fitted!")
        return self.target_scaler.inverse_transform(scaled_target)

preprocessor = DataPreprocessor() # Global instance of DataPreprocessor


def test_preprocessor(): # Test the preprocessor workflow
    from data_fetcher import fetcher
    from data_cleaner import cleaner
    from feature_engineering import engineer

    df = fetcher.fetch_stock("AAPL", "2025-08-01", "2025-08-31")# Fetch Apple stock data
    df = cleaner.clean_data(df) #Clean raw data
    df_features = engineer.add_features(df) # Add engineered features
    
    result = preprocessor.preprocess(df_features) # Run preprocessing pipeline
    print("\nPreprocessor Test Results:")
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test shape: {result['X_test'].shape}")
    print(f"y_train shape: {result['y_train'].shape}")
    print(f"y_test shape: {result['y_test'].shape}")
    print(f"X_last_window shape: {result['X_last_window'].shape}")
    print(f"Feature columns: {result['feature_columns']}")
    print("All tests passed!") # Confirmation message
    return True

if __name__ == "__main__": # Run test if executed as script
    test_preprocessor()
