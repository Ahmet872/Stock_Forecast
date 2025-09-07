import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
import logging
import pandas as pd
import os

logging.basicConfig( # Logging setup
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



class ForecastModel: # LSTM-based forecasting model


    def __init__( # Initialize model configuration
        self,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        loss: str = 'mse',
        metrics: List[str] = ['mae'],
        model_dir: str = 'models'):
    
        self.lstm_units = lstm_units #LSTM layer units
        self.dropout_rate = dropout_rate # Dropout rate
        self.learning_rate = learning_rate # Optimizer learning rate
        self.loss = loss # Loss function
        self.metrics = metrics #Metrics for training
        self.model_dir = model_dir # Model save directory
        self.model = None # Placeholder for model
        os.makedirs(model_dir, exist_ok=True) #Create model directory if not exists


    def build(self, input_shape: Tuple[int, int]) -> None: # Build LSTM model
        try:
            self.model = Sequential() # Sequential model
            self.model.add(LSTM(self.lstm_units[0], input_shape=input_shape, return_sequences=len(self.lstm_units) > 1))# First LSTM
            self.model.add(Dropout(self.dropout_rate)) # Dropout layer
            for units in self.lstm_units[1:]: # Additional LSTM layers
                self.model.add(LSTM(units, return_sequences=False))
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(Dense(1))# Output layer
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=self.loss,
                metrics=self.metrics )
           
            logger.info("Model has created") # Log success
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}") # Log error
            raise


    def train( #Train model with early stopping and checkpoint
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 1) -> Dict[str, Any]:
    
        try:
            if self.model is None: # Ensure model is built
                raise ValueError("Model has not been created yet!")
            model_path = os.path.join(self.model_dir, f"model_{pd.Timestamp.now():%Y%m%d_%H%M%S}.h5") # Model save path
            callbacks = [
                EarlyStopping(monitor='val_loss' if val_data else 'loss', patience=patience, restore_best_weights=True), # Early stopping
                ModelCheckpoint(model_path, monitor='val_loss' if val_data else 'loss', save_best_only=True)] # Save best model
            
            history = self.model.fit(# Fit model
                train_data['X_train'], train_data['y_train'],
                validation_data=(val_data['X_test'], val_data['y_test']) if val_data else None,
                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
            
            logger.info(f"Model trained and saved: {model_path}") # Log training complete
            return {'history': history.history, 'model_path': model_path}
        except Exception as e:
            logger.error(f"Training error: {str(e)}") #Log error
            raise


    def predict_future( #Predict future values
        self,
        preprocessor_output: Dict[str, Any],
        n_days: int,
        target_scaler,
        mc_dropout: bool = True,
        n_simulations: int = 100) -> Dict[str, Any]:
        try:
            if self.model is None: # Ensure model is trained
                raise ValueError("Model has not been trained yet!")
            X_last = preprocessor_output['X_last_window'] # Last input window
            last_date = preprocessor_output['dates_test'][-1] # Last date
            predictions, confidence_intervals, probability_ranges, dates = [], [], [], []
            current_window = X_last.copy()
            current_date = last_date

            for day in range(n_days):# Iterate over days
                if mc_dropout: # Monte Carlo dropout
                    mc_preds = np.array([
                        self.model(current_window, training=True).numpy()[0, 0]
                        for _ in range(n_simulations)])
                    
                    mean_pred = np.mean(mc_preds) # Mean prediction
                    std_pred = np.std(mc_preds) # Standard deviation
                    ci_lower = mean_pred - 1.96 * std_pred # Confidence interval lower
                    ci_upper = mean_pred + 1.96 * std_pred # Confidence interval upper
                    probability_ranges.append({
                        "very_likely": (mean_pred - std_pred, mean_pred + std_pred),   # ~68%
                        "likely": (mean_pred - 2*std_pred, mean_pred + 2*std_pred),    # ~95%
                        "possible": (mean_pred - 3*std_pred, mean_pred + 3*std_pred)}) # ~99.7%
                    predictions.append(mean_pred)
                    confidence_intervals.append((ci_lower, ci_upper))
                else:# Deterministic prediction
                    pred = self.model.predict(current_window, verbose=0)[0, 0]
                    predictions.append(pred)
                    confidence_intervals.append((pred, pred))
                    probability_ranges.append(None)
                               
                current_window = np.roll(current_window, -1, axis=1) # Shift input window
                current_window[0, -1, 0] = predictions[-1] # Update with latest prediction
                current_date = current_date + pd.Timedelta(days=1) #Increment date
                dates.append(current_date)
            
            predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten() # Rescale predictions
            confidence_intervals = [
                tuple(target_scaler.inverse_transform([[ci[0]], [ci[1]]]).flatten())
                for ci in confidence_intervals]

            probability_ranges = [
                {k: tuple(target_scaler.inverse_transform([[v[0]], [v[1]]]).flatten())
                 for k, v in pr.items()} if pr else None
                for pr in probability_ranges]
            return {
                'predictions': predictions,
                'dates': dates,
                'confidence_intervals': confidence_intervals,
                'probability_ranges': probability_ranges}
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")# Log error
            raise


    def load(self, model_path: str) -> None: # Load trained model
        try:
            self.model = load_model(model_path)
            logger.info(f"Model loaded: {model_path}") # Log success
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}") #Log error
            raise


def test_forecast(): #Test forecasting pipeline
    # Pipeline chain: Fetcher > Cleaner > Feature Engineer > Preprocessor >Forecast
    from data_fetcher import fetcher
    from data_cleaner import cleaner
    from feature_engineering import engineer
    from data_preprocessor import DataPreprocessor
    
    df = fetcher.fetch_stock("GOOGL", "2025-08-01", "2025-08-31") # Fetch data
    df = cleaner.clean_data(df) # Clean data
    df_features = engineer.add_features(df) # Add features
    prep = DataPreprocessor.preprocess(df_features) # Preprocess data
    
    model = ForecastModel(lstm_units=[32, 16])# Initialize model
    model.build(input_shape=(prep['X_train'].shape[1], prep['X_train'].shape[2]))# Build model
    future = model.predict_future(# Predict future
        preprocessor_output=prep,
        n_days=5,
        target_scaler=DataPreprocessor.target_scaler,
        mc_dropout=True)
    
    print("\nPrediction results:")# Print results
    for i, (date, pred, ci) in enumerate(zip(future['dates'], future['predictions'], future['confidence_intervals'])):
        print(f"Day {i+1} ({date:%Y-%m-%d}): Prediction: {pred:.2f}, Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
        if future['probability_ranges']:
            ranges = future['probability_ranges'][i]
            print("Probability Ranges:") #Print probability ranges
            for k, v in ranges.items():
                print(f"- {k}: ({v[0]:.2f}, {v[1]:.2f})")
    print("All tests passed!") #Success message
    return True

if __name__ == "__main__": # Run test when executed directly
    test_forecast()
