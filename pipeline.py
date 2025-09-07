import pandas as pd  
import time  
import logging  
import yaml  
from typing import Dict, Any  
from functools import wraps  
from datetime import datetime  
from pathlib import Path  
import json  

# module importings
from data_fetcher import fetcher 
from data_cleaner import cleaner  
from feature_engineering import engineer  
from data_preprocessor import DataPreprocessor  
from forecast import ForecastModel
from backtest import BacktestEngine 

BASE_DIR = Path(__file__).parent # Base directory of this script
LOGS_DIR = BASE_DIR / "logs"  
OUTPUTS_DIR = BASE_DIR / "outputs"  
MODELS_DIR = BASE_DIR / "models"  
SCALERS_DIR = BASE_DIR / "scalers"  

for dir_path in [LOGS_DIR, OUTPUTS_DIR, MODELS_DIR, SCALERS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) # Create directories if they do not exist

logging.basicConfig(
    level=logging.INFO,# Set logging level to INFO
    format="%(asctime)s | %(levelname)s | %(message)s", # Define log format
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"), 
        logging.StreamHandler()])  
logger = logging.getLogger(__name__)  


def timing(func): # Decorator to measure function execution time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time() # Start timer
        result = func(*args, **kwargs)  
        elapsed = time.time() - start  
        logger.info(f"{func.__name__} completed. Duration: {elapsed:.2f} sec")# Log execution time
        return result  
    return wrapper



class TradingPipeline: # Main trading pipeline class


    def __init__(self, config: Dict[str, Any]):
        self.config = config# Store config
        self._init_components()# Initializing the pipeline components


    def _init_components(self) -> None: 
        try:
            data_cfg = self.config['data'] #Extract data configuration
            self.ticker = data_cfg['ticker']  
            self.start_date = data_cfg['start_date']  
            self.end_date = data_cfg['end_date']  
            self.interval = data_cfg.get('interval', '1d')  
            self.n_days_to_predict = data_cfg.get('n_days_to_predict', 5)  

            fe_cfg = self.config.get('feature_engineering', {})# Feature engineering config will be used in future
            self.fe = engineer  

            prep_cfg = self.config.get('preprocessor', {}) # Preprocessor config
            self.preprocessor = DataPreprocessor(
                scaler_type=prep_cfg.get('scaler_type', 'minmax'), 
                window_size=prep_cfg.get('window_size', 10),  
                test_size=prep_cfg.get('test_size', 0.2))  

            model_cfg = self.config.get('model', {}) # Model configration
            self.model = ForecastModel(
                lstm_units=model_cfg.get('lstm_units', [64, 32]),  
                dropout_rate=model_cfg.get('dropout_rate', 0.2),  
                learning_rate=model_cfg.get('learning_rate', 0.001),  
                loss=model_cfg.get('loss', 'mse'), 
                metrics=model_cfg.get('metrics', ['mae']),  
                model_dir=str(MODELS_DIR))

            self.backtest = BacktestEngine(
                reports_path=str(OUTPUTS_DIR / "reports"),
                signals_path=str(OUTPUTS_DIR / "signals")) #Signals path

        except KeyError as e:
            logger.error(f"Required config key missing: {str(e)}")# Log missing key
            raise
        except Exception as e:
            logger.error(f"Pipeline initialization error: {str(e)}")# Log other errors
            raise


    @timing
    def fetch_and_clean(self) -> pd.DataFrame: 
        try:
            df_raw = fetcher.fetch_stock(
                self.ticker,
                self.start_date,
                self.end_date,
                interval=self.interval) # Fetch stock data

            df_clean = cleaner.clean_data(df_raw) #Clean raw data
            if df_clean.empty:
                raise ValueError("Data fetching failed or returned empty DataFrame.") # Error if empty
            return df_clean  # Return cleaned DataFrame
        except Exception as e:
            logger.error(f"Data fetching/cleaning error: {str(e)}") # Log errors
            raise


    @timing
    def run(self) -> Dict[str, Any]:#Executing the entire stock forecasting workflow and outputs results (everything)
        try:
            df_clean = self.fetch_and_clean() # Step 1: Data preparation
            current_price = df_clean['Close'].iloc[-1] # Latest closing price

            df_features = engineer.add_features(df_clean) # Step 2: Feature engineering

            prep_output = self.preprocessor.preprocess( #Step 3: Preprocessing
                df=df_features,
                target_col='Close')

            self.model.build(input_shape=( # Step 4: Model building
                prep_output['X_train'].shape[1],
                prep_output['X_train'].shape[2]))
            train_results = self.model.train(
                train_data={
                    'X_train': prep_output['X_train'],
                    'y_train': prep_output['y_train']},
                val_data={
                    'X_test': prep_output['X_test'],
                    'y_test': prep_output['y_test']})

            future = self.model.predict_future( # Step 5:Future predictions
                preprocessor_output=prep_output,
                n_days=self.n_days_to_predict,
                target_scaler=self.preprocessor.target_scaler,
                mc_dropout=True)

            signals = self.backtest.calculate_signals( # Step 6: Backtest & signal generation
                forecast_output=future,
                current_price=current_price)

            metrics = {} # Step 7: Performance evaluation
            if len(prep_output['X_test']) > 0:
                metrics = self.backtest.evaluate_predictions(
                    y_true=prep_output['y_test'],
                    y_pred=self.model.model.predict(prep_output['X_test']).flatten(),
                    confidence_intervals=future['confidence_intervals'])
            else:
                logger.warning("No test data available, metrics could not be calculated.")# Warn if no test data

            report = self.backtest.generate_report(# Step 8:Generate report
                signals=signals,
                metrics=metrics,
                additional_info={
                    "Stock": self.ticker,
                    "Model": "LSTM",
                    "Prediction Days": self.n_days_to_predict,
                    "Current Price": f"{current_price:.2f}",
                    "Training Dates": f"{self.start_date} - {self.end_date}"} )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")# Step 9: Timestamp
            results_path = OUTPUTS_DIR / f"results_{timestamp}.json"
            results = {
                "signals": signals,
                "current_price": current_price,
                "metrics": metrics,
                "report_path": str(OUTPUTS_DIR / f"report_{timestamp}.md"),
                "model_path": train_results['model_path']}

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)# Save results to JSON
            logger.info(f"Pipeline completed successfully. Results: {results_path}")  # Log success
            return results  # Return final output
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)  # Log pipeline errors
            raise


def load_config(path: str) -> Dict[str, Any]:# Load YAML configuration file
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) # Parse YAML into dict
    except Exception as e:
        logger.error(f"Config loading error: {str(e)}") # Log loading errors
        raise

if __name__ == "__main__":
    import argparse # Command-line arguments parsing
    parser = argparse.ArgumentParser(description="Trading Pipeline CLI") # CLI description
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file") # Config file argument
    args = parser.parse_args() # Parse CLI args
    cfg = load_config(args.config) # Load pipeline config
    pipeline = TradingPipeline(cfg) # initialize pipeline instance
    output = pipeline.run() # Execute pipeline
    logger.info(f"Pipeline output keys: {list(output.keys())}") #Log output 
