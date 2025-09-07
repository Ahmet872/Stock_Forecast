import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



class BacktestEngine:# Backtesting and reporting engine


    def __init__( # Initialize paths
        self,
        reports_path: str = "reports",
        signals_path: str = "signals"):
    
        self.reports_path = reports_path # Path to save reports
        self.signals_path = signals_path # Path to save signals
        os.makedirs(reports_path, exist_ok=True) # Create directory if not exists
        os.makedirs(signals_path, exist_ok=True)


    def calculate_signals( # Generate trading signals
        self,
        forecast_output: Dict[str, Any],
        current_price: float) -> List[Dict]:
    
        predictions = forecast_output['predictions'] # Forecasted values
        confidence_intervals = forecast_output['confidence_intervals'] # CI
        probability_ranges = forecast_output['probability_ranges'] # Probabilities
        dates = forecast_output['dates'] # Dates
        signals = []
        
        for i, (pred, ci, pr, date) in enumerate(zip(predictions, confidence_intervals, probability_ranges, dates), 1):
            direction = "UPWARD" if pred > current_price else "DOWNWARD" # Determine direction
            change_pct = ((pred - current_price) / current_price) * 100 # % change
            risk_level = self._calculate_risk_level(change_pct, ci) # Risk

            probability_levels = { # Format probability ranges
                "Very Likely": f"[{pr['very_likely'][0]:.2f} - {pr['very_likely'][1]:.2f}]" if pr else "-",
                "Likely": f"[{pr['likely'][0]:.2f} - {pr['likely'][1]:.2f}]" if pr else "-",
                "Possible": f"[{pr['possible'][0]:.2f} - {pr['possible'][1]:.2f}]" if pr else "-"}
            
            signal = { # Create signal dictionary
                "day": i,
                "date": date.strftime("%Y-%m-%d"),
                "prediction": f"{pred:.2f}",
                "direction": direction,
                "change_%": f"{change_pct:.2f}%",
                "risk_level": risk_level,
                "confidence_interval": f"[{ci[0]:.2f} - {ci[1]:.2f}]",
                "probability_levels": probability_levels}
            signals.append(signal)
        self._save_json(signals, self.signals_path, "signals") #Save signals to JSON
        return signals


    def _calculate_risk_level(self, change_pct: float, confidence_interval: Tuple[float, float]) -> str:# Determning Risk logic
        
        interval_width = confidence_interval[1] - confidence_interval[0]
        abs_change = abs(change_pct)
        if interval_width > 10 or abs_change > 10:
            return "HIGH"
        elif interval_width > 5 or abs_change > 5:
            return "MEDIUM"
        else:
            return "LOW"


    def evaluate_predictions( # Evaluate model performance
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_intervals: List[Tuple[float, float]]) -> Dict[str, float]:
    
        mae = mean_absolute_error(y_true, y_pred) # MAE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE
        r2 = r2_score(y_true, y_pred) # R^2
        direction_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) # Direction match
        in_interval = sum(1 for y, (lower, upper) in zip(y_true, confidence_intervals) if lower <= y <= upper) # CI coverage
        interval_coverage = in_interval / len(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 # MAPE
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
            "Direction_Accuracy": direction_accuracy,
            "Confidence_Interval_Coverage": interval_coverage}
        
        logger.info("Performance metrics calculated") # Log metrics
        return metrics


    def generate_report( # Generate markdown report
        self,
        signals: List[Dict],
        metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict] = None) -> str:
    
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC") # Timestamp
        report = [
            "# Prediction Report",
            f"\nGenerated: {now}"]
        
        if additional_info: # General info section
            report.append("\n## General Information\n")
            for key, value in additional_info.items():
                report.append(f"- **{key}:** {value}")
        if metrics: # Metrics section
            report.append("\n## Performance Metrics\n")
            for metric, value in metrics.items():
                report.append(f"- **{metric}:** {value:.4f}")
        report.append("\n## Daily Predictions\n")
        for signal in signals: # Daily predictions
            report.extend([
                f"### Day {signal['day']} ({signal['date']})",
                f"- **Prediction:** {signal['prediction']}",
                f"- **Direction:** {signal['direction']}",
                f"- **Change:** {signal['change_%']}",
                f"- **Risk Level:** {signal['risk_level']}",
                f"- **Confidence Interval:** {signal['confidence_interval']}",
                "**Probability Levels:**"])
            
            for level, range_str in signal['probability_levels'].items(): # Probability ranges
                report.append(f"- *{level}:* {range_str}")
            report.append("")
        report_text = "\n".join(report)
        self._save_md(report_text, self.reports_path, "report") # Save report
        return report_text


    def _save_json(self, obj: Any, path: str, prefix: str) -> None: # Save JSON helper
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(path, f"{prefix}_{timestamp}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info(f"{prefix.capitalize()} saved: {filepath}")


    def _save_md(self, text: str, path: str, prefix: str) -> None: # Save Markdown helper
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(path, f"{prefix}_{timestamp}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Report saved: {filepath}")


def test_backtest(): # Test function for backtesting


    from data_fetcher import fetcher
    from data_cleaner import cleaner
    from feature_engineering import engineer
    from data_preprocessor import DataPreprocessor
    from forecast import ForecastModel

    df = fetcher.fetch_stock("GOOGL", "2025-08-01", "2025-08-31") # Fetch sample data
    df = cleaner.clean_data(df) # Clean data
    current_price = df['Close'].iloc[-1] # Last close price
    df_features = engineer.add_features(df) # Add features
    prep = DataPreprocessor.preprocess(df_features) # Preprocess for model
    model = ForecastModel(lstm_units=[32, 16]) # Initialize LSTM
    model.build(input_shape=(prep['X_train'].shape[1], prep['X_train'].shape[2])) # Build model
    model.train( # Quick train for test
        train_data={'X_train': prep['X_train'], 'y_train': prep['y_train']},
        val_data={'X_test': prep['X_test'], 'y_test': prep['y_test']},
        epochs=3)
    
    future = model.predict_future( # Forecast future
        preprocessor_output=prep,
        n_days=5,
        target_scaler=DataPreprocessor.target_scaler,
        mc_dropout=True)
    
    backtest = BacktestEngine() # Initialize backtest engine
    signals = backtest.calculate_signals(forecast_output=future, current_price=current_price) # Generate signals
    metrics = backtest.evaluate_predictions( # Evaluate predictions
        y_true=prep['y_test'],
        y_pred=model.model.predict(prep['X_test']).flatten(),
        confidence_intervals=future['confidence_intervals'])
    
    additional_info = { # Extra info
        "Stock": "GOOGL",
        "Model": "LSTM",
        "Prediction Days": 5,
        "Current Price": f"{current_price:.2f}"}
    
    report = backtest.generate_report(signals, metrics, additional_info) # Generate report
    print("\nReport and signals generated!") # Log completion
    return True


if __name__ == "__main__":
    test_backtest()
