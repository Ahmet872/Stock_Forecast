import pandas as pd 
from flask import Flask, render_template, request, jsonify, session 
from datetime import datetime, timedelta  
import logging 
import yaml 
import os
from typing import Dict, Any 
from pipeline import TradingPipeline 
from pathlib import Path 

app = Flask(__name__) # Initializes Flask app
app.secret_key = os.urandom(24) #Random secret key for sessions

BASE_DIR = Path(__file__).parent # Projecting base directory
CONFIG_PATH = BASE_DIR / "config.yaml"  
OUTPUT_DIR = BASE_DIR / "outputs"  
LOG_DIR = BASE_DIR / "logs"  

for dir_path in [OUTPUT_DIR, LOG_DIR]:# Ensure required directories exist
    dir_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig( #logging setup
    filename=LOG_DIR / 'app.log',
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | [%(name)s] %(message)s')
logger = logging.getLogger(__name__) # Logger instance


class ConfigManager: # Handles loading and updating configuration files
    

    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]: # Load config from YAML
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) # Parse YAML
        except Exception as e:
            logger.error(f"Config loading error: {str(e)}")# Log error
            raise

    @staticmethod
    def update_config(path: Path, updates: Dict[str, Any], section: str = 'data') -> None:#Update YAML config
        try:
            config = ConfigManager.load_config(path)# Load current config
            config[section].update(updates) #Apply updates
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False) # Write updated config
        except Exception as e:
            logger.error(f"Config update error: {str(e)}") # Log error
            raise



class PredictionFormatter: # Formats pipeline outputs into structured prediction results


    @staticmethod
    def format_results(pipeline_output: Dict[str, Any]) -> Dict[str, Any]: # Format prediction results
        try:
            signals = pipeline_output.get('signals', [])# Extract prediction signals
            formatted_predictions = [
                {'day': signal['day'],
                    'date': signal['date'],
                    'price': signal['prediction'],
                    'direction': signal['direction'],
                    'change': signal['change_%'],
                    'risk_level': signal['risk_level'],
                    'confidence_interval': signal['confidence_interval'],
                    'probabilities': signal['probability_levels']} for signal in signals]
    
            return {
                'predictions': formatted_predictions,# Predictions list
                'current_price': pipeline_output.get('current_price', 0.0), # Current price
                'metrics': pipeline_output.get('metrics', {}), #Metrics data
                'report_path': pipeline_output.get('report_path')} #Report file path
        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")# log error details
            raise


@app.route('/')
def home():# Home page route
    try:
        if 'user_id' not in session:# Assign user ID if missing
            session['user_id'] = request.headers.get('X-User-ID', 'anonymous')
        today = datetime.utcnow() #Current UTC time
        default_start = (today - timedelta(days=180)).strftime('%Y-%m-%d')# Default start date
        default_end = today.strftime('%Y-%m-%d') # Default end date
        return render_template(
            'index.html',
            default_start=default_start,
            default_end=default_end,
            current_time=today.strftime('%Y-%m-%d %H:%M:%S'),
            user=session['user_id'])
    except Exception as e:
        logger.error(f"Home page error: {str(e)}") # Log error
        return render_template('error.html', error=str(e)), 500


@app.route('/predict', methods=['POST'])
def predict(): # Prediction endpoint
    try:
        data = request.form # Get form data
        ticker = data.get('ticker', '').upper() # Stock ticker
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        forecast_days = int(data.get('forecast_days', 5))# Number of prediction days
        if not ticker: # Validation: ticker required
            return jsonify({'error': 'Stock code is required!'}), 400
        try:
            start = pd.to_datetime(start_date) # Parse start date
            end = pd.to_datetime(end_date)#Parse end date
            if start >= end: #Ensure date order
                return jsonify({'error': 'Start date must be before end date!'}), 400
        except ValueError as e:
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
        if not 1 <= forecast_days <= 30: #Check forecast range
            return jsonify({'error': 'Prediction days must be between 1-30!'}), 400
        ConfigManager.update_config( # Update config file
            CONFIG_PATH,
            {'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'n_days_to_predict': forecast_days})
            
        pipeline = TradingPipeline(ConfigManager.load_config(CONFIG_PATH)) #Initialize pipeline
        results = pipeline.run()  # Run pipeline
        formatted_results = PredictionFormatter.format_results(results) # Format results
        return jsonify({'success': True, 'results': formatted_results}) # Return JSON
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True) # Log error with traceback
        return jsonify({'error': str(e)}), 500
    

@app.route('/health')
def health_check():# Health check endpoint
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0',
        'config_path': str(CONFIG_PATH),
        'output_dir': str(OUTPUT_DIR)})


@app.errorhandler(404)
def page_not_found(e): #Handle 404 errors
    return render_template('error.html', error="Page not found."), 404


@app.errorhandler(500)
def internal_server_error(e): # Handle 500 errors
    return render_template('error.html', error="Server error occurred."), 500


@app.errorhandler(Exception)
def handle_exception(e): # Handle unexpected exceptions
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    return render_template('error.html', error=str(e)), 500


@app.template_filter('format_date')
def format_date(date_str: str) -> str:# Custom date formatting filter
    try:
        return pd.to_datetime(date_str).strftime('%d.%m.%Y')
    except:
        return date_str


@app.template_filter('format_number')
def format_number(value: float) -> str: # Custom number formatting filter
    try:
        return f"{float(value):,.2f}"
    except:
        return str(value)

if __name__ == '__main__':
    app.run(  # Run Flask app
        host='0.0.0.0',  # Accessible from all network interfaces
        port=5000,  # Port number
        debug=True)  # Debug mode enabled
