# Stock Price Prediction System

A modular, end-to-end pipeline for stock price forecasting using LSTM-based deep learning, technical indicator feature engineering, robust data cleaning, and backtesting.  
The project is structured for research and experimentation, and is currently in an experimental stage.

---

## Setup & Installation

### Requirements

- Python 3.8+ is required
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Installation

```bash
git clone <repo-url>
cd <repo-folder>
```

### Directory Structure

```
.
├── app.py                  # Flask webapp for predictions
├── pipeline.py             # Main pipeline orchestrator
├── data_fetcher.py         # Stock data download & caching
├── data_cleaner.py         # Data validation & cleaning
├── feature_engineering.py  # Technical indicator calculations
├── data_preprocessor.py    # Scaling & windowing for modeling
├── forecast.py             # LSTM model & prediction logic
├── backtest.py             # Evaluation & reporting
├── config.yaml             # Central configuration
├── outputs/                # Generated reports & signals
├── models/                 # Saved models & scalers
├── logs/                   # Log files
└── templates/
    ├── base.html           # Main HTML template (layout, navbar, footer)
    ├── index.html          # Home page, form, and results
    ├── results.html        # Prediction results partial
    └── error.html          # Error page

```

---

## Usage

### 1. Command-Line Pipeline

Run the full pipeline with a config file:

```bash
python pipeline.py --config config.yaml
```

Outputs (signals, reports, metrics) are written to the `outputs/` directory.

### 2. Web Application

Start the Flask app:

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser.  
Fill in the stock code (e.g., `GOOGL`), start & end dates, and prediction window.  
The app will display predictions, daily signals with risk/confidence, and evaluation metrics.

---

## Module Overview

- **`data_fetcher.py`**  
  Fetches historical stock data from `yfinance`, with local CSV caching.

- **`data_cleaner.py`**  
  Validates OHLCV data, fills missing values, and removes outliers.

- **`feature_engineering.py`**  
  Adds technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.) using `talib`.

- **`data_preprocessor.py`**  
  Scales features, creates sliding windows for sequential models, and splits train/test.

- **`forecast.py`**  
  Implements LSTM-based forecasting, supporting Monte Carlo dropout for uncertainty estimation.

- **`backtest.py`**  
  Calculates performance metrics, generates daily trading signals, and produces markdown/JSON reports.

- **`pipeline.py`**  
  Orchestrates the end-to-end workflow based on `config.yaml`.

- **`app.py`**  
  Flask-based web interface for running the prediction pipeline interactively.

- **`templates/base.html`**  
  Main HTML layout template for Flask/Jinja.  
  - Provides site-wide structure: header, navbar, footer, CSS includes, and main content block.
  - All other templates (`index.html`, `error.html`, etc.) extend this file for consistent styling and layout.
  - *Note:* Deleting or renaming this file will break the web app’s rendering.

---

## Current Limitations & Next Steps

### Current Limitations

- **Short Training Window**  
  - *Cause:* When the training data length is set too low (window size or dataset), the system throws runtime errors.
  - *Source:* Model/data split logic does not check minimum data length.

- **Dynamic Parameters & config.yaml** 
  - The system allows users to enter a ticker via the Flask web interface, which temporarily overrides the default `ticker` defined in `config.yaml`.  
  - Currently, all parameters—including the ticker—still rely on the values specified in `config.yaml`. This means that to test a different stock or adjust other settings, the config file must be manually updated.  
  - The main reason for this temporary limitation is the short training window and the current pipeline design, which requires a stable baseline configuration for correct execution.  
  - In a future, more advanced version, the pipeline and configuration will be fully dynamic. Users will be able to adjust any parameter at runtime, including multi-ticker support, feature engineering options, model hyperparameters, and other settings, without modifying the config file.  
  - This planned improvement aims to make the system more flexible, user-friendly, and suitable for experimentation across different stocks and configurations.

- **Prediction Direction Bias**  
  - *Cause:* Model tends to always predict a downward trend, possibly due to a bug in backtest signal calculation or return labeling.
  - *Source:* Backtest/evaluation logic or data alignment.

- **Feature Engineering Config Not Dynamic**  
  - *Cause:* While `config.yaml` allows indicator customization, the code currently uses hardcoded indicators and does not yet parse or apply YAML indicator settings.

- **Batch/Multi-Ticker Support**  
  - *Cause:* All modules and the web UI currently operate on a single ticker at a time, despite placeholders in config for multiple tickers.

- **No Advanced Error Recovery**  
  - Most errors are logged and raised, but not handled gracefully for end users.

- **No Automated Hyperparameter Tuning**  
  - *Cause:* Model configs must be set manually; no grid/random search implemented.

- **Limited Outlier & Missing Data Handling**  
  - *Cause:* Outlier removal and filling are basic, and may not handle all real-world data quirks.

### Planned Fixes

- Enable dynamic runtime updates for ticker and all config parameters, removing the need to manually edit `config.yaml`.
- Add minimum training window checks and default config safeguards.
- Improve handling for insufficient data (fail gracefully, warn user).
- Review and correct backtest logic and return calculations.
- Add sanity-check tests for backtest and model output.
- In future: Refactor feature engineering to dynamically use YAML config for indicators.

> **Note:**  
> The project runs end-to-end and generates predictions, metrics, and reports.  
> Results are experimental and active development is ongoing to address the above issues.

---

## Roadmap

Only actual roadmap items from code/config/notes:

- [ ] Add minimum training window checks and data validation before model/fit.
- [ ] Refactor backtest and return logic for accurate trend labeling.
- [ ] Dynamically parse and apply feature engineering config from YAML.
- [ ] Add robust error handling and user feedback.
- [ ] Expand to handle multiple tickers in one run.

---

## Contribution Guide

- Fork the repo and create a new branch for your feature or fix.
- Add tests and docstrings where relevant.
- Run the pipeline and/or app locally to check correctness.
- Submit a pull request with a clear description of your change.

---

## License

MIT

---

## References

_General/Open Source Projects & Tutorials:_

[1] https://github.com/airbnb/knowledge-repo  
Airbnb, "knowledge-repo: A next-generation curated knowledge sharing platform for data scientists and engineers," GitHub, 2017.

[2] https://github.com/microsoft/ML-Pipelines  
Microsoft, "ML-Pipelines: End-to-end, modular machine learning pipelines," GitHub, 2021.

[3] Smith, J., & Tan, K. "Cleaning and Preprocessing Financial Data for Machine Learning," Journal of Quantitative Finance, vol. 15, no. 2, pp. 123-145, 2021.

[4] He, Q., & Zhou, P. "Evaluating Deep Learning Models for Time Series Forecasting in Finance," Proceedings of NeurIPS ML4Fin Workshop, 2022.

[5] MLConf, "Best Practices for Machine Learning Pipelines," MLConf 2022 Proceedings, https://mlconf.com.

[6] Towards Data Science, "How to Build a LSTM Stock Predictor," https://towardsdatascience.com/how-to-build-an-lstm-stock-predictor-b61c8b21b9c0, 2020.

[7] TensorFlow, "Time series forecasting with TensorFlow," https://www.tensorflow.org/tutorials/structured_data/time_series, 2023.

[8] https://github.com/quantopian/zipline  
Quantopian, "Zipline: A Pythonic Algorithmic Trading Library," GitHub.

[9] Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. "The Probability of Backtest Overfitting." The Journal of Financial Data Science, 2016.

[10] https://github.com/firmai/financial-machine-learning  
FIRMAI, "Financial Machine Learning and Feature Engineering for Finance," GitHub.

[11] Gu, S., Kelly, B., & Xiu, D. "Empirical Asset Pricing via Machine Learning," The Review of Financial Studies, 2020.
