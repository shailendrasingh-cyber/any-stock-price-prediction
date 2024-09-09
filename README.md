# Stock Price Prediction App

This is a Streamlit-based web application that predicts stock prices using various machine learning models. Users can visualize real-time stock data, apply technical indicators, and forecast future prices based on historical data. The app fetches data from Yahoo Finance and provides interactive visualizations and predictions.

## Features

- **Download Historical Stock Data:** Fetch and visualize historical stock data from Yahoo Finance.
- **Technical Indicators:**
  - Close Price
  - Bollinger Bands (BB)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
- **Stock Price Prediction:** Predict future stock prices using multiple machine learning models:
  - Linear Regression
  - Random Forest Regressor
  - Extra Trees Regressor
  - K-Nearest Neighbors Regressor
  - XGBoost Regressor
- **Forecast Visualization:** View predictions for future stock prices and compare them with actual data.
- **Interactive UI:** Built with Streamlit for a user-friendly experience.
- **Made by Shailendra Singh**

## How It Works

The app processes stock data by:

1. **Fetching Data:** Retrieves historical stock data from Yahoo Finance.
2. **Technical Indicators:** Calculates technical indicators such as Bollinger Bands, MACD, RSI, SMA, and EMA.
3. **Model Training:** Trains various machine learning models on the historical stock data.
4. **Prediction:** Predicts future stock prices based on the trained models.
5. **Visualization:** Displays the predictions and visualizations interactively.

## Getting Started

### Prerequisites

- Python 3.x installed
- `pip` installed

### Libraries Used

- `streamlit`
- `yfinance`
- `ta`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/shailendrasingh-cyber/stock-price-prediction-app.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd Any stock-price-prediction-app
    ```

3. **Install the Required Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

5. **Interact with the App:** Open the URL provided by Streamlit in your web browser.

### Example

To use the app:

1. **Enter a Stock Symbol:** Input the stock symbol in the sidebar.
2. **Select Date Range:** Choose the start and end dates for historical data.
3. **Visualize Technical Indicators:** Select and view various technical indicators.
4. **Make Predictions:** Choose a machine learning model and specify the forecast duration.
5. **View Forecasts:** See the predicted stock prices and corresponding visualizations.


## Contributing

If you’d like to contribute to this project, please submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
