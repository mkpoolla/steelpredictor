# Steel Price Predictor

A Streamlit application that fetches real-time and historical steel prices, performs forecasting using linear regression, and provides alerts for significant price changes.

## Features

- Fetch real-time and historical steel prices from the Metals API
- Clean and transform price data using pandas
- Forecast future steel prices using linear regression
- Generate alerts for significant price changes (>5% in next 7 days)
- Interactive dashboard with price trends, forecasts, and alerts

## Project Structure

```
steelpredictor/
├── data/                # Generated directory for storing price data
├── models/              # Generated directory for storing trained models
├── alerts/              # Generated directory for storing price alerts
├── scripts/
│   ├── fetch_prices.py  # Script to fetch steel prices from API
│   ├── transform_data.py # Script to clean and preprocess data
│   ├── predict_prices.py # Script to forecast future prices
│   └── check_alerts.py  # Script to generate price alerts
├── dashboard/
│   └── app.py           # Streamlit dashboard application
└── README.md            # This file
```

## Setup Instructions

1. Clone the repository

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your Metals API key:
   ```
   METALS_API_KEY=your_api_key_here
   ```
   Note: If you don't have a Metals API key, the application will generate sample data for testing.

## Running the Application

1. Launch the Streamlit dashboard:
   ```
   cd /path/to/steelpredictor
   streamlit run dashboard/app.py
   ```

2. The dashboard will open in your web browser. Click the "Refresh Data" button in the sidebar to:
   - Fetch the latest steel prices
   - Clean and transform the data
   - Generate price predictions
   - Check for price alerts

3. Use the date range selectors in the sidebar to zoom in on specific time periods.

## Data Pipeline

The application follows this workflow:

1. **Fetch Prices**: Gets real-time and historical steel prices from the Metals API or generates sample data
2. **Transform Data**: Cleans the data and adds features for forecasting
3. **Predict Prices**: Uses linear regression to forecast steel prices for the next 90 days
4. **Check Alerts**: Identifies significant price changes in the forecast
5. **Dashboard**: Displays all the information in an interactive UI

## License

This project is open-source and available under the MIT License.

## Notes

- The application generates sample data if no API key is provided, allowing for testing without an API subscription.
- Predictions are based on historical patterns and should be used as a reference only.
- The alert threshold is set to 5% by default but can be modified in the code.
