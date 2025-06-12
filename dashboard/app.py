#!/usr/bin/env python3
"""
Streamlit UI to visualize steel price trends, predictions, and alerts
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root directory to Python path
PROJ_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJ_DIR))

# Import project modules
from scripts.fetch_prices import fetch_and_save_prices
from scripts.transform_data import transform_data
from scripts.predict_prices import predict_prices
from scripts.check_alerts import check_alerts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(PROJ_DIR, "data")
ALERTS_DIR = os.path.join(PROJ_DIR, "alerts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Steel Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
    }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .increase-alert {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .decrease-alert {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
    }
    .alert-date {
        font-weight: bold;
    }
    .alert-change {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

def load_historical_data():
    """
    Load historical steel price data
    """
    try:
        file_path = os.path.join(DATA_DIR, "historical_steel_prices.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            st.warning("No historical data found. Please refresh data.")
            return None
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

def load_predictions():
    """
    Load price predictions data
    """
    try:
        file_path = os.path.join(DATA_DIR, "price_predictions.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            st.warning("No prediction data found. Please run predictions.")
            return None
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None

def load_alerts():
    """
    Load price alerts data
    """
    try:
        file_path = os.path.join(ALERTS_DIR, "price_alerts.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                alerts = json.load(f)
            return alerts
        else:
            return []
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")
        return []

def run_data_pipeline(fetch_data=True):
    """
    Run the complete or partial data pipeline
    """
    if fetch_data:
        with st.spinner('Fetching steel prices...'):
            fetch_and_save_prices()
        
        with st.spinner('Transforming data...'):
            transform_data()
    
    with st.spinner('Predicting prices...'):
        predict_prices()
    
    with st.spinner('Checking for price alerts...'):
        check_alerts()
    
    st.success('Price predictions updated successfully!')

def main():
    # Header
    st.markdown('<h1 class="main-header">Steel Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time steel price forecasting and alerts</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Create a container for buttons
    button_col1, button_col2 = st.sidebar.columns(2)
    
    # Data refresh button
    if button_col1.button('Refresh Data', help="Fetch new data and run predictions"):
        run_data_pipeline(fetch_data=True)
        
    # Prediction only button
    if button_col2.button('Run Predictions', help="Run predictions using existing data"):
        run_data_pipeline(fetch_data=False)
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    historical_data = load_historical_data()
    predictions_data = load_predictions()
    
    min_date = None
    max_date = None
    
    if historical_data is not None:
        min_date = historical_data['date'].min().date()
        if predictions_data is not None:
            max_date = predictions_data['date'].max().date()
        else:
            max_date = historical_data['date'].max().date()
    
    if min_date is not None and max_date is not None:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date + timedelta(days=(max_date - min_date).days // 2),
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=start_date,
            max_value=max_date
        )
    else:
        start_date = datetime.now().date() - timedelta(days=30)
        end_date = datetime.now().date() + timedelta(days=30)
    
    # Download data
    st.sidebar.subheader("Download Data")
    if historical_data is not None:
        csv = historical_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Historical Data",
            data=csv,
            file_name="steel_prices_historical.csv",
            mime="text/csv"
        )
    
    if predictions_data is not None:
        csv = predictions_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Predictions Data",
            data=csv,
            file_name="steel_prices_predictions.csv",
            mime="text/csv"
        )
    
    # Main content area - three columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Steel Price Trends and Predictions")
        
        if historical_data is not None and not historical_data.empty:
            # Filter data by selected date range
            filtered_historical = historical_data[
                (historical_data['date'].dt.date >= start_date) & 
                (historical_data['date'].dt.date <= end_date)
            ]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(
                filtered_historical['date'], 
                filtered_historical['price'],
                color='blue', 
                marker='o', 
                linestyle='-', 
                linewidth=1, 
                markersize=3,
                label='Historical Prices'
            )
            
            # Plot predictions if available
            if predictions_data is not None and not predictions_data.empty:
                filtered_predictions = predictions_data[
                    (predictions_data['date'].dt.date >= start_date) & 
                    (predictions_data['date'].dt.date <= end_date)
                ]
                
                if not filtered_predictions.empty:
                    # Find where historical data ends
                    last_historical_date = historical_data['date'].max()
                    
                    ax.plot(
                        filtered_predictions['date'], 
                        filtered_predictions['predicted_price'],
                        color='red', 
                        marker='x', 
                        linestyle='--', 
                        linewidth=1, 
                        markersize=3,
                        label='Predicted Prices'
                    )
                    
                    # Add a vertical line to separate historical and prediction
                    ax.axvline(x=last_historical_date, color='gray', linestyle='-', linewidth=1)
            
            ax.set_title('Steel Price Trend and Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD/ton)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set y-axis to start from zero
            ax.set_ylim(bottom=0)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            st.pyplot(fig)
            
            # Show current price
            if not filtered_historical.empty:
                current_price = filtered_historical.iloc[-1]['price']
                st.metric(
                    label="Current Steel Price", 
                    value=f"${current_price:.2f} per ton"
                )
            
            # Price statistics
            st.subheader("Price Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("Average Price", f"${filtered_historical['price'].mean():.2f}")
            
            with stats_col2:
                st.metric("Minimum Price", f"${filtered_historical['price'].min():.2f}")
            
            with stats_col3:
                st.metric("Maximum Price", f"${filtered_historical['price'].max():.2f}")
            
            # Price change over time periods
            if len(filtered_historical) > 1:
                st.subheader("Price Changes")
                
                change_col1, change_col2, change_col3 = st.columns(3)
                
                # Last week change
                week_cutoff = pd.Timestamp(datetime.now().date() - timedelta(days=7))
                week_data = historical_data[historical_data['date'] >= week_cutoff]
                if len(week_data) >= 2:
                    week_change = ((week_data['price'].iloc[-1] / week_data['price'].iloc[0]) - 1) * 100
                    with change_col1:
                        st.metric("Last Week", f"{week_change:.2f}%", delta=f"{week_change:.2f}%")
                
                # Last month change
                month_cutoff = pd.Timestamp(datetime.now().date() - timedelta(days=30))
                month_data = historical_data[historical_data['date'] >= month_cutoff]
                if len(month_data) >= 2:
                    month_change = ((month_data['price'].iloc[-1] / month_data['price'].iloc[0]) - 1) * 100
                    with change_col2:
                        st.metric("Last Month", f"{month_change:.2f}%", delta=f"{month_change:.2f}%")
                
                # Last quarter change
                quarter_cutoff = pd.Timestamp(datetime.now().date() - timedelta(days=90))
                quarter_data = historical_data[historical_data['date'] >= quarter_cutoff]
                if len(quarter_data) >= 2:
                    quarter_change = ((quarter_data['price'].iloc[-1] / quarter_data['price'].iloc[0]) - 1) * 100
                    with change_col3:
                        st.metric("Last Quarter", f"{quarter_change:.2f}%", delta=f"{quarter_change:.2f}%")
        else:
            st.info("No historical price data available. Please click 'Refresh Data' in the sidebar.")
    
    with col2:
        st.subheader("Price Alerts")
        alerts = load_alerts()
        
        if alerts:
            # Filter alerts for the next 7 days
            today = datetime.now().date()
            recent_alerts = [
                alert for alert in alerts 
                if datetime.strptime(alert['date'], '%Y-%m-%d').date() <= (today + timedelta(days=7))
            ]
            
            if recent_alerts:
                for alert in recent_alerts:
                    alert_type = alert['alert_type']
                    alert_class = "increase-alert" if alert_type == "increase" else "decrease-alert"
                    emoji = "🔴" if alert_type == "increase" else "🟢"
                    
                    st.markdown(f"""
                    <div class="alert-box {alert_class}">
                        <span class="alert-date">{alert['date']}</span>: {emoji} 
                        <span class="alert-change">{alert['percentage_change']}%</span> price {alert_type} predicted
                        <br>From ${alert['current_price']} to ${alert['predicted_price']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts for the next 7 days")
        else:
            st.info("No price alerts detected")
        
        # Price forecast summary
        st.subheader("Price Forecast Summary")
        if predictions_data is not None and not predictions_data.empty:
            # Get predictions for specific time periods
            next_7_days = predictions_data[predictions_data['date'] <= pd.Timestamp(datetime.now() + timedelta(days=7))]
            next_30_days = predictions_data[predictions_data['date'] <= pd.Timestamp(datetime.now() + timedelta(days=30))]
            next_90_days = predictions_data[predictions_data['date'] <= pd.Timestamp(datetime.now() + timedelta(days=90))]
            
            # Current price
            if historical_data is not None and not historical_data.empty:
                current_price = historical_data.iloc[-1]['price']
                
                # Calculate expected changes
                if not next_7_days.empty:
                    price_7d = next_7_days.iloc[-1]['predicted_price']
                    change_7d = ((price_7d / current_price) - 1) * 100
                    st.metric(
                        label="7-Day Forecast", 
                        value=f"${price_7d:.2f}", 
                        delta=f"{change_7d:.2f}%"
                    )
                
                if not next_30_days.empty:
                    # Calculate point forecast for 30 days
                    price_30d = next_30_days.iloc[-1]['predicted_price']
                    change_30d = ((price_30d / current_price) - 1) * 100
                    
                    # Calculate average price over next month (30 days)
                    avg_price_next_month = next_30_days['predicted_price'].mean()
                    avg_change_next_month = ((avg_price_next_month / current_price) - 1) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="30-Day Forecast", 
                            value=f"${price_30d:.2f}", 
                            delta=f"{change_30d:.2f}%"
                        )
                    with col2:
                        st.metric(
                            label="Next Month Average", 
                            value=f"${avg_price_next_month:.2f}", 
                            delta=f"{avg_change_next_month:.2f}%"
                        )
                
                if not next_90_days.empty:
                    price_90d = next_90_days.iloc[-1]['predicted_price']
                    change_90d = ((price_90d / current_price) - 1) * 100
                    st.metric(
                        label="90-Day Forecast", 
                        value=f"${price_90d:.2f}", 
                        delta=f"{change_90d:.2f}%"
                    )
        else:
            st.info("No prediction data available")
        
        # Add information about the model
        st.subheader("About the Model")
        st.info("""
        This application uses a linear regression model to forecast steel prices based on historical trends.
        The model analyzes patterns in historical data, including:
        
        - Price lag features
        - Rolling statistics
        - Seasonal patterns
        
        Predictions are updated when you click the 'Refresh Data' button.
        """)

if __name__ == "__main__":
    main()
