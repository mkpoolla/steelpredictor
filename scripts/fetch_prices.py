#!/usr/bin/env python3
"""
Fetch real-time and historical steel prices from Metals API
"""
import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv('METALS_API_KEY')
BASE_URL = "https://metals-api.com/api"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_price():
    """
    Get the latest steel price from the Metals API
    """
    try:
        url = f"{BASE_URL}/latest"
        params = {
            'access_key': API_KEY,
            'base': 'USD',
            'symbols': 'STEEL'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Failed to fetch latest price: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching latest price: {str(e)}")
        return None

def get_historical_prices(days=365):
    """
    Get historical steel prices for the last specified number of days
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    try:
        url = f"{BASE_URL}/timeseries"
        params = {
            'access_key': API_KEY,
            'base': 'USD',
            'symbols': 'STEEL',
            'start_date': start_date_str,
            'end_date': end_date_str
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Failed to fetch historical prices: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching historical prices: {str(e)}")
        return None

def save_data_to_csv(data, filename):
    """
    Save the API response data to a CSV file
    """
    try:
        output_file = os.path.join(OUTPUT_DIR, filename)
        
        # For timeseries data
        if 'rates' in data and isinstance(data['rates'], dict):
            rows = []
            for date, rates in data['rates'].items():
                if 'STEEL' in rates:
                    rows.append({
                        'date': date,
                        'price': rates['STEEL']
                    })
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved data to {output_file}")
            return output_file
        
        # For latest price data
        elif 'rates' in data and isinstance(data['rates'], dict) and 'STEEL' in data['rates']:
            df = pd.DataFrame([{
                'date': datetime.now().strftime("%Y-%m-%d"),
                'price': data['rates']['STEEL']
            }])
            df['date'] = pd.to_datetime(df['date'])
            df.to_csv(output_file, index=False)
            logger.info(f"Saved data to {output_file}")
            return output_file
        else:
            logger.error("Unexpected data structure")
            return None
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        return None

def fetch_and_save_prices():
    """
    Fetch and save both latest and historical steel prices
    """
    # Create dummy data if API key is not available
    if not API_KEY:
        logger.warning("No API key found. Creating dummy data instead.")
        return create_dummy_data()
    
    # Get latest price
    latest_data = get_latest_price()
    if latest_data:
        save_data_to_csv(latest_data, "latest_steel_price.csv")
    
    # Get historical prices
    historical_data = get_historical_prices(days=365)
    if historical_data:
        return save_data_to_csv(historical_data, "historical_steel_prices.csv")
    
    return None

def create_dummy_data():
    """
    Create dummy data for testing when API key is not available
    """
    # Create dummy historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic prices with trends and some randomness
    import numpy as np
    base_price = 600  # Starting price in USD/ton
    trend = np.linspace(0, 50, len(date_range))  # Gradual upward trend
    seasonality = 20 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 60))  # 60-day cycle
    noise = np.random.normal(0, 10, len(date_range))  # Random noise
    
    prices = base_price + trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'price': prices
    })
    
    # Save to CSV
    historical_file = os.path.join(OUTPUT_DIR, "historical_steel_prices.csv")
    df.to_csv(historical_file, index=False)
    logger.info(f"Created dummy historical data at {historical_file}")
    
    # Save latest price
    latest_file = os.path.join(OUTPUT_DIR, "latest_steel_price.csv")
    latest_df = df.iloc[-1:].copy()
    latest_df.to_csv(latest_file, index=False)
    logger.info(f"Created dummy latest price data at {latest_file}")
    
    return historical_file

if __name__ == "__main__":
    logger.info("Fetching steel prices from API...")
    output_file = fetch_and_save_prices()
    if output_file:
        logger.info(f"Successfully saved steel prices to {output_file}")
    else:
        logger.error("Failed to fetch and save steel prices")
