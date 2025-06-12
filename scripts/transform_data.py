#!/usr/bin/env python3
"""
Clean and preprocess steel price data
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJ_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_historical_data():
    """
    Load historical steel price data from CSV
    """
    try:
        input_file = os.path.join(DATA_DIR, "historical_steel_prices.csv")
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.error(f"File not found: {input_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return None

def clean_data(df):
    """
    Clean and preprocess the data:
    - Sort by date
    - Handle missing values
    - Remove outliers
    - Add features for forecasting
    """
    if df is None or df.empty:
        logger.error("No data to clean")
        return None
    
    try:
        # Make a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Sort by date
        df_clean = df_clean.sort_values('date')
        
        # Handle missing values
        if df_clean['price'].isna().any():
            logger.info("Handling missing values in price column")
            # Interpolate missing prices
            df_clean['price'] = df_clean['price'].interpolate(method='linear')
        
        # Remove outliers (prices that are 3 standard deviations away from the mean)
        price_mean = df_clean['price'].mean()
        price_std = df_clean['price'].std()
        lower_bound = price_mean - 3 * price_std
        upper_bound = price_mean + 3 * price_std
        
        outliers = df_clean[(df_clean['price'] < lower_bound) | (df_clean['price'] > upper_bound)]
        if not outliers.empty:
            logger.info(f"Found {len(outliers)} outliers in the data")
            df_clean = df_clean[(df_clean['price'] >= lower_bound) & (df_clean['price'] <= upper_bound)]
        
        # Add features for forecasting
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['year'] = df_clean['date'].dt.year
        
        # Add lag features
        df_clean['price_lag_1'] = df_clean['price'].shift(1)
        df_clean['price_lag_7'] = df_clean['price'].shift(7)
        df_clean['price_lag_30'] = df_clean['price'].shift(30)
        
        # Add rolling statistics
        df_clean['price_rolling_mean_7'] = df_clean['price'].rolling(window=7).mean()
        df_clean['price_rolling_std_7'] = df_clean['price'].rolling(window=7).std()
        df_clean['price_rolling_mean_30'] = df_clean['price'].rolling(window=30).mean()
        
        # Add price change features
        df_clean['price_change_1'] = df_clean['price'].pct_change(1)
        df_clean['price_change_7'] = df_clean['price'].pct_change(7)
        df_clean['price_change_30'] = df_clean['price'].pct_change(30)
        
        # Drop rows with NaN values created by lag and rolling features
        df_clean = df_clean.dropna()
        
        return df_clean
    
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        return None

def save_processed_data(df):
    """
    Save processed data to CSV
    """
    if df is None or df.empty:
        logger.error("No data to save")
        return None
    
    try:
        output_file = os.path.join(DATA_DIR, "processed_steel_prices.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        return None

def transform_data():
    """
    Main function to transform data
    """
    # Load historical data
    logger.info("Loading historical steel price data...")
    df = load_historical_data()
    
    if df is None or df.empty:
        logger.error("Failed to load historical data")
        return None
    
    # Clean and preprocess data
    logger.info("Cleaning and preprocessing data...")
    df_clean = clean_data(df)
    
    if df_clean is None or df_clean.empty:
        logger.error("Failed to clean data")
        return None
    
    # Save processed data
    logger.info("Saving processed data...")
    output_file = save_processed_data(df_clean)
    
    return output_file

if __name__ == "__main__":
    output_file = transform_data()
    if output_file:
        logger.info(f"Successfully transformed data and saved to {output_file}")
    else:
        logger.error("Failed to transform data")
