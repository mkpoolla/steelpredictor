#!/usr/bin/env python3
"""
Trigger alerts if price increases more than 5% in forecast
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
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
ALERTS_DIR = os.path.join(PROJ_DIR, "alerts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

def load_predictions():
    """
    Load price predictions data
    """
    try:
        input_file = os.path.join(DATA_DIR, "price_predictions.csv")
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.error(f"File not found: {input_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return None

def load_latest_price():
    """
    Load latest steel price data
    """
    try:
        input_file = os.path.join(DATA_DIR, "latest_steel_price.csv")
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            return df['price'].iloc[0]
        else:
            # Try to get the latest price from historical data
            historical_file = os.path.join(DATA_DIR, "historical_steel_prices.csv")
            if os.path.exists(historical_file):
                df = pd.read_csv(historical_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                return df['price'].iloc[0]
            else:
                logger.error("No price data found")
                return None
    except Exception as e:
        logger.error(f"Error loading latest price: {str(e)}")
        return None

def check_price_alerts(predictions_df, current_price, threshold=0.05):
    """
    Check if predicted prices exceed the threshold percentage change
    """
    if predictions_df is None or predictions_df.empty:
        logger.error("No predictions data to check for alerts")
        return []
    
    if current_price is None:
        logger.error("No current price available for alert checking")
        return []
    
    try:
        # Filter predictions for the next 7 days
        next_7_days = predictions_df[predictions_df['date'] <= datetime.now() + timedelta(days=7)]
        
        if next_7_days.empty:
            logger.warning("No predictions available for the next 7 days")
            return []
        
        # Calculate percentage change from current price
        next_7_days['pct_change'] = (next_7_days['predicted_price'] / current_price) - 1
        
        # Find days with significant price changes
        significant_changes = next_7_days[abs(next_7_days['pct_change']) >= threshold]
        
        if significant_changes.empty:
            logger.info("No significant price changes detected in the next 7 days")
            return []
        
        # Create alert records
        alerts = []
        for _, row in significant_changes.iterrows():
            alert = {
                'date': row['date'].strftime('%Y-%m-%d'),
                'current_price': round(current_price, 2),
                'predicted_price': round(row['predicted_price'], 2),
                'percentage_change': round(row['pct_change'] * 100, 2),
                'alert_type': 'increase' if row['pct_change'] > 0 else 'decrease',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            alerts.append(alert)
        
        # Save alerts
        alerts_file = os.path.join(ALERTS_DIR, "price_alerts.json")
        
        # Load existing alerts if available
        existing_alerts = []
        if os.path.exists(alerts_file):
            try:
                with open(alerts_file, 'r') as f:
                    existing_alerts = json.load(f)
            except:
                existing_alerts = []
        
        # Add new alerts
        all_alerts = existing_alerts + alerts
        
        # Save to file
        with open(alerts_file, 'w') as f:
            json.dump(all_alerts, f, indent=2)
        
        if alerts:
            logger.info(f"Generated {len(alerts)} price alerts")
        
        return alerts
    
    except Exception as e:
        logger.error(f"Error checking price alerts: {str(e)}")
        return []

def check_alerts():
    """
    Main function to check for price alerts
    """
    # Load predictions
    logger.info("Loading price predictions...")
    predictions_df = load_predictions()
    
    if predictions_df is None or predictions_df.empty:
        logger.error("Failed to load predictions")
        return None
    
    # Load current price
    logger.info("Loading current steel price...")
    current_price = load_latest_price()
    
    if current_price is None:
        logger.error("Failed to load current price")
        return None
    
    # Check for alerts
    logger.info("Checking for significant price changes...")
    alerts = check_price_alerts(predictions_df, current_price)
    
    return alerts

if __name__ == "__main__":
    alerts = check_alerts()
    if alerts:
        logger.info(f"Generated {len(alerts)} price alerts")
        for alert in alerts:
            alert_type = alert['alert_type'].upper()
            logger.info(f"ALERT {alert_type}: {alert['percentage_change']}% change predicted on {alert['date']}")
    else:
        logger.info("No price alerts generated")
