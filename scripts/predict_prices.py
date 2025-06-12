#!/usr/bin/env python3
"""
Run linear regression to forecast steel prices for next 7/30/90 days
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import pickle
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
MODELS_DIR = os.path.join(PROJ_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_processed_data():
    """
    Load processed steel price data
    """
    try:
        input_file = os.path.join(DATA_DIR, "processed_steel_prices.csv")
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.error(f"File not found: {input_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        return None

def prepare_features(df):
    """
    Prepare features for model training
    """
    if df is None or df.empty:
        logger.error("No data to prepare features")
        return None, None
    
    try:
        # Select features for training
        features = [
            'day_of_week', 'month', 'year', 
            'price_lag_1', 'price_lag_7', 'price_lag_30',
            'price_rolling_mean_7', 'price_rolling_std_7', 'price_rolling_mean_30',
            'price_change_1', 'price_change_7', 'price_change_30'
        ]
        
        # Make sure all features exist
        existing_features = [f for f in features if f in df.columns]
        if len(existing_features) < len(features):
            logger.warning(f"Some features are missing: {set(features) - set(existing_features)}")
        
        X = df[existing_features]
        y = df['price']
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None, None

def train_linear_regression(X, y):
    """
    Train a linear regression model
    """
    if X is None or y is None:
        logger.error("No data to train model")
        return None
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        # Save the model
        model_file = os.path.join(MODELS_DIR, "linear_regression_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved model to {model_file}")
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def generate_future_features(df, days=90):
    """
    Generate feature values for future dates for prediction
    """
    if df is None or df.empty:
        logger.error("No data to generate future features")
        return None
    
    try:
        # Get the last date in the dataset
        last_date = df['date'].max()
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        future_df = pd.DataFrame({'date': future_dates})
        
        # Extract date features
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['year'] = future_df['date'].dt.year
        
        # For lag features, use the last available values
        last_prices = df['price'].iloc[-30:].tolist()  # Get last 30 days of prices
        
        # Initialize lists for lag features
        price_lag_1_values = []
        price_lag_7_values = []
        price_lag_30_values = []
        
        # Initialize lists for rolling features
        rolling_mean_7_values = []
        rolling_std_7_values = []
        rolling_mean_30_values = []
        
        # Initialize lists for price change features
        price_change_1_values = []
        price_change_7_values = []
        price_change_30_values = []
        
        # Temporary list to store predicted prices
        predicted_prices = []
        
        # Populate lag and rolling features iteratively
        for i in range(days):
            # For the first prediction, use historical values
            if i == 0:
                price_lag_1 = last_prices[-1]  # Last price
                price_lag_7 = last_prices[-7] if len(last_prices) >= 7 else last_prices[-1]
                price_lag_30 = last_prices[-30] if len(last_prices) >= 30 else last_prices[-1]
                
                # Calculate rolling statistics
                rolling_mean_7 = np.mean(last_prices[-7:]) if len(last_prices) >= 7 else np.mean(last_prices)
                rolling_std_7 = np.std(last_prices[-7:]) if len(last_prices) >= 7 else np.std(last_prices)
                rolling_mean_30 = np.mean(last_prices[-30:]) if len(last_prices) >= 30 else np.mean(last_prices)
                
                # Calculate price changes
                price_change_1 = (last_prices[-1] / last_prices[-2] - 1) if len(last_prices) >= 2 else 0
                price_change_7 = (last_prices[-1] / last_prices[-8] - 1) if len(last_prices) >= 8 else 0
                price_change_30 = (last_prices[-1] / last_prices[-31] - 1) if len(last_prices) >= 31 else 0
            
            # For subsequent predictions, use previous predictions
            else:
                price_lag_1 = predicted_prices[-1]
                price_lag_7 = predicted_prices[-7] if len(predicted_prices) >= 7 else (last_prices + predicted_prices)[-7]
                price_lag_30 = predicted_prices[-30] if len(predicted_prices) >= 30 else (last_prices + predicted_prices)[-30]
                
                # Calculate rolling statistics using both historical and predicted prices
                prices_so_far = last_prices + predicted_prices
                rolling_window_7 = prices_so_far[-7:]
                rolling_window_30 = prices_so_far[-30:]
                
                rolling_mean_7 = np.mean(rolling_window_7)
                rolling_std_7 = np.std(rolling_window_7)
                rolling_mean_30 = np.mean(rolling_window_30)
                
                # Calculate price changes
                price_change_1 = (prices_so_far[-1] / prices_so_far[-2] - 1)
                price_change_7 = (prices_so_far[-1] / prices_so_far[-8] - 1) if len(prices_so_far) >= 8 else 0
                price_change_30 = (prices_so_far[-1] / prices_so_far[-31] - 1) if len(prices_so_far) >= 31 else 0
            
            # Store values for this future date
            price_lag_1_values.append(price_lag_1)
            price_lag_7_values.append(price_lag_7)
            price_lag_30_values.append(price_lag_30)
            
            rolling_mean_7_values.append(rolling_mean_7)
            rolling_std_7_values.append(rolling_std_7)
            rolling_mean_30_values.append(rolling_mean_30)
            
            price_change_1_values.append(price_change_1)
            price_change_7_values.append(price_change_7)
            price_change_30_values.append(price_change_30)
            
            # Use the last known price as a placeholder instead of None
            # This allows us to build features, and these will be overridden in predict_future
            placeholder_price = last_prices[-1] if i == 0 else predicted_prices[-1]
            predicted_prices.append(placeholder_price)
        
        # Add lag features to future_df
        future_df['price_lag_1'] = price_lag_1_values
        future_df['price_lag_7'] = price_lag_7_values
        future_df['price_lag_30'] = price_lag_30_values
        
        # Add rolling features
        future_df['price_rolling_mean_7'] = rolling_mean_7_values
        future_df['price_rolling_std_7'] = rolling_std_7_values
        future_df['price_rolling_mean_30'] = rolling_mean_30_values
        
        # Add price change features
        future_df['price_change_1'] = price_change_1_values
        future_df['price_change_7'] = price_change_7_values
        future_df['price_change_30'] = price_change_30_values
        
        return future_df, last_prices
    
    except Exception as e:
        logger.error(f"Error generating future features: {str(e)}")
        return None, None

def predict_future(model, future_df, last_prices, days=90):
    """
    Make predictions for future dates
    """
    if model is None or future_df is None or future_df.empty:
        logger.error("Missing model or future features for prediction")
        return None
    
    try:
        # List to store predictions
        predictions = []
        
        # Make predictions one by one, updating features after each prediction
        for i in range(days):
            # Get features for the current future date
            features = [
                'day_of_week', 'month', 'year', 
                'price_lag_1', 'price_lag_7', 'price_lag_30',
                'price_rolling_mean_7', 'price_rolling_std_7', 'price_rolling_mean_30',
                'price_change_1', 'price_change_7', 'price_change_30'
            ]
            
            # Make sure all features exist
            existing_features = [f for f in features if f in future_df.columns]
            
            X_future = future_df.iloc[i:i+1][existing_features]
            
            # Make a prediction
            prediction = model.predict(X_future)[0]
            predictions.append(prediction)
            
            # Update lag, rolling, and change features for next predictions
            if i < days - 1:
                all_prices = last_prices + predictions
                
                # Update lag features
                future_df.loc[future_df.index[i+1], 'price_lag_1'] = predictions[-1]
                if i + 1 >= 7:
                    future_df.loc[future_df.index[i+1], 'price_lag_7'] = predictions[i-6]
                else:
                    future_df.loc[future_df.index[i+1], 'price_lag_7'] = all_prices[-7]
                
                if i + 1 >= 30:
                    future_df.loc[future_df.index[i+1], 'price_lag_30'] = predictions[i-29]
                else:
                    future_df.loc[future_df.index[i+1], 'price_lag_30'] = all_prices[-30]
                
                # Update rolling features
                rolling_window_7 = all_prices[-7:]
                rolling_window_30 = all_prices[-30:]
                
                future_df.loc[future_df.index[i+1], 'price_rolling_mean_7'] = np.mean(rolling_window_7)
                future_df.loc[future_df.index[i+1], 'price_rolling_std_7'] = np.std(rolling_window_7)
                future_df.loc[future_df.index[i+1], 'price_rolling_mean_30'] = np.mean(rolling_window_30)
                
                # Update price change features
                future_df.loc[future_df.index[i+1], 'price_change_1'] = (all_prices[-1] / all_prices[-2] - 1)
                future_df.loc[future_df.index[i+1], 'price_change_7'] = (all_prices[-1] / all_prices[-8] - 1)
                future_df.loc[future_df.index[i+1], 'price_change_30'] = (all_prices[-1] / all_prices[-31] - 1)
        
        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({
            'date': future_df['date'],
            'predicted_price': predictions
        })
        
        # Save predictions
        predictions_file = os.path.join(DATA_DIR, "price_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Saved predictions to {predictions_file}")
        
        return predictions_df
    
    except Exception as e:
        logger.error(f"Error predicting future prices: {str(e)}")
        return None

def predict_prices():
    """
    Main function to predict prices
    """
    # Load processed data
    logger.info("Loading processed steel price data...")
    df = load_processed_data()
    
    if df is None or df.empty:
        logger.error("Failed to load processed data")
        return None
    
    # Prepare features
    logger.info("Preparing features for model training...")
    X, y = prepare_features(df)
    
    if X is None or y is None:
        logger.error("Failed to prepare features")
        return None
    
    # Train model
    logger.info("Training linear regression model...")
    model = train_linear_regression(X, y)
    
    if model is None:
        logger.error("Failed to train model")
        return None
    
    # Generate future features
    logger.info("Generating features for future prediction...")
    future_df, last_prices = generate_future_features(df, days=90)
    
    if future_df is None:
        logger.error("Failed to generate future features")
        return None
    
    # Predict future prices
    logger.info("Predicting future steel prices...")
    predictions_df = predict_future(model, future_df, last_prices, days=90)
    
    return predictions_df

if __name__ == "__main__":
    predictions_df = predict_prices()
    if predictions_df is not None:
        logger.info("Successfully predicted future steel prices")
    else:
        logger.error("Failed to predict future steel prices")
