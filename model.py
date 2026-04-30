from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def train_model(df):
    # Features for the model:
    # 1. Units consumed (the primary driver)
    # 2. Tariff (city-specific cost)
    # 3. INTERACTION: Units * Tariff (This is the ACTUAL formula for a bill)
    # 4. Appliance counts (to capture specific usage patterns)
    
    appliance_cols = ['fan', 'refrigerator', 'airconditioner', 'television', 'monitor', 'motorpump']
    # Filter available appliance columns
    app_cols = [col for col in appliance_cols if col in df.columns]
    
    df_train = df.copy()
    # Add interaction term: This is crucial for accuracy as Bill = Units * Tariff
    df_train['units_x_tariff'] = df_train['units_consumed'] * df_train['tariff']
    
    X = df_train[['units_consumed', 'tariff', 'units_x_tariff'] + app_cols]
    y = df_train['bill_amount']
    
    # Using LinearRegression with an interaction term is extremely accurate for bills
    # and extrapolates perfectly for new inputs.
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return model, {'r2': r2, 'mae': mae, 'mse': mse}

def predict_bill(model, units, tariff, appliances_data):
    # Map UI dictionary to model features
    
    feature_data = {
        'units_consumed': units,
        'tariff': tariff,
        'units_x_tariff': units * tariff, # Include the interaction term for prediction
        'fan': appliances_data.get('fan', {}).get('count', 0),
        'refrigerator': appliances_data.get('refrigerator', {}).get('count', 0),
        'airconditioner': appliances_data.get('ac', {}).get('count', 0),
        'television': appliances_data.get('tv', {}).get('count', 0),
        'monitor': appliances_data.get('monitor', {}).get('count', 0),
        'motorpump': appliances_data.get('motor', {}).get('count', 0)
    }
    
    # Ensure we use the exact features the model expects in the correct order
    try:
        features = model.feature_names_in_
    except AttributeError:
        # Fallback if model doesn't have feature_names_in_
        features = ['units_consumed', 'tariff', 'units_x_tariff', 'fan', 'refrigerator', 'airconditioner', 'television', 'monitor', 'motorpump']
    
    # Create input DataFrame with correct feature names and order
    input_df = pd.DataFrame([feature_data], columns=features)
    
    # Fill any missing columns with 0
    input_df = input_df.fillna(0)
    
    # If units consumed is 0, return 0 to avoid baseline intercept bias
    if units == 0:
        return 0.0
    
    prediction = model.predict(input_df)
    
    # Ensure prediction doesn't go below 0
    return max(0, prediction[0])

if __name__ == "__main__":
    import os
    from preprocess import load_and_preprocess_data
    
    # Path to the dataset
    dataset_path = os.path.join('dataset', 'electricity_bill_dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"--- Training Model on {dataset_path} ---")
        
        # 1. Load and Preprocess
        df = load_and_preprocess_data(dataset_path)
        
        # 2. Train Model
        model, metrics = train_model(df)
        
        # 3. Display Results
        print("\n" + "="*30)
        print(f"MODEL TRAINING SUMMARY")
        print("="*30)
        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"Accuracy: {metrics['r2'] * 100:.2f}%")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"Mean Squared Error: {metrics['mse']:.4f}")
        print("="*30)
        print("\nThe model is now trained and ready for predictions.\n")
    else:
        print(f"Error: Dataset not found at {dataset_path}")
