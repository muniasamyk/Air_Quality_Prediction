import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define the file name for the saved model
MODEL_PATH = "air_quality_model.pkl"

# Model Loading Function for Flask App
def load_model():
    """Loads the trained model from a file."""
    try:
        # joblib is used for saving/loading scikit-learn objects
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Run this script once to train and save the model.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 2.Prediction Function for Flask App
def predict_quality(model, input_data: np.ndarray):
    if model is None:
        return np.array([0.0])
    
    
    prediction = model.predict(input_data)
    return prediction


# Training and Saving Logic 

def train_and_save_model():
    """Performs data cleaning, training, evaluation, and saves the model."""
    print("--- Starting Model Training Process ---")
    
    try:
        df = pd.read_csv('C:/Users/ms900/OneDrive/Desktop/Air_Quality_Prediction/AirQualityUCI.csv')
    except FileNotFoundError:
        print("ERROR: Could not find the CSV file. Please check the file path.")
        return

    print("Dataset Loaded!")

    # Data Preprocessing 
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df = df.drop(columns=["Date", "Time", "DateTime"])

    # Replace the missing value placeholder with NaN and then drop rows
    df = df.replace(-200, np.nan).dropna()


    # Select Features & Target
    target = "C6H6(GT)"

    features = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
        "PT08.S5(O3)", "T", "RH", "AH", "Hour", "Day", "Month"
    ]

    # Check if the number of features matches the expected 15
    if len(features) != 15:
        print(f"Feature count mismatch: Expected 15, got {len(features)}")
        return

    X = df[features]
    y = df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=15
    )
    model.fit(X_train, y_train)

    # Evaluate Model
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print("\nMODEL RESULTS:")
    print(f"Target Feature: {target}")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel Successfully Saved as {MODEL_PATH}")

# This ensures the training only happens when you run the script directly

if __name__ == '__main__':
    train_and_save_model()
    

