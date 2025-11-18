import joblib
import pandas as pd

# Load model
model = joblib.load("air_quality_model.pkl")

# Correct feature order
features = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
    "PT08.S5(O3)", "T", "RH", "AH", "Hour", "Day", "Month"
]

# Input sample
input_values = [[2.3, 1200, 80, 900, 140, 700, 65, 1200, 1000, 20.5, 45, 0.5, 14, 25, 10]]

# Make DataFrame
input_df = pd.DataFrame(input_values, columns=features)

print(input_df)
# Predict
prediction = model.predict(input_df)

print("Predicted C6H6(GT):", prediction[0])
