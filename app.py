from flask import Flask, request, jsonify, render_template
import numpy as np
from train_model import load_model, predict_quality

app = Flask(__name__)

# Load the model
model = load_model()

# FRONTEND ROUTES


@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predicting.html")

# API ROUTE


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features_order = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
        "PT08.S5(O3)", "T", "RH", "AH", "Hour", "Day", "Month"
    ]

    input_list = [float(data[f]) for f in features_order]
    input_array = np.array([input_list])

    result = predict_quality(model, input_array)

    return jsonify({"prediction": float(result[0])})


if __name__ == "__main__":
    app.run(debug=True)

