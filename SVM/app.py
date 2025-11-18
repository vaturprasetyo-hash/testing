from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import tempfile
import os

app = Flask(_name_)

# ===============================
# 1. Download model dari Google Drive
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=1Gjoja0smtWN2B9gR0x0ZAmfQO_lJo_SW"
SCALER_URL = "https://drive.google.com/uc?id=1Gg3bSRdffy03UH5VyTLJjao74w8G54YG"

def download_file(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

# Temp directory untuk file model
model_path = os.path.join(tempfile.gettempdir(), "svm_model.pkl")
scaler_path = os.path.join(tempfile.gettempdir(), "scaler.pkl")

if not os.path.exists(model_path):
    download_file(MODEL_URL, model_path)
if not os.path.exists(scaler_path):
    download_file(SCALER_URL, scaler_path)

# ===============================
# 2. Load model & scaler
# ===============================
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ===============================
# 3. Helper parsing
# ===============================
def parse_bp(blood_pressure):
    """Format input '120/80' → 120, 80"""
    try:
        s, d = blood_pressure.split('/')
        return float(s), float(d)
    except:
        return 0, 0

def parse_glucose(glucose):
    """Format '150a' → 150 | '200b' → 200"""
    try:
        num = ''.join([c for c in glucose if c.isdigit()])
        return float(num)
    except:
        return 0


# ===============================
# 4. Endpoint Prediksi
# ===============================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    data = request.get_json()

    # Ambil & parsing data
    gender = 1 if data["gender"].lower() == "laki-laki" else 0
    age = float(data["age"])
    hr = float(data["heart_rate"])
    spo2 = float(data["spo2"])
    temp = float(data["temperature"])
    glucose = parse_glucose(data["glucose"])
    sistol, diastol = parse_bp(data["blood_pressure"])

    # Susun array fitur
    X = np.array([[gender, age, glucose, sistol, diastol, spo2, temp, hr]])

    # Scale fitur
    X_scaled = scaler.transform(X)

    # Prediksi
    pred = model.predict(X_scaled)[0]

    return jsonify({"risk": pred})


@app.route("/", methods=["GET"])
def home():
    return "SVM Backend Running!"


if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)