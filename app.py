from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# =====================================================
# 1. Load Model & Scaler
# =====================================================
MODEL_PATH = "svm_model_rbf.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =====================================================
# 2. Helper Function
# =====================================================
def parse_bp(bp):
    """Parse '120/80' menjadi float sistol & diastol"""
    try:
        if not bp or "/" not in bp:
            return 0.0, 0.0
        s, d = bp.split("/")
        return float(s), float(d)
    except:
        return 0.0, 0.0


def parse_glucose(glucose):
    """
    Format yang diterima:
        '150a', '200b', '160' (default = a)
    Output:
        nilai glucose, tipe (1=puasa, 2=2jam PP)
    """
    s = str(glucose).strip().lower()
    if s == "" or s == "nan":
        return 0.0, 1  # default puasa

    tipe = 1  # default 'a'
    if s[-1] in ["a", "b"]:
        tipe = 1 if s[-1] == "a" else 2
        s = s[:-1]

    try:
        val = float("".join(c for c in s if c.isdigit()))
    except:
        val = 0.0

    return val, tipe


# =====================================================
# 3. Endpoint Prediksi
# =====================================================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    try:
        data = request.get_json()

        # -------------- Extract data --------------
        gender = data.get("gender", "").lower()
        gender_num = 1 if gender == "laki-laki" else 0

        age = float(data.get("age", 0))
        hr = float(data.get("heart_rate", 0))
        spo2 = float(data.get("spo2", 0))
        temp = float(data.get("temperature", 0))

        glucose_raw = data.get("glucose", "0")
        glucose_val, glucose_type = parse_glucose(glucose_raw)

        sistol, diastol = parse_bp(data.get("blood_pressure", "0/0"))

        # -------------- Format fitur sesuai training --------------
        X = np.array([[
            gender_num,
            age,
            glucose_val,
            glucose_type,   # 1 = puasa, 2 = 2jam PP
            sistol,
            diastol,
            spo2,
            temp,
            hr
        ]])

        # -------------- Scale & Predict --------------
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return jsonify({
            "risk": pred,
            "glucose_value": glucose_val,
            "glucose_type": glucose_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "SVM Backend Ready! 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)