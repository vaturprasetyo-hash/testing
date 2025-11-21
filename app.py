from flask import flask, request, jsonify
import joblib
import numpy as np

app = flask(__name__)

# ===============================
# 1. Load model & scaler (langsung dari folder)
# ===============================
MODEL_PATH = "svm_model_rbf.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# 2. Helper parsing
# ===============================
def parse_bp(bp):
    """Format input '120/80' → (120, 80)"""
    if not bp or "/" not in bp:
        return 0.0, 0.0
    try:
        s, d = bp.split("/")
        return float(s), float(d)
    except:
        return 0.0, 0.0


def parse_glucose(glucose):
    """Format '150a' → 150, '200b' → 200"""
    try:
        num = "".join([c for c in str(glucose) if c.isdigit()])
        return float(num)
    except:
        return 0.0


# ===============================
# 3. Endpoint Prediksi
# ===============================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    try:
        data = request.get_json()

        gender = data.get("gender", "").lower()
        gender_num = 1 if gender == "laki-laki" else 0

        age = float(data.get("age", 0))
        hr = float(data.get("heart_rate", 0))
        spo2 = float(data.get("spo2", 0))
        temp = float(data.get("temperature", 0))

        glucose = parse_glucose(data.get("glucose", "0"))
        sistol, diastol = parse_bp(data.get("blood_pressure", "0/0"))

        X = np.array([[gender_num, age, glucose, sistol, diastol,
                       spo2, temp, hr]])

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return jsonify({"risk": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "SVM Backend Ready!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
