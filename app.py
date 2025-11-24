from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)   # 🔥 Izinkan akses dari semua domain

# =====================================================
# 1. Load Model & Scaler
# =====================================================
MODEL_PATH = "svm_model_rbf.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =====================================================
# 2. Helper Functions
# =====================================================

def normalize_float(x):
    """Membulatkan nilai float ke integer terdekat."""
    try:
        return round(float(str(x).replace(",", ".")))
    except:
        return 0


def parse_bp(bp):
    """Parse tekanan darah + pembulatan desimal."""
    try:
        if not bp or "/" not in bp:
            return 0, 0

        s, d = bp.split("/")

        # ganti koma ke titik lalu float
        s = float(s.replace(",", "."))
        d = float(d.replace(",", "."))

        return round(s), round(d)
    except:
        return 0, 0


def parse_glucose(glucose):
    """
    Mendukung format:
    - 120a
    - 150.5a
    - 180,7b
    - 200 (default puasa)
    """
    s = str(glucose).strip().lower()
    if s == "" or s == "nan":
        return 0, 1

    tipe = 1
    if s[-1] in ["a", "b"]:
        tipe = 1 if s[-1] == "a" else 2
        s = s[:-1]

    s = s.replace(",", ".")  # dukung koma
    try:
        val = float(s)
    except:
        val = 0.0

    val = round(val)  # model dilatih dengan integer

    return val, tipe


# =====================================================
# 3. Endpoint Prediksi
# =====================================================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    try:
        data = request.get_json()

        gender = data.get("gender", "").lower()
        gender_num = 1 if gender == "laki-laki" else 0

        age = normalize_float(data.get("age", 0))
        hr = normalize_float(data.get("heart_rate", 0))
        spo2 = normalize_float(data.get("spo2", 0))
        temp = normalize_float(data.get("temperature", 0))

        glucose_raw = data.get("glucose", "0")
        glucose_val, glucose_type = parse_glucose(glucose_raw)

        sistol, diastol = parse_bp(data.get("blood_pressure", "0/0"))

        # Susunan fitur HARUS sesuai model latih!!
        X = np.array([[
            gender_num,
            age,
            glucose_val,
            glucose_type,
            sistol,
            diastol,
            spo2,
            temp,
            hr
        ]])

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
