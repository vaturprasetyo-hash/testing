from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# =====================================================
# 1. Load Model & Scaler
# =====================================================
MODEL_PATH = "svm_model_rbf.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================================
# 2. Medical Range
# =====================================================
def clamp(value, lo, hi):
    try:
        value = float(value)
    except:
        return lo
    return max(lo, min(value, hi))

MEDICAL_LIMITS = {
    "temp": (33, 42),
    "spo2": (50, 100),
    "hr": (30, 200),
    "glucose": (40, 500),
    "sistol": (70, 250),
    "diastol": (40, 150),
}

# =====================================================
# 3. Helper Functions
# =====================================================
def parse_bp(bp):
    try:
        if not bp or "/" not in bp:
            return 0.0, 0.0

        s, d = bp.split("/")

        sistol = clamp(float(s.replace(",", ".")), *MEDICAL_LIMITS["sistol"])
        diastol = clamp(float(d.replace(",", ".")), *MEDICAL_LIMITS["diastol"])

        return float(sistol), float(diastol)
    except:
        return 0.0, 0.0


def parse_glucose(glucose):
    """
    Latihannya pakai:
    df2["Gula_Nilai"] = df2["Gula Darah"].str.extract(r"(\d+)").astype(float)

    Maka:
    - ekstrak digit saja
    - bulatkan
    - tipe a/b tetap didukung
    """
    s = str(glucose).strip().lower()
    if s == "" or s == "nan":
        return 0, 1

    tipe = 1
    if s[-1] in ["a", "b"]:
        tipe = 1 if s[-1] == "a" else 2
        s = s[:-1]

    s = s.replace(",", ".")

    digits = "".join([c for c in s if c.isdigit()])
    if digits == "":
        val = 0.0
    else:
        val = float(digits)

    val = clamp(val, *MEDICAL_LIMITS["glucose"])
    val = round(val)  # WAJIB karena model dilatih angka bulat

    return val, tipe

# =====================================================
# 4. Endpoint Prediksi
# =====================================================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    try:
        data = request.get_json()

        # gender → integer
        gender = data.get("gender", "").lower()
        gender_num = 1 if gender == "laki-laki" else 0

        # umur → integer
        age = int(float(str(data.get("age", 0)).replace(",", ".")))

        # nilai sensor
        hr = clamp(data.get("heart_rate", 0), *MEDICAL_LIMITS["hr"])
        spo2 = clamp(data.get("spo2", 0), *MEDICAL_LIMITS["spo2"])
        temp = clamp(data.get("temperature", 0), *MEDICAL_LIMITS["temp"])

        glucose_val, glucose_type = parse_glucose(data.get("glucose", "0"))
        sistol, diastol = parse_bp(data.get("blood_pressure", "0/0"))

        # urutan fitur = HARUS sama dengan training!!!
        X = np.array([[
            gender_num,
            age,
            glucose_val,
            glucose_type,
            sistol,
            diastol,
            float(spo2),
            float(temp),
            float(hr)
        ]])

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return jsonify({
            "risk": pred,
            "glucose_value": glucose_val,
            "glucose_type": glucose_type,
            "msg": "Prediction OK — aligned with training dataset"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "SVM Backend Ready! (Aligned with Training) 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
