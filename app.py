from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)   # Izinkan akses dari semua domain

# =====================================================
# 1. Load Model & Scaler
# =====================================================
MODEL_PATH = "svm_model_rbf.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =====================================================
# 2. Medical Range (Toleransi Nilai Ekstrem)
# =====================================================
def clamp(value, lo, hi):
    """Membatasi nilai agar tetap di rentang medis yang mungkin."""
    try:
        value = float(value)
    except:
        return lo
    return max(lo, min(value, hi))


# batas medis realistis
MEDICAL_LIMITS = {
    "temp": (33, 42),       # suhu tubuh manusia
    "spo2": (50, 100),      # saturasi
    "hr": (30, 200),        # detak jantung
    "glucose": (40, 500),   # gula darah mg/dL
    "sistol": (70, 250),    # tekanan darah
    "diastol": (40, 150),
}


# =====================================================
# 3. Helper Functions
# =====================================================
def normalize_float(x):
    """Membulatkan nilai float ke integer, dengan dukungan koma."""
    try:
        return round(float(str(x).replace(",", ".")))
    except:
        return 0


def parse_bp(bp):
    """Parse tekanan darah + pembulatan + batas medis."""
    try:
        if not bp or "/" not in bp:
            return 0, 0

        s, d = bp.split("/")

        sistol = clamp(float(s.replace(",", ".")), *MEDICAL_LIMITS["sistol"])
        diastol = clamp(float(d.replace(",", ".")), *MEDICAL_LIMITS["diastol"])

        return round(sistol), round(diastol)
    except:
        return 0, 0


def parse_glucose(glucose):
    """
    Mendukung format:
    - 120a
    - 150.5a
    - 180,7b
    - 200 (default puasa)

    Ditambah:
    - batas medis untuk nilai glucose
    """
    s = str(glucose).strip().lower()
    if s == "" or s == "nan":
        return 0, 1  # default puasa

    tipe = 1
    if s[-1] in ["a", "b"]:
        tipe = 1 if s[-1] == "a" else 2
        s = s[:-1]

    s = s.replace(",", ".")  # dukung koma

    try:
        val = float(s)
    except:
        val = 0.0

    # batas medis
    val = clamp(val, *MEDICAL_LIMITS["glucose"])
    val = round(val)

    return val, tipe


# =====================================================
# 4. Endpoint Prediksi
# =====================================================
@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    try:
        data = request.get_json()

        gender = data.get("gender", "").lower()
        gender_num = 1 if gender == "laki-laki" else 0

        age = normalize_float(data.get("age", 0))

        # nilai sensor dibulatkan + batas medis
        hr = clamp(normalize_float(data.get("heart_rate", 0)), *MEDICAL_LIMITS["hr"])
        spo2 = clamp(normalize_float(data.get("spo2", 0)), *MEDICAL_LIMITS["spo2"])
        temp = clamp(normalize_float(data.get("temperature", 0)), *MEDICAL_LIMITS["temp"])

        glucose_raw = data.get("glucose", "0")
        glucose_val, glucose_type = parse_glucose(glucose_raw)

        sistol, diastol = parse_bp(data.get("blood_pressure", "0/0"))

        # Susunan fitur HARUS SAMA dengan model latih
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
            "glucose_type": glucose_type,
            "msg": "Prediction OK (with medical-range filtering)"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "SVM Backend Ready with Medical Filters! 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
