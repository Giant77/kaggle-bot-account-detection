import uuid
import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import warnings
import traceback

# ------------------------------
# Inisialisasi FastAPI app
# ------------------------------
app = FastAPI(title="User Bot Prediction API", description="API untuk memprediksi apakah user adalah bot atau bukan", version="1.0")

# ------------------------------
# Skema data input dari user
# ------------------------------
class UserInput(BaseModel):
    NAME: str
    GENDER: str  # "Male" atau "Female"
    EMAIL_ID: str
    IS_GLOGIN: bool
    FOLLOWER_COUNT: int
    FOLLOWING_COUNT: int
    DATASET_COUNT: int
    CODE_COUNT: int
    DISCUSSION_COUNT: int
    AVG_NB_READ_TIME_MIN: float
    TOTAL_VOTES_GAVE_NB: int
    TOTAL_VOTES_GAVE_DS: int
    TOTAL_VOTES_GAVE_DC: int

# ------------------------------
# Load model & scaler
# ------------------------------
scaler = joblib.load('../../model/minmax_scaler.pkl')  # Path relatif dari lokasi main.py
model = joblib.load('../../model/ET_model.pkl')         # Model yang dilatih sebelumnya

# Daftar fitur yang dibutuhkan model (harus sama seperti saat training)
FEATURES_FOR_SCALER = [
    'IS_GLOGIN', 'FOLLOWER_COUNT', 'FOLLOWING_COUNT',
    'DATASET_COUNT', 'CODE_COUNT', 'DISCUSSION_COUNT',
    'AVG_NB_READ_TIME_MIN', 'TOTAL_VOTES_GAVE_NB',
    'TOTAL_VOTES_GAVE_DS', 'TOTAL_VOTES_GAVE_DC',
    'GENDER_Female', 'GENDER_Male' 
]

# ------------------------------
# Endpoint Prediksi
# ------------------------------
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    try:
        # Konversi inputan user menjadi DataFrame
        data = pd.DataFrame([user_input.dict()])
        original_index = data.index 

        # Hilangkan kolom yang tidak digunakan model
        data_to_process = data.drop(columns=["NAME", "EMAIL_ID"], errors='ignore')

        # Encode kolom gender jadi dua kolom binary
        data_to_process['GENDER_Male'] = (data_to_process['GENDER'] == 'Male').astype(int)
        data_to_process['GENDER_Female'] = (data_to_process['GENDER'] == 'Female').astype(int)
        data_to_process = data_to_process.drop('GENDER', axis=1)

        # Pastikan semua fitur yang dibutuhkan tersedia
        for col in FEATURES_FOR_SCALER:
            if col not in data_to_process.columns:
                data_to_process[col] = 0  # default value

        # Ambil kolom fitur dan skalakan
        data_for_scaler = data_to_process[FEATURES_FOR_SCALER]
        scaled_data_np = scaler.transform(data_for_scaler)

        # Buat DataFrame hasil scaling
        processed_df = pd.DataFrame(scaled_data_np, columns=FEATURES_FOR_SCALER, index=original_index)

        # Prediksi
        prediction = model.predict(processed_df)

        return {
            "prediction": int(prediction[0]),
            "message": "1 berarti Bot, 0 berarti Bukan Bot"
        }

    except Exception as e:
        traceback_str = traceback.format_exc()
        return {
            "error": str(e),
            "trace": traceback_str
        }

# ------------------------------
# Menjalankan server saat file ini dijalankan langsung
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
