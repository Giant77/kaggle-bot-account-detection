import uuid
import uvicorn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union, List

# membuat instance FastAPI
app = FastAPI(
    title="Kaggle Bot Detection API",
    description="API untuk mendeteksi akun bot pada Kaggle",
    version="1.0.0"
)

# mendefinisikan schema input 
class UserInput(BaseModel):
    GENDER: str = Field(..., description="Gender of the user (Male or Female)")
    IS_GLOGIN: bool = Field(..., description="Whether the user used Google login")
    FOLLOWER_COUNT: int = Field(..., ge=0, description="Number of followers")
    FOLLOWING_COUNT: int = Field(..., ge=0, description="Number of following")
    CODE_COUNT: int = Field(..., ge=0, description="Number of notebooks created")
    DISCUSSION_COUNT: int = Field(..., ge=0, description="Number of discussions participated")
    AVG_NB_READ_TIME_MIN: float = Field(..., ge=0.0, description="Average notebook read time in minutes")
    TOTAL_VOTES_GAVE_NB: int = Field(..., ge=0, description="Total votes gave to notebooks")
    TOTAL_VOTES_GAVE_DS: int = Field(..., ge=0, description="Total votes gave to datasets")
    TOTAL_VOTES_GAVE_DC: int = Field(..., ge=0, description="Total votes gave to discussion comments")

    class Config:
        schema_extra = {
            "example": {
                "GENDER": "Male",
                "IS_GLOGIN": True,
                "FOLLOWER_COUNT": 10,
                "FOLLOWING_COUNT": 20,
                "CODE_COUNT": 5,
                "DISCUSSION_COUNT": 3,
                "AVG_NB_READ_TIME_MIN": 15.5,
                "TOTAL_VOTES_GAVE_NB": 25,
                "TOTAL_VOTES_GAVE_DS": 10,
                "TOTAL_VOTES_GAVE_DC": 5
            }
        }

# Load model dari setiap file yang tersedia
try:
    DT_model = joblib.load('../model/DT_model.pkl')
    ET_model = joblib.load('../model/ET_model.pkl')
    SVM_model = joblib.load('../model/SVM_model.pkl')
    LR_model = joblib.load('../model/LR_model.pkl')  # Asumsi model logistic regression juga ada
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

# Fungsi untuk preprocessing data
def preprocess_data(data):
    # Convert IS_GLOGIN to object/string type for one-hot encoding
    data['IS_GLOGIN'] = data['IS_GLOGIN'].astype(str)
    
    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data)
    
    # Check if both gender columns exist, if not add missing columns
    required_gender_cols = ['GENDER_Female', 'GENDER_Male']
    for col in required_gender_cols:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    
    # Check if both IS_GLOGIN columns exist
    required_login_cols = ['IS_GLOGIN_False', 'IS_GLOGIN_True']
    for col in required_login_cols:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    
    # Apply min-max scaling
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_encoded)
    
    # Convert back to DataFrame with column names
    data_normalized_df = pd.DataFrame(data_normalized, columns=data_encoded.columns)
    
    return data_normalized_df

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded properly. Please check server logs.")
    
    # Ubah input menjadi format yang sesuai (pandas DataFrame)
    data = pd.DataFrame([user_input.dict()])
    
    # Preprocessing data
    processed_data = preprocess_data(data)
    
    # Prediksi dengan model
    try:
        # Menggunakan ensemble voting dari semua model yang tersedia
        dt_pred = DT_model.predict(processed_data)[0] if 'DT_model' in globals() else None
        et_pred = ET_model.predict(processed_data)[0] if 'ET_model' in globals() else None
        svm_pred = SVM_model.predict(processed_data)[0] if 'SVM_model' in globals() else None
        lr_pred = LR_model.predict(processed_data)[0] if 'LR_model' in globals() else None
        
        # Mengumpulkan semua prediksi yang tersedia
        predictions = [p for p in [dt_pred, et_pred, svm_pred, lr_pred] if p is not None]
        
        # Voting mayoritas
        if len(predictions) > 0:
            final_prediction = 1 if sum(predictions) / len(predictions) >= 0.5 else 0
        else:
            raise HTTPException(status_code=500, detail="No models available for prediction")
            
        # Mengambil probabilitas (jika tersedia)
        probabilities = {}
        try:
            if 'DT_model' in globals() and hasattr(DT_model, 'predict_proba'):
                probabilities['DT'] = float(DT_model.predict_proba(processed_data)[0][1])
            if 'ET_model' in globals() and hasattr(ET_model, 'predict_proba'):
                probabilities['ET'] = float(ET_model.predict_proba(processed_data)[0][1])
            if 'SVM_model' in globals() and hasattr(SVM_model, 'predict_proba'):
                probabilities['SVM'] = float(SVM_model.predict_proba(processed_data)[0][1])
            if 'LR_model' in globals() and hasattr(LR_model, 'predict_proba'):
                probabilities['LR'] = float(LR_model.predict_proba(processed_data)[0][1])
        except Exception as e:
            print(f"Error calculating probabilities: {e}")
    
        # Format output
        result = {
            "is_bot": bool(final_prediction),
            "prediction": int(final_prediction),
            "prediction_text": "Bot" if final_prediction == 1 else "Not Bot"
        }
        
        # Tambahkan probabilitas jika tersedia
        if probabilities:
            result["model_probabilities"] = probabilities
            if len(probabilities) > 0:
                result["avg_probability"] = sum(probabilities.values()) / len(probabilities)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Endpoint untuk info API
@app.get("/", summary="Mendapatkan informasi tentang API")
async def root():
    return {
        "message": "Kaggle Bot Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online",
        "models_loaded": models_loaded
    }

# Endpoint untuk health check
@app.get("/health", summary="Memeriksa status API")
async def health():
    return {"status": "healthy", "models_loaded": models_loaded}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)