# Backend - Kaggle Bot Detection API

API backend untuk aplikasi deteksi bot akun Kaggle.

## Struktur Direktori

```
backend/
  ├── __init__.py        # File inisialisasi package
  ├── main.py            # Aplikasi FastAPI utama
  ├── model.py           # Kelas pengelola model
  ├── test_main.py       # File pengujian
  └── requirements.txt   # Dependensi
```

## Spesifikasi

Backend ini dibuat menggunakan FastAPI dan terintegrasi dengan model machine learning untuk mendeteksi akun bot pada platform Kaggle. API menyediakan endpoint untuk melakukan prediksi berdasarkan data pengguna.

## Cara Menjalankan

1. Pastikan Python 3.8+ sudah terinstall
2. Install semua dependensi:

```bash
pip install -r requirements.txt
```

3. Pastikan model ML (`DT_model.pkl`, `ET_model.pkl`, `SVM_model.pkl`, dll) tersedia di direktori `../model/`.
4. Jalankan server:

```bash
python main.py
```

5. Server akan berjalan di `http://localhost:8000`
6. Akses dokumentasi API di `http://localhost:8000/docs`

## Endpoint

- `GET /`: Informasi tentang API
- `GET /health`: Memeriksa status API
- `POST /predict/`: Melakukan prediksi apakah user tergolong bot atau bukan

## Contoh Penggunaan

```python
import requests
import json

# Data pengguna
user_data = {
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

# Kirim request ke API
response = requests.post("http://localhost:8000/predict/", json=user_data)

# Tampilkan hasil
print(json.dumps(response.json(), indent=2))
```

## Pengujian

Untuk menjalankan tes:

```bash
pytest test_main.py -v
```

## Model yang Digunakan

Backend menggunakan model machine learning yang sudah dilatih sebelumnya:

1. **Decision Tree (DT_model.pkl)**
2. **Extra Trees (ET_model.pkl)**
3. **Support Vector Machine (SVM_model.pkl)**
4. **Logistic Regression (LR_model.pkl)** (jika tersedia)

Prediksi akhir dibuat berdasarkan ensemble voting dari semua model yang tersedia.
