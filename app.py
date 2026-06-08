import os
import pickle
import numpy as np
import cv2
import urllib.request
from flask import Flask, render_template, request, redirect

# PENGAMAN IMPORT MODUL FITUR
try:
    from core.feature import extract_features
except ImportError:
    from code.feature import extract_features

app = Flask(__name__)

# --- KONFIGURASI PATH UNTUK VERCEL (/tmp) ---
UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
MODEL_DIR = os.path.join('/tmp', 'model')
MODEL_NAME = 'random_forest_plant.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MODEL_URL = "https://www.dropbox.com/scl/fi/9qs24p7j8jb0yqc98f6ck/random_forest_plant.pkl?rlkey=h89j6kyxxf5dom124jv5rdpiu&st=ckxuvlg8&dl=1"

CLASS_LABELS = {0: 'Sehat (Healthy)', 1: 'Bercak Daun (Leaf Mold)', 2: 'Daun Terbakar (Early Blight)'}

def download_model_if_not_exists():
    """Fungsi download hanya berjalan di dalam runtime folder /tmp"""
    if not os.path.exists(MODEL_PATH):
        print("Model tidak ditemukan di /tmp! Memulai unduhan otomatis dari cloud...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Unduhan selesai! Model berhasil disimpan di /tmp.")
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")
            raise e

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # 1. Folder upload baru dibuat saat user menekan tombol submit POST
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # 2. Proses download & load model dijalankan AMAN di dalam fase runtime POST
            download_model_if_not_exists()
            
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            
            # 3. Ekstraksi fitur dan prediksi model
            features = extract_features(file_path)
            
            if features is not None:
                features = features.reshape(1, -1)
                prediction_id = model.predict(features)[0]
                result = CLASS_LABELS[prediction_id]
                
                probabilities = model.predict_proba(features)[0]
                confidence = round(probabilities[prediction_id] * 100, 2)
            else:
                result = "Gambar tidak dapat diproses oleh sistem."
                confidence = 0

            return render_template('index.html', result=result, confidence=confidence, image_path=file_path)

    # Tampilan bersih saat pertama kali membuka web (GET)
    return render_template('index.html', result=None, confidence=None, image_path=None)

# Membantu Vercel mengenali instansiasi aplikasi WSGI dengan benar
app = app

if __name__ == '__main__':
    app.run(debug=True)