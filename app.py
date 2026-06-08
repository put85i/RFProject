import os
import pickle
import numpy as np
import cv2
import urllib.request
from flask import Flask, render_template, request, redirect
from core.feature import extract_features  # Mengambil fungsi ekstraksi dari folder code

app = Flask(__name__)

# --- KONFIGURASI PATH UNTUK VERCEL (/tmp) ---
UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
MODEL_DIR = os.path.join('/tmp', 'model')
MODEL_NAME = 'random_forest_plant.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# LINK GOOGLE DRIVE YANG SUDAH JADI DIRECT DOWNLOAD LINK
MODEL_URL = "https://docs.google.com/uc?export=download&id=1j81BR2KqT3qEUKtEqIhNwyLj6Bq36dbP"
CLASS_LABELS = {0: 'Sehat (Healthy)', 1: 'Bercak Daun (Leaf Mold)', 2: 'Daun Terbakar (Early Blight)'}

def download_model_if_not_exists():
    """Fungsi untuk mengunduh model otomatis jika belum ada di server"""
    if not os.path.exists(MODEL_PATH):
        print("Model tidak ditemukan! Memulai unduhan otomatis dari cloud...")
        # Pastikan folder 'model/' sudah dibuat
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            # Proses mengunduh file dari URL ke MODEL_PATH
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Unduhan selesai! Model berhasil disimpan.")
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")
            raise e
    else:
        print("Model sudah tersedia, siap memuat ke sistem.")

# Jalankan fungsi unduhan sebelum melakukan pickle.load
download_model_if_not_exists()

# Memuat model Random Forest ke memori aplikasi
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Mapping angka prediksi kembali menjadi teks nama penyakit
CLASS_LABELS = {0: 'Sehat (Healthy)', 1: 'Bercak Daun (Leaf Mold)', 2: 'Daun Terbakar (Early Blight)'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek apakah pengguna sudah memilih file
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Simpan file yang diunggah ke folder static/uploads/
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # 2. Ekstrak fitur dari gambar yang diunggah menggunakan OpenCV
            features = extract_features(file_path)
            
            if features is not None:
                # 3. Lakukan prediksi menggunakan model Random Forest
                features = features.reshape(1, -1) # Ubah ke bentuk 2D array untuk model
                prediction_id = model.predict(features)[0]
                result = CLASS_LABELS[prediction_id]
                
                # Hitung probabilitas/kemiripan hasil prediksi
                probabilities = model.predict_proba(features)[0]
                confidence = round(probabilities[prediction_id] * 100, 2)
            else:
                result = "Gambar tidak dapat diproses oleh sistem."
                confidence = 0

            # Kirim hasil prediksi dan jalur foto kembali ke halaman web saat proses POST sukses
            return render_template('index.html', result=result, confidence=confidence, image_path=file_path)

    # PERBAIKAN UTAMA: Saat di-refresh (GET), paksa semua variabel hasil menjadi None.
    # Ini akan membuat tag {% if result %} di HTML bernilai False dan otomatis menghilangkan kotak hijau hasil analisis.
    return render_template('index.html', result=None, confidence=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)