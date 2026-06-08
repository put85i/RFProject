import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from core.feature import extract_features

def train_model_process():
    # 1. Tentukan path lokasi dataset dan inisialisasi list data
    # Sesuaikan path jika letak folder dataset_raw berbeda
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_raw'))
    
    # Mapping nama folder menjadi label angka untuk keperluan Machine Learning
    categories = {'Sehat': 0, 'Bercak_Daun': 1, 'Daun_Terbakar': 2}
    
    X = [] # List untuk menyimpan vektor fitur (angka hasil OpenCV)
    y = [] # List untuk menyimpan target label (0, 1, atau 2)

    print("=== Memulai Proses Ekstraksi Fitur Gambar ===")
    
    # 2. Perulangan untuk membaca setiap folder kategori
    for category, label in categories.items():
        folder_path = os.path.join(base_dir, category)
        
        # Cek apakah folder tersebut ada di laptopmu
        if not os.path.exists(folder_path):
            print(f"Peringatan: Folder {category} tidak ditemukan di {folder_path}")
            continue

        print(f"Sedang memproses kategori: {category}...")
        
        # Ambil semua file gambar di dalam folder tersebut
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Pastikan file yang dibaca adalah gambar (.jpg, .jpeg, .png)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Panggil fungsi extract_features dari file feature_extractor.py
                features = extract_features(img_path)
                
                # Jika gambar berhasil diproses, masukkan ke dalam dataset array
                if features is not None:
                    X.append(features)
                    y.append(label)

    # Ubah list menjadi numpy array agar bisa diproses Scikit-Learn
    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("Error: Tidak ada data gambar yang berhasil diekstrak. Periksa kembali folder dataset_raw kamu.")
        return

    print(f"\nTotal data gambar yang berhasil diekstrak: {len(X)}")

    # 3. Membagi data menjadi Data Training (80%) dan Data Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Mencoba melatih algoritma Random Forest...")
    
    # 4. Inisialisasi dan pelatihan model Random Forest Classifier
    # n_estimators=100 artinya kita membangun 100 pohon keputusan (Decision Trees)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluasi akurasi model menggunakan Data Testing
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=== HASIL EVALUASI MODEL ===")
    print(f"Akurasi Model: {accuracy * 100:.2f}%")
    print("\nLaporan Klasifikasi Lengkap:")
    print(classification_report(y_test, y_pred, target_names=list(categories.keys())))

    # 6. Menyimpan model hasil training ke folder 'model/'
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(model_dir, exist_ok=True) # Buat folder model jika belum ada
    
    model_path = os.path.join(model_dir, 'random_forest_plant.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Sukses! Model AI telah disimpan di: {model_path}")

if __name__ == '__main__':
    train_model_process()