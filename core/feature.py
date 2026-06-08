import cv2
import numpy as np

def extract_features(image_path, size=(128, 128)):
    # 1. Membaca gambar menggunakan OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 2. Resize gambar agar ukuran seragam
    img_resized = cv2.resize(img, size)
    
    # --- A. EKSTRAKSI FITUR WARNA ---
    # Perbaikan: Langsung gunakan cv2.COLOR_BGR2HSV (ini adalah integer)
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Hitung rata-rata nilai Hue, Saturation, dan Value
    mean_color = cv2.mean(hsv_img)[:3] 
    
    # --- B. EKSTRAKSI FITUR TEKSTUR ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges == 255) / (size[0] * size[1])
    
    # --- C. MENGGABUNGKAN FITUR ---
    feature_vector = np.array([mean_color[0], mean_color[1], mean_color[2], edge_density])
    
    return feature_vector