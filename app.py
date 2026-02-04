import streamlit as st
import cv2
import pickle
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Smart Parking ResNet", layout="wide")
st.title("🐢 Smart Parking System (ResNet50 Engine)")

# ==========================================
# 2. LOAD MODEL RESNET & KOORDINAT
# ==========================================
@st.cache_resource
def load_resnet_model():
    # Pastikan file model .h5 Anda ada di sini
    model = load_model('model_parkir_resnet50.h5')
    return model

try:
    model = load_resnet_model()
    st.sidebar.success("Model ResNet50 Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .h5 ada. Error: {e}")
    st.stop()

# Load Posisi Parkir (Hasil dari picker.py)
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    st.error("File 'CarParkPos' tidak ditemukan! Jalankan picker.py dulu untuk menggambar kotak.")
    posList = []

# ==========================================
# 3. SIDEBAR CONFIG
# ==========================================
st.sidebar.header("Pengaturan")
width_slot = st.sidebar.number_input("Lebar Slot (px)", value=100)
height_slot = st.sidebar.number_input("Tinggi Slot (px)", value=100)

# ResNet butuh ukuran input spesifik (biasanya 224x224 untuk ResNet50 default)
# Tapi tergantung training Anda di parkir.ipynb pakai ukuran berapa?
# Jika Anda pakai default transfer learning, biasanya butuh resize.
img_size_model = st.sidebar.slider("Ukuran Input Model", 32, 256, 100) 

# Frame Skipping (PENTING untuk ResNet biar gak lag)
frame_skip = st.sidebar.slider("Kecepatan (Frame Skip)", 1, 30, 5)

uploaded_file = st.sidebar.file_uploader("Upload Video CCTV", type=['mp4', 'avi'])

# ==========================================
# 4. FUNGSI DETEKSI
# ==========================================
def check_parking_space(img, processed_img):
    space_counter = 0

    for pos in posList:
        x, y = pos

        # 1. CROP GAMBAR (Potong sesuai kotak)
        img_crop = processed_img[y:y+height_slot, x:x+width_slot]
        
        # 2. RESIZE (Sesuai input model ResNet)
        try:
            img_resize = cv2.resize(img_crop, (img_size_model, img_size_model))
            img_normalized = img_resize / 255.0  # Normalisasi 0-1
            img_reshape = np.reshape(img_normalized, (1, img_size_model, img_size_model, 3))

            # 3. PREDIKSI
            prediction = model.predict(img_reshape, verbose=0)
            
            # Asumsi Output Model: [Prob_Empty, Prob_Occupied]
            # Cek akurasi training Anda, mana index 0 mana index 1.
            # Di sini kita asumsi Index 0 = Empty, Index 1 = Occupied (Mobil)
            class_idx = np.argmax(prediction)
            
            # Jika kelas == 0 (Empty)
            if class_idx == 0: 
                color = (0, 255, 0) # Hijau
                thickness = 2
                space_counter += 1
                text_status = "Free"
            else:
                color = (0, 0, 255) # Merah
                thickness = 2
                text_status = "Car"

            # 4. GAMBAR HASIL DI FRAME ASLI
            cv2.rectangle(img, pos, (pos[0] + width_slot, pos[1] + height_slot), color, thickness)
            cv2.putText(img, text_status, (x, y + height_slot - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        except Exception as e:
            # Kadang error kalau kotak keluar batas video
            pass

    # Tampilkan Text Total
    cv2.rectangle(img, (0, 0), (250, 50), (0,0,0), -1)
    cv2.putText(img, f'Free: {space_counter}/{len(posList)}', (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

# ==========================================
# 5. LOOPING VIDEO STREAMLIT
# ==========================================
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    frame_count = 0
    
    # Placeholder untuk gambar terakhir agar tidak blank saat skip frame
    last_frame = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Optimasi: Hanya proses ResNet setiap N frame
        if frame_count % frame_skip == 0:
            # Proses Deteksi
            # ResNet biasanya dilatih dengan RGB, OpenCV baca BGR
            # Jadi convert dulu untuk input model (kalau training pake RGB)
            img_rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Gambar kotak di frame asli
            frame = check_parking_space(frame, img_rgb_input)
            last_frame = frame
        else:
            # Gunakan hasil gambar terakhir biar video tetap mulus
            if last_frame is not None:
                frame = last_frame

        # Tampilkan
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_display, channels="RGB", use_column_width=True)

    cap.release()
else:
    st.info("Silakan upload video parkir untuk memulai.")
