import streamlit as st
import cv2
import numpy as np
import pickle
import os
import tensorflow as tf
import gdown
import tempfile

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Smart Parking AI", layout="wide")

st.title("🚗 Smart Parking System (ResNet Engine)")
st.markdown("Sistem deteksi parkir berbasis Deep Learning (ResNet50) dengan integrasi Cloud Storage.")

# ==========================================
# 2. KONFIGURASI GOOGLE DRIVE (EDIT DISINI!)
# ==========================================
# Ganti ID di bawah ini dengan ID File .h5 dari Link Share Google Drive Anda
# Contoh Link: https://drive.google.com/file/d/1A-BcDeFg.../view
# ID-nya adalah: 1A-BcDeFg...
GDRIVE_FILE_ID = '1zLQ3-BoCn-9PCzFC2c2cYnkyIEzafc9n' 

MODEL_FILENAME = 'model_parkir_resnet50.h5'
POS_FILENAME = 'CarParkPos'

# ==========================================
# 3. FUNGSI DOWNLOAD & LOAD MODEL
# ==========================================
@st.cache_resource
def load_model_and_mapping():
    # A. Cek & Download Model
    if not os.path.exists(MODEL_FILENAME):
        if GDRIVE_FILE_ID == 'MASUKKAN_ID_GOOGLE_DRIVE_DISINI':
            st.error("⚠️ ID Google Drive belum diisi di kodingan app.py!")
            st.stop()
            
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        st.info(f"sedang mendownload model '{MODEL_FILENAME}' dari Google Drive... Harap tunggu.")
        try:
            gdown.download(url, MODEL_FILENAME, quiet=False)
            st.success("Download Model Selesai!")
        except Exception as e:
            st.error(f"Gagal download model: {e}")
            st.stop()

    # B. Load Model ke Memory
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
    except Exception as e:
        st.error(f"File model rusak atau tidak kompatibel: {e}")
        st.stop()

    # C. Load Mapping Posisi
    posList = []
    if os.path.exists(POS_FILENAME):
        with open(POS_FILENAME, 'rb') as f:
            posList = pickle.load(f)
    else:
        st.warning(f"⚠️ File '{POS_FILENAME}' tidak ditemukan. Upload file mapping ke GitHub!")
    
    return model, posList

# Load Data (Cached)
model, posList = load_model_and_mapping()

if len(posList) > 0:
    st.sidebar.success(f"✅ Mapping Terbaca: {len(posList)} Slot")
else:
    st.sidebar.error("❌ Mapping Kosong/Tidak Ditemukan")

# ==========================================
# 4. SIDEBAR PENGATURAN
# ==========================================
st.sidebar.header("🔧 Pengaturan Deteksi")

# Slider Ukuran Slot (Bisa disesuaikan jika kotak kurang pas)
slot_width = st.sidebar.number_input("Lebar Slot (px)", value=100)
slot_height = st.sidebar.number_input("Tinggi Slot (px)", value=100)

# Input Size Model (Sesuaikan dengan training Anda, biasanya 48, 64, 100, atau 224)
img_size_input = st.sidebar.selectbox("Ukuran Input Model (Resize)", [100, 224, 64, 48], index=0)

# Frame Skip (Biar gak lag)
frame_skip = st.sidebar.slider("Kecepatan (Frame Skip)", 1, 30, 10, help="Semakin tinggi semakin cepat, tapi patah-patah.")

# Toggle Invert Class (Jaga-jaga kalau Merah/Hijau terbalik)
invert_class = st.sidebar.checkbox("Tukar Warna (Invert Class)", value=False)

uploaded_file = st.sidebar.file_uploader("📂 Upload Video CCTV", type=['mp4', 'avi', 'mov'])

# ==========================================
# 5. FUNGSI UTAMA (MAIN LOOP)
# ==========================================
def process_video():
    if uploaded_file is None:
        st.info("👈 Silakan upload video parkir di menu sebelah kiri.")
        return

    # Simpan video ke temp file biar bisa dibaca OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    st_frame = st.empty() # Placeholder gambar
    st_text = st.empty()  # Placeholder teks statistik
    
    frame_count = 0
    # Placeholder frame terakhir untuk efek smooth saat skipping
    last_processed_frame = None 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Video habis, loop ulang dari awal (optional)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Optimasi: Hanya proses ResNet setiap N frame
        if frame_count % frame_skip != 0:
            if last_processed_frame is not None:
                st_frame.image(last_processed_frame, channels="RGB", use_column_width=True)
            continue

        # Proses Frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_clean = frame.copy() # Gambar bersih untuk crop

        free_spots = 0
        total_spots = len(posList)

        for pos in posList:
            x, y = pos
            
            # 1. Validasi Koordinat (Biar gak error out of bounds)
            h_img, w_img, _ = frame.shape
            if x + slot_width > w_img or y + slot_height > h_img:
                continue 

            # 2. Crop Gambar Slot
            img_crop = img_clean[y:y+slot_height, x:x+slot_width]
            
            # 3. Pre-process untuk ResNet
            try:
                # Resize sesuai input model
                img_resized = cv2.resize(img_crop, (img_size_input, img_size_input))
                img_norm = img_resized / 255.0 # Normalisasi 0-1
                img_reshape = np.reshape(img_norm, (1, img_size_input, img_size_input, 3))

                # 4. Prediksi
                prediction = model.predict(img_reshape, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction)

                # Logika Penentuan Kelas
                # Default: 0 = Empty (Hijau), 1 = Occupied (Merah)
                if invert_class:
                    is_empty = (class_idx == 1)
                else:
                    is_empty = (class_idx == 0)

                if is_empty:
                    color = (0, 255, 0) # Hijau (RGB untuk Streamlit nanti di-convert)
                    cv_color = (0, 255, 0) # BGR untuk OpenCV
                    text_label = "Free"
                    free_spots += 1
                else:
                    color = (255, 0, 0) # Merah
                    cv_color = (0, 0, 255)
                    text_label = "Car"

                # 5. Gambar Visualisasi
                cv2.rectangle(frame, (x, y), (x + slot_width, y + slot_height), cv_color, 2)
                
                # Tambah background teks biar terbaca
                cv2.rectangle(frame, (x, y + slot_height - 20), (x + slot_width, y + slot_height), cv_color, -1)
                cv2.putText(frame, f"{text_label}", (x + 5, y + slot_height - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            except Exception as e:
                pass # Skip error crop

        # Update Info Statistik
        st_text.markdown(f"### 🅿️ Status Parkir: **{free_spots}** Kosong / **{total_spots}** Total")
        
        # Tampilkan Video
        # Convert balik ke RGB untuk Streamlit
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_processed_frame = frame_display
        st_frame.image(frame_display, channels="RGB", use_column_width=True)

    cap.release()

# Jalankan App
process_video()
