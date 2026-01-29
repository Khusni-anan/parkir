import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import datetime
import os
import gdown

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Sistem Parkir ResNet50", layout="wide")

# ==========================================
# 2. KONFIGURASI SLOT PARKIR (DATABASE KOORDINAT)
# ==========================================
# Ini adalah Estimasi Koordinat untuk gambar 'carParkImg.jpg'
# Fokus pada area tengah (Vertical Parking)
PARKING_ROIS = {
    # --- KOLOM 1 (PALING KIRI) ---
    "A-01": [45, 103, 100, 32], # <-- Ini data dari kamu tadi
    "A-02": [45, 145, 100, 32],
    "A-03": [45, 187, 100, 32],
    "A-04": [45, 229, 100, 32],
    "A-05": [45, 271, 100, 32],
    "A-06": [45, 313, 100, 32],
    "A-07": [45, 355, 100, 32],
    "A-08": [45, 397, 100, 32],
    "A-09": [45, 439, 100, 32],
    "A-10": [45, 481, 100, 32],
    "A-11": [45, 523, 100, 32],
    "A-12": [45, 565, 100, 32],

    # --- KOLOM 2 (KIRI DALAM) ---
    "B-01": [158, 103, 100, 32],
    "B-02": [158, 145, 100, 32],
    "B-03": [158, 187, 100, 32],
    "B-04": [158, 229, 100, 32],
    "B-05": [158, 271, 100, 32],
    "B-06": [158, 313, 100, 32],
    "B-07": [158, 355, 100, 32],
    "B-08": [158, 397, 100, 32],
    "B-09": [158, 439, 100, 32],
    "B-10": [158, 481, 100, 32],
    "B-11": [158, 523, 100, 32],
    "B-12": [158, 565, 100, 32],

    # --- KOLOM 3 (TENGAH KIRI) ---
    "C-01": [365, 103, 100, 32],
    "C-02": [365, 145, 100, 32],
    "C-03": [365, 187, 100, 32],
    "C-04": [365, 229, 100, 32],
    "C-05": [365, 271, 100, 32],
    "C-06": [365, 313, 100, 32],
    "C-07": [365, 355, 100, 32],
    "C-08": [365, 397, 100, 32],
    "C-09": [365, 439, 100, 32],
    "C-10": [365, 481, 100, 32],
    "C-11": [365, 523, 100, 32],
    "C-12": [365, 565, 100, 32],

    # --- KOLOM 4 (TENGAH KANAN) ---
    "D-01": [478, 103, 100, 32],
    "D-02": [478, 145, 100, 32],
    "D-03": [478, 187, 100, 32],
    "D-04": [478, 229, 100, 32],
    "D-05": [478, 271, 100, 32],
    "D-06": [478, 313, 100, 32],
    "D-07": [478, 355, 100, 32],
    "D-08": [478, 397, 100, 32],
    "D-09": [478, 439, 100, 32],
    "D-10": [478, 481, 100, 32],
    "D-11": [478, 523, 100, 32],
    "D-12": [478, 565, 100, 32],

    # --- KOLOM 5 (KANAN DALAM) ---
    "E-01": [690, 103, 100, 32],
    "E-02": [690, 145, 100, 32],
    "E-03": [690, 187, 100, 32],
    "E-04": [690, 229, 100, 32],
    "E-05": [690, 271, 100, 32],
    "E-06": [690, 313, 100, 32],
    "E-07": [690, 355, 100, 32],
    "E-08": [690, 397, 100, 32],
    "E-09": [690, 439, 100, 32],
    "E-10": [690, 481, 100, 32],
    "E-11": [690, 523, 100, 32],
    "E-12": [690, 565, 100, 32],

    # --- KOLOM 6 (PALING KANAN) ---
    "F-01": [803, 103, 100, 32],
    "F-02": [803, 145, 100, 32],
    "F-03": [803, 187, 100, 32],
    "F-04": [803, 229, 100, 32],
    "F-05": [803, 271, 100, 32],
    "F-06": [803, 313, 100, 32],
    "F-07": [803, 355, 100, 32],
    "F-08": [803, 397, 100, 32],
    "F-09": [803, 439, 100, 32],
    "F-10": [803, 481, 100, 32],
    "F-11": [803, 523, 100, 32],
    "F-12": [803, 565, 100, 32],
}

# ==========================================
# 3. FUNGSI DOWNLOAD & LOAD MODEL
# ==========================================
def download_model_from_drive():
    file_id = '1zLQ3-BoCn-9PCzFC2c2cYnkyIEzafc9n'
    output = 'model_parkir_resnet50.h5'
    if not os.path.exists(output):
        st.warning("⚠️ Sedang mendownload model... (Mohon Tunggu)")
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output, quiet=False)
            st.success("✅ Download Selesai!")
        except Exception as e:
            st.error(f"Gagal download model: {e}")
            return None
    return output

@st.cache_resource
def load_learner():
    path = download_model_from_drive()
    if path: 
        try:
            return tf.keras.models.load_model(path)
        except:
            return None
    return None

model = load_learner()

# ==========================================
# 4. FUNGSI PRE-PROCESSING
# ==========================================
def preprocess_image(roi_image):
    img = cv2.resize(roi_image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================
# 5. USER INTERFACE
# ==========================================
st.title("🚗 Smart Parking System (ResNet50)")

# --- SIDEBAR: MODE SETUP ---
st.sidebar.header("🔧 Pengaturan")
st.sidebar.info("Gunakan mode ini jika kotak parkir meleset/tidak pas.")
setup_mode = st.sidebar.checkbox("Aktifkan Mode Setup Koordinat")

col_cctv, col_info = st.columns([2, 1])

# --- KOLOM KIRI: CCTV ---
with col_cctv:
    st.subheader("📡 Monitor CCTV")
    uploaded_file = st.file_uploader("Upload Gambar CCTV", type=['jpg', 'png', 'jpeg'])
    
    empty_slots = []
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- LOGIKA A: MODE SETUP (JIKA DIAKTIFKAN) ---
        if setup_mode:
            st.warning("⚠️ MODE SETUP AKTIF: Geser slider untuk mencari koordinat baru.")
            
            # Slider Pengatur Kotak
            col_x, col_y = st.columns(2)
            with col_x:
                x_val = st.slider("Posisi X (Kiri-Kanan)", 0, frame.shape[1], 375)
                w_val = st.slider("Lebar Kotak (Width)", 10, 200, 60)
            with col_y:
                y_val = st.slider("Posisi Y (Atas-Bawah)", 0, frame.shape[0], 125)
                h_val = st.slider("Tinggi Kotak (Height)", 10, 200, 90)
            
            # Gambar kotak kuning (preview)
            cv2.rectangle(frame_rgb, (x_val, y_val), (x_val+w_val, y_val+h_val), (0, 255, 255), 3)
            st.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Tampilkan Kodingan untuk di-Copy
            st.success("👇 Copy kode ini & ganti di bagian PARKING_ROIS:")
            st.code(f'"SLOT-BARU": [{x_val}, {y_val}, {w_val}, {h_val}],')

        # --- LOGIKA B: MODE NORMAL (DETEKSI) ---
        else:
            if model is not None:
                for slot_id, (x, y, w, h) in PARKING_ROIS.items():
                    # Safety check agar tidak error jika kotak keluar gambar
                    if y+h > frame.shape[0] or x+w > frame.shape[1]: continue

                    roi_img = frame[y:y+h, x:x+w]
                    if roi_img.size != 0:
                        processed_input = preprocess_image(roi_img)
                        prediction = model.predict(processed_input, verbose=0)[0][0]
                        
                        # Threshold 0.5
                        is_occupied = prediction > 0.5
                        confidence = prediction if is_occupied else 1 - prediction
                        
                        if is_occupied:
                            color = (255, 0, 0) # Merah
                            thickness = 2
                            label_text = ""
                        else:
                            color = (0, 255, 0) # Hijau
                            thickness = 3
                            empty_slots.append(slot_id)
                            label_text = "KOSONG"
                        
                        # Gambar Kotak & Label
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, thickness)
                        # Label Background
                        cv2.rectangle(frame_rgb, (x, y-15), (x+w, y), color, -1)
                        cv2.putText(frame_rgb, slot_id, (x+2, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Real-time Detection")

# --- KOLOM KANAN: STATUS ---
with col_info:
    # Hanya tampilkan info jika TIDAK dalam mode setup
    if not setup_mode:
        st.subheader("📊 Status Parkir")
        
        if uploaded_file is None:
            st.info("Upload gambar CCTV dulu.")
        else:
            total_slots = len(PARKING_ROIS)
            available_slots = len(empty_slots)
            
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Total Slot", total_slots)
            col_s2.metric("Tersedia", available_slots)
            
            st.divider()
            
            st.subheader("🖨️ Kiosk Tiket")
            if st.button("CETAK TIKET", type="primary", use_container_width=True):
                if available_slots > 0:
                    slot_pilihan = empty_slots[0]
                    waktu = datetime.datetime.now().strftime("%H:%M")
                    
                    st.success(f"✅ Tiket Dicetak: {slot_pilihan}")
                    st.markdown(f"""
                    <div style="border:1px dashed #000; padding:10px; text-align:center;">
                        <h3>TIKET PARKIR</h3>
                        <p>Slot: <b>{slot_pilihan}</b></p>
                        <p>Jam: {waktu}</p>
                        <img src="https://bwipjs-api.metafloor.com/?bcid=code128&text={slot_pilihan}&scale=2" width="70%">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("⛔ Parkiran Penuh!")
