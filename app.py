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
    # --- KOLOM TENGAH KIRI ---
    "TL-01": [375, 125, 60, 90],  # Mobil Silver/Putih Atas
    "TL-02": [375, 220, 60, 90],  # Mobil Hitam/Gelap
    "TL-03": [375, 315, 60, 90],  # Pickup Merah
    "TL-04": [375, 410, 60, 90],  # Mobil Abu-abu
    "TL-05": [375, 505, 60, 90],  # Mobil Merah
    
    # --- KOLOM TENGAH KANAN ---
    "TR-01": [470, 125, 60, 90],  # Mobil Hitam Atas
    "TR-02": [470, 220, 60, 90],  # Mobil Hitam
    "TR-03": [470, 315, 60, 90],  # Mobil Putih (Dekat orang lewat)
    "TR-04": [470, 410, 60, 90],  # Mobil Merah Marun
    "TR-05": [470, 505, 60, 90],  # Mobil Hitam
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
