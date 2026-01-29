import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
import gdown
import time

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Sistem Parkir ResNet50 - Video Support", layout="wide")

# ==========================================
# 2. DATABASE KOORDINAT (Y Step = 50)
# ==========================================
PARKING_ROIS = {
    # --- KOLOM 1 (PALING KIRI - X=45) ---
    "A-01": [45, 103, 100, 32],
    "A-02": [45, 153, 100, 32],  # 103 + 50
    "A-03": [45, 203, 100, 32],
    "A-04": [45, 253, 100, 32],
    "A-05": [45, 303, 100, 32],
    "A-06": [45, 353, 100, 32],
    "A-07": [45, 403, 100, 32],
    "A-08": [45, 453, 100, 32],
    "A-09": [45, 503, 100, 32],
    "A-10": [45, 553, 100, 32],
    "A-11": [45, 603, 100, 32],
    "A-12": [45, 653, 100, 32],

    # --- KOLOM 2 (KIRI DALAM - X=158) ---
    "B-01": [158, 103, 100, 32],
    "B-02": [158, 153, 100, 32],
    "B-03": [158, 203, 100, 32],
    "B-04": [158, 253, 100, 32],
    "B-05": [158, 303, 100, 32],
    "B-06": [158, 353, 100, 32],
    "B-07": [158, 403, 100, 32],
    "B-08": [158, 453, 100, 32],
    "B-09": [158, 503, 100, 32],
    "B-10": [158, 553, 100, 32],
    "B-11": [158, 603, 100, 32],
    "B-12": [158, 653, 100, 32],

    # --- KOLOM 3 (TENGAH KIRI - X=398) ---
    "C-01": [398, 103, 100, 32],
    "C-02": [398, 153, 100, 32],
    "C-03": [398, 203, 100, 32],
    "C-04": [398, 253, 100, 32],
    "C-05": [398, 303, 100, 32],
    "C-06": [398, 353, 100, 32],
    "C-07": [398, 403, 100, 32],
    "C-08": [398, 453, 100, 32],
    "C-09": [398, 503, 100, 32],
    "C-10": [398, 553, 100, 32],
    "C-11": [398, 603, 100, 32],
    "C-12": [398, 653, 100, 32],

    # --- KOLOM 4 (TENGAH KANAN - X=511) ---
    "D-01": [511, 103, 100, 32],
    "D-02": [511, 153, 100, 32],
    "D-03": [511, 203, 100, 32],
    "D-04": [511, 253, 100, 32],
    "D-05": [511, 303, 100, 32],
    "D-06": [511, 353, 100, 32],
    "D-07": [511, 403, 100, 32],
    "D-08": [511, 453, 100, 32],
    "D-09": [511, 503, 100, 32],
    "D-10": [511, 553, 100, 32],
    "D-11": [511, 603, 100, 32],
    "D-12": [511, 653, 100, 32],

    # --- KOLOM 5 (KANAN DALAM - X=751) ---
    "E-01": [751, 103, 100, 32],
    "E-02": [751, 153, 100, 32],
    "E-03": [751, 203, 100, 32],
    "E-04": [751, 253, 100, 32],
    "E-05": [751, 303, 100, 32],
    "E-06": [751, 353, 100, 32],
    "E-07": [751, 403, 100, 32],
    "E-08": [751, 453, 100, 32],
    "E-09": [751, 503, 100, 32],
    "E-10": [751, 553, 100, 32],
    "E-11": [751, 603, 100, 32],
    "E-12": [751, 653, 100, 32],

    # --- KOLOM 6 (PALING KANAN - X=864) ---
    "F-01": [864, 103, 100, 32],
    "F-02": [864, 153, 100, 32],
    "F-03": [864, 203, 100, 32],
    "F-04": [864, 253, 100, 32],
    "F-05": [864, 303, 100, 32],
    "F-06": [864, 353, 100, 32],
    "F-07": [864, 403, 100, 32],
    "F-08": [864, 453, 100, 32],
    "F-09": [864, 503, 100, 32],
    "F-10": [864, 553, 100, 32],
    "F-11": [864, 603, 100, 32],
    "F-12": [864, 653, 100, 32],
}

# ==========================================
# 3. FUNGSI UTILITIES
# ==========================================
@st.cache_resource
def load_learner():
    # ID File Model (Pastikan ID ini benar file ResNet50 kamu)
    file_id = '1zLQ3-BoCn-9PCzFC2c2cYnkyIEzafc9n'
    output = 'model_parkir_resnet50.h5'
    
    if not os.path.exists(output):
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output, quiet=True)
        except:
            return None
            
    if os.path.exists(output):
        return tf.keras.models.load_model(output)
    return None

model = load_learner()

def preprocess_image(roi_image):
    img = cv2.resize(roi_image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Fungsi Inti Deteksi (Dipakai untuk Gambar & Video)
def process_frame(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    empty_slots = []
    
    if model is not None:
        for slot_id, (x, y, w, h) in PARKING_ROIS.items():
            if y+h > frame.shape[0] or x+w > frame.shape[1]: continue

            roi_img = frame[y:y+h, x:x+w]
            if roi_img.size != 0:
                # Prediksi
                processed_input = preprocess_image(roi_img)
                prediction = model.predict(processed_input, verbose=0)[0][0]
                
                is_occupied = prediction > 0.5
                
                # Visualisasi
                if is_occupied:
                    color = (255, 0, 0) # Merah
                else:
                    color = (0, 255, 0) # Hijau
                    empty_slots.append(slot_id)
                
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_rgb, slot_id, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
    return frame_rgb, empty_slots

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.title("📹 Smart Parking System (Video Support)")

# Pilihan Mode Input
mode = st.radio("Pilih Mode Input:", ["Gambar (Foto)", "Video (CCTV)"], horizontal=True)

col_tampil, col_status = st.columns([3, 1])

# --- LOGIKA MODE VIDEO ---
if mode == "Video (CCTV)":
    with col_tampil:
        uploaded_video = st.file_uploader("Upload Video CCTV (.mp4)", type=['mp4', 'avi'])
        
        # Placeholder untuk Video Player
        video_placeholder = st.empty()
        
        if uploaded_video is not None:
            # Simpan video ke file sementara biar bisa dibaca OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            st.info("Sedang memproses video... (Tekan 'Stop' di browser untuk berhenti)")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Video habis
                
                # Resize video biar gak kegedean & berat
                frame = cv2.resize(frame, (1000, 700)) 
                
                # PROSES DETEKSI PER FRAME
                result_frame, slots_kosong = process_frame(frame, model)
                
                # Tampilkan ke Layar
                video_placeholder.image(result_frame, channels="RGB", use_column_width=True)
                
                # Update Status di Kanan Real-time
                with col_status:
                    st.metric("Slot Kosong", len(slots_kosong))
                    # Tampilkan list slot kosong pertama
                    if len(slots_kosong) > 0:
                        st.success(f"Rekomen: {slots_kosong[0]}")
                    else:
                        st.error("PENUH")

            cap.release()

# --- LOGIKA MODE GAMBAR (Seperti Biasa) ---
elif mode == "Gambar (Foto)":
    with col_tampil:
        uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png'])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            result_frame, slots_kosong = process_frame(frame, model)
            st.image(result_frame, channels="RGB", use_column_width=True)
            
            with col_status:
                st.metric("Total Slot", len(PARKING_ROIS))
                st.metric("Tersedia", len(slots_kosong))
                if len(slots_kosong) > 0:
                    st.success(f"Silakan ke: {slots_kosong[0]}")
