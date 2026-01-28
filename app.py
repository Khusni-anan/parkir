import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import datetime

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Sistem Parkir ResNet50", layout="wide")

# KOORDINAT SLOT PARKIR (Dummy / Contoh)
# Tugas Kamu: Sesuaikan angka [x, y, w, h] ini dengan gambar CCTV aslimu nanti.
# Caranya: Buka gambar di Paint, arahkan mouse, catat koordinatnya.
PARKING_ROIS = {
    "A-01": [50, 360, 120, 100], 
    "A-02": [180, 360, 120, 100],
    "A-03": [310, 360, 120, 100],
    "B-01": [50, 500, 120, 100],
    "B-02": [180, 500, 120, 100],
    "B-03": [310, 500, 120, 100]
}

@st.cache_resource
def load_learner():
    # Load model ResNet50V2 High Accuracy
    try:
        model = tf.keras.models.load_model('model_parkir_resnet50.h5')
        return model
    except Exception as e:
        return None

model = load_learner()

# ==========================================
# 2. FUNGSI PRE-PROCESSING (Wajib sama dgn Training)
# ==========================================
def preprocess_image(roi_image):
    # 1. Resize ke 128x128 (Sesuai Training ResNet50 di Colab)
    img = cv2.resize(roi_image, (128, 128))
    # 2. Normalisasi (0-255 jadi 0-1)
    img = img / 255.0
    # 3. Tambah dimensi batch (jadi 1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================
# 3. USER INTERFACE (STREAMLIT)
# ==========================================
st.title("🚗 Smart Parking System (ResNet50)")
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .metric-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.write("Sistem deteksi ketersediaan slot parkir berbasis **Deep Learning (ResNet50V2)** dengan akurasi **99.6%**.")

col_cctv, col_info = st.columns([2, 1])

# --- KOLOM KIRI: CCTV & DETEKSI ---
with col_cctv:
    st.subheader("📡 Monitor CCTV")
    uploaded_file = st.file_uploader("Upload Gambar Simulasi CCTV", type=['jpg', 'png', 'jpeg'])
    
    empty_slots = [] # List slot kosong
    
    if uploaded_file is not None:
        # Baca file gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cek apakah model berhasil di-load
        if model is None:
            st.error("❌ File 'model_parkir_resnet50.h5' tidak ditemukan! Silakan upload ke folder proyek.")
        else:
            # LOOPING DETEKSI PER SLOT
            for slot_id, (x, y, w, h) in PARKING_ROIS.items():
                # Safety check: Pastikan koordinat tidak keluar batas gambar
                img_h, img_w, _ = frame.shape
                if y+h > img_h or x+w > img_w:
                    continue 

                # 1. Crop Gambar
                roi_img = frame[y:y+h, x:x+w]
                
                # 2. Prediksi
                if roi_img.size != 0:
                    processed_input = preprocess_image(roi_img)
                    prediction = model.predict(processed_input, verbose=0)[0][0]
                    
                    # Logic: 0 = Empty, 1 = Occupied (Sesuai urutan folder alphabet)
                    # Kita pakai threshold 0.5
                    is_occupied = prediction > 0.5
                    confidence = prediction if is_occupied else 1 - prediction
                    
                    # 3. Visualisasi (Hijau = Kosong, Merah = Isi)
                    if is_occupied:
                        color = (255, 0, 0) # Merah
                        label = f"ISI ({confidence:.0%})"
                        thickness = 2
                    else:
                        color = (0, 255, 0) # Hijau
                        label = "KOSONG"
                        thickness = 3
                        empty_slots.append(slot_id)
                    
                    # Gambar kotak
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, thickness)
                    # Gambar label background biar tulisan jelas
                    cv2.rectangle(frame_rgb, (x, y-20), (x+w, y), color, -1) 
                    cv2.putText(frame_rgb, f"{slot_id}", (x+5, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Tampilkan Gambar Final
            st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Real-time Detection")

# --- KOLOM KANAN: TIKET & STATUS ---
with col_info:
    st.subheader("📊 Status Parkir")
    
    total_slots = len(PARKING_ROIS)
    available_slots = len(empty_slots)
    
    # Tampilkan Metrik
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Total Slot", total_slots)
    col_stat2.metric("Tersedia", available_slots, delta_color="normal")
    
    st.divider()
    
    st.subheader("🖨️ Kiosk Tiket")
    st.write("Tekan tombol jika ada kendaraan masuk.")
    
    if st.button("CETAK TIKET MASUK", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ CCTV belum aktif!")
        elif available_slots == 0:
            st.error("⛔ MAAF PARKIRAN PENUH!")
        else:
            # Algoritma: Pilih slot kosong pertama
            assigned_slot = empty_slots[0]
            waktu_masuk = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            
            st.success(f"✅ Tiket Dicetak untuk Slot: {assigned_slot}")
            
            # Simulasi Struk Thermal Printer
            ticket_html = f"""
            <div style="
                background-color: #fff; 
                border: 1px dashed #000; 
                padding: 15px; 
                width: 100%; 
                font-family: 'Courier New', Courier, monospace;
                text-align: center;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                color: black;
            ">
                <h4 style="margin:0;">MALL SKRIPSI</h4>
                <p style="font-size: 12px; margin:5px;">Jl. Teknologi No. 1</p>
                <hr style="border-top: 1px dashed #000;">
                <div style="text-align: left; font-size: 14px;">
                    <p>Waktu : {waktu_masuk}</p>
                    <p>Gate  : UTAMA-01</p>
                </div>
                <hr style="border-top: 1px dashed #000;">
                <h2 style="font-size: 32px; margin: 10px 0;">{assigned_slot}</h2>
                <p style="font-size: 12px;">Silakan parkir di area tersebut</p>
                <hr style="border-top: 1px dashed #000;">
                <img src="https://bwipjs-api.metafloor.com/?bcid=code128&text={assigned_slot}&scale=2" alt="barcode" style="width:80%;">
                <p style="font-size: 10px; margin-top:5px;">Terima Kasih</p>
            </div>
            """
            st.markdown(ticket_html, unsafe_allow_html=True)
            
            # Di sini bisa tambahkan kode library printer thermal sungguhan jika ada alatnya
            # import escpos...
