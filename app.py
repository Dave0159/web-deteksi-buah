import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Buah YOLOv8", layout="centered")
st.title("Deteksi Buah & Objek YOLOv8")

st.sidebar.header("Pengaturan Model")

MODEL_OPTIONS = {
    "Model Terbaik (best.pt)": "best.pt",
    "Model Pembanding (last.pt / low.pt)": "last.pt",
    "Model Dasar YOLO (yolov8n.pt)": "yolov8n.pt",
    "Upload Model Lain (.pt)...": "custom_upload"
}

selected_label = st.sidebar.selectbox("Pilih Model yang Digunakan:", list(MODEL_OPTIONS.keys()))
selected_target = MODEL_OPTIONS[selected_label]

uploaded_model = None
if selected_target == "custom_upload":
    uploaded_model = st.sidebar.file_uploader("Upload file model (.pt)", type=["pt"])

@st.cache_resource
def load_local_model(path):
    return YOLO(path)

@st.cache_resource
def load_uploaded_model(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return YOLO(tmp_path)

active_model = None
if selected_target == "custom_upload":
    if uploaded_model is not None:
        active_model = load_uploaded_model(uploaded_model.getvalue())
        st.sidebar.success(f"Aktif: {uploaded_model.name}")
    else:
        st.sidebar.info("Silakan unggah file .pt di atas.")
else:
    # Cek apakah nama filenya last.pt atau low.pt
    if selected_target == "last.pt" and not os.path.exists("last.pt") and os.path.exists("low.pt"):
        selected_target = "low.pt"
        
    if os.path.exists(selected_target) or selected_target == "yolov8n.pt":
        active_model = load_local_model(selected_target)
        st.sidebar.success(f"Model Aktif: {selected_target}")
    else:
        st.sidebar.error(f"File '{selected_target}' tidak ditemukan di repository.")

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
filter_buah_coco = st.sidebar.checkbox("Filter khusus buah saja (untuk yolov8n.pt)", value=True)

COCO_FRUIT_IDS = [46, 47, 49]  # banana, apple, orange
COLOR_BGR = {
    "apple": (40, 40, 220),
    "orange": (0, 150, 255),
    "banana": (0, 200, 255)
}

def detect_and_draw(img_bgr, model, conf=0.5, is_coco=False):
    if is_coco and filter_buah_coco:
        results = model.predict(img_bgr, conf=conf, classes=COCO_FRUIT_IDS, verbose=False)[0]
    else:
        results = model.predict(img_bgr, conf=conf, verbose=False)[0]

    dets = [
        (results.names[int(c)], float(s), tuple(int(v) for v in b))
        for b, s, c in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)
    ]

    out = img_bgr.copy()
    for name, s, (x1, y1, x2, y2) in dets:
        col = COLOR_BGR.get(name.lower(), (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{name} {s:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out, dets

mode = st.radio("Pilih Metode Input:", ["Upload Gambar", "Kamera Real-time (Snapshot)"])

input_image = None
if mode == "Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded)
else:
    camera_photo = st.camera_input("Ambil foto")
    if camera_photo:
        input_image = Image.open(camera_photo)

if input_image is not None and active_model is not None:
    img_rgb = np.array(input_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    is_coco_model = (selected_target == "yolov8n.pt")
    result_bgr, detections = detect_and_draw(img_bgr, active_model, conf=conf_thresh, is_coco=is_coco_model)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    st.image(result_rgb, caption=f"Hasil Deteksi ({selected_target})", use_container_width=True)
    if detections:
        for name, score, _ in detections:
            st.success(f"Terdeteksi: **{name}** (Confidence: {score:.2%})")
    else:
        st.warning("Tidak ada objek yang terdeteksi.")
