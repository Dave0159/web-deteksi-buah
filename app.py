import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Buah YOLOv8", layout="centered")
st.title("Deteksi Buah (Apple / Orange / Banana)")

@st.cache_resource
def load_models():
    custom_model = YOLO("best.pt")
    coco_model = YOLO("yolov8n.pt")
    return custom_model, coco_model

model, coco = load_models()
COCO_IDS = [46, 47, 49]
COLOR_BGR = {"apple": (40, 40, 220), "orange": (0, 150, 255), "banana": (0, 200, 255)}

def detect_and_draw(img_bgr, conf=0.5):
    def unpack(res):
        return [(res.names[int(c)], float(s), tuple(int(v) for v in b))
                for b, s, c in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls)]

    dets = unpack(model.predict(img_bgr, conf=conf, verbose=False)[0])
    if not dets:
        dets = unpack(coco.predict(img_bgr, conf=conf, classes=COCO_IDS, verbose=False)[0])

    out = img_bgr.copy()
    for name, s, (x1, y1, x2, y2) in dets:
        col = COLOR_BGR.get(name, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{name} {s:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out, dets

conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
mode = st.radio("Pilih Metode Input:", ["Upload Gambar", "Kamera Real-time (Snapshot)"])

input_image = None
if mode == "Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded)
else:
    camera_photo = st.camera_input("Ambil foto buah")
    if camera_photo:
        input_image = Image.open(camera_photo)

if input_image is not None:
    img_rgb = np.array(input_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    result_bgr, detections = detect_and_draw(img_bgr, conf=conf_thresh)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    st.image(result_rgb, caption="Hasil Deteksi", use_container_width=True)
    if detections:
        for name, score, _ in detections:
            st.success(f"Terdeteksi: **{name}** (Confidence: {score:.2%})")
    else:
        st.warning("Tidak ada buah yang terdeteksi.")