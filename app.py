import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import gdown

st.set_page_config(
    page_title="PPE Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model():
    model_path = "model/best.pt"
    
    # If the model is not found locally, download it directly from Google Drive
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        file_id = "1grn2HPMzbjUg-bMzWt0WfdTy_mQn_a4g"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception:
            pass # Fallback below if download fails
    
    # Check if download succeeded or if it was already there
    if os.path.exists(model_path):
        return YOLO(model_path), True
    
    # Absolute Fallback to the generic pre-trained model if no internet / failed
    return YOLO("model/yolov8n.pt"), False

def main():
    st.title("PPE & Safety Equipment Detection System")
    st.markdown("Construction site safety monitoring using YOLOv8. Upload an image to analyze it.")
    
    model, is_custom = load_yolo_model()
    
    if not is_custom:
        st.warning("⚠️ **Custom model not found.** Using generic YOLOv8 nano (detects general objects). Please train the custom model using `src/train.ipynb` to detect PPE.")
    
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Adjust the minimum confidence score required for detections."
    )
    
    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Processing..."):
            results = model.predict(image, conf=confidence_threshold, verbose=False)
            
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.subheader("Results")
            st.image(annotated_img, use_container_width=True)

if __name__ == "__main__":
    main()
