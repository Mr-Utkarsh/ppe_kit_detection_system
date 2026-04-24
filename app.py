import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="PPE Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model():
    is_custom = os.path.exists("model/best.pt")
    return YOLO("model/best.pt" if is_custom else "model/yolov8n.pt"), is_custom

def main():
    st.title("PPE & Safety Equipment Detection System")
    st.markdown("Construction site safety monitoring using YOLOv8. Upload an image to analyze it.")
    
    model, is_custom = load_yolo_model()
    
    if not is_custom:
        st.warning("Custom weights not found. Using base YOLOv8n. Run train.ipynb to train the PPE model.")
    
    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Processing..."):
            results = model.predict(image, conf=0.25, verbose=False)
            
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.subheader("Results")
            st.image(annotated_img, use_container_width=True)

if __name__ == "__main__":
    main()
