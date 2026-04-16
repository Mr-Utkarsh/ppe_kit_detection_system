import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model load so it doesn't try to pull it from disk
# every time the user clicks a button or moves the slider.
@st.cache_resource
def load_yolo_model():
    return YOLO("model/yolov8n.pt")

def main():
    st.title("Object Detection System")
    st.markdown("Object detection using YOLOv8. Upload an image to analyze it.")
    
    model = load_yolo_model()
    
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
