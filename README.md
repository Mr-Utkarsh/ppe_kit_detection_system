# PPE & Safety Equipment Detection System

A safety monitoring application using YOLOv8 and Streamlit. It analyzes construction site images to identify and localize Personal Protective Equipment (PPE) like hardhats, safety vests, masks, and gloves.

## Features
- Fast object detection using custom-trained YOLOv8
- Clean web dashboard interface via Streamlit
- Drag and drop local image uploads for instant analysis

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/your-username/PPE_Detection_System.git
cd PPE_Detection_System
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App
To start the web application locally:
```bash
streamlit run app.py
```
The application will launch on `localhost:8501`. 
*Note: You must run the `src/train.ipynb` notebook to train the custom PPE detection model. Until then, the app will gracefully fall back to the generic YOLOv8 nano model and show a warning.*
# ppe_kit_detection_system
