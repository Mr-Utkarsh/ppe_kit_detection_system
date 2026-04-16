# Object Detection System

A simple object detection application using YOLOv8 and Streamlit. It analyzes static images to identify and localize objects within them.

## Features
- Fast object detection using YOLOv8 nano
- Clean web dashboard interface via Streamlit
- Drag and drop local image uploads for instant analysis

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/your-username/Object_Detection_System.git
cd Object_Detection_System
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
The application will launch on `localhost:8501`. On the first run, the `yolov8n.pt` model weights will be automatically loaded from the `model/` directory.
