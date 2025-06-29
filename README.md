# Face Recognition Web App

A simple web application for face recognition that can identify between Adams, Jeremy, and Samuel using a trained MobileNetV2 model.

## Features

- 📷 Camera integration for real-time photo capture
- 🎯 Face recognition using deep learning
- 📊 Confidence scores for all predictions
- 🎨 Modern, responsive UI design

## Requirements

- Python 3.8+
- Web browser with camera access
- The trained model.h5 file (already included)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and go to:
```
http://localhost:5000
```

## How to Use

1. Click **"Start Camera"** to activate your webcam
2. Position yourself in front of the camera
3. Click **"Capture Photo"** to take a picture
4. Click **"Predict"** to identify the person in the photo
5. View the results with confidence scores

## Model Information

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224x224 RGB images
- **Classes**: Adams, Jeremy, Samuel
- **Accuracy**: 100% on test set

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL

## Notes

- Ensure good lighting for better recognition accuracy
- The camera needs permission to access your webcam
- Works best with clear, front-facing photos 