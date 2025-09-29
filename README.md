# Number Plate Detection and Recognition Model 🚘🔍

This project implements an Automatic Number Plate Recognition (ANPR) system using OpenCV, TensorFlow/Keras, and deep learning.
It detects license plates in images, segments individual characters, and recognizes them using a trained Convolutional Neural Network (CNN).

## ✨ Features
---
License Plate Detection using Haar Cascade Classifier (indian_license_plate.xml).

Region of Interest (ROI) Extraction with bounding box expansion for better accuracy.

Character Segmentation using contour detection and Connected Components Analysis.

CNN-based Character Recognition trained on 36 classes (0-9, A-Z).

Custom F1-Score Metric implementation for evaluation.

Early Stopping Callback to stop training once model achieves high accuracy.

Visualizations of each step: detection, segmentation, and final recognition.

---

## 🏗️ Project Workflow

---
Preprocessing

Extract license plate region using Haar Cascade.

Convert to grayscale, thresholding, and binarization.

Character Segmentation

Contour and connected components filtering.

Resize and normalize segmented characters.

---
Model Training

CNN trained on dataset with augmentation (ImageDataGenerator).

Classes: 0–9, A–Z.

Metrics: Loss + Micro F1-Score.

---
Prediction

Segmented characters are passed through trained CNN.

Characters are concatenated to form the license plate string.

---
Visualization

Draw bounding box around detected plate.

Overlay recognized plate number on the image.
```
bash
📂 Repository Structure
├── data/                     # Dataset & Haar cascade XML
├── models/                   # Saved trained models (optional)
├── notebooks/                # Jupyter notebooks / experiments
├── src/
│   ├── detect_plate.py       # Plate detection logic
│   ├── segment_characters.py # Character segmentation
│   ├── train_model.py        # Model training
│   ├── recognize.py          # Recognition pipeline
│   └── utils.py              # Helper functions
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```
⚙️ Installation

Clone the repository:
```
bash
git clone https://github.com/yourusername/NumberPlateDetectionModel.git
```
cd NumberPlateDetectionModel

---
Create & activate virtual environment:

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

---
Install dependencies:

pip install -r requirements.txt

▶️ Usage
Detect and Recognize Number Plate from Image
from src.detect_plate import detect_plate
from src.segment_characters import segment_characters
from src.recognize import show_results
import cv2

# Load image
img = cv2.imread("data/car.jpg")

# Detect plate
output_img, plate = detect_plate(img)

# Segment characters
chars = segment_characters(plate)

# Recognize plate number
plate_number = show_results(chars)
print("Detected Plate:", plate_number)

## 📊 Results
---
Examples from test images:

Input Image	Detected Plate	Predicted Number

	✅	0L8CAF5030

	✅	KA29Z999

	✅	MH20EE7602

	✅	HR26DK8337
## 🔮 Future Improvements

---
Replace Haar Cascade with YOLO/SSD/RetinaNet for robust detection.

Expand dataset for more diverse plate styles & fonts.

Deploy as a Flask/FastAPI web app or mobile app.

Real-time detection from video streams.

## 📦 Requirements
---
Python 3.8+

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

scikit-learn

Install via:

pip install -r requirements.txt

---
📝 Author

Mriduparna Bania

