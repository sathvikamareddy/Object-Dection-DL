<div align="center">
  
# 🧠 Object Detection Using Deep Learning (YOLOv8 + Tkinter)

</div>

## 📌 Project Overview

This project is a **GUI-based Object Detection System** built using **YOLOv8**, **OpenCV**, and **Tkinter**.
It allows users to detect objects in:

* 📷 Uploaded images
* 🎥 Live webcam feed

Detected objects are displayed with **bounding boxes and confidence scores**.

---

## 🚀 Features

* Upload and detect objects in images
* Real-time object detection using webcam
* Displays detected object names
* Simple and user-friendly GUI
* Lightweight YOLOv8 model (`yolov8n.pt`)

---

## 🛠️ Technologies Used

* Python
* Tkinter (GUI)
* OpenCV
* PIL (Pillow)
* Ultralytics YOLOv8
* Threading

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sathvikamareddy/object-detection-yolo.git
cd object-detection-yolo
```

### 2️⃣ Install Dependencies

```bash
pip install ultralytics opencv-python pillow
```

---

## 📁 Download YOLO Model

Download the YOLOv8 model file:

```bash
yolov8n.pt
```

👉 It will automatically download when you run the code (if not present).

---

## ▶️ How to Run

```bash
python app.py
```

---

## 🎯 How It Works

1. Load YOLOv8 model
2. User selects:

   * Upload image OR
   * Start webcam
3. Model detects objects
4. Displays:

   * Bounding boxes
   * Object names
   * Confidence scores

---

## 📸 Output

* Detected objects are highlighted in **green boxes**
* Names and confidence scores are shown on screen

---

## ⚠️ Requirements

* Python 3.8+
* Webcam (for live detection)

---

## 🔧 Future Improvements

* Add video file detection
* Save detected results
* Improve UI design
* Add custom model support


