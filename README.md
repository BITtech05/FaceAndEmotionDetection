# 🎭 Realtime Face + Emotion + Valence/Arousal App  

> **Realtime Human-Aware AI Interface** — Face Recognition, Emotion Detection, Valence/Arousal Visualization — all in a slick PyQt5 GUI.  

![Demo Banner](https://user-images.githubusercontent.com/your-demo-banner.png) <!-- Replace with actual demo GIF/screenshot -->

---

## ✨ Features
- 🔴 **Live Webcam Feed** — smooth, threaded video capture.  
- 😃 **Emotion Recognition** — powered by [DeepFace](https://github.com/serengil/deepface) (age, gender, emotions).  
- 🎭 **Face Mesh Landmarks** — fast, lightweight via [MediaPipe](https://developers.google.com/mediapipe).  
- 📊 **Realtime Valence/Arousal Plot** — continuous affective state visualization with **PyQtGraph**.  
- 📈 **Emotion Probability Bars** — see your feelings update live.  
- 🧑‍🤝‍🧑 **Face Recognition + DB** — known vs unknown detection with embedding storage.  
- 💾 **Save Unknowns** — one-click “Save Face” to grow your database.  
- ⚡ **Threaded & Optimized** — no UI freeze; heavy models throttled intelligently.  
- 🔧 **Upgrade-Ready** — modular, extendable for GPU acceleration, ONNX, FAISS, or custom VA models.  

---

## 🗺️ Project Map / Dataflow  

```
Webcam → CaptureThread → AnalyzerThread ───────▶ GUI
          |                       |               ├─ Video Feed + Overlays
          |                       ├─ Emotion Probs → Bar Graph
          |                       ├─ Valence/Arousal → Realtime Plot
          |                       └─ Face Embeddings → Recognition DB
          └─ Save Unknown Button → face_db.npz update
```

---

## 🚀 Quickstart

### 1. Install Requirements  
```bash
# optional: create virtual environment
python -m venv venv
source venv/bin/activate   # linux/mac
venv\Scripts\activate      # windows

# install dependencies
pip install --upgrade pip
pip install numpy opencv-python mediapipe pyqt5 pyqtgraph deepface scikit-learn imutils
```

⚡ **TensorFlow / PyTorch note:** DeepFace needs one backend. For CPU only:  
```bash
pip install tensorflow-cpu
```

For GPU acceleration, install `tensorflow` or `torch` with CUDA drivers configured.  

---

### 2. Run the App  
```bash
python realtime_face_emotion_app.py
```

---

## 🎥 What You Get
- **Main Window**  
  - Webcam feed with bounding boxes + mesh overlays  
  - Recognized face names (or “Unknown”)  
- **Right Panel**  
  - Emotion bar chart (probabilities per frame)  
  - Realtime valence/arousal plot  
- **Controls**  
  - ✅ `Save Unknown` — add new identities to DB  

---

## ⚡ Troubleshooting

- **Camera not found** → try changing index `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`  
- **DeepFace errors** → ensure internet (for first-time model download) and TensorFlow installed  
- **Mediapipe crash** → try `pip install opencv-python==4.7.0.72`  
- **Laggy UI** → increase `analyzer_throttle_sec` or run GPU-optimized models  

---

## 🚧 Upgrade Ideas
- 🎯 Add **tracking** (persistent IDs for smoother plots).  
- 🚀 Switch to **ONNX/TensorRT** for GPU speed.  
- 📚 Train your own **Valence/Arousal regressor** (AffectNet, Aff-Wild2).  
- ⚡ Integrate **FAISS/SVM** for large-scale recognition.  
- 🔒 Encrypt DB for privacy.  

---

## 🖼️ Screenshots  
| Emotion Bars | Valence/Arousal Plot | Face Mesh Overlay |
|--------------|----------------------|-------------------|
| ![bars](https://user-images.githubusercontent.com/emotion-bars.png) | ![plot](https://user-images.githubusercontent.com/valence-plot.png) | ![mesh](https://user-images.githubusercontent.com/facemesh.png) |

---

## 🛠️ Tech Stack
- [Python](https://www.python.org/)  
- [OpenCV](https://opencv.org/) (video capture & display)  
- [MediaPipe](https://developers.google.com/mediapipe) (landmarks)  
- [DeepFace](https://github.com/serengil/deepface) (age, gender, emotion, embeddings)  
- [PyQt5](https://pypi.org/project/PyQt5/) (GUI framework)  
- [PyQtGraph](http://www.pyqtgraph.org/) (real-time plots)  

---

## 🧑‍💻 Author
Made by [Bittesh](https://github.com/BITtech05)  
