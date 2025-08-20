"""
Realtime Face + Emotion + Valence/Arousal App
Option B — Full workable PyQt5 + PyQtGraph application

CONTENTS OF THIS FILE
1) Project map / Outline (what this app does, components, dataflow)
2) Installation notes (pip commands)
3) Explanations of upgrade points and possible error points (and how to fix them)
4) The full runnable Python code (single-file) with detailed comments and TODOs

USAGE SUMMARY
- Launch this script locally on a machine with a webcam.
- It creates a GUI showing live video, an emotion-bar, a valence/arousal realtime plot, and a "Save unknown" button.
- It uses MediaPipe Face Mesh (fast) for landmarks and DeepFace for age/gender/emotion + embeddings (recognition).

IMPORTANT NOTES BEFORE RUNNING
- DeepFace requires a deep-learning backend (TensorFlow or PyTorch). On CPU-only machines install `tensorflow` or `tensorflow-cpu`.
- MediaPipe may require a specific opencv version; if you get issues, try `pip install opencv-python==4.7.0.72` (or the latest compatible release).

INSTALL (recommended)
# create venv (optional but recommended)
# python -m venv venv && source venv/bin/activate  # linux/mac
# venv\Scripts\activate  # windows
pip install --upgrade pip
pip install numpy opencv-python mediapipe pyqt5 pyqtgraph deepface scikit-learn imutils
# If you have GPU and want speedups: install tensorflow with GPU support separately (and correct CUDA drivers)

PROJECT MAP / OUTLINE (precise work plan for Option B)
1. Video capture thread
   - Capture frames continuously from webcam using OpenCV
   - Emit raw frames to the UI (fast, low-latency)
2. Analyzer thread
   - Receives latest frames (no backlog) and runs lightweight MediaPipe Face Mesh every frame or every Nth frame
   - Computes face bounding boxes from landmarks and simple AU-like metrics (eye openness, mouth opening)
   - Throttles heavy DeepFace calls (emotion/age/gender + embedding) to run at lower frequency (configurable)
   - Emits analysis results (per-face: bbox, landmarks, emotion_probs, age, gender, embedding) to main UI
3. Main UI (PyQt)
   - Displays video frames and overlays (bbox, name, face mesh)
   - Shows realtime valence/arousal line plot (PyQtGraph) updated from analyzer
   - Shows realtime emotion probability bar graph
   - "Save Unknown" button: captures recent crop(s) from selected unknown face and adds to local DB
4. Face DB
   - A small local database: face_db.npz (names list + embeddings list)
   - Compare embeddings with cosine similarity to detect known/unknown
   - "Save" flow stores several aligned crops and adds an averaged embedding to the DB
5. Smoothing & postprocessing
   - EMA smoothing for valence/arousal and emotion probabilities to reduce jitter
   - Simple throttling for heavy models

ERROR / TROUBLESHOOTING POINTS (common) + FIXES
- Camera fails to open: check device index (0, 1, ...) or permissions; test with `opencv` sample script
- DeepFace import or model download fails: ensure you have internet for first run (DeepFace downloads models), and install tensorflow
- Mediapipe runtime error: try pip installing a matching opencv version
- UI freezes: ensure heavy work is offloaded to threads (this app does that). If UI still hangs, increase throttling for DeepFace (analyzer_throttle_sec)
- Incorrect detection / many false positives: tune face embedding threshold `RECOG_THRESHOLD` and increase samples when saving

UPGRADE SUGGESTIONS (where to invest later)
- Use persistent face IDs (implement simple tracking with centroid matching) to keep per-person VA history
- Replace DeepFace calls with ONNX / TensorRT optimized models to run very fast on GPU
- Train a proper Valence/Arousal regressor on AffectNet/Aff-Wild2 for continuous VA instead of mapping from categories
- Use a small local classifier (SVM/FAISS) for faster retrieval at scale
- Add secure encryption for stored images/embeddings if privacy required

------------------------------------------------------------------
# BEGIN CODE
------------------------------------------------------------------

import sys
import os
import time
import math
import json
import threading
from collections import deque

import numpy as np
import cv2

# PyQt5 imports
from PyQt5 import QtWidgets, QtCore, QtGui

# pyqtgraph for realtime plotting
import pyqtgraph as pg

# MediaPipe face mesh
import mediapipe as mp

# DeepFace for emotion/age/gender/embedding
from deepface import DeepFace

# sklearn for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# small helper for image alignment/cropping
import imutils

# ------------------ CONFIG ------------------
# Tune these settings according to your hardware
VIDEO_SRC = 0  # webcam index (0 default). Change to file path or RTSP if needed
MAX_FACES = 2  # how many faces to process/track concurrently in analyzer
ANALYZER_FPS = 6  # approximate frequency at which heavy analysis runs (frames/sec)
ANALYZE_EVERY_N_FRAMES = max(1, int(30 / ANALYZER_FPS))
DEEPFACE_MODEL_NAME = 'ArcFace'  # model used for embeddings by DeepFace
RECOG_THRESHOLD = 0.45  # cosine similarity threshold for recognition (0..1). Higher -> stricter
DB_PATH = 'face_db.npz'  # file used to store names + embeddings
SAVE_DIR = 'saved_faces'  # where to dump captured face crops when "Save" clicked
VA_SMOOTH_ALPHA = 0.4  # EMA alpha for valence/arousal smoothing
EMO_SMOOTH_ALPHA = 0.5  # EMA alpha for emotion vector smoothing
PLOT_LENGTH = 300  # number of points to show in VA plot

# small mapping from categorical emotion -> (valence, arousal)
# This is an approximate circumplex mapping. For production, train a regressor.
EMOTION_VA = {
    'happy': (0.9, 0.6),
    'surprise': (0.2, 0.8),
    'neutral': (0.0, 0.0),
    'sad': (-0.7, -0.2),
    'angry': (-0.9, 0.7),
    'fear': (-0.8, 0.8),
    'disgust': (-0.6, 0.4)
}

# ------------------ UTILITIES ------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_db(path=DB_PATH):
    """Load names and embeddings from .npz file. Returns (names_list, embeddings_list).
    Embeddings are numpy arrays (not normalized necessarily)."""
    if not os.path.exists(path):
        return [], []
    try:
        npz = np.load(path, allow_pickle=True)
        names = list(npz['names'].tolist()) if 'names' in npz else []
        embs = list(npz['embeddings'].tolist()) if 'embeddings' in npz else []
        # convert embeddings to np arrays
        embs = [np.array(e) for e in embs]
        return names, embs
    except Exception as e:
        print("Failed to load DB:", e)
        return [], []


def save_db(names, embeddings, path=DB_PATH):
    try:
        # convert to object arrays for saving
        np.savez_compressed(path, names=np.array(names, dtype=object), embeddings=np.array(embeddings, dtype=object))
    except Exception as e:
        print("Failed to save DB:", e)


def l2_norm(x):
    x = np.array(x, dtype=float)
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def compute_embedding(face_img):
    """Compute a normalized embedding vector for a face crop using DeepFace.
    Returns None on error.
    face_img: BGR numpy array (as from OpenCV) or RGB — we pass BGR; DeepFace will handle typical arrays.
    """
    try:
        # DeepFace.represent can accept numpy arrays. Enforce detection False so it doesn't fail on imperfect crops.
        rep = DeepFace.represent(img_path=face_img, model_name=DEEPFACE_MODEL_NAME, enforce_detection=False)
        # DeepFace.represent output format varies by version: handle common cases
        emb = None
        if isinstance(rep, list):
            # sometimes list of lists or list of dicts
            if len(rep) == 0:
                return None
            first = rep[0]
            if isinstance(first, dict) and 'embedding' in first:
                emb = np.array(first['embedding'])
            elif isinstance(first, (list, np.ndarray)):
                emb = np.array(first)
            else:
                emb = np.array(rep)
        elif isinstance(rep, dict) and 'embedding' in rep:
            emb = np.array(rep['embedding'])
        else:
            emb = np.array(rep)

        if emb is None:
            return None
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb
    except Exception as e:
        # DeepFace might print a lot; we catch failures gracefully
        print("compute_embedding error:", type(e), e)
        return None


def analyze_face(face_img):
    """Run DeepFace.analyze on a face crop for emotion/age/gender.
    Returns a dict or None on error.
    """
    try:
        res = DeepFace.analyze(img_path=face_img, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        return res
    except Exception as e:
        print("analyze_face error:", type(e), e)
        return None


def emotion_probs_to_va(probs):
    """Map emotion probabilities dict -> weighted valence/arousal using EMOTION_VA table"""
    v_sum = 0.0
    a_sum = 0.0
    s = 0.0
    for emo, p in probs.items():
        if emo in EMOTION_VA:
            v, a = EMOTION_VA[emo]
            v_sum += v * p
            a_sum += a * p
            s += p
    if s <= 0:
        return 0.0, 0.0
    return v_sum / s, a_sum / s


def find_best_match(embedding, names, embeddings, threshold=RECOG_THRESHOLD):
    """Return (name, score) if best match >= threshold else (None, score).
    If embeddings empty, return (None, 0.0)
    """
    if embedding is None or len(embeddings) == 0:
        return None, 0.0
    try:
        embs = np.vstack(embeddings)
        # ensure normalized
        embs = np.array([l2_norm(e) for e in embs])
        embedding = l2_norm(embedding)
        sims = cosine_similarity([embedding], embs)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            return names[best_idx], best_score
        else:
            return None, best_score
    except Exception as e:
        print("find_best_match error:", e)
        return None, 0.0

# ------------------ THREADS ------------------

class VideoCaptureThread(QtCore.QThread):
    """Capture frames in a background thread and emit them as numpy arrays.
    This keeps the UI responsive.
    """
    frame_ready = QtCore.pyqtSignal(object)  # emits BGR numpy array

    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print("ERROR: Cannot open video source", self.src)
            return
        # Try to set a reasonable size (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break
            # emit frame as BGR numpy array
            self.frame_ready.emit(frame)
            # a tiny sleep to yield
            self.msleep(10)  # ~1000/10 = 100 fps cap; actual depends on camera

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class AnalyzerThread(QtCore.QThread):
    """Analyze frames with MediaPipe (fast) and DeepFace (heavy, throttled).
    Emits a dict with analysis results.
    """
    analysis_ready = QtCore.pyqtSignal(object)  # emits dict with keys: timestamp, faces (list)

    def __init__(self, max_faces=MAX_FACES, analyze_every_n=ANALYZE_EVERY_N_FRAMES):
        super().__init__()
        self.latest_frame = None
        self.lock = threading.Lock()
        self._running = True
        self.analyze_every_n = analyze_every_n
        self.counter = 0
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_faces)
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_deepface_time = 0.0
        self.deepface_throttle_sec = 0.8  # interval between heavy DeepFace calls

    @QtCore.pyqtSlot(object)
    def submit_frame(self, frame):
        """Called from capture thread: store latest frame for analysis (no backlog)."""
        with self.lock:
            self.latest_frame = frame.copy()
            self.counter += 1

    def run(self):
        while self._running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    # copy and let newer frames replace latest_frame
                    frame = self.latest_frame.copy()
                    # don't set latest_frame to None — we just process the most recent
            if frame is None:
                self.msleep(10)
                continue

            # Light processing: MediaPipe face mesh (fast)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.mp_face_mesh.process(rgb)
            except Exception as e:
                print("MediaPipe processing error:", e)
                results = None

            faces_out = []
            if results and results.multi_face_landmarks:
                h, w = frame.shape[:2]
                for face_idx, lm in enumerate(results.multi_face_landmarks):
                    # compute bbox from landmarks
                    xs = [p.x for p in lm.landmark]
                    ys = [p.y for p in lm.landmark]
                    minx = int(max(0, min(xs) * w) - 10)
                    miny = int(max(0, min(ys) * h) - 10)
                    maxx = int(min(w, max(xs) * w) + 10)
                    maxy = int(min(h, max(ys) * h) + 10)
                    # crop face with safety checks
                    try:
                        face_crop = frame[miny:maxy, minx:maxx]
                        if face_crop.size == 0:
                            continue
                        # optional resize for stable processing
                        small_crop = cv2.resize(face_crop, (224, 224))
                    except Exception:
                        continue

                    face_data = {
                        'bbox': (minx, miny, maxx, maxy),
                        'landmarks': lm,  # MediaPipe landmarks object
                        'emotion_probs': None,
                        'age': None,
                        'gender': None,
                        'embedding': None,
                        'known_name': None,
                        'known_score': 0.0,
                        'raw_crop': face_crop
                    }

                    # Decide to run heavy DeepFace analysis (throttled)
                    do_deepface = (time.time() - self.last_deepface_time) > self.deepface_throttle_sec
                    if do_deepface:
                        # analyze (emotion/age/gender)
                        df_res = analyze_face(face_crop)
                        if df_res is not None:
                            # DeepFace.analyze typically gives 'emotion': {'angry':..}, 'age' , 'gender'
                            if 'emotion' in df_res:
                                # ensure probabilities are normalized
                                emo = df_res['emotion']
                                # DeepFace may provide a dict of scores; normalize to sum 1
                                total = sum(emo.values()) if isinstance(emo, dict) else 1.0
                                if isinstance(emo, dict) and total > 0:
                                    probs = {k: float(v) / total for k, v in emo.items()}
                                else:
                                    probs = emo
                                face_data['emotion_probs'] = probs
                            if 'age' in df_res:
                                face_data['age'] = int(round(df_res['age']))
                            if 'gender' in df_res:
                                face_data['gender'] = str(df_res['gender'])

                        # embedding
                        emb = compute_embedding(face_crop)
                        face_data['embedding'] = emb
                        self.last_deepface_time = time.time()

                    faces_out.append(face_data)

            # emit analysis result (may be empty list)
            out = {
                'timestamp': time.time(),
                'frame_shape': frame.shape,
                'faces': faces_out
            }
            self.analysis_ready.emit(out)

            # small sleep to avoid tight loop
            self.msleep(5)

    def stop(self):
        self._running = False
        try:
            self.mp_face_mesh.close()
        except Exception:
            pass
        self.wait()

# ------------------ MAIN UI ------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Affect & Recognition — Option B")
        self.setGeometry(50, 50, 1200, 700)

        # Load DB
        self.names, self.embeddings = load_db()
        print("Loaded DB: ", self.names)

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: video and controls
        left_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(left_vbox, 3)

        # Video display (QLabel)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: #000")
        left_vbox.addWidget(self.video_label)

        # Controls: Save Unknown button
        controls = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Unknown (first unknown)")
        self.save_button.clicked.connect(self.on_save_unknown)
        controls.addWidget(self.save_button)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        controls.addWidget(self.status_label)

        left_vbox.addLayout(controls)

        # Right: plots and info
        right_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(right_vbox, 2)

        # Emotion bar - we use a simple custom widget using pyqtgraph BarGraphItem
        self.emotion_plot = pg.PlotWidget(title="Emotion probabilities")
        self.emotion_plot.setYRange(0, 1)
        self.emotion_plot.setMouseEnabled(False, False)
        right_vbox.addWidget(self.emotion_plot)
        self.emotion_bars = None
        self.emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # VA plot
        self.va_plot = pg.PlotWidget(title="Valence (V) and Arousal (A)")
        self.va_plot.addLegend()
        self.va_plot.setLabel('left', 'VA value', units='')
        self.va_plot.setLabel('bottom', 'time', units='samples')
        self.va_plot.setYRange(-1, 1)
        self.va_plot.setMouseEnabled(False, False)
        right_vbox.addWidget(self.va_plot)

        self.va_curve_v = self.va_plot.plot(pen='y', name='Valence')
        self.va_curve_a = self.va_plot.plot(pen='r', name='Arousal')
        self.va_buffer_v = deque(maxlen=PLOT_LENGTH)
        self.va_buffer_a = deque(maxlen=PLOT_LENGTH)
        self.plot_x = list(range(PLOT_LENGTH))

        # small info textbox
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        right_vbox.addWidget(self.info_text)

        # state
        self.current_frame = None
        self.latest_analysis = None
        self.selected_face_index = 0

        # smoothing state
        self.smoothed_va = (0.0, 0.0)
        self.smoothed_emotion = {k: 0.0 for k in self.emotion_categories}

        # Threads
        self.capture_thread = VideoCaptureThread(VIDEO_SRC)
        self.analyzer_thread = AnalyzerThread(max_faces=MAX_FACES)

        # Connect signals
        self.capture_thread.frame_ready.connect(self.on_frame_ready)
        self.analyzer_thread.analysis_ready.connect(self.on_analysis_ready)
        self.capture_thread.frame_ready.connect(self.analyzer_thread.submit_frame)

        # Start threads
        self.capture_thread.start()
        self.analyzer_thread.start()

        # Close event handling
        self._closing = False

    def closeEvent(self, event):
        self._closing = True
        self.capture_thread.stop()
        self.analyzer_thread.stop()
        event.accept()

    @QtCore.pyqtSlot(object)
    def on_frame_ready(self, frame):
        """Receive a raw BGR frame from capture thread and display it with overlays from latest analysis.
        Keep this function lightweight — just drawing and display.
        """
        self.current_frame = frame
        display_frame = frame.copy()

        # If we have latest_analysis, draw overlays
        if self.latest_analysis and 'faces' in self.latest_analysis:
            for i, f in enumerate(self.latest_analysis['faces']):
                (minx, miny, maxx, maxy) = f.get('bbox', (0, 0, 0, 0))
                # draw bbox
                cv2.rectangle(display_frame, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
                # draw name or unknown
                name = f.get('known_name', None)
                score = f.get('known_score', 0.0)
                label = f"{name} ({score:.2f})" if name else "Unknown"
                cv2.putText(display_frame, label, (minx, miny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # draw face mesh landmarks (if available)
                lm = f.get('landmarks', None)
                if lm is not None:
                    h, w = display_frame.shape[:2]
                    for p in lm.landmark:
                        x = int(min(max(0, p.x * w), w - 1))
                        y = int(min(max(0, p.y * h), h - 1))
                        cv2.circle(display_frame, (x, y), 1, (0, 200, 200), -1)

        # convert BGR to QImage and show
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        scaled = qt_image.scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))

    @QtCore.pyqtSlot(object)
    def on_analysis_ready(self, analysis):
        """Receive latest analysis from the analyzer thread. Update internal state and UI widgets.
        analysis: {timestamp, frame_shape, faces: [ {bbox, landmarks, emotion_probs, age, gender, embedding } ] }
        """
        self.latest_analysis = analysis
        faces = analysis.get('faces', [])

        # Try recognition for each face embedding
        for f in faces:
            emb = f.get('embedding', None)
            if emb is not None and len(self.embeddings) > 0:
                name, score = find_best_match(emb, self.names, self.embeddings)
                f['known_name'] = name
                f['known_score'] = score
            else:
                f['known_name'] = None
                f['known_score'] = 0.0

        # Update emotion bar and VA using first face if available (you can expand to multiple faces)
        if len(faces) > 0:
            f0 = faces[0]
            probs = f0.get('emotion_probs', None)
            if probs is None:
                # if DeepFace didn't run this cycle, keep previous values
                probs = {k: self.smoothed_emotion.get(k, 0.0) for k in self.emotion_categories}
            # ensure all keys are present
            normalized = {k: float(probs.get(k, 0.0)) for k in self.emotion_categories}

            # smooth emotion vector (EMA)
            for k in self.emotion_categories:
                prev = self.smoothed_emotion.get(k, 0.0)
                nxt = EMO_SMOOTH_ALPHA * normalized[k] + (1 - EMO_SMOOTH_ALPHA) * prev
                self.smoothed_emotion[k] = nxt

            # compute VA from probs
            v, a = emotion_probs_to_va(normalized)
            # smooth VA
            sv = VA_SMOOTH_ALPHA * v + (1 - VA_SMOOTH_ALPHA) * self.smoothed_va[0]
            sa = VA_SMOOTH_ALPHA * a + (1 - VA_SMOOTH_ALPHA) * self.smoothed_va[1]
            self.smoothed_va = (sv, sa)

            # append to buffers
            self.va_buffer_v.append(sv)
            self.va_buffer_a.append(sa)

            # update plots
            self.update_emotion_bar(self.smoothed_emotion)
            self.update_va_plot()

            # update info text
            info = f"Detected {len(faces)} face(s)\n"
            info += f"Age: {f0.get('age')}  Gender: {f0.get('gender')}\n"
            info += f"Name: {f0.get('known_name')}  Score: {f0.get('known_score'):.2f}\n"
            info += f"VA: {self.smoothed_va}\n"
            self.info_text.setPlainText(info)

        else:
            # no faces: decay smoothed values toward neutral
            self.smoothed_va = (0.95 * self.smoothed_va[0], 0.95 * self.smoothed_va[1])
            for k in self.emotion_categories:
                self.smoothed_emotion[k] *= 0.95
            self.va_buffer_v.append(self.smoothed_va[0])
            self.va_buffer_a.append(self.smoothed_va[1])
            self.update_emotion_bar(self.smoothed_emotion)
            self.update_va_plot()

    def update_emotion_bar(self, emotion_dict):
        # draw a simple bar chart for the emotions
        values = [emotion_dict.get(k, 0.0) for k in self.emotion_categories]
        x = np.arange(len(values))
        width = 0.6
        self.emotion_plot.clear()
        bg = pg.BarGraphItem(x=x, height=values, width=width, brush=(100, 150, 200, 200))
        self.emotion_plot.addItem(bg)
        # label ticks
        ticks = [list(zip(x, self.emotion_categories))]
        ax = self.emotion_plot.getAxis('bottom')
        ax.setTicks(ticks)

    def update_va_plot(self):
        # pad buffers to length if needed
        xs = np.arange(len(self.va_buffer_v))
        v = np.array(self.va_buffer_v)
        a = np.array(self.va_buffer_a)
        self.va_curve_v.setData(xs, v)
        self.va_curve_a.setData(xs, a)

    def on_save_unknown(self):
        """Called when user clicks Save Unknown. Saves first unknown face from latest_analysis.
        Procedure:
         - take several crops (N) from recent frames for the chosen face
         - compute embeddings for each and average them
         - add to DB and save
        """
        if not self.latest_analysis or len(self.latest_analysis.get('faces', [])) == 0:
            self.status_label.setText("Status: No face to save")
            return

        # pick first unknown face
        unknown_face = None
        for f in self.latest_analysis['faces']:
            if f.get('known_name') is None:
                unknown_face = f
                break
        if unknown_face is None:
            self.status_label.setText("Status: No unknown faces detected right now")
            return

        # ask user for name
        name, ok = QtWidgets.QInputDialog.getText(self, 'Save person', 'Enter name for the person:')
        if not ok or not name:
            self.status_label.setText("Save cancelled")
            return

        ensure_dir(SAVE_DIR)
        # capture several crops from current_frame around the face bbox
        minx, miny, maxx, maxy = unknown_face['bbox']
        crops = []
        # try to grab a few slightly jittered crops for robustness
        for dx, dy in [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]:
            x1 = max(0, minx + dx)
            y1 = max(0, miny + dy)
            x2 = min(self.current_frame.shape[1], maxx + dx)
            y2 = min(self.current_frame.shape[0], maxy + dy)
            crop = self.current_frame[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                crops.append(crop)

        # compute embeddings for each crop and average
        embs = []
        for i, c in enumerate(crops):
            emb = compute_embedding(c)
            if emb is not None:
                embs.append(emb)
                # save raw crop image for reference
                fname = os.path.join(SAVE_DIR, f"{name}_{int(time.time())}_{i}.jpg")
                cv2.imwrite(fname, c)

        if len(embs) == 0:
            self.status_label.setText("Failed to compute embeddings for saved crops")
            return

        mean_emb = np.mean(np.vstack(embs), axis=0)
        mean_emb = l2_norm(mean_emb)

        # append to DB
        self.names.append(name)
        self.embeddings.append(mean_emb)
        save_db(self.names, self.embeddings)

        self.status_label.setText(f"Saved {name} with {len(embs)} samples")

# ------------------ MAIN ------------------

if __name__ == '__main__':
    ensure_dir(SAVE_DIR)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print("Application error:", e)

"""  
Notes:
- This program is meant as a well-documented, runnable starting point. It is intentionally pragmatic:
  - MediaPipe face mesh is used every cycle (very fast)
  - DeepFace analyze/represent are throttled to avoid UI freezes
  - The "Save" flow stores a small set of crops for the new identity and saves an averaged embedding

- Further improvements you may implement:
  - Persistent face tracking to maintain per-person history and stable VA curves
  - Offload DeepFace model initialization earlier (right now DeepFace may download models on first run)
  - Use a background job queue (QThreadPool + QRunnable) for embeddings and analysis so analyzer thread does not block
  - Integrate a trained VA regressor (on AffectNet/Aff-Wild2) if you require accurate continuous VA values
  - Convert DeepFace models to ONNX / use TensorRT for faster inference on GPU

Possible runtime gotchas:
- If DeepFace throws errors on first run: it may be downloading large models (several hundred MB). Ensure internet access and patience.
- If the UI is still sluggish: increase analyzer_throttle_sec or ANALYZER_FPS reduction.
- If mediapipe fails to import: ensure you installed its dependencies and match opencv/pip versions.

Further Moves:
- Reduce this to a minimal MVP cut (remove PyQtGraph) for easier testing on low-end machines
- Add persistent identity thumbnails in the UI and a small settings panel for thresholds
- Convert DeepFace calls to use a preloaded model instance (faster startup)

"""
