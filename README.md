The system uses a decoupled, event-driven architecture. Frame captures are offloaded from thread blocks, transformed into `mp.Image` instances, and evaluated asynchronously via local task binaries.

---

## 📂 Modular Pipeline Deep Dives

### 1. Kinematic Hand Gesture Tracking (`hand.py`)
A custom tracking pipeline that renders hand skeletons onto a responsive interface.
* **Core Logic:** Extracts 21 3D landmarks per hand without legacy pipeline bindings.
* **UI Pattern:** Frame vectors are serialized and projected onto a custom **Tkinter Canvas** element using a structured processing rate wrapper.
* **Performance:** Uses `opencv-python-headless` to eliminate local window handling crashes in isolated runtime clusters.

### 2. Biometric Drowsiness Alert Framework (`drowsiness.py`)
A facial feature tracking framework that identifies operator fatigue by monitoring facial motion.
* **Mathematical Foundation:** Evaluates eye state using the **Eye Aspect Ratio (EAR)** calculation:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\|p_1 - p_4\|}$$
* **Detection Mechanism:** Tracks distances between vertical eyelid coordinates relative to horizontal anchor points. Sustained decreases below an established threshold ($< 0.22$) for consecutive frames trigger an immediate drowsiness alert.

### 3. Biomechanical Pose Analytics Counter (`pushup_counter.py`)
An automated physical fitness movement tracker that provides real-time biomechanical feedback.
* **Kinematic Evaluation:** Isolates and extracts spatial coordinates for key joint clusters: **Shoulder** ($A$), **Elbow** ($B$), and **Wrist** ($C$).
* **Vector Trigonometry:** Calculates the real-time angle ($\theta$) of the elbow joint using the dot product inversion theorem:
  $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}\right)$$
* **State Machine Tracking:** Transitions smoothly between state parameters based on form execution. Increments repetition values when the arm shifts from a deep contraction ($\theta \le 90^\circ$) back to full extension ($\theta \ge 160^\circ$).

---

## 🛠️ Installation & Environment Setup

Follow these clean installation steps to configure your environment, initialize requirements, and fetch the necessary model binaries:

```bash
# Initialize and activate isolated system environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade dependency installers and install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Download model binaries required by the MediaPipe Tasks API
mkdir -p models
curl -L -o models/hand_landmarker.task "https://googleapis.com"
curl -L -o models/face_landmarker.task "https://googleapis.com"
curl -L -o models/pose_landmarker.task "https://googleapis.com"
```

---

## 🚀 Execution Guide

Run your preferred tracking pipeline using the following terminal commands:

```bash
# Run the Hand Gesture Skeletal App
python modules/hand.py

# Run the Drowsiness Detection Pipeline
python modules/drowsiness.py

# Run the Biomechanical Push-Up Counter
python modules/pushup_counter.py
```
