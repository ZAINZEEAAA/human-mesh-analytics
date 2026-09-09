"""
drowsiness.py - AI Driver Sleep & Drowsiness Detection System with IGI 2 Alarm Sound & MediaPipe Tasks

Features:
- Dual Input Modes: Seamlessly analyzes both Recorded Video Files (with scrubber timeline & controls) and Live Webcams.
- IGI 2 Mission 1 Military Base Siren Alarm: High-impact dual-pulse klaxon alarm sound when sleep or severe drowsiness is detected.
- Built strictly with the modern MediaPipe Tasks API (mediapipe.tasks.python.vision.FaceLandmarker).
- Absolutely NO deprecated mp.solutions used.
- Auto-downloads the official Google MediaPipe face landmarker model if not present.
- Multi-Modal Drowsiness & Fatigue Analytics:
  1. Eye Closure / Micro-Sleep: 3D Eye Aspect Ratio (EAR) + Blendshape Blink Fusion with continuous closure duration timer.
  2. Yawn & Fatigue Detection: Mouth Aspect Ratio (MAR) + Jaw Openness duration tracker & cumulative yawn counter.
  3. Head Slump / Micro-Sleep Nodding: 3D Head Pose Pitch analysis detecting forward head dropping.
  4. Gaze Distraction Detection: Detects driver looking away from road for prolonged intervals.
  5. Composite Drowsiness Risk Index (0% - 100% Danger Level).
- Multi-Level Visual & Audio Alarm:
  - Visual HUD: Green (Awake) -> Amber (Drowsy / Yawning) -> Flashing Red (EMERGENCY SLEEP ALERT).
  - Authentic IGI 2 Base Security Alarm Sound (`igi2_alarm.wav`) with non-blocking asynchronous audio engine and Mute toggle.
- Premium Tkinter dark-themed cockpit UI with live biometric meters, playback controls, and sensitivity sliders.
- Fully compatible with Python 3.11, 3.12, 3.13, and Python 3.14+.
"""

import os
import sys
import time
import math
import queue
import wave
import threading
import urllib.request
import ssl
from typing import List, Tuple, Optional, Dict, Any

# Ensure UTF-8 stdout/stderr on Windows console
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional Windows sound support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# MediaPipe modern Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Constants & Model Configuration
# ---------------------------------------------------------------------------
MODEL_FILENAME = "face_landmarker.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
]
ALARM_WAV_FILENAME = "igi2_alarm.wav"

# Eye Landmark Indices (3D EAR calculation)
# Left Eye
LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_VERTICAL = [(160, 144), (158, 153)]
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]

# Right Eye
RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERTICAL = [(385, 380), (387, 373)]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]

# Mouth Indices (MAR calculation)
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = [(13, 14), (81, 178), (311, 402)]
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]


# ---------------------------------------------------------------------------
# IGI 2 Alarm Sound Generator Helper
# ---------------------------------------------------------------------------
def ensure_igi2_alarm_wav(filename: str = ALARM_WAV_FILENAME) -> str:
    """
    Synthesizes the authentic Project I.G.I. 2 (Mission 1) military base dual-pulse klaxon alarm sound.
    Structure per cycle: Pulse 1 (0.32s sweep) + Gap (0.08s) + Pulse 2 (0.32s sweep) + Rest (0.45s).
    """
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        return filename

    print(f"[Audio] Synthesizing IGI 2 base alarm sound: '{filename}'...")
    sample_rate = 44100
    total_audio = []
    duration_loops = 3  # 3 full cycles

    for _ in range(duration_loops):
        # Pulse 1 (Rising pitch sweep 820 Hz -> 1380 Hz with rich overtone harmonics)
        t1 = np.linspace(0, 0.32, int(sample_rate * 0.32), endpoint=False)
        freq1 = np.linspace(820, 1380, len(t1))
        phase1 = 2 * np.pi * np.cumsum(freq1) / sample_rate
        wave1 = 0.70 * np.sin(phase1) + 0.25 * np.sin(2 * phase1) + 0.10 * np.sin(3 * phase1)

        # Attack & Decay Envelopes
        env1 = np.ones_like(t1)
        att = int(sample_rate * 0.03)
        dec = int(sample_rate * 0.05)
        env1[:att] = np.linspace(0, 1, att)
        env1[-dec:] = np.linspace(1, 0, dec)
        wave1 *= env1
        total_audio.extend(wave1)

        # Short Gap
        total_audio.extend(np.zeros(int(sample_rate * 0.08)))

        # Pulse 2
        t2 = np.linspace(0, 0.32, int(sample_rate * 0.32), endpoint=False)
        freq2 = np.linspace(820, 1380, len(t2))
        phase2 = 2 * np.pi * np.cumsum(freq2) / sample_rate
        wave2 = 0.70 * np.sin(phase2) + 0.25 * np.sin(2 * phase2) + 0.10 * np.sin(3 * phase2)
        env2 = np.ones_like(t2)
        env2[:att] = np.linspace(0, 1, att)
        env2[-dec:] = np.linspace(1, 0, dec)
        wave2 *= env2
        total_audio.extend(wave2)

        # Rest Gap before next dual-pulse sequence
        total_audio.extend(np.zeros(int(sample_rate * 0.45)))

    audio_data = np.array(total_audio, dtype=np.float32)
    # Peak normalization
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        print(f"[Audio] Successfully generated IGI 2 alarm sound '{filename}' ({os.path.getsize(filename)} bytes)")
    except Exception as e:
        print(f"[Audio] Failed saving alarm wav file: {e}")

    return filename


# ---------------------------------------------------------------------------
# Model Downloader Helper
# ---------------------------------------------------------------------------
def ensure_face_model(model_path: str = MODEL_FILENAME) -> str:
    """Ensures the MediaPipe face landmarker task model is downloaded."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:
        return model_path

    print(f"[MediaPipe Tasks] Model '{model_path}' not found locally. Downloading...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in MODEL_URLS:
        try:
            print(f"[MediaPipe Tasks] Fetching from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(model_path, "wb") as out_file:
                total_length = response.getheader("content-length")
                if total_length:
                    total_length = int(total_length)
                downloaded = 0
                block_size = 65536
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_length:
                        percent = downloaded / total_length * 100
                        sys.stdout.write(f"\rDownloading model: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)")
                        sys.stdout.flush()
                sys.stdout.write("\n")
            print(f"[MediaPipe Tasks] Successfully downloaded {model_path} ({os.path.getsize(model_path)} bytes)")
            return model_path
        except Exception as e:
            print(f"[MediaPipe Tasks] Failed downloading from {url}: {e}")
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except OSError:
                    pass

    raise RuntimeError(
        f"Unable to download MediaPipe face model '{model_path}'. "
        f"Please check your internet connection or manually download the model."
    )


# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker Engine (Modern Tasks API)
# ---------------------------------------------------------------------------
class MediaPipeFaceEngine:
    """Wrapper around mediapipe.tasks.python.vision.FaceLandmarker."""

    def __init__(self, model_path: str, min_confidence: float = 0.5):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.landmarker: Optional[vision.FaceLandmarker] = None
        self._lock = threading.Lock()
        self.rebuild_landmarker()

    def rebuild_landmarker(self, min_confidence: Optional[float] = None):
        """Re-initializes the FaceLandmarker with updated confidence threshold."""
        with self._lock:
            if min_confidence is not None:
                self.min_confidence = min_confidence

            if self.landmarker is not None:
                try:
                    self.landmarker.close()
                except Exception:
                    pass
                self.landmarker = None

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self.min_confidence,
                min_face_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_rgb_frame(self, rgb_frame: np.ndarray):
        """Runs face landmark & blendshape detection synchronously on an RGB image."""
        with self._lock:
            if self.landmarker is None:
                return None
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self.landmarker.detect(mp_image)

    def close(self):
        """Cleans up the MediaPipe landmarker."""
        with self._lock:
            if self.landmarker is not None:
                try:
                    self.landmarker.close()
                except Exception:
                    pass
                self.landmarker = None


# ---------------------------------------------------------------------------
# IGI 2 Base Audio Alarm Controller (Threaded & Asynchronous)
# ---------------------------------------------------------------------------
class AudioAlarmController:
    """Manages IGI 2 Military Alarm audio playback asynchronously."""

    def __init__(self, alarm_wav: str = ALARM_WAV_FILENAME):
        self.enabled = True
        self.alarm_wav = alarm_wav
        self.is_playing = False
        self.current_alert_level = "NORMAL"  # 'NORMAL', 'WARNING', 'CRITICAL'
        self._lock = threading.Lock()

    def set_alert_level(self, level: str):
        """Updates alarm state: plays continuous IGI 2 alarm on CRITICAL, stops on NORMAL/WARNING."""
        if not HAS_WINSOUND or not self.enabled:
            if self.is_playing:
                self.stop_alarm()
            return

        with self._lock:
            if level == "CRITICAL":
                if not self.is_playing and os.path.exists(self.alarm_wav):
                    try:
                        # Asynchronous looped playback of the IGI 2 base alarm sound
                        winsound.PlaySound(self.alarm_wav, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                        self.is_playing = True
                    except Exception as e:
                        print(f"[Audio Alarm] PlaySound error: {e}")
            else:
                if self.is_playing:
                    try:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass
                    self.is_playing = False

    def trigger_test_alarm(self):
        """Plays one cycle of the IGI 2 alarm as an instant audio test."""
        if HAS_WINSOUND and os.path.exists(self.alarm_wav):
            threading.Thread(
                target=lambda: winsound.PlaySound(self.alarm_wav, winsound.SND_FILENAME | winsound.SND_ASYNC),
                daemon=True,
            ).start()

    def stop_alarm(self):
        """Stops any currently playing audio immediately."""
        with self._lock:
            if HAS_WINSOUND:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
            self.is_playing = False

    def close(self):
        """Cleans up audio resources."""
        self.stop_alarm()


# ---------------------------------------------------------------------------
# Driver Drowsiness & Sleep Detector Analytics
# ---------------------------------------------------------------------------
class DrowsinessDetector:
    """Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Head Slump, and Risk Score."""

    def __init__(self, eye_alarm_sec: float = 1.5, yawn_thresh_mar: float = 0.55):
        self.eye_alarm_sec = eye_alarm_sec
        self.yawn_thresh_mar = yawn_thresh_mar

        # Eye closure tracking
        self.eye_closed_start_time: Optional[float] = None
        self.current_eye_closed_duration = 0.0
        self.total_microsleeps = 0

        # Yawn tracking
        self.yawn_start_time: Optional[float] = None
        self.is_yawning = False
        self.total_yawns = 0
        self.recent_yawns: List[float] = []

        # Distraction tracking
        self.distracted_start_time: Optional[float] = None
        self.is_distracted = False

        # Overall Status
        self.status_level = "NORMAL"  # 'NORMAL', 'WARNING', 'CRITICAL'
        self.status_message = "Awake & Alert - Driver Focused"
        self.drowsiness_score = 0.0   # 0% to 100%

    @staticmethod
    def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_ear(self, landmarks: List[Tuple[int, int]], corners: Tuple[int, int], verticals: List[Tuple[int, int]]) -> float:
        """Calculates Eye Aspect Ratio (EAR)."""
        c1, c2 = corners
        horiz_dist = self.calculate_distance(landmarks[c1], landmarks[c2])
        if horiz_dist == 0:
            return 0.3

        vert_sum = 0.0
        for top, bot in verticals:
            vert_sum += self.calculate_distance(landmarks[top], landmarks[bot])

        return vert_sum / (2.0 * horiz_dist)

    def calculate_mar(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculates Mouth Aspect Ratio (MAR)."""
        c1, c2 = MOUTH_CORNERS
        horiz_dist = self.calculate_distance(landmarks[c1], landmarks[c2])
        if horiz_dist == 0:
            return 0.0

        vert_sum = 0.0
        for top, bot in MOUTH_VERTICAL:
            vert_sum += self.calculate_distance(landmarks[top], landmarks[bot])

        return vert_sum / (2.0 * horiz_dist)

    @staticmethod
    def extract_head_pose(matrix_4x4: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """Extracts Pitch, Yaw, Roll in degrees from 4x4 transformation matrix."""
        if matrix_4x4 is None or matrix_4x4.shape != (4, 4):
            return 0.0, 0.0, 0.0

        r = matrix_4x4[:3, :3]
        sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(r[2, 1], r[2, 2])
            yaw = math.atan2(-r[2, 0], sy)
            roll = math.atan2(r[1, 0], r[0, 0])
        else:
            pitch = math.atan2(-r[1, 2], r[1, 1])
            yaw = math.atan2(-r[2, 0], sy)
            roll = 0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def process_face(
        self,
        landmarks: List[Tuple[int, int]],
        blendshapes: Dict[str, float],
        matrix: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Analyzes facial features and returns comprehensive driver alertness telemetry."""
        now = time.time()

        if len(landmarks) < 468:
            self.status_level = "NORMAL"
            self.status_message = "No Driver Face Detected"
            return {
                "detected": False,
                "status_level": "NORMAL",
                "status_message": "No Driver Face Detected",
                "drowsiness_score": 0,
                "ear_left": 0.3,
                "ear_right": 0.3,
                "ear_avg": 0.3,
                "mar": 0.0,
                "eyes_closed_sec": 0.0,
                "is_eyes_closed": False,
                "is_yawning": False,
                "total_yawns": self.total_yawns,
                "total_microsleeps": self.total_microsleeps,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "gaze": "Center Road",
            }

        # 1. Eye Aspect Ratio & Eye Closure Tracking
        ear_left = self.calculate_ear(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_VERTICAL)
        ear_right = self.calculate_ear(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_VERTICAL)
        ear_avg = (ear_left + ear_right) / 2.0

        blink_l = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_r = blendshapes.get("eyeBlinkRight", 0.0)
        blink_avg = (blink_l + blink_r) / 2.0

        is_eyes_closed = (ear_avg < 0.20) or (blink_avg > 0.65)

        if is_eyes_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
            self.current_eye_closed_duration = now - self.eye_closed_start_time
        else:
            if self.eye_closed_start_time is not None:
                if self.current_eye_closed_duration >= self.eye_alarm_sec:
                    self.total_microsleeps += 1
            self.eye_closed_start_time = None
            self.current_eye_closed_duration = 0.0

        # 2. Mouth Aspect Ratio & Yawn Detection
        mar = self.calculate_mar(landmarks)
        jaw_open = blendshapes.get("jawOpen", 0.0)
        is_mouth_open = (mar > self.yawn_thresh_mar) or (jaw_open > 0.50)

        if is_mouth_open:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            elif (now - self.yawn_start_time) >= 1.2 and not self.is_yawning:
                self.is_yawning = True
                self.total_yawns += 1
                self.recent_yawns.append(now)
        else:
            self.yawn_start_time = None
            self.is_yawning = False

        self.recent_yawns = [t for t in self.recent_yawns if now - t <= 180]

        # 3. 3D Head Pose & Nodding Analysis
        pitch, yaw, roll = self.extract_head_pose(matrix)
        is_head_slumped = (pitch < -18.0) or (pitch > 35.0)

        # 4. Gaze & Distraction Analysis
        look_left = (blendshapes.get("eyeLookOutLeft", 0.0) + blendshapes.get("eyeLookInRight", 0.0)) / 2.0
        look_right = (blendshapes.get("eyeLookInLeft", 0.0) + blendshapes.get("eyeLookOutRight", 0.0)) / 2.0
        look_down = (blendshapes.get("eyeLookDownLeft", 0.0) + blendshapes.get("eyeLookDownRight", 0.0)) / 2.0

        if abs(yaw) > 30.0 or look_left > 0.60 or look_right > 0.60 or look_down > 0.65:
            if self.distracted_start_time is None:
                self.distracted_start_time = now
            is_distracted = (now - self.distracted_start_time) >= 2.5
        else:
            self.distracted_start_time = None
            is_distracted = False

        # 5. Composite Drowsiness Risk Calculation (0% - 100%)
        risk = 0.0
        if is_eyes_closed:
            risk += min(70.0, (self.current_eye_closed_duration / self.eye_alarm_sec) * 70.0)
        risk += min(30.0, len(self.recent_yawns) * 15.0)
        if is_head_slumped:
            risk += 35.0
        if is_distracted:
            risk += 25.0

        drowsiness_score = int(min(100.0, max(0.0, risk)))

        # 6. Multi-Level State Classification
        if self.current_eye_closed_duration >= self.eye_alarm_sec or (is_eyes_closed and is_head_slumped):
            self.status_level = "CRITICAL"
            self.status_message = f"🚨 IGI 2 ALARM: DRIVER ASLEEP! ({self.current_eye_closed_duration:.1f}s)"
        elif self.current_eye_closed_duration >= 0.7:
            self.status_level = "WARNING"
            self.status_message = f"⚠️ DROWSINESS: EYES CLOSING ({self.current_eye_closed_duration:.1f}s)"
        elif self.is_yawning:
            self.status_level = "WARNING"
            self.status_message = "🥱 YAWN DETECTED: DRIVER FATIGUE"
        elif is_head_slumped:
            self.status_level = "WARNING"
            self.status_message = "⚠️ HEAD NODDING / SLUMP DETECTED"
        elif is_distracted:
            self.status_level = "WARNING"
            self.status_message = "⚠️ DISTRACTION: EYES OFF ROAD"
        elif drowsiness_score >= 40:
            self.status_level = "WARNING"
            self.status_message = f"⚠️ HIGH FATIGUE RISK ({drowsiness_score}%)"
        else:
            self.status_level = "NORMAL"
            self.status_message = "🟢 AWAKE & ALERT - DRIVER FOCUSED"

        return {
            "detected": True,
            "status_level": self.status_level,
            "status_message": self.status_message,
            "drowsiness_score": drowsiness_score,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_avg": ear_avg,
            "mar": mar,
            "eyes_closed_sec": self.current_eye_closed_duration,
            "is_eyes_closed": is_eyes_closed,
            "is_yawning": self.is_yawning,
            "is_head_slumped": is_head_slumped,
            "is_distracted": is_distracted,
            "total_yawns": self.total_yawns,
            "total_microsleeps": self.total_microsleeps,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "gaze": "Looking Left" if look_left > 0.4 else "Looking Right" if look_right > 0.4 else "Looking Down" if look_down > 0.4 else "Center Road",
        }


# ---------------------------------------------------------------------------
# High-Performance Cockpit HUD & Visual Renderer (No mp.solutions)
# ---------------------------------------------------------------------------
class DrowsinessVisualRenderer:
    """Renders driver eyes, mouth, face contours, and cockpit alert HUD."""

    @staticmethod
    def draw_path(img_bgr: np.ndarray, landmarks: List[Tuple[int, int]], indices: List[int], color: Tuple[int, int, int], thickness: int = 2):
        """Draws connected contour lines."""
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                cv2.line(img_bgr, landmarks[idx1], landmarks[idx2], color, thickness, cv2.LINE_AA)

    @staticmethod
    def render_drowsiness_hud(
        img_bgr: np.ndarray,
        landmarks: List[Tuple[int, int]],
        telemetry: Dict[str, Any],
        show_contours: bool = True,
        show_hud: bool = True,
    ):
        """Draws driver facial contours, status banners, and warning borders."""
        h, w, _ = img_bgr.shape

        if not telemetry.get("detected", False) or len(landmarks) < 468:
            return

        level = telemetry.get("status_level", "NORMAL")
        status_msg = telemetry.get("status_message", "")
        is_eyes_closed = telemetry.get("is_eyes_closed", False)
        is_yawning = telemetry.get("is_yawning", False)
        drowsiness_score = telemetry.get("drowsiness_score", 0)

        # Choose Alert Colors (BGR)
        if level == "CRITICAL":
            theme_color = (0, 0, 255)       # Bright Red
            eye_color = (0, 0, 255)
            mouth_color = (0, 140, 255)
        elif level == "WARNING":
            theme_color = (0, 180, 255)     # Amber / Orange
            eye_color = (0, 180, 255) if is_eyes_closed else (50, 255, 120)
            mouth_color = (0, 0, 255) if is_yawning else (100, 180, 255)
        else:
            theme_color = (50, 255, 120)    # Green
            eye_color = (50, 255, 120)
            mouth_color = (120, 180, 255)

        # 1. Draw Eye & Mouth Contours
        if show_contours:
            DrowsinessVisualRenderer.draw_path(img_bgr, landmarks, LEFT_EYE_CONTOUR, eye_color, 2)
            DrowsinessVisualRenderer.draw_path(img_bgr, landmarks, RIGHT_EYE_CONTOUR, eye_color, 2)
            DrowsinessVisualRenderer.draw_path(img_bgr, landmarks, MOUTH_CONTOUR, mouth_color, 2)

            if len(landmarks) >= 478:
                cv2.circle(img_bgr, landmarks[468], 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img_bgr, landmarks[473], 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 2. Critical Flashing Border if Asleep
        if level == "CRITICAL":
            cv2.rectangle(img_bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 8, cv2.LINE_AA)

        # 3. Top-Left Cockpit Status Pill
        if show_hud:
            badge_w, badge_h = 240, 75
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (16, 16), (16 + badge_w, 16 + badge_h), (14, 18, 26), -1)
            cv2.addWeighted(overlay, 0.82, img_bgr, 0.18, 0, img_bgr)
            cv2.rectangle(img_bgr, (16, 16), (16 + badge_w, 16 + badge_h), theme_color, 1, cv2.LINE_AA)

            ear_avg = telemetry.get("ear_avg", 0.3)
            mar = telemetry.get("mar", 0.0)
            cv2.putText(img_bgr, "DRIVER MONITOR", (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 160, 190), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, f"EAR: {ear_avg:.2f} | MAR: {mar:.2f}", (26, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, f"Risk Score: {drowsiness_score}%", (26, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, theme_color, 1, cv2.LINE_AA)

            # Bottom Alert Banner
            banner_y = h - 45
            cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + 35), (14, 18, 26), -1)
            cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)
            cv2.rectangle(img_bgr, (20, banner_y), (w - 20, banner_y + 35), theme_color, 2, cv2.LINE_AA)
            cv2.putText(img_bgr, status_msg, (35, banner_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Modern Tkinter Driver Drowsiness Studio Application
# ---------------------------------------------------------------------------
class DrowsinessStudioApp:
    """Main Desktop UI Application for Driver Sleep & Drowsiness Detection."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Driver Drowsiness & Sleep Detector — IGI 2 Siren Edition")
        self.root.geometry("1280x850")
        self.root.minsize(1080, 750)
        self.root.configure(bg="#0d1117")

        # Application state
        self.is_running = False
        self.is_paused = False
        self.input_source_mode = "WEBCAM"  # 'WEBCAM' or 'VIDEO_FILE'
        self.video_filepath: Optional[str] = None
        self.camera_index = 0

        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Video metadata
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.video_fps = 30.0

        # UI Settings
        self.var_mirror = tk.BooleanVar(value=True)
        self.var_contours = tk.BooleanVar(value=True)
        self.var_hud = tk.BooleanVar(value=True)
        self.var_loop_video = tk.BooleanVar(value=True)
        self.var_sound_alert = tk.BooleanVar(value=True)
        self.var_cam_idx = tk.IntVar(value=0)
        self.var_eye_alarm_sec = tk.DoubleVar(value=1.5)
        self.var_yawn_mar = tk.DoubleVar(value=0.55)

        # Performance counters
        self.fps = 0.0
        self.latency_ms = 0.0

        # Thread-safe queue
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Ensure IGI 2 Alarm WAV exists & Initialize MediaPipe Engine
        ensure_igi2_alarm_wav(ALARM_WAV_FILENAME)
        self._init_mediapipe_engine()

        self.detector = DrowsinessDetector(
            eye_alarm_sec=self.var_eye_alarm_sec.get(),
            yawn_thresh_mar=self.var_yawn_mar.get(),
        )
        self.audio_alarm = AudioAlarmController(ALARM_WAV_FILENAME)

        # Build GUI Layout
        self._build_ui()

        # Start Stream
        self.start_stream()

        # Periodic GUI update tick
        self.root.after(15, self._gui_update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_mediapipe_engine(self):
        """Initializes model downloader and FaceLandmarker."""
        try:
            model_path = ensure_face_model(MODEL_FILENAME)
            self.engine = MediaPipeFaceEngine(model_path=model_path, min_confidence=0.5)
            self.model_status_text = "Model: face_landmarker.task + IGI 2 Alarm"
        except Exception as e:
            messagebox.showerror("Model Init Error", f"Failed to initialize Face Landmarker:\n{e}")
            sys.exit(1)

    def _build_ui(self):
        """Constructs modern dark cockpit widgets."""
        # Top Header Bar
        header_frame = tk.Frame(self.root, bg="#161b22", height=60, highlightthickness=1, highlightbackground="#30363d")
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="🚨 I.G.I. 2 DRIVER SLEEP GUARDIAN",
            font=("Segoe UI", 15, "bold"),
            fg="#58a6ff",
            bg="#161b22",
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="• MediaPipe Tasks & IGI 2 Mission 1 Siren (Python 3.14 Ready)",
            font=("Segoe UI", 10),
            fg="#8b949e",
            bg="#161b22",
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=14)

        self.lbl_header_fps = tk.Label(
            header_frame,
            text="⚡ FPS: -- | Latency: -- ms",
            font=("Consolas", 11, "bold"),
            fg="#3fb950",
            bg="#21262d",
            padx=12,
            pady=4,
        )
        self.lbl_header_fps.pack(side=tk.RIGHT, padx=20, pady=12)

        self.lbl_model_badge = tk.Label(
            header_frame,
            text=self.model_status_text,
            font=("Segoe UI", 9),
            fg="#c9d1d9",
            bg="#1f242c",
            padx=10,
            pady=4,
        )
        self.lbl_model_badge.pack(side=tk.RIGHT, padx=8, pady=12)

        # Body Container (Left: Video, Right: Telemetry & Controls)
        body_frame = tk.Frame(self.root, bg="#0d1117")
        body_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Left Column: Video Viewport
        video_col = tk.Frame(body_frame, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Video Header
        video_header = tk.Frame(video_col, bg="#1c2128", height=36)
        video_header.pack(fill=tk.X, side=tk.TOP)
        video_header.pack_propagate(False)

        self.lbl_viewport_title = tk.Label(
            video_header,
            text="📹 DRIVER CAMERA STREAM",
            font=("Segoe UI", 10, "bold"),
            fg="#c9d1d9",
            bg="#1c2128",
        )
        self.lbl_viewport_title.pack(side=tk.LEFT, padx=14, pady=8)

        self.lbl_stream_status = tk.Label(
            video_header,
            text="● STREAM ACTIVE",
            font=("Segoe UI", 9, "bold"),
            fg="#3fb950",
            bg="#1c2128",
        )
        self.lbl_stream_status.pack(side=tk.RIGHT, padx=14, pady=8)

        # Video Canvas / Label
        self.lbl_video = tk.Label(video_col, bg="#080a0f", text="Starting stream...", fg="#8b949e", font=("Segoe UI", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # Scrubber Bar (For Video File playback)
        self.scrub_frame = tk.Frame(video_col, bg="#161b22", height=34)
        self.scrub_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 6))

        self.btn_play_pause = tk.Button(
            self.scrub_frame,
            text="⏸ Pause",
            font=("Segoe UI", 8, "bold"),
            bg="#21262d",
            fg="#c9d1d9",
            bd=0,
            padx=8,
            command=self.toggle_play_pause,
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=(0, 6))

        self.scale_scrub = tk.Scale(
            self.scrub_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#161b22",
            fg="#58a6ff",
            troughcolor="#21262d",
            highlightthickness=0,
            command=self._on_scrub_change,
        )
        self.scale_scrub.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_timecode = tk.Label(self.scrub_frame, text="00:00 / 00:00", font=("Consolas", 8), fg="#8b949e", bg="#161b22", padx=6)
        self.lbl_timecode.pack(side=tk.RIGHT)

        # Right Column: Sidebar
        sidebar_col = tk.Frame(body_frame, bg="#0d1117", width=430)
        sidebar_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        sidebar_col.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_col, bg="#0d1117", highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar_col, orient="vertical", command=sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(sidebar_canvas, bg="#0d1117")

        self.sidebar_content.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", width=410)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build Sidebar Panels
        self._build_input_source_section()
        self._build_alert_status_section()
        self._build_biometrics_section()
        self._build_controls_section()

    def _build_input_source_section(self):
        """Builds input selector: Live Webcam vs. Video File Loader."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  📥 INPUT SOURCE  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        sec.pack(fill=tk.X, pady=(0, 10), padx=4)

        pad = tk.Frame(sec, bg="#161b22", padx=10, pady=8)
        pad.pack(fill=tk.X)

        row_modes = tk.Frame(pad, bg="#161b22")
        row_modes.pack(fill=tk.X, pady=(0, 6))

        self.btn_mode_webcam = tk.Button(
            row_modes,
            text="📷 Live Webcam",
            font=("Segoe UI", 9, "bold"),
            bg="#238636",
            fg="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=5,
            command=self.set_source_webcam,
        )
        self.btn_mode_webcam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_mode_file = tk.Button(
            row_modes,
            text="📁 Video File",
            font=("Segoe UI", 9, "bold"),
            bg="#21262d",
            fg="#c9d1d9",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=5,
            command=self.browse_video_file,
        )
        self.btn_mode_file.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        self.lbl_file_info = tk.Label(
            pad,
            text="Source: Live Camera Feed",
            font=("Segoe UI", 8),
            fg="#8b949e",
            bg="#161b22",
            anchor="w",
        )
        self.lbl_file_info.pack(fill=tk.X)

    def _build_alert_status_section(self):
        """Builds prominent alert status banner & danger risk meter."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  🚨 DRIVER ALERTNESS STATUS  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        sec.pack(fill=tk.X, pady=(0, 10), padx=4)

        pad = tk.Frame(sec, bg="#161b22", padx=10, pady=8)
        pad.pack(fill=tk.X)

        # Status Card
        self.status_card = tk.Frame(pad, bg="#21262d", padx=10, pady=10, highlightthickness=1, highlightbackground="#30363d")
        self.status_card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_status_icon = tk.Label(self.status_card, text="🟢", font=("Segoe UI Emoji", 20), bg="#21262d")
        self.lbl_status_icon.pack(side=tk.LEFT, padx=(2, 8))

        s_meta = tk.Frame(self.status_card, bg="#21262d")
        s_meta.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_status_title = tk.Label(s_meta, text="AWAKE & ALERT", font=("Segoe UI", 12, "bold"), fg="#3fb950", bg="#21262d", anchor="w")
        self.lbl_status_title.pack(fill=tk.X)

        self.lbl_status_desc = tk.Label(s_meta, text="Driver Focused on Road", font=("Segoe UI", 8), fg="#8b949e", bg="#21262d", anchor="w")
        self.lbl_status_desc.pack(fill=tk.X)

        # Danger Risk Meter
        tk.Label(pad, text="Drowsiness Risk Level:", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22", anchor="w").pack(fill=tk.X, pady=(4, 2))
        risk_row = tk.Frame(pad, bg="#161b22")
        risk_row.pack(fill=tk.X)

        self.risk_canvas = tk.Canvas(risk_row, bg="#21262d", height=12, highlightthickness=0)
        self.risk_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.risk_bar = self.risk_canvas.create_rectangle(0, 0, 0, 12, fill="#3fb950", width=0)

        self.lbl_risk_pct = tk.Label(risk_row, text="0%", font=("Segoe UI", 9, "bold"), fg="#3fb950", bg="#161b22", width=5, anchor="e")
        self.lbl_risk_pct.pack(side=tk.RIGHT)

    def _build_biometrics_section(self):
        """Builds biometric meters (Eyes, Yawns, Head Slump, Gaze)."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  👁️ BIOMETRIC FATIGUE TELEMETRY  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        sec.pack(fill=tk.X, pady=(0, 10), padx=4)

        pad = tk.Frame(sec, bg="#161b22", padx=10, pady=8)
        pad.pack(fill=tk.X)

        # Eye Closed Timer Row
        self.lbl_eye_timer = tk.Label(pad, text="⏱️ Eyes Closed Duration: 0.0s (Normal)", font=("Consolas", 9, "bold"), fg="#3fb950", bg="#161b22", anchor="w")
        self.lbl_eye_timer.pack(fill=tk.X, pady=1)

        # EAR & MAR values
        self.lbl_ear_mar = tk.Label(pad, text="EAR: 0.30 (Open) | MAR: 0.05 (Closed)", font=("Consolas", 8), fg="#c9d1d9", bg="#161b22", anchor="w")
        self.lbl_ear_mar.pack(fill=tk.X, pady=1)

        # Yawns & Microsleeps counters
        self.lbl_fatigue_counters = tk.Label(pad, text="🥱 Yawns: 0 | 🚨 Micro-sleeps: 0", font=("Segoe UI", 9, "bold"), fg="#58a6ff", bg="#161b22", anchor="w")
        self.lbl_fatigue_counters.pack(fill=tk.X, pady=2)

        # Head Nodding & Gaze
        self.lbl_head_pose = tk.Label(pad, text="Head Pitch: +0.0° | Gaze: Center Road", font=("Consolas", 8), fg="#8b949e", bg="#161b22", anchor="w")
        self.lbl_head_pose.pack(fill=tk.X, pady=1)

    def _build_controls_section(self):
        """Builds alarm toggles, sensitivity sliders, and options."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  🔊 IGI 2 ALARM & SETTINGS  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        sec.pack(fill=tk.X, pady=(0, 10), padx=4)

        pad = tk.Frame(sec, bg="#161b22", padx=10, pady=8)
        pad.pack(fill=tk.X)

        # Sound Alert Controls Row
        snd_row = tk.Frame(pad, bg="#161b22")
        snd_row.pack(fill=tk.X, pady=(0, 6))

        def _on_sound_toggle():
            self.audio_alarm.enabled = self.var_sound_alert.get()
            if not self.audio_alarm.enabled:
                self.audio_alarm.stop_alarm()

        cb_snd = tk.Checkbutton(
            snd_row,
            text="🔊 IGI 2 Alarm Sound",
            variable=self.var_sound_alert,
            font=("Segoe UI", 9, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            selectcolor="#21262d",
            activebackground="#161b22",
            command=_on_sound_toggle,
        )
        cb_snd.pack(side=tk.LEFT)

        tk.Button(
            snd_row,
            text="🚨 Test Siren",
            font=("Segoe UI", 8, "bold"),
            bg="#da3633",
            fg="#ffffff",
            activebackground="#b62324",
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=2,
            command=self.audio_alarm.trigger_test_alarm,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Visual Toggles
        grid = tk.Frame(pad, bg="#161b22")
        grid.pack(fill=tk.X, pady=(0, 8))

        tk.Checkbutton(grid, text="Mirror Video", variable=self.var_mirror, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=0, sticky="w")
        tk.Checkbutton(grid, text="Eye / Mouth Contours", variable=self.var_contours, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=1, sticky="w", padx=8)
        tk.Checkbutton(grid, text="On-Video HUD", variable=self.var_hud, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(grid, text="Loop Video File", variable=self.var_loop_video, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=1, sticky="w", padx=8)

        # Sliders
        lbl_eye = tk.Label(pad, text=f"Eye Closure Alarm Threshold: {self.var_eye_alarm_sec.get():.1f}s", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22", anchor="w")
        lbl_eye.pack(fill=tk.X)

        def _on_eye_change(val):
            v = float(val)
            lbl_eye.config(text=f"Eye Closure Alarm Threshold: {v:.1f}s")
            self.detector.eye_alarm_sec = v

        tk.Scale(
            pad,
            from_=0.8,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.var_eye_alarm_sec,
            command=_on_eye_change,
            bg="#161b22",
            fg="#58a6ff",
            troughcolor="#21262d",
            highlightthickness=0,
            showvalue=False,
        ).pack(fill=tk.X, pady=(0, 6))

    # -----------------------------------------------------------------------
    # Stream & Background Worker
    # -----------------------------------------------------------------------
    def set_source_webcam(self):
        """Switches to Live Webcam mode."""
        if self.input_source_mode == "WEBCAM" and self.is_running:
            return

        self.input_source_mode = "WEBCAM"
        self.btn_mode_webcam.config(bg="#238636", fg="#ffffff")
        self.btn_mode_file.config(bg="#21262d", fg="#c9d1d9")
        self.lbl_viewport_title.config(text="📹 DRIVER CAMERA STREAM")
        self.lbl_file_info.config(text="Source: Live Camera Feed")
        self.var_mirror.set(True)

        self.stop_stream()
        self.start_stream()

    def browse_video_file(self):
        """Opens a file picker to choose a video file for drowsiness testing."""
        filepath = filedialog.askopenfilename(
            title="Select Driver Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        self.video_filepath = filepath
        self.input_source_mode = "VIDEO_FILE"
        self.btn_mode_file.config(bg="#238636", fg="#ffffff")
        self.btn_mode_webcam.config(bg="#21262d", fg="#c9d1d9")
        self.var_mirror.set(False)

        basename = os.path.basename(filepath)
        self.lbl_viewport_title.config(text=f"🎬 VIDEO FILE: {basename}")
        self.lbl_file_info.config(text=f"File: {basename}")

        self.stop_stream()
        self.start_stream()

    def toggle_play_pause(self):
        """Toggles video playback pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_play_pause.config(text="▶ Play", bg="#238636")
        else:
            self.btn_play_pause.config(text="⏸ Pause", bg="#21262d")

    def _on_scrub_change(self, val):
        """Seeks video position."""
        if self.input_source_mode == "VIDEO_FILE" and self.cap is not None and self.video_total_frames > 0:
            target_frame = int((float(val) / 100.0) * self.video_total_frames)
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            except Exception:
                pass

    def start_stream(self):
        """Starts capture thread."""
        if self.is_running:
            return

        self.stop_event.clear()
        self.is_running = True
        self.is_paused = False
        self.btn_play_pause.config(text="⏸ Pause", bg="#21262d")
        self.lbl_stream_status.config(text="● STREAM ACTIVE", fg="#3fb950")

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def stop_stream(self):
        """Stops capture thread."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.audio_alarm.stop_alarm()
        self.lbl_stream_status.config(text="○ STOPPED", fg="#8b949e")

    def _open_capture_source(self) -> Optional[cv2.VideoCapture]:
        """Opens capture device or video file safely."""
        if self.input_source_mode == "VIDEO_FILE" and self.video_filepath:
            if not os.path.exists(self.video_filepath):
                return None
            cap = cv2.VideoCapture(self.video_filepath)
            if cap and cap.isOpened():
                self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                return cap
            return None
        else:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(self.camera_index, backend)
                    if cap and cap.isOpened():
                        for prop, val in [
                            (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                            (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                            (cv2.CAP_PROP_FPS, 30),
                        ]:
                            try:
                                cap.set(prop, val)
                            except Exception:
                                pass
                        return cap
                    if cap:
                        cap.release()
                except Exception:
                    pass

            try:
                cap = cv2.VideoCapture(self.camera_index)
                if cap and cap.isOpened():
                    return cap
            except Exception:
                pass
            return None

    def _capture_worker(self):
        """Background thread for grabbing frames, MediaPipe detection, and rendering."""
        print(f"[Driver Monitor] Initializing source: {self.input_source_mode}...")
        self.cap = self._open_capture_source()

        if self.cap is None or not self.cap.isOpened():
            print("[Driver Monitor] Failed to open capture source.")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "Camera Unavailable" if self.input_source_mode == "WEBCAM" else "Could not open video file"
            cv2.putText(placeholder, msg, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 140, 255), 2, cv2.LINE_AA)
            display_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((display_rgb, {}))
            self.is_running = False
            return

        print("[Driver Monitor] Capture opened successfully. Beginning drowsiness detection loop.")
        fps_frame_counter = 0
        fps_timer = time.time()

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[Driver Monitor] Frame read exception: {e}")
                time.sleep(0.05)
                continue

            if not ret or frame is None:
                if self.input_source_mode == "VIDEO_FILE" and self.var_loop_video.get():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.02)
                    continue

            if self.input_source_mode == "VIDEO_FILE":
                self.video_current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            if self.var_mirror.get():
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()

            # Convert BGR to RGB for MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform Face Landmark & Blendshape Detection
            try:
                detection_result = self.engine.detect_rgb_frame(rgb_frame)
            except Exception as e:
                print(f"[MediaPipe Tasks] Face inference exception: {e}")
                detection_result = None

            t_infer = (time.perf_counter() - t0) * 1000.0

            # Extract Landmark Pixel Coordinates & Blendshapes
            landmarks_pixel: List[Tuple[int, int]] = []
            blendshapes: Dict[str, float] = {}
            matrix = None
            h, w, _ = frame.shape

            if detection_result and detection_result.face_landmarks:
                first_face = detection_result.face_landmarks[0]
                for lm in first_face:
                    px = max(0, min(w - 1, int(lm.x * w)))
                    py = max(0, min(h - 1, int(lm.y * h)))
                    landmarks_pixel.append((px, py))

                if detection_result.face_blendshapes:
                    for cat in detection_result.face_blendshapes[0]:
                        blendshapes[cat.category_name] = float(cat.score)

                if detection_result.facial_transformation_matrixes:
                    matrix = detection_result.facial_transformation_matrixes[0]

            # Process Drowsiness & Sleep Telemetry
            telemetry = self.detector.process_face(landmarks_pixel, blendshapes, matrix)

            # Update Audio Alarm Controller state with IGI 2 sound
            level = telemetry.get("status_level", "NORMAL")
            self.audio_alarm.set_alert_level(level)

            # Render Contours & Cockpit HUD
            DrowsinessVisualRenderer.render_drowsiness_hud(
                frame,
                landmarks_pixel,
                telemetry,
                show_contours=self.var_contours.get(),
                show_hud=self.var_hud.get(),
            )

            # FPS calculation
            fps_frame_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                self.fps = fps_frame_counter / elapsed
                self.latency_ms = t_infer
                fps_frame_counter = 0
                fps_timer = time.time()

            # Convert processed frame to RGB for Tkinter display
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((display_rgb, telemetry))

            # Pace video file playback
            if self.input_source_mode == "VIDEO_FILE" and self.video_fps > 0:
                frame_delay = 1.0 / self.video_fps
                time.sleep(max(0.005, frame_delay - (time.perf_counter() - t0)))

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[Driver Monitor] Worker thread exited cleanly.")

    # -----------------------------------------------------------------------
    # GUI Periodic Update Loop
    # -----------------------------------------------------------------------
    def _gui_update_loop(self):
        """Pulls latest processed frame from queue and updates Tkinter widgets."""
        try:
            if not self.frame_queue.empty():
                display_rgb, telemetry = self.frame_queue.get_nowait()

                # Update Header FPS & Latency
                self.lbl_header_fps.config(
                    text=f"⚡ FPS: {self.fps:.1f} | Latency: {self.latency_ms:.1f} ms",
                    fg="#3fb950" if self.fps >= 20 else "#d29922",
                )

                widget_w = max(320, self.lbl_video.winfo_width())
                widget_h = max(240, self.lbl_video.winfo_height())
                img_h, img_w, _ = display_rgb.shape

                scale = min(widget_w / img_w, widget_h / img_h)
                new_w = max(1, int(img_w * scale))
                new_h = max(1, int(img_h * scale))

                pil_img = Image.fromarray(display_rgb)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                tk_img = ImageTk.PhotoImage(image=pil_img)

                self.lbl_video.img_tk = tk_img
                self.lbl_video.configure(image=tk_img, text="")

                # Update Telemetry UI
                self._update_telemetry_ui(telemetry)

                # Update Video Timeline if in Video File mode
                if self.input_source_mode == "VIDEO_FILE" and self.video_total_frames > 0:
                    pct = (self.video_current_frame / self.video_total_frames) * 100.0
                    self.scale_scrub.set(pct)
                    cur_sec = int(self.video_current_frame / (self.video_fps or 30.0))
                    tot_sec = int(self.video_total_frames / (self.video_fps or 30.0))
                    self.lbl_timecode.config(text=f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

        except Exception:
            pass

        self.root.after(16, self._gui_update_loop)

    def _update_telemetry_ui(self, telemetry: Dict[str, Any]):
        """Updates GUI cards with live driver fatigue metrics."""
        if not telemetry:
            return

        level = telemetry.get("status_level", "NORMAL")
        status_msg = telemetry.get("status_message", "")
        drowsiness_score = telemetry.get("drowsiness_score", 0)
        eyes_closed_sec = telemetry.get("eyes_closed_sec", 0.0)
        is_eyes_closed = telemetry.get("is_eyes_closed", False)
        ear_avg = telemetry.get("ear_avg", 0.3)
        mar = telemetry.get("mar", 0.0)
        yawns = telemetry.get("total_yawns", 0)
        microsleeps = telemetry.get("total_microsleeps", 0)
        pitch = telemetry.get("pitch", 0.0)
        gaze = telemetry.get("gaze", "Center Road")

        # Update Status Card
        if level == "CRITICAL":
            self.lbl_status_icon.config(text="🚨")
            self.lbl_status_title.config(text="IGI 2 ALARM: ASLEEP!", fg="#f85149")
            self.status_card.config(highlightbackground="#f85149")
        elif level == "WARNING":
            self.lbl_status_icon.config(text="⚠️")
            self.lbl_status_title.config(text="WARNING: DROWSY", fg="#f0883e")
            self.status_card.config(highlightbackground="#f0883e")
        else:
            self.lbl_status_icon.config(text="🟢")
            self.lbl_status_title.config(text="AWAKE & ALERT", fg="#3fb950")
            self.status_card.config(highlightbackground="#30363d")

        self.lbl_status_desc.config(text=status_msg)

        # Danger Risk Meter
        canvas_w = max(10, self.risk_canvas.winfo_width())
        bar_w = int(canvas_w * (drowsiness_score / 100.0))
        risk_color = "#f85149" if drowsiness_score >= 60 else "#f0883e" if drowsiness_score >= 35 else "#3fb950"
        self.risk_canvas.coords(self.risk_bar, 0, 0, bar_w, 12)
        self.risk_canvas.itemconfig(self.risk_bar, fill=risk_color)
        self.lbl_risk_pct.config(text=f"{drowsiness_score}%", fg=risk_color)

        # Eye Closed Timer
        if is_eyes_closed:
            self.lbl_eye_timer.config(
                text=f"⏱️ Eyes Closed: {eyes_closed_sec:.1f}s (ALERT)",
                fg="#f85149" if eyes_closed_sec >= self.detector.eye_alarm_sec else "#f0883e",
            )
        else:
            self.lbl_eye_timer.config(text="⏱️ Eyes Open: (Normal Focus)", fg="#3fb950")

        self.lbl_ear_mar.config(text=f"EAR: {ear_avg:.2f} | MAR: {mar:.2f} ({'Yawning' if telemetry.get('is_yawning') else 'Normal'})")
        self.lbl_fatigue_counters.config(text=f"🥱 Total Yawns: {yawns} | 🚨 Micro-sleeps: {microsleeps}")
        self.lbl_head_pose.config(text=f"Head Pitch: {pitch:+.1f}° | Gaze: {gaze}")

    def on_closing(self):
        """Gracefully closes resources and terminates window."""
        print("[App] Closing Drowsiness Studio...")
        self.stop_stream()
        if hasattr(self, "audio_alarm") and self.audio_alarm:
            self.audio_alarm.close()
        if hasattr(self, "engine") and self.engine:
            self.engine.close()
        try:
            self.root.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("[AI DRIVER DROWSINESS & SLEEP DETECTOR] IGI 2 Alarm Edition")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"MediaPipe Version: {mp.__version__}")
    print("Tasks API Vision: FaceLandmarker + IGI 2 Military Klaxon Siren")
    print("=" * 65)

    root = tk.Tk()
    app = DrowsinessStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
