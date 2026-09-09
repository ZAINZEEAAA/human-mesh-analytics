"""
face.py - Professional 478-Landmark Face Mesh, Detail & Blendshape Analyzer

Features:
- Built strictly with the modern MediaPipe Tasks API (mediapipe.tasks.python.vision.FaceLandmarker).
- Absolutely NO deprecated mp.solutions used.
- Auto-downloads the official Google MediaPipe face landmarker model (with blendshapes and 3D matrix).
- Tracks all 478 3D landmarks (468 Dense Face Mesh + 10 Iris/Pupil tracking points).
- Analyzes 52 facial blendshapes: Smile, Blinks, Jaw Openness, Brow Elevation, Gaze Direction, and Emotion states.
- Computes real-time 3D Head Pose (Yaw, Pitch, Roll) with visual 3D coordinate axis projection.
- Premium Tkinter dark-themed studio UI with live telemetry cards, meters, and customizable visual toggles.
- Threaded camera capture pipeline with exception-safe OpenCV backend fallback.
- Fully compatible with Python 3.11, 3.12, 3.13, and Python 3.14+.
"""

import os
import sys
import time
import math
import queue
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

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

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

# ---------------------------------------------------------------------------
# Facial Topology Indices (478 Landmarks)
# ---------------------------------------------------------------------------
# Face Oval / Jawline
FACIAL_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

# Lips (Outer and Inner Contours)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# Eyes & Eyebrows
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Nose bridge and contours
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
NOSE_BOTTOM = [98, 97, 2, 326, 327]

# Precision Iris Landmarks (Landmarks 468 to 477)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_CONTOUR = [469, 470, 471, 472, 469]

RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_CONTOUR = [474, 475, 476, 477, 474]

# Selected Key Triangulation Connections for Mesh Mode
TESSELLATION_KEY_CONNECTIONS = [
    # Forehead & Brow bridge
    (10, 109), (10, 338), (109, 67), (338, 297), (67, 103), (297, 332),
    (103, 54), (332, 284), (54, 21), (284, 251), (21, 162), (251, 389),
    # Cheeks & Midface
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
    (1, 19), (19, 94), (94, 2), (2, 164), (164, 0), (0, 11),
    (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    (18, 200), (200, 199), (199, 175), (175, 152),
    # Left Cheek diagonals
    (116, 123), (123, 147), (147, 213), (213, 192), (192, 214), (214, 210), (210, 211),
    # Right Cheek diagonals
    (345, 352), (352, 376), (376, 433), (433, 416), (416, 434), (434, 430), (430, 431),
]

# Color Palette (BGR)
CLR_OVAL = (255, 230, 0)         # Cyan
CLR_EYES = (80, 255, 120)        # Emerald Green
CLR_IRIS = (255, 40, 220)        # Electric Magenta
CLR_PUPIL = (255, 255, 255)      # Pure White
CLR_BROWS = (0, 210, 255)        # Neon Amber
CLR_NOSE = (230, 230, 250)       # Ice Silver
CLR_LIPS = (100, 120, 255)       # Coral Rose
CLR_MESH = (70, 80, 110)         # Subtle Dark Blue-Grey


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
        f"Please check internet connection or manually download the model."
    )


# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker Engine (Modern Tasks API)
# ---------------------------------------------------------------------------
class MediaPipeFaceEngine:
    """Wrapper around mediapipe.tasks.python.vision.FaceLandmarker."""

    def __init__(self, model_path: str, num_faces: int = 2, min_confidence: float = 0.5):
        self.model_path = model_path
        self.num_faces = num_faces
        self.min_confidence = min_confidence
        self.landmarker: Optional[vision.FaceLandmarker] = None
        self._lock = threading.Lock()
        self.rebuild_landmarker()

    def rebuild_landmarker(self, num_faces: Optional[int] = None, min_confidence: Optional[float] = None):
        """Re-initializes the FaceLandmarker with updated options."""
        with self._lock:
            if num_faces is not None:
                self.num_faces = num_faces
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
                num_faces=self.num_faces,
                min_face_detection_confidence=self.min_confidence,
                min_face_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_rgb_frame(self, rgb_frame: np.ndarray):
        """Runs face landmark detection on an RGB numpy image."""
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
# Face Detail, Head Pose & Expression Analyzer
# ---------------------------------------------------------------------------
class FaceTelemetryExtractor:
    """Analyzes 52 blendshapes, calculates Euler angles, and detects expressions."""

    @staticmethod
    def parse_blendshapes(blendshape_list) -> Dict[str, float]:
        """Converts Category blendshape objects to a name -> score dict."""
        bs_dict: Dict[str, float] = {}
        if not blendshape_list:
            return bs_dict
        for cat in blendshape_list:
            bs_dict[cat.category_name] = float(cat.score)
        return bs_dict

    @staticmethod
    def compute_head_pose_from_matrix(matrix_4x4: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        Extracts Yaw, Pitch, Roll in degrees from the 4x4 facial transformation matrix.
        Returns: (pitch_deg, yaw_deg, roll_deg)
        """
        if matrix_4x4 is None or matrix_4x4.shape != (4, 4):
            return 0.0, 0.0, 0.0

        r = matrix_4x4[:3, :3]

        # Decompose rotation matrix into Euler angles (ZYX convention)
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

        # Convert radians to degrees
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        roll_deg = math.degrees(roll)

        return pitch_deg, yaw_deg, roll_deg

    @staticmethod
    def detect_facial_expression(bs: Dict[str, float]) -> Tuple[str, str, float]:
        """
        Determines the prominent facial mood / expression based on blendshape scores.
        Returns: (emoji, expression_name, confidence)
        """
        smile_l = bs.get("mouthSmileLeft", 0.0)
        smile_r = bs.get("mouthSmileRight", 0.0)
        smile = (smile_l + smile_r) / 2.0

        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)

        jaw_open = bs.get("jawOpen", 0.0)
        brow_up = (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0) + bs.get("browInnerUp", 0.0)) / 3.0
        brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        pucker = bs.get("mouthPucker", 0.0)
        cheek_puff = bs.get("cheekPuff", 0.0)

        # Classification heuristics
        if blink_l > 0.55 and blink_r < 0.25:
            return "😉", "Wink (Left Eye)", blink_l
        if blink_r > 0.55 and blink_l < 0.25:
            return "😉", "Wink (Right Eye)", blink_r
        if blink_l > 0.65 and blink_r > 0.65:
            return "😌", "Eyes Closed / Blinking", (blink_l + blink_r) / 2.0

        if smile > 0.40:
            return "😄", "Smiling / Joyful", smile
        if smile > 0.20 and jaw_open > 0.25:
            return "😃", "Laughing", (smile + jaw_open) / 2.0

        if jaw_open > 0.45 and brow_up > 0.25:
            return "😲", "Surprised / Shocked", (jaw_open + brow_up) / 2.0
        if jaw_open > 0.20:
            return "🗣️", "Speaking / Talking", jaw_open

        if brow_down > 0.40:
            return "😠", "Frowning / Focused", brow_down
        if pucker > 0.40:
            return "😙", "Pucker / Whistling", pucker
        if cheek_puff > 0.40:
            return "🐡", "Cheek Puff", cheek_puff
        if brow_up > 0.35:
            return "🤨", "Raised Eyebrows", brow_up

        return "😐", "Neutral Face", 1.0 - max(smile, jaw_open, brow_up, brow_down)

    @staticmethod
    def detect_gaze_direction(bs: Dict[str, float]) -> str:
        """Determines eye gaze direction from blendshapes."""
        look_left = (bs.get("eyeLookOutLeft", 0.0) + bs.get("eyeLookInRight", 0.0)) / 2.0
        look_right = (bs.get("eyeLookInLeft", 0.0) + bs.get("eyeLookOutRight", 0.0)) / 2.0
        look_up = (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)) / 2.0
        look_down = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2.0

        max_gaze = max(look_left, look_right, look_up, look_down)
        if max_gaze < 0.30:
            return "👀 Center Forward"
        if look_left == max_gaze:
            return "👈 Looking Left"
        if look_right == max_gaze:
            return "👉 Looking Right"
        if look_up == max_gaze:
            return "👆 Looking Up"
        return "👇 Looking Down"


# ---------------------------------------------------------------------------
# Professional Face Renderer (Antialiased, No mp.solutions)
# ---------------------------------------------------------------------------
class FaceDetailRenderer:
    """Renders 478 landmarks, contours, irises, and 3D pose axes on OpenCV frames."""

    @staticmethod
    def draw_path(img_bgr: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int = 2):
        """Draws connected line strips."""
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            cv2.line(img_bgr, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_pose_axes(
        img_bgr: np.ndarray,
        nose_pt: Tuple[int, int],
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
        axis_length: int = 60,
    ):
        """Draws 3D coordinate orientation axes (X=Red/Pitch, Y=Green/Yaw, Z=Blue/Roll)."""
        x0, y0 = nose_pt

        # Convert Euler degrees to radians
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        roll = math.radians(roll_deg)

        # X-axis (Pitch / Down-Up) in image projection
        x_end = int(x0 + axis_length * (math.cos(yaw) * math.cos(roll)))
        y_end = int(y0 + axis_length * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)))
        cv2.line(img_bgr, (x0, y0), (x_end, y_end), (0, 0, 255), 2, cv2.LINE_AA)  # Red = X

        # Y-axis (Yaw / Left-Right)
        x_end_y = int(x0 - axis_length * (math.cos(yaw) * math.sin(roll)))
        y_end_y = int(y0 + axis_length * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)))
        cv2.line(img_bgr, (x0, y0), (x_end_y, y_end_y), (0, 255, 0), 2, cv2.LINE_AA)  # Green = Y

        # Z-axis (Roll / Camera Normal)
        x_end_z = int(x0 + axis_length * math.sin(yaw))
        y_end_z = int(y0 - axis_length * math.sin(pitch) * math.cos(yaw))
        cv2.line(img_bgr, (x0, y0), (x_end_z, y_end_z), (255, 120, 0), 2, cv2.LINE_AA)  # Blue = Z

        cv2.circle(img_bgr, (x0, y0), 4, (255, 255, 255), -1, cv2.LINE_AA)

    @staticmethod
    def render_face(
        img_bgr: np.ndarray,
        detection_result,
        show_contours: bool = True,
        show_irises: bool = True,
        show_mesh: bool = False,
        show_pose_axis: bool = True,
        show_hud: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Renders face details onto the BGR image and returns extracted telemetry.
        """
        telemetry_list: List[Dict[str, Any]] = []
        if detection_result is None or not detection_result.face_landmarks:
            return telemetry_list

        h, w, _ = img_bgr.shape
        num_faces = len(detection_result.face_landmarks)

        for i in range(num_faces):
            landmarks = detection_result.face_landmarks[i]
            
            # Convert normalized landmarks to pixel coordinates
            pixel_pts: List[Tuple[int, int]] = []
            xs: List[int] = []
            ys: List[int] = []
            for lm in landmarks:
                px = max(0, min(w - 1, int(lm.x * w)))
                py = max(0, min(h - 1, int(lm.y * h)))
                pixel_pts.append((px, py))
                xs.append(px)
                ys.append(py)

            # Extract Blendshapes
            bs_dict: Dict[str, float] = {}
            if i < len(detection_result.face_blendshapes):
                bs_dict = FaceTelemetryExtractor.parse_blendshapes(detection_result.face_blendshapes[i])

            # Extract 3D Transformation Matrix & Euler Head Pose
            matrix = None
            if i < len(detection_result.facial_transformation_matrixes):
                matrix = detection_result.facial_transformation_matrixes[i]
            pitch_deg, yaw_deg, roll_deg = FaceTelemetryExtractor.compute_head_pose_from_matrix(matrix)

            # Expression classification
            emoji, mood_name, mood_conf = FaceTelemetryExtractor.detect_facial_expression(bs_dict)
            gaze_dir = FaceTelemetryExtractor.detect_gaze_direction(bs_dict)

            # Iris positions & Inter-Pupillary Distance (IPD)
            left_iris_pos = pixel_pts[LEFT_IRIS_CENTER] if len(pixel_pts) > LEFT_IRIS_CENTER else (0, 0)
            right_iris_pos = pixel_pts[RIGHT_IRIS_CENTER] if len(pixel_pts) > RIGHT_IRIS_CENTER else (0, 0)
            ipd_px = int(math.hypot(right_iris_pos[0] - left_iris_pos[0], right_iris_pos[1] - left_iris_pos[1]))

            # Key metric summary for UI cards
            telemetry_list.append({
                "face_idx": i,
                "emoji": emoji,
                "expression": mood_name,
                "expression_conf": mood_conf,
                "gaze": gaze_dir,
                "smile_pct": int(((bs_dict.get("mouthSmileLeft", 0) + bs_dict.get("mouthSmileRight", 0)) / 2.0) * 100),
                "blink_l_pct": int(bs_dict.get("eyeBlinkLeft", 0) * 100),
                "blink_r_pct": int(bs_dict.get("eyeBlinkRight", 0) * 100),
                "jaw_open_pct": int(bs_dict.get("jawOpen", 0) * 100),
                "brow_raise_pct": int(bs_dict.get("browOuterUpLeft", 0) * 100),
                "pitch": pitch_deg,
                "yaw": yaw_deg,
                "roll": roll_deg,
                "left_iris": left_iris_pos,
                "right_iris": right_iris_pos,
                "ipd_px": ipd_px,
                "total_landmarks": len(landmarks),
            })

            # 1. Draw Dense Wireframe Mesh (Tessellation) if enabled
            if show_mesh and len(pixel_pts) >= 468:
                for idx1, idx2 in TESSELLATION_KEY_CONNECTIONS:
                    if idx1 < len(pixel_pts) and idx2 < len(pixel_pts):
                        cv2.line(img_bgr, pixel_pts[idx1], pixel_pts[idx2], CLR_MESH, 1, cv2.LINE_AA)

            # 2. Draw Facial Contours if enabled
            if show_contours and len(pixel_pts) >= 468:
                # Face Oval / Jawline
                oval_pts = [pixel_pts[idx] for idx in FACIAL_OVAL if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, oval_pts, CLR_OVAL, 2)

                # Eyebrows
                lb_pts = [pixel_pts[idx] for idx in LEFT_EYEBROW if idx < len(pixel_pts)]
                rb_pts = [pixel_pts[idx] for idx in RIGHT_EYEBROW if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, lb_pts, CLR_BROWS, 2)
                FaceDetailRenderer.draw_path(img_bgr, rb_pts, CLR_BROWS, 2)

                # Eye Contours
                le_pts = [pixel_pts[idx] for idx in LEFT_EYE if idx < len(pixel_pts)]
                re_pts = [pixel_pts[idx] for idx in RIGHT_EYE if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, le_pts, CLR_EYES, 2)
                FaceDetailRenderer.draw_path(img_bgr, re_pts, CLR_EYES, 2)

                # Nose
                nose_pts = [pixel_pts[idx] for idx in NOSE_BRIDGE if idx < len(pixel_pts)]
                nose_bot = [pixel_pts[idx] for idx in NOSE_BOTTOM if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, nose_pts, CLR_NOSE, 2)
                FaceDetailRenderer.draw_path(img_bgr, nose_bot, CLR_NOSE, 2)

                # Lips
                lip_out_pts = [pixel_pts[idx] for idx in LIPS_OUTER if idx < len(pixel_pts)]
                lip_in_pts = [pixel_pts[idx] for idx in LIPS_INNER if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, lip_out_pts, CLR_LIPS, 2)
                FaceDetailRenderer.draw_path(img_bgr, lip_in_pts, CLR_LIPS, 1)

            # 3. Draw Precision Irises & Pupils (468-477)
            if show_irises and len(pixel_pts) >= 478:
                # Left Iris
                li_pts = [pixel_pts[idx] for idx in LEFT_IRIS_CONTOUR if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, li_pts, CLR_IRIS, 2)
                cv2.circle(img_bgr, left_iris_pos, 3, CLR_PUPIL, -1, cv2.LINE_AA)

                # Right Iris
                ri_pts = [pixel_pts[idx] for idx in RIGHT_IRIS_CONTOUR if idx < len(pixel_pts)]
                FaceDetailRenderer.draw_path(img_bgr, ri_pts, CLR_IRIS, 2)
                cv2.circle(img_bgr, right_iris_pos, 3, CLR_PUPIL, -1, cv2.LINE_AA)

            # 4. Draw 3D Head Pose Axes on Nose Tip
            if show_pose_axis and len(pixel_pts) > 1:
                nose_tip = pixel_pts[1]  # Landmark 1 is the nose tip
                FaceDetailRenderer.draw_pose_axes(img_bgr, nose_tip, pitch_deg, yaw_deg, roll_deg, axis_length=50)

            # 5. Cybernetic Bounding Box & HUD Label
            if show_hud and xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                pad_x = int((max_x - min_x) * 0.08) + 8
                pad_y = int((max_y - min_y) * 0.08) + 8
                box_x1 = max(0, min_x - pad_x)
                box_y1 = max(0, min_y - pad_y)
                box_x2 = min(w - 1, max_x + pad_x)
                box_y2 = min(h - 1, max_y + pad_y)

                # Modern corner brackets
                bracket_len = min(20, (box_x2 - box_x1) // 5, (box_y2 - box_y1) // 5)
                bracket_clr = (80, 255, 120)
                
                cv2.line(img_bgr, (box_x1, box_y1), (box_x1 + bracket_len, box_y1), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x1, box_y1), (box_x1, box_y1 + bracket_len), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y1), (box_x2 - bracket_len, box_y1), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y1), (box_x2, box_y1 + bracket_len), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x1, box_y2), (box_x1 + bracket_len, box_y2), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x1, box_y2), (box_x1, box_y2 - bracket_len), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y2), (box_x2 - bracket_len, box_y2), bracket_clr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y2), (box_x2, box_y2 - bracket_len), bracket_clr, 2, cv2.LINE_AA)

                # Floating pill tag above face
                tag_text = f"Face #{i+1}: {mood_name} | Pose ({int(pitch_deg)}°, {int(yaw_deg)}°, {int(roll_deg)}°)"
                tag_y = max(24, box_y1 - 8)
                font_scale = 0.48
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                pill_x1 = box_x1
                pill_y1 = tag_y - text_h - 6
                pill_x2 = box_x1 + text_w + 14
                pill_y2 = tag_y + baseline + 2

                overlay = img_bgr.copy()
                cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (18, 22, 30), -1)
                cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
                cv2.rectangle(img_bgr, (pill_x1, pill_y1), (pill_x2, pill_y2), bracket_clr, 1, cv2.LINE_AA)

                cv2.putText(
                    img_bgr,
                    tag_text,
                    (pill_x1 + 7, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return telemetry_list


# ---------------------------------------------------------------------------
# Modern Tkinter Studio GUI Application
# ---------------------------------------------------------------------------
class FaceVisionStudioApp:
    """Main Desktop UI Application for Detailed Face Analysis."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Vision Studio — 478D Detail & Blendshape Analyzer")
        self.root.geometry("1280x850")
        self.root.minsize(1080, 750)
        self.root.configure(bg="#0d1117")

        # Application state
        self.is_running = False
        self.camera_index = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Settings variables
        self.var_mirror = tk.BooleanVar(value=True)
        self.var_contours = tk.BooleanVar(value=True)
        self.var_irises = tk.BooleanVar(value=True)
        self.var_mesh = tk.BooleanVar(value=False)
        self.var_pose = tk.BooleanVar(value=True)
        self.var_hud = tk.BooleanVar(value=True)
        self.var_confidence = tk.DoubleVar(value=0.5)
        self.var_cam_idx = tk.IntVar(value=0)

        # Performance counters
        self.fps = 0.0
        self.latency_ms = 0.0

        # Thread-safe queue
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Initialize MediaPipe Face Engine
        self._init_mediapipe_face_engine()

        # Build GUI Layout
        self._build_ui()

        # Start Camera Stream
        self.start_camera()

        # Periodic GUI update tick
        self.root.after(15, self._gui_update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_mediapipe_face_engine(self):
        """Initializes model downloader and FaceLandmarker engine."""
        try:
            model_path = ensure_face_model(MODEL_FILENAME)
            self.engine = MediaPipeFaceEngine(
                model_path=model_path,
                num_faces=2,
                min_confidence=self.var_confidence.get(),
            )
            self.model_status_text = "Model: face_landmarker.task (478 Landmarks & 52 Blendshapes)"
        except Exception as e:
            messagebox.showerror("Model Init Error", f"Failed to initialize Face Landmarker:\n{e}")
            sys.exit(1)

    def _build_ui(self):
        """Builds modern dark-themed studio widgets."""
        # Header Bar
        header_frame = tk.Frame(self.root, bg="#161b22", height=60, highlightthickness=1, highlightbackground="#30363d")
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="✨ FACE VISION STUDIO",
            font=("Segoe UI", 16, "bold"),
            fg="#58a6ff",
            bg="#161b22",
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="• 478 3D Landmarks & 52 Blendshapes (Python 3.14 Ready)",
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

        video_header = tk.Frame(video_col, bg="#1c2128", height=36)
        video_header.pack(fill=tk.X, side=tk.TOP)
        video_header.pack_propagate(False)

        tk.Label(
            video_header,
            text="📹 HIGH-PRECISION FACIAL FEED",
            font=("Segoe UI", 10, "bold"),
            fg="#c9d1d9",
            bg="#1c2128",
        ).pack(side=tk.LEFT, padx=14, pady=8)

        self.lbl_stream_status = tk.Label(
            video_header,
            text="● STREAM ACTIVE",
            font=("Segoe UI", 9, "bold"),
            fg="#3fb950",
            bg="#1c2128",
        )
        self.lbl_stream_status.pack(side=tk.RIGHT, padx=14, pady=8)

        self.lbl_video = tk.Label(video_col, bg="#080a0f", text="Starting camera...", fg="#8b949e", font=("Segoe UI", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Right Column: Sidebar
        sidebar_col = tk.Frame(body_frame, bg="#0d1117", width=440)
        sidebar_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        sidebar_col.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_col, bg="#0d1117", highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar_col, orient="vertical", command=sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(sidebar_canvas, bg="#0d1117")

        self.sidebar_content.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", width=420)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build Sidebar Panels
        self._build_expression_telemetry_section()
        self._build_head_pose_section()
        self._build_iris_geometric_section()
        self._build_controls_section()

    def _build_expression_telemetry_section(self):
        """Builds expression & 52-blendshape meters."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  🎭 FACIAL EXPRESSION & BLENDSHAPES  ",
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

        # Primary Mood / Expression Card
        mood_card = tk.Frame(pad, bg="#21262d", padx=8, pady=6, highlightthickness=1, highlightbackground="#30363d")
        mood_card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_mood_icon = tk.Label(mood_card, text="😐", font=("Segoe UI Emoji", 20), bg="#21262d")
        self.lbl_mood_icon.pack(side=tk.LEFT, padx=(2, 8))

        m_meta = tk.Frame(mood_card, bg="#21262d")
        m_meta.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_mood_name = tk.Label(m_meta, text="Neutral Face", font=("Segoe UI", 11, "bold"), fg="#c9d1d9", bg="#21262d", anchor="w")
        self.lbl_mood_name.pack(fill=tk.X)

        self.lbl_gaze = tk.Label(m_meta, text="Gaze: Center Forward", font=("Segoe UI", 8), fg="#8b949e", bg="#21262d", anchor="w")
        self.lbl_gaze.pack(fill=tk.X)

        # Blendshape Live Progress Bars
        self.meter_smile = self._create_meter_row(pad, "Smile Intensity", "#3fb950")
        self.meter_blink_l = self._create_meter_row(pad, "Left Eye Blink", "#58a6ff")
        self.meter_blink_r = self._create_meter_row(pad, "Right Eye Blink", "#58a6ff")
        self.meter_jaw = self._create_meter_row(pad, "Jaw Openness (Speech)", "#d29922")
        self.meter_brow = self._create_meter_row(pad, "Brow Elevation", "#bc8cff")

    def _create_meter_row(self, parent: tk.Widget, label_text: str, bar_color: str) -> Dict[str, Any]:
        """Creates a meter row with label, progress bar, and percentage text."""
        row = tk.Frame(parent, bg="#161b22")
        row.pack(fill=tk.X, pady=2)

        lbl_title = tk.Label(row, text=label_text, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", width=18, anchor="w")
        lbl_title.pack(side=tk.LEFT)

        # Custom canvas progress bar
        canvas = tk.Canvas(row, bg="#21262d", height=10, width=130, highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=6)
        bar_id = canvas.create_rectangle(0, 0, 0, 10, fill=bar_color, width=0)

        lbl_pct = tk.Label(row, text="0%", font=("Segoe UI", 8, "bold"), fg="#8b949e", bg="#161b22", width=5, anchor="e")
        lbl_pct.pack(side=tk.LEFT)

        return {"canvas": canvas, "bar": bar_id, "pct_lbl": lbl_pct, "color": bar_color}

    def _update_meter(self, meter: Dict[str, Any], pct_val: int):
        """Updates canvas meter fill width."""
        w = max(0, min(130, int(130 * (pct_val / 100.0))))
        meter["canvas"].coords(meter["bar"], 0, 0, w, 10)
        meter["pct_lbl"].config(text=f"{pct_val}%", fg="#3fb950" if pct_val > 50 else "#8b949e")

    def _build_head_pose_section(self):
        """Builds 3D head pose orientation telemetry (Pitch, Yaw, Roll)."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  📐 3D HEAD POSE (EULER ANGLES)  ",
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

        row = tk.Frame(pad, bg="#161b22")
        row.pack(fill=tk.X)

        # Pitch Badge
        self.lbl_pitch = self._create_pose_badge(row, "Pitch (Up/Down)", "0.0°")
        self.lbl_yaw = self._create_pose_badge(row, "Yaw (Turn L/R)", "0.0°")
        self.lbl_roll = self._create_pose_badge(row, "Roll (Tilt)", "0.0°")

    def _create_pose_badge(self, parent: tk.Widget, title: str, init_val: str) -> tk.Label:
        """Creates a modern styled numeric angle badge."""
        f = tk.Frame(parent, bg="#21262d", padx=6, pady=4, highlightthickness=1, highlightbackground="#30363d")
        f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Label(f, text=title, font=("Segoe UI", 7), fg="#8b949e", bg="#21262d").pack()
        lbl_val = tk.Label(f, text=init_val, font=("Consolas", 10, "bold"), fg="#58a6ff", bg="#21262d")
        lbl_val.pack()
        return lbl_val

    def _build_iris_geometric_section(self):
        """Builds iris and pupil tracking metrics."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  👁️ IRIS & GEOMETRIC DETAIL  ",
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

        self.lbl_iris_coords = tk.Label(
            pad,
            text="Left Iris: (-, -) | Right Iris: (-, -)",
            font=("Consolas", 8),
            fg="#c9d1d9",
            bg="#161b22",
            anchor="w",
        )
        self.lbl_iris_coords.pack(fill=tk.X, pady=1)

        self.lbl_ipd = tk.Label(
            pad,
            text="Inter-Pupillary Distance (IPD): -- px | 478 3D Joints",
            font=("Consolas", 8),
            fg="#8b949e",
            bg="#161b22",
            anchor="w",
        )
        self.lbl_ipd.pack(fill=tk.X, pady=1)

    def _build_controls_section(self):
        """Builds camera controls and visualization toggles."""
        ctrl = tk.LabelFrame(
            self.sidebar_content,
            text="  ⚙️ CONTROLS & VISUALIZATION  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        ctrl.pack(fill=tk.X, pady=(0, 10), padx=4)

        pad = tk.Frame(ctrl, bg="#161b22", padx=10, pady=10)
        pad.pack(fill=tk.X)

        # Button Row
        btn_row = tk.Frame(pad, bg="#161b22")
        btn_row.pack(fill=tk.X, pady=(0, 8))

        self.btn_toggle_cam = tk.Button(
            btn_row,
            text="⏹ Stop Camera",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#da3633",
            activebackground="#b62324",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=5,
            command=self.toggle_camera,
            cursor="hand2",
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        # Camera Selector
        cam_box = tk.Frame(btn_row, bg="#21262d", padx=4, pady=2)
        cam_box.pack(side=tk.RIGHT)
        tk.Label(cam_box, text="Cam #", font=("Segoe UI", 8), fg="#c9d1d9", bg="#21262d").pack(side=tk.LEFT)
        tk.Spinbox(
            cam_box,
            from_=0,
            to=5,
            width=2,
            textvariable=self.var_cam_idx,
            font=("Segoe UI", 9, "bold"),
            bg="#161b22",
            fg="#58a6ff",
            buttonbackground="#30363d",
            command=self.switch_camera_device,
        ).pack(side=tk.LEFT, padx=2)

        # Checkboxes
        grid = tk.Frame(pad, bg="#161b22")
        grid.pack(fill=tk.X, pady=(0, 8))

        tk.Checkbutton(grid, text="Mirror View", variable=self.var_mirror, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=0, sticky="w")
        tk.Checkbutton(grid, text="Facial Contours", variable=self.var_contours, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=1, sticky="w", padx=10)
        tk.Checkbutton(grid, text="Iris & Pupils", variable=self.var_irises, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(grid, text="3D Pose Axis", variable=self.var_pose, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=1, sticky="w", padx=10)
        tk.Checkbutton(grid, text="468 Mesh Wireframe", variable=self.var_mesh, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=2, column=0, sticky="w")
        tk.Checkbutton(grid, text="HUD Bounding Box", variable=self.var_hud, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=2, column=1, sticky="w", padx=10)

        # Confidence Slider
        sl_box = tk.Frame(pad, bg="#161b22")
        sl_box.pack(fill=tk.X)

        lbl_sl = tk.Label(sl_box, text="Min Detection Confidence: 0.50", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22")
        lbl_sl.pack(anchor="w")

        def _on_conf_change(val):
            v = float(val)
            lbl_sl.config(text=f"Min Detection Confidence: {v:.2f}")
            self.engine.rebuild_landmarker(min_confidence=v)

        tk.Scale(
            sl_box,
            from_=0.1,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.var_confidence,
            command=_on_conf_change,
            bg="#161b22",
            fg="#58a6ff",
            troughcolor="#21262d",
            highlightthickness=0,
            showvalue=False,
        ).pack(fill=tk.X, pady=(2, 0))

    # -----------------------------------------------------------------------
    # Camera Stream & Background Capture Thread
    # -----------------------------------------------------------------------
    def start_camera(self):
        """Starts the video capture worker thread."""
        if self.is_running:
            return

        self.camera_index = self.var_cam_idx.get()
        self.stop_event.clear()
        self.is_running = True

        self.btn_toggle_cam.config(text="⏹ Stop Camera", bg="#da3633", activebackground="#b62324")
        self.lbl_stream_status.config(text="● STREAM ACTIVE", fg="#3fb950")

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def stop_camera(self):
        """Stops the video capture worker thread."""
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

        self.btn_toggle_cam.config(text="▶ Start Camera", bg="#238636", activebackground="#2ea043")
        self.lbl_stream_status.config(text="○ CAMERA STOPPED", fg="#8b949e")
        self.lbl_video.config(image="", text="Camera is paused.\nClick 'Start Camera' to resume.", fg="#8b949e")
        self._reset_telemetry()

    def toggle_camera(self):
        """Toggles camera on or off."""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def switch_camera_device(self):
        """Switches to the selected camera index."""
        new_idx = self.var_cam_idx.get()
        if new_idx != self.camera_index:
            was_running = self.is_running
            self.stop_camera()
            self.camera_index = new_idx
            if was_running:
                self.start_camera()

    def _open_capture_device(self, index: int) -> Optional[cv2.VideoCapture]:
        """Tries multiple capture backends to open the camera safely."""
        backends = []
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
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
                if cap is not None:
                    cap.release()
            except Exception as e:
                print(f"[Camera] Backend {backend} failed for index {index}: {e}")

        try:
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass

        return None

    def _capture_worker(self):
        """Background thread for camera capture, inference, and rendering."""
        print(f"[Camera] Initializing VideoCapture device index {self.camera_index}...")
        self.cap = self._open_capture_device(self.camera_index)

        if self.cap is None or not self.cap.isOpened():
            print(f"[Camera] Could not open camera device at index {self.camera_index}.")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                f"Camera #{self.camera_index} not detected or unavailable",
                (35, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (100, 140, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                placeholder,
                "Select another camera index in the Control Panel",
                (50, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 170, 190),
                1,
                cv2.LINE_AA,
            )
            display_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((display_rgb, []))
            self.is_running = False
            return

        print("[Camera] VideoCapture opened successfully. Beginning FaceLandmarker loop.")
        fps_frame_counter = 0
        fps_timer = time.time()

        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[Camera] Error reading frame: {e}")
                time.sleep(0.05)
                continue

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.var_mirror.get():
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()

            # Convert BGR to RGB for MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform Face Landmark & Blendshape Detection
            try:
                detection_result = self.engine.detect_rgb_frame(rgb_frame)
            except Exception as e:
                print(f"[MediaPipe Tasks] Inference exception: {e}")
                detection_result = None

            t_infer = (time.perf_counter() - t0) * 1000.0

            # Render Skeletons & HUD directly onto the frame (BGR)
            telemetry = FaceDetailRenderer.render_face(
                frame,
                detection_result,
                show_contours=self.var_contours.get(),
                show_irises=self.var_irises.get(),
                show_mesh=self.var_mesh.get(),
                show_pose_axis=self.var_pose.get(),
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

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[Camera] Capture worker thread exited cleanly.")

    # -----------------------------------------------------------------------
    # GUI Periodic Update Loop
    # -----------------------------------------------------------------------
    def _gui_update_loop(self):
        """Pulls latest frame from queue and updates Tkinter widgets."""
        try:
            if not self.frame_queue.empty():
                display_rgb, telemetry = self.frame_queue.get_nowait()

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

                self._update_telemetry(telemetry)

        except Exception:
            pass

        self.root.after(16, self._gui_update_loop)

    def _update_telemetry(self, telemetry: List[Dict[str, Any]]):
        """Updates GUI telemetry cards and meters."""
        if telemetry:
            data = telemetry[0]  # Primary face

            # Mood Card
            self.lbl_mood_icon.config(text=data["emoji"])
            self.lbl_mood_name.config(text=f"{data['expression']} ({int(data['expression_conf']*100)}%)")
            self.lbl_gaze.config(text=f"Gaze: {data['gaze']}")

            # Blendshape Meters
            self._update_meter(self.meter_smile, data["smile_pct"])
            self._update_meter(self.meter_blink_l, data["blink_l_pct"])
            self._update_meter(self.meter_blink_r, data["blink_r_pct"])
            self._update_meter(self.meter_jaw, data["jaw_open_pct"])
            self._update_meter(self.meter_brow, data["brow_raise_pct"])

            # 3D Head Pose
            self.lbl_pitch.config(text=f"{data['pitch']:+.1f}°")
            self.lbl_yaw.config(text=f"{data['yaw']:+.1f}°")
            self.lbl_roll.config(text=f"{data['roll']:+.1f}°")

            # Iris Coordinates & IPD
            lx, ly = data["left_iris"]
            rx, ry = data["right_iris"]
            self.lbl_iris_coords.config(text=f"Left Iris: ({lx}, {ly}) | Right Iris: ({rx}, {ry})")
            self.lbl_ipd.config(text=f"Inter-Pupillary Distance: {data['ipd_px']} px | 478 3D Joints")
        else:
            self._reset_telemetry()

    def _reset_telemetry(self):
        """Resets all telemetry displays to neutral/idle."""
        self.lbl_mood_icon.config(text="😐")
        self.lbl_mood_name.config(text="No Face Detected")
        self.lbl_gaze.config(text="Gaze: --")
        self._update_meter(self.meter_smile, 0)
        self._update_meter(self.meter_blink_l, 0)
        self._update_meter(self.meter_blink_r, 0)
        self._update_meter(self.meter_jaw, 0)
        self._update_meter(self.meter_brow, 0)
        self.lbl_pitch.config(text="0.0°")
        self.lbl_yaw.config(text="0.0°")
        self.lbl_roll.config(text="0.0°")
        self.lbl_iris_coords.config(text="Left Iris: (-, -) | Right Iris: (-, -)")
        self.lbl_ipd.config(text="Inter-Pupillary Distance: -- px | 0 Joints")

    def on_closing(self):
        """Gracefully shuts down resources and closes window."""
        print("[App] Closing Face Vision Studio...")
        self.stop_camera()
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
    print("[FACE VISION STUDIO] Professional 478D Landmark & Blendshape Tracker")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"MediaPipe Version: {mp.__version__}")
    print("Tasks API Vision: FaceLandmarker (478 Landmarks + 52 Blendshapes + 3D Pose)")
    print("=" * 65)

    root = tk.Tk()
    app = FaceVisionStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
