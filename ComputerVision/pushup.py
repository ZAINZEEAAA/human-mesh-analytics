"""
pushup.py - AI Pushup Counter & Form Analyzer using Modern MediaPipe Tasks API & Tkinter

Features:
- Dual Input Modes: Seamlessly processes both Video Files (with seek bar & playback controls) and Live Webcams.
- Built strictly with the modern MediaPipe Tasks API (mediapipe.tasks.python.vision.PoseLandmarker).
- Absolutely NO deprecated mp.solutions used.
- Auto-downloads the official Google MediaPipe pose landmarker model if not present.
- Biomechanical Angle & Depth Tracking:
  - Left & Right Elbow angles (Shoulder-Elbow-Wrist).
  - Back & Hip alignment angle (Shoulder-Hip-Ankle) for real-time posture analysis.
  - Continuous 0-100% rep depth percentage meter.
- Intelligent Hysteresis State Machine:
  - Accurately tracks UP -> GOING_DOWN -> DOWN -> GOING_UP -> UP transitions.
  - Distinguishes Good Reps vs. Bad Form Reps (sagging back, incomplete depth).
  - Real-time AI form coach feedback banner ("Perfect Form", "Go Lower", "Straighten Back").
  - Computes Cadence (Reps Per Minute), Total Workout Time, and Calories burned estimate.
- Premium Tkinter dark-themed studio UI with video playback scrubber, live angle overlays, and interactive sliders.
- Threaded video pipeline for smooth 60 FPS UI performance.
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
from tkinter import ttk, messagebox, filedialog

# MediaPipe modern Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Constants & Model Configuration
# ---------------------------------------------------------------------------
MODEL_FILENAME = "pose_landmarker_full.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
]

# 33 Pose Landmark Topology Connections
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Head / Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]


# ---------------------------------------------------------------------------
# Model Downloader Helper
# ---------------------------------------------------------------------------
def ensure_pose_model(model_path: str = MODEL_FILENAME) -> str:
    """Ensures the MediaPipe pose landmarker task model is downloaded."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:
        return model_path

    print(f"[MediaPipe Tasks] Pose Model '{model_path}' not found locally. Downloading...")
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
        f"Unable to download MediaPipe pose model '{model_path}'. "
        f"Please check your internet connection or manually download the model file."
    )


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker Engine (Modern Tasks API)
# ---------------------------------------------------------------------------
class MediaPipePoseEngine:
    """Wrapper around mediapipe.tasks.python.vision.PoseLandmarker."""

    def __init__(self, model_path: str, min_confidence: float = 0.5):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.landmarker: Optional[vision.PoseLandmarker] = None
        self._lock = threading.Lock()
        self.rebuild_landmarker()

    def rebuild_landmarker(self, min_confidence: Optional[float] = None):
        """Re-initializes PoseLandmarker with updated confidence threshold."""
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
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_confidence,
                min_pose_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                output_segmentation_masks=False,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect_rgb_frame(self, rgb_frame: np.ndarray):
        """Runs pose estimation synchronously on an RGB numpy image."""
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
# Biomechanical Pushup Rep & Form Analyzer
# ---------------------------------------------------------------------------
class PushupFormAnalyzer:
    """Calculates biomechanical joint angles, depth, and tracks rep state transitions."""

    def __init__(self, up_angle: float = 155.0, down_angle: float = 90.0):
        self.up_angle = up_angle
        self.down_angle = down_angle

        # Rep counters
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0

        # State machine: 'UP', 'GOING_DOWN', 'DOWN', 'GOING_UP'
        self.state = "UP"
        self.rep_min_elbow_angle = 180.0
        self.rep_min_back_angle = 180.0
        self.rep_max_back_angle = 0.0

        # Coaching Feedback Banner
        self.feedback_message = "Ready - Assume Pushup Position"
        self.feedback_type = "NEUTRAL"  # 'SUCCESS', 'WARNING', 'ERROR', 'NEUTRAL'

        # Timers & Pace
        self.start_time = time.time()
        self.rep_timestamps: List[float] = []

    def reset_counters(self):
        """Resets all rep counters and metrics."""
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.state = "UP"
        self.rep_min_elbow_angle = 180.0
        self.rep_min_back_angle = 180.0
        self.rep_max_back_angle = 0.0
        self.feedback_message = "Counter Reset - Start Pushups"
        self.feedback_type = "NEUTRAL"
        self.start_time = time.time()
        self.rep_timestamps.clear()

    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculates interior angle (in degrees) at vertex b formed by points a-b-c."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])

        if mag_ba * mag_bc == 0:
            return 180.0

        cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cosine))

    def process_pose(self, landmarks_pixel: List[Tuple[int, int]], visibility_list: List[float]) -> Dict[str, Any]:
        """
        Computes elbow & back angles, updates rep state machine, and evaluates form.
        """
        if len(landmarks_pixel) < 33:
            return {
                "detected": False,
                "elbow_angle": 180.0,
                "back_angle": 180.0,
                "depth_pct": 0,
                "state": self.state,
                "total_reps": self.total_reps,
                "good_reps": self.good_reps,
                "bad_reps": self.bad_reps,
                "feedback": "No Pose Detected in Frame",
                "feedback_type": "NEUTRAL",
                "active_side": "None",
            }

        # Landmarks:
        # 11: Left Shoulder, 13: Left Elbow, 15: Left Wrist, 23: Left Hip, 27: Left Ankle
        # 12: Right Shoulder, 14: Right Elbow, 16: Right Wrist, 24: Right Hip, 28: Right Ankle
        l_sh, l_el, l_wr = landmarks_pixel[11], landmarks_pixel[13], landmarks_pixel[15]
        r_sh, r_el, r_wr = landmarks_pixel[12], landmarks_pixel[14], landmarks_pixel[16]

        l_hip, l_ank = landmarks_pixel[23], landmarks_pixel[27]
        r_hip, r_ank = landmarks_pixel[24], landmarks_pixel[28]

        # Calculate joint angles
        l_elbow_ang = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_ang = self.calculate_angle(r_sh, r_el, r_wr)

        l_back_ang = self.calculate_angle(l_sh, l_hip, l_ank)
        r_back_ang = self.calculate_angle(r_sh, r_hip, r_ank)

        # Visibility heuristic to pick active side
        l_vis = (visibility_list[11] + visibility_list[13] + visibility_list[15]) / 3.0 if visibility_list else 1.0
        r_vis = (visibility_list[12] + visibility_list[14] + visibility_list[16]) / 3.0 if visibility_list else 1.0

        if abs(l_vis - r_vis) > 0.15:
            if l_vis > r_vis:
                active_side = "Left"
                elbow_ang = l_elbow_ang
                back_ang = l_back_ang
                active_elbow_pt = l_el
                active_hip_pt = l_hip
            else:
                active_side = "Right"
                elbow_ang = r_elbow_ang
                back_ang = r_back_ang
                active_elbow_pt = r_el
                active_hip_pt = r_hip
        else:
            # Both sides visible -> use average
            active_side = "Both"
            elbow_ang = (l_elbow_ang + r_elbow_ang) / 2.0
            back_ang = (l_back_ang + r_back_ang) / 2.0
            active_elbow_pt = l_el if l_vis >= r_vis else r_el
            active_hip_pt = l_hip if l_vis >= r_vis else r_hip

        # Compute Depth Percentage (0% at top UP position, 100% at full DOWN depth)
        span = max(1.0, self.up_angle - self.down_angle)
        raw_depth = ((self.up_angle - elbow_ang) / span) * 100.0
        depth_pct = max(0, min(100, int(raw_depth)))

        # Update rep-tracking extremes
        self.rep_min_elbow_angle = min(self.rep_min_elbow_angle, elbow_ang)
        self.rep_min_back_angle = min(self.rep_min_back_angle, back_ang)
        self.rep_max_back_angle = max(self.rep_max_back_angle, back_ang)

        # -------------------------------------------------------------------
        # Hysteresis State Machine
        # -------------------------------------------------------------------
        if self.state == "UP":
            if depth_pct >= 25:
                self.state = "GOING_DOWN"
                self.feedback_message = "Going Down... Keep Chest Lowering"
                self.feedback_type = "NEUTRAL"
            else:
                self.feedback_message = "Ready - Push Down When Ready"
                self.feedback_type = "NEUTRAL"

        elif self.state == "GOING_DOWN":
            if depth_pct >= 90 or elbow_ang <= self.down_angle:
                self.state = "DOWN"
                self.feedback_message = "Good Depth! Now Push Up"
                self.feedback_type = "SUCCESS"
            elif depth_pct < 15:
                # Aborted downward movement before reaching bottom
                self.state = "UP"
                self.rep_min_elbow_angle = 180.0

        elif self.state == "DOWN":
            if elbow_ang > self.down_angle + 12:
                self.state = "GOING_UP"
                self.feedback_message = "Pushing Up... Full Arm Extension"
                self.feedback_type = "NEUTRAL"

        elif self.state == "GOING_UP":
            if elbow_ang >= self.up_angle - 6 or depth_pct <= 5:
                # Rep Finished!
                self.total_reps += 1
                now = time.time()
                self.rep_timestamps.append(now)

                # Form Quality Check
                is_back_straight = (150.0 <= self.rep_min_back_angle <= 190.0)
                is_full_depth = (self.rep_min_elbow_angle <= self.down_angle + 5)

                if is_back_straight and is_full_depth:
                    self.good_reps += 1
                    self.feedback_message = f"Rep #{self.total_reps}: PERFECT FORM! ✅"
                    self.feedback_type = "SUCCESS"
                elif not is_back_straight:
                    self.bad_reps += 1
                    if self.rep_min_back_angle < 150.0:
                        self.feedback_message = f"Rep #{self.total_reps}: ⚠️ Hips Sagging (Straighten Back)"
                    else:
                        self.feedback_message = f"Rep #{self.total_reps}: ⚠️ Hips Too High"
                    self.feedback_type = "WARNING"
                else:
                    self.bad_reps += 1
                    self.feedback_message = f"Rep #{self.total_reps}: ⚠️ Incomplete Depth (Go Lower)"
                    self.feedback_type = "WARNING"

                # Reset for next rep
                self.state = "UP"
                self.rep_min_elbow_angle = 180.0
                self.rep_min_back_angle = 180.0
                self.rep_max_back_angle = 0.0

        # Calculate Reps Per Minute (RPM)
        recent = [t for t in self.rep_timestamps if time.time() - t <= 60]
        rpm = len(recent)

        # Estimated Calories (approx 0.35 - 0.45 kcal per pushup)
        calories = self.total_reps * 0.40

        return {
            "detected": True,
            "elbow_angle": elbow_ang,
            "l_elbow_ang": l_elbow_ang,
            "r_elbow_ang": r_elbow_ang,
            "back_angle": back_ang,
            "depth_pct": depth_pct,
            "state": self.state,
            "total_reps": self.total_reps,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "accuracy": int((self.good_reps / max(1, self.total_reps)) * 100),
            "feedback": self.feedback_message,
            "feedback_type": self.feedback_type,
            "active_side": active_side,
            "elbow_pt": active_elbow_pt,
            "hip_pt": active_hip_pt,
            "rpm": rpm,
            "calories": calories,
        }


# ---------------------------------------------------------------------------
# High-Performance Pose & HUD Renderer (No mp.solutions)
# ---------------------------------------------------------------------------
class PushupVisualRenderer:
    """Draws skeleton, joint angle badges, depth gauges, and feedback banners."""

    @staticmethod
    def draw_skeleton_and_hud(
        img_bgr: np.ndarray,
        landmarks_pixel: List[Tuple[int, int]],
        telemetry: Dict[str, Any],
        show_skeleton: bool = True,
        show_angles: bool = True,
        show_hud: bool = True,
    ):
        """Draws pushup tracking visuals onto the OpenCV frame."""
        h, w, _ = img_bgr.shape

        if not telemetry.get("detected", False) or len(landmarks_pixel) < 33:
            return

        # Choose skeleton color based on form state
        fb_type = telemetry.get("feedback_type", "NEUTRAL")
        if fb_type == "SUCCESS":
            bone_color = (50, 255, 120)  # Neon Green
            joint_color = (255, 255, 255)
        elif fb_type == "WARNING":
            bone_color = (0, 180, 255)   # Amber / Orange
            joint_color = (0, 220, 255)
        else:
            bone_color = (255, 230, 0)   # Cyan / Electric Blue
            joint_color = (230, 240, 255)

        # 1. Draw Pose Bones
        if show_skeleton:
            for idx1, idx2 in POSE_CONNECTIONS:
                if idx1 < len(landmarks_pixel) and idx2 < len(landmarks_pixel):
                    pt1 = landmarks_pixel[idx1]
                    pt2 = landmarks_pixel[idx2]
                    cv2.line(img_bgr, pt1, pt2, bone_color, 3, cv2.LINE_AA)

            # Draw Joint Circles
            for idx, pt in enumerate(landmarks_pixel):
                if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    cv2.circle(img_bgr, pt, 6, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(img_bgr, pt, 8, bone_color, 2, cv2.LINE_AA)
                else:
                    cv2.circle(img_bgr, pt, 3, (160, 170, 200), -1, cv2.LINE_AA)

        # 2. Draw Angle Badges at Key Joints
        if show_angles:
            elbow_pt = telemetry.get("elbow_pt", (0, 0))
            hip_pt = telemetry.get("hip_pt", (0, 0))
            elbow_ang = telemetry.get("elbow_angle", 180.0)
            back_ang = telemetry.get("back_angle", 180.0)

            # Elbow Angle Badge
            if elbow_pt != (0, 0):
                ang_str = f"Elbow: {int(elbow_ang)}°"
                cv2.rectangle(img_bgr, (elbow_pt[0] - 50, elbow_pt[1] - 28), (elbow_pt[0] + 50, elbow_pt[1] - 6), (20, 24, 32), -1)
                cv2.rectangle(img_bgr, (elbow_pt[0] - 50, elbow_pt[1] - 28), (elbow_pt[0] + 50, elbow_pt[1] - 6), bone_color, 1, cv2.LINE_AA)
                cv2.putText(img_bgr, ang_str, (elbow_pt[0] - 44, elbow_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Back Alignment Angle Badge
            if hip_pt != (0, 0):
                back_str = f"Back: {int(back_ang)}°"
                back_badge_clr = (50, 255, 120) if 150 <= back_ang <= 190 else (0, 140, 255)
                cv2.rectangle(img_bgr, (hip_pt[0] - 45, hip_pt[1] - 28), (hip_pt[0] + 45, hip_pt[1] - 6), (20, 24, 32), -1)
                cv2.rectangle(img_bgr, (hip_pt[0] - 45, hip_pt[1] - 28), (hip_pt[0] + 45, hip_pt[1] - 6), back_badge_clr, 1, cv2.LINE_AA)
                cv2.putText(img_bgr, back_str, (hip_pt[0] - 40, hip_pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw On-Video Rep Counter & Depth Progress Bar (HUD)
        if show_hud:
            total_reps = telemetry.get("total_reps", 0)
            good_reps = telemetry.get("good_reps", 0)
            depth_pct = telemetry.get("depth_pct", 0)
            state = telemetry.get("state", "UP")
            feedback = telemetry.get("feedback", "")

            # Top-Left Rep Badge
            badge_w, badge_h = 220, 95
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (16, 16), (16 + badge_w, 16 + badge_h), (14, 18, 26), -1)
            cv2.addWeighted(overlay, 0.82, img_bgr, 0.18, 0, img_bgr)
            cv2.rectangle(img_bgr, (16, 16), (16 + badge_w, 16 + badge_h), (50, 60, 80), 1, cv2.LINE_AA)

            # Total Reps text
            cv2.putText(img_bgr, "PUSHUPS", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 160, 190), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, f"{total_reps}", (28, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img_bgr, f"Good: {good_reps}", (125, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(img_bgr, f"State: {state}", (125, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

            # Vertical Depth Progress Bar on Left side
            bar_x = 16 + badge_w + 14
            bar_y = 16
            bar_h = 95
            bar_w = 16
            fill_h = int((depth_pct / 100.0) * bar_h)

            cv2.rectangle(img_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 24, 34), -1)
            cv2.rectangle(img_bgr, (bar_x, bar_y + (bar_h - fill_h)), (bar_x + bar_w, bar_y + bar_h), (50, 255, 120) if depth_pct >= 85 else (0, 180, 255), -1)
            cv2.rectangle(img_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 80, 100), 1, cv2.LINE_AA)

            # Bottom Feedback Coach Banner
            banner_y = h - 45
            cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + 35), (14, 18, 26), -1)
            cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)
            banner_border = (50, 255, 120) if fb_type == "SUCCESS" else (0, 180, 255) if fb_type == "WARNING" else (70, 80, 100)
            cv2.rectangle(img_bgr, (20, banner_y), (w - 20, banner_y + 35), banner_border, 1, cv2.LINE_AA)
            cv2.putText(img_bgr, f"COACH: {feedback}", (35, banner_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Modern Tkinter Pushup Studio Application
# ---------------------------------------------------------------------------
class PushupStudioApp:
    """Main Desktop UI Application for Pushup Counting & Form Coaching."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Pushup Studio — MediaPipe Tasks API")
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
        self.var_source = tk.StringVar(value="WEBCAM")
        self.var_mirror = tk.BooleanVar(value=True)
        self.var_skeleton = tk.BooleanVar(value=True)
        self.var_angles = tk.BooleanVar(value=True)
        self.var_hud = tk.BooleanVar(value=True)
        self.var_loop_video = tk.BooleanVar(value=True)
        self.var_cam_idx = tk.IntVar(value=0)
        self.var_up_angle = tk.DoubleVar(value=155.0)
        self.var_down_angle = tk.DoubleVar(value=90.0)

        # Performance counters
        self.fps = 0.0
        self.latency_ms = 0.0

        # Thread-safe queue
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Setup MediaPipe Model & Form Analyzer
        self._init_mediapipe_pose_engine()
        self.analyzer = PushupFormAnalyzer(
            up_angle=self.var_up_angle.get(),
            down_angle=self.var_down_angle.get(),
        )

        # Build GUI Layout
        self._build_ui()

        # Start Camera Stream by default
        self.start_stream()

        # Periodic GUI update tick
        self.root.after(15, self._gui_update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_mediapipe_pose_engine(self):
        """Initializes model downloader and PoseLandmarker."""
        try:
            model_path = ensure_pose_model(MODEL_FILENAME)
            self.engine = MediaPipePoseEngine(model_path=model_path, min_confidence=0.5)
            self.model_status_text = "Model: pose_landmarker_full.task (33 3D Pose Joints)"
        except Exception as e:
            messagebox.showerror("Model Init Error", f"Failed to initialize Pose Landmarker:\n{e}")
            sys.exit(1)

    def _build_ui(self):
        """Builds dark-themed modern studio widgets."""
        # Top Header Bar
        header_frame = tk.Frame(self.root, bg="#161b22", height=60, highlightthickness=1, highlightbackground="#30363d")
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="💪 AI PUSHUP STUDIO",
            font=("Segoe UI", 16, "bold"),
            fg="#58a6ff",
            bg="#161b22",
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="• Video & Live Webcam Biomechanical Rep Counter (Python 3.14 Ready)",
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

        # Body Container (Left: Video & Playback, Right: Telemetry & Controls)
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
            text="📹 LIVE WEBCAM FEED",
            font=("Segoe UI", 10, "bold"),
            fg="#c9d1d9",
            bg="#1c2128",
        )
        self.lbl_viewport_title.pack(side=tk.LEFT, padx=14, pady=8)

        self.lbl_stream_status = tk.Label(
            video_header,
            text="● ACTIVE",
            font=("Segoe UI", 9, "bold"),
            fg="#3fb950",
            bg="#1c2128",
        )
        self.lbl_stream_status.pack(side=tk.RIGHT, padx=14, pady=8)

        # Video Canvas / Label
        self.lbl_video = tk.Label(video_col, bg="#080a0f", text="Starting stream...", fg="#8b949e", font=("Segoe UI", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # Video File Scrubber Bar (Only active in Video File mode)
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

        # Right Column: Sidebar (Rep Telemetry, Controls, Video Loader)
        sidebar_col = tk.Frame(body_frame, bg="#0d1117", width=420)
        sidebar_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        sidebar_col.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_col, bg="#0d1117", highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar_col, orient="vertical", command=sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(sidebar_canvas, bg="#0d1117")

        self.sidebar_content.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", width=400)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build Sidebar Panels
        self._build_input_source_section()
        self._build_rep_counter_section()
        self._build_biomechanics_section()
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

        # Toggle Buttons for Mode
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

        # Current File indicator label
        self.lbl_file_info = tk.Label(
            pad,
            text="Source: Built-in / USB Camera",
            font=("Segoe UI", 8),
            fg="#8b949e",
            bg="#161b22",
            anchor="w",
        )
        self.lbl_file_info.pack(fill=tk.X)

    def _build_rep_counter_section(self):
        """Builds large rep counter cards & accuracy badge."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  🏆 PUSHUP REP TELEMETRY  ",
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

        # Big Rep Display Card
        counter_card = tk.Frame(pad, bg="#21262d", padx=12, pady=10, highlightthickness=1, highlightbackground="#30363d")
        counter_card.pack(fill=tk.X, pady=(0, 8))

        # Total Reps
        col_total = tk.Frame(counter_card, bg="#21262d")
        col_total.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(col_total, text="TOTAL REPS", font=("Segoe UI", 8, "bold"), fg="#8b949e", bg="#21262d").pack(anchor="w")
        self.lbl_total_reps = tk.Label(col_total, text="0", font=("Segoe UI", 28, "bold"), fg="#58a6ff", bg="#21262d")
        self.lbl_total_reps.pack(anchor="w")

        # Good vs Bad Reps
        col_sub = tk.Frame(counter_card, bg="#21262d")
        col_sub.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_good_reps = tk.Label(col_sub, text="Good: 0 ✅", font=("Segoe UI", 10, "bold"), fg="#3fb950", bg="#21262d", anchor="e")
        self.lbl_good_reps.pack(fill=tk.X)

        self.lbl_bad_reps = tk.Label(col_sub, text="Bad: 0 ⚠️", font=("Segoe UI", 10, "bold"), fg="#f85149", bg="#21262d", anchor="e")
        self.lbl_bad_reps.pack(fill=tk.X)

        self.lbl_accuracy = tk.Label(col_sub, text="Form: 100%", font=("Segoe UI", 9), fg="#c9d1d9", bg="#21262d", anchor="e")
        self.lbl_accuracy.pack(fill=tk.X, pady=(2, 0))

        # Current Phase & Workout Stats
        stats_row = tk.Frame(pad, bg="#161b22")
        stats_row.pack(fill=tk.X, pady=(0, 6))

        self.lbl_phase_badge = tk.Label(
            stats_row,
            text="Phase: UP",
            font=("Segoe UI", 9, "bold"),
            fg="#58a6ff",
            bg="#21262d",
            padx=8,
            pady=3,
        )
        self.lbl_phase_badge.pack(side=tk.LEFT)

        self.lbl_rpm_badge = tk.Label(
            stats_row,
            text="⚡ Pace: 0 RPM",
            font=("Segoe UI", 8),
            fg="#c9d1d9",
            bg="#21262d",
            padx=8,
            pady=3,
        )
        self.lbl_rpm_badge.pack(side=tk.LEFT, padx=6)

        self.lbl_cal_badge = tk.Label(
            stats_row,
            text="🔥 0.0 kcal",
            font=("Segoe UI", 8),
            fg="#f0883e",
            bg="#21262d",
            padx=8,
            pady=3,
        )
        self.lbl_cal_badge.pack(side=tk.RIGHT)

        # Reset Reps Button
        tk.Button(
            pad,
            text="🔄 Reset Rep Counter",
            font=("Segoe UI", 8, "bold"),
            bg="#30363d",
            fg="#c9d1d9",
            activebackground="#21262d",
            relief=tk.FLAT,
            bd=0,
            pady=4,
            command=self.reset_counters,
            cursor="hand2",
        ).pack(fill=tk.X)

    def _build_biomechanics_section(self):
        """Builds joint angles & depth meters."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  📐 BIOMECHANICAL ANGLES  ",
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

        # Live Angle Readouts
        self.lbl_elbow_angle = tk.Label(pad, text="Elbow Angle: 180° (Arms Extended)", font=("Consolas", 9, "bold"), fg="#58a6ff", bg="#161b22", anchor="w")
        self.lbl_elbow_angle.pack(fill=tk.X, pady=1)

        self.lbl_back_angle = tk.Label(pad, text="Back Alignment: 180° (Straight)", font=("Consolas", 9, "bold"), fg="#3fb950", bg="#161b22", anchor="w")
        self.lbl_back_angle.pack(fill=tk.X, pady=1)

        # Pushup Depth Progress Bar
        tk.Label(pad, text="Pushup Depth Percentage:", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22", anchor="w").pack(fill=tk.X, pady=(6, 2))
        
        depth_row = tk.Frame(pad, bg="#161b22")
        depth_row.pack(fill=tk.X)

        self.depth_canvas = tk.Canvas(depth_row, bg="#21262d", height=12, highlightthickness=0)
        self.depth_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.depth_bar = self.depth_canvas.create_rectangle(0, 0, 0, 12, fill="#3fb950", width=0)

        self.lbl_depth_pct = tk.Label(depth_row, text="0%", font=("Segoe UI", 9, "bold"), fg="#c9d1d9", bg="#161b22", width=5, anchor="e")
        self.lbl_depth_pct.pack(side=tk.RIGHT)

    def _build_controls_section(self):
        """Builds controls, angle thresholds, and visual toggles."""
        sec = tk.LabelFrame(
            self.sidebar_content,
            text="  ⚙️ SETTINGS & THRESHOLDS  ",
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

        # Visual Toggles
        grid = tk.Frame(pad, bg="#161b22")
        grid.pack(fill=tk.X, pady=(0, 8))

        tk.Checkbutton(grid, text="Mirror Video", variable=self.var_mirror, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=0, sticky="w")
        tk.Checkbutton(grid, text="Pose Skeleton", variable=self.var_skeleton, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=0, column=1, sticky="w", padx=8)
        tk.Checkbutton(grid, text="Joint Angles", variable=self.var_angles, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=0, sticky="w")
        tk.Checkbutton(grid, text="On-Video HUD", variable=self.var_hud, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=1, column=1, sticky="w", padx=8)
        tk.Checkbutton(grid, text="Loop Video File", variable=self.var_loop_video, font=("Segoe UI", 8), fg="#c9d1d9", bg="#161b22", selectcolor="#21262d", activebackground="#161b22").grid(row=2, column=0, sticky="w")

        # Threshold Sliders
        lbl_down = tk.Label(pad, text=f"Down Threshold (Target Depth): {int(self.var_down_angle.get())}°", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22", anchor="w")
        lbl_down.pack(fill=tk.X, pady=(4, 0))

        def _on_down_change(val):
            v = float(val)
            lbl_down.config(text=f"Down Threshold (Target Depth): {int(v)}°")
            self.analyzer.down_angle = v

        tk.Scale(
            pad,
            from_=60,
            to=110,
            orient=tk.HORIZONTAL,
            variable=self.var_down_angle,
            command=_on_down_change,
            bg="#161b22",
            fg="#58a6ff",
            troughcolor="#21262d",
            highlightthickness=0,
            showvalue=False,
        ).pack(fill=tk.X, pady=(0, 6))

        lbl_up = tk.Label(pad, text=f"Up Threshold (Top Extension): {int(self.var_up_angle.get())}°", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22", anchor="w")
        lbl_up.pack(fill=tk.X)

        def _on_up_change(val):
            v = float(val)
            lbl_up.config(text=f"Up Threshold (Top Extension): {int(v)}°")
            self.analyzer.up_angle = v

        tk.Scale(
            pad,
            from_=140,
            to=175,
            orient=tk.HORIZONTAL,
            variable=self.var_up_angle,
            command=_on_up_change,
            bg="#161b22",
            fg="#58a6ff",
            troughcolor="#21262d",
            highlightthickness=0,
            showvalue=False,
        ).pack(fill=tk.X)

    # -----------------------------------------------------------------------
    # Source Switching & Video File Management
    # -----------------------------------------------------------------------
    def set_source_webcam(self):
        """Switches to Live Webcam mode."""
        if self.input_source_mode == "WEBCAM" and self.is_running:
            return

        self.input_source_mode = "WEBCAM"
        self.btn_mode_webcam.config(bg="#238636", fg="#ffffff")
        self.btn_mode_file.config(bg="#21262d", fg="#c9d1d9")
        self.lbl_viewport_title.config(text="📹 LIVE WEBCAM FEED")
        self.lbl_file_info.config(text="Source: Built-in / USB Camera")
        self.var_mirror.set(True)

        self.stop_stream()
        self.start_stream()

    def browse_video_file(self):
        """Opens a file picker to choose a video file for pushup analysis."""
        filepath = filedialog.askopenfilename(
            title="Select Pushup Video File",
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
        """Seeks to the frame percentage in the video file."""
        if self.input_source_mode == "VIDEO_FILE" and self.cap is not None and self.video_total_frames > 0:
            target_frame = int((float(val) / 100.0) * self.video_total_frames)
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            except Exception:
                pass

    def reset_counters(self):
        """Resets rep tracking."""
        self.analyzer.reset_counters()

    # -----------------------------------------------------------------------
    # Stream & Background Worker
    # -----------------------------------------------------------------------
    def start_stream(self):
        """Starts video or webcam capture thread."""
        if self.is_running:
            return

        self.stop_event.clear()
        self.is_running = True
        self.is_paused = False
        self.btn_play_pause.config(text="⏸ Pause", bg="#21262d")
        self.lbl_stream_status.config(text="● ACTIVE", fg="#3fb950")

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def stop_stream(self):
        """Stops video or webcam capture thread."""
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

        self.lbl_stream_status.config(text="○ STOPPED", fg="#8b949e")

    def _open_capture_source(self) -> Optional[cv2.VideoCapture]:
        """Opens either the video file or webcam safely."""
        if self.input_source_mode == "VIDEO_FILE" and self.video_filepath:
            if not os.path.exists(self.video_filepath):
                print(f"[Video] File not found: {self.video_filepath}")
                return None
            cap = cv2.VideoCapture(self.video_filepath)
            if cap and cap.isOpened():
                self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                return cap
            return None
        else:
            # Webcam Mode
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
        """Background thread for grabbing frames, running pose inference, and rendering."""
        print(f"[Stream] Initializing capture source: {self.input_source_mode}...")
        self.cap = self._open_capture_source()

        if self.cap is None or not self.cap.isOpened():
            print("[Stream] Failed to open capture source.")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "Camera Unavailable" if self.input_source_mode == "WEBCAM" else "Could not open video file"
            cv2.putText(placeholder, msg, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(placeholder, "Click 'Video File' to browse a file or check webcam", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 190), 1, cv2.LINE_AA)
            display_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((display_rgb, {}))
            self.is_running = False
            return

        print("[Stream] Capture opened successfully. Running pushup detection loop.")
        fps_frame_counter = 0
        fps_timer = time.time()

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"[Stream] Frame read exception: {e}")
                time.sleep(0.05)
                continue

            if not ret or frame is None:
                if self.input_source_mode == "VIDEO_FILE" and self.var_loop_video.get():
                    # Loop video from start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.02)
                    continue

            # Update video position
            if self.input_source_mode == "VIDEO_FILE":
                self.video_current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Mirror mode for webcam
            if self.var_mirror.get():
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()

            # Convert BGR to RGB for MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform Pose Landmark Detection
            try:
                detection_result = self.engine.detect_rgb_frame(rgb_frame)
            except Exception as e:
                print(f"[MediaPipe Tasks] Pose inference exception: {e}")
                detection_result = None

            t_infer = (time.perf_counter() - t0) * 1000.0

            # Extract Landmark Pixel Coordinates & Visibilities
            landmarks_pixel: List[Tuple[int, int]] = []
            visibility_list: List[float] = []
            h, w, _ = frame.shape

            if detection_result and detection_result.pose_landmarks:
                first_pose = detection_result.pose_landmarks[0]
                for lm in first_pose:
                    px = max(0, min(w - 1, int(lm.x * w)))
                    py = max(0, min(h - 1, int(lm.y * h)))
                    landmarks_pixel.append((px, py))
                    visibility_list.append(getattr(lm, "visibility", 1.0) or 1.0)

            # Process Pushup Biomechanics & Rep Counting
            telemetry = self.analyzer.process_pose(landmarks_pixel, visibility_list)

            # Render Skeletons, Joint Badges & On-Screen HUD
            PushupVisualRenderer.draw_skeleton_and_hud(
                frame,
                landmarks_pixel,
                telemetry,
                show_skeleton=self.var_skeleton.get(),
                show_angles=self.var_angles.get(),
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

            # Pace video playback to normal FPS if reading from video file
            if self.input_source_mode == "VIDEO_FILE" and self.video_fps > 0:
                frame_delay = 1.0 / self.video_fps
                time.sleep(max(0.005, frame_delay - (time.perf_counter() - t0)))

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[Stream] Capture worker thread exited cleanly.")

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

                # Resize image dynamically to fit video viewport
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

                # Update Telemetry Readouts
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
        """Updates GUI cards with live rep stats and angles."""
        if not telemetry:
            return

        total = telemetry.get("total_reps", 0)
        good = telemetry.get("good_reps", 0)
        bad = telemetry.get("bad_reps", 0)
        accuracy = telemetry.get("accuracy", 100)
        state = telemetry.get("state", "UP")
        elbow_ang = telemetry.get("elbow_angle", 180.0)
        back_ang = telemetry.get("back_angle", 180.0)
        depth_pct = telemetry.get("depth_pct", 0)
        rpm = telemetry.get("rpm", 0)
        calories = telemetry.get("calories", 0.0)

        # Rep Counters
        self.lbl_total_reps.config(text=str(total))
        self.lbl_good_reps.config(text=f"Good: {good} ✅")
        self.lbl_bad_reps.config(text=f"Bad: {bad} ⚠️")
        self.lbl_accuracy.config(text=f"Form: {accuracy}%")

        # Phase & Workout Badges
        self.lbl_phase_badge.config(
            text=f"Phase: {state}",
            fg="#3fb950" if state == "DOWN" else "#58a6ff" if state == "UP" else "#d29922",
        )
        self.lbl_rpm_badge.config(text=f"⚡ Pace: {rpm} RPM")
        self.lbl_cal_badge.config(text=f"🔥 {calories:.1f} kcal")

        # Angles
        self.lbl_elbow_angle.config(text=f"Elbow Angle: {int(elbow_ang)}°")
        self.lbl_back_angle.config(
            text=f"Back Alignment: {int(back_ang)}° ({'Straight' if 150 <= back_ang <= 190 else 'Check Posture'})",
            fg="#3fb950" if 150 <= back_ang <= 190 else "#f85149",
        )

        # Depth Bar
        canvas_w = max(10, self.depth_canvas.winfo_width())
        bar_w = int(canvas_w * (depth_pct / 100.0))
        self.depth_canvas.coords(self.depth_bar, 0, 0, bar_w, 12)
        self.depth_canvas.itemconfig(self.depth_bar, fill="#3fb950" if depth_pct >= 85 else "#58a6ff")
        self.lbl_depth_pct.config(text=f"{depth_pct}%")

    def on_closing(self):
        """Gracefully closes resources, background threads, and window."""
        print("[App] Closing AI Pushup Studio...")
        self.stop_stream()
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
    print("[AI PUSHUP STUDIO] Biomechanical Rep Counter & Form Analyzer")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"MediaPipe Version: {mp.__version__}")
    print("Tasks API Vision: PoseLandmarker (33 3D Joints, Dual Video & Webcam)")
    print("=" * 65)

    root = tk.Tk()
    app = PushupStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
