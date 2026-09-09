"""
hand.py - Hand Skeleton & Gesture Tracking using Modern MediaPipe Tasks API & Tkinter

Features:
- Built strictly with the modern MediaPipe Tasks API (mediapipe.tasks.python.vision.GestureRecognizer).
- Absolutely NO deprecated mp.solutions used.
- Auto-downloads the official Google MediaPipe gesture recognizer model if not present.
- Premium Tkinter dark-themed interface with live video feed, telemetry cards, and controls.
- Full 21-landmark 3D hand skeleton rendering with finger-specific neon color coding and joint glows.
- Real-time gesture classification (Open Palm, Closed Fist, Pointing Up, Victory, Thumb Up, Thumb Down, ILoveYou).
- Threaded camera pipeline with robust error handling for high FPS and non-blocking UI.
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
MODEL_FILENAME = "gesture_recognizer.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
]

# 21 Hand Landmarks Connections (Skeleton Topology)
HAND_PALM_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)
]
HAND_THUMB_CONNECTIONS = [(1, 2), (2, 3), (3, 4)]
HAND_INDEX_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
HAND_MIDDLE_CONNECTIONS = [(9, 10), (10, 11), (11, 12)]
HAND_RING_CONNECTIONS = [(13, 14), (14, 15), (15, 16)]
HAND_PINKY_CONNECTIONS = [(17, 18), (18, 19), (19, 20)]

FINGER_CONNECTIONS = [
    (HAND_PALM_CONNECTIONS, (230, 230, 245), 2),     # Palm: Soft Ice White/Silver
    (HAND_THUMB_CONNECTIONS, (0, 140, 255), 3),      # Thumb: Vivid Orange
    (HAND_INDEX_CONNECTIONS, (255, 230, 0), 3),      # Index: Electric Cyan
    (HAND_MIDDLE_CONNECTIONS, (80, 255, 120), 3),    # Middle: Neon Green
    (HAND_RING_CONNECTIONS, (0, 230, 255), 3),       # Ring: Neon Yellow/Gold
    (HAND_PINKY_CONNECTIONS, (255, 60, 200), 3),     # Pinky: Neon Magenta
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGERTIP_COLORS = [
    (0, 140, 255),    # Thumb Tip (Orange)
    (255, 230, 0),    # Index Tip (Cyan)
    (80, 255, 120),   # Middle Tip (Green)
    (0, 230, 255),    # Ring Tip (Gold)
    (255, 60, 200),   # Pinky Tip (Magenta)
]

GESTURE_EMOJIS: Dict[str, Tuple[str, str]] = {
    "None": ("❓", "Unrecognized"),
    "Unrecognized": ("❓", "Unrecognized"),
    "Closed_Fist": ("✊", "Closed Fist"),
    "Open_Palm": ("🖐️", "Open Palm"),
    "Pointing_Up": ("☝️", "Pointing Up"),
    "Thumb_Down": ("👎", "Thumb Down"),
    "Thumb_Up": ("👍", "Thumb Up"),
    "Victory": ("✌️", "Victory / Peace"),
    "ILoveYou": ("🤟", "I Love You"),
}


# ---------------------------------------------------------------------------
# Model Downloader Helper
# ---------------------------------------------------------------------------
def ensure_model_file(model_path: str = MODEL_FILENAME) -> str:
    """Ensures the MediaPipe gesture recognizer task model is downloaded."""
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
        f"Unable to download MediaPipe model '{model_path}'. "
        f"Please verify internet connectivity or manually download the model file."
    )


# ---------------------------------------------------------------------------
# Gesture Recognizer Engine (Modern Tasks API)
# ---------------------------------------------------------------------------
class MediaPipeGestureEngine:
    """Wrapper around mediapipe.tasks.python.vision.GestureRecognizer."""

    def __init__(self, model_path: str, num_hands: int = 2, min_confidence: float = 0.5):
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_confidence = min_confidence
        self.recognizer: Optional[vision.GestureRecognizer] = None
        self._lock = threading.Lock()
        self.rebuild_recognizer()

    def rebuild_recognizer(self, num_hands: Optional[int] = None, min_confidence: Optional[float] = None):
        """Re-initializes the recognizer with updated parameters."""
        with self._lock:
            if num_hands is not None:
                self.num_hands = num_hands
            if min_confidence is not None:
                self.min_confidence = min_confidence

            if self.recognizer is not None:
                try:
                    self.recognizer.close()
                except Exception:
                    pass
                self.recognizer = None

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_confidence,
                min_hand_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def process_rgb_frame(self, rgb_frame: np.ndarray):
        """Runs gesture recognition synchronously on an RGB numpy image."""
        with self._lock:
            if self.recognizer is None:
                return None
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self.recognizer.recognize(mp_image)

    def close(self):
        """Cleans up the MediaPipe recognizer."""
        with self._lock:
            if self.recognizer is not None:
                try:
                    self.recognizer.close()
                except Exception:
                    pass
                self.recognizer = None


# ---------------------------------------------------------------------------
# Skeleton & Visual Renderer (No mp.solutions)
# ---------------------------------------------------------------------------
class HandSkeletonRenderer:
    """High-performance antialiased renderer for hand skeleton, joints, and HUD."""

    @staticmethod
    def draw_skeleton(
        img_bgr: np.ndarray,
        recognition_result,
        show_bones: bool = True,
        show_glow: bool = True,
        show_hud: bool = True,
    ) -> List[Dict[str, Any]]:
        telemetry: List[Dict[str, Any]] = []
        if recognition_result is None or not recognition_result.hand_landmarks:
            return telemetry

        h, w, _ = img_bgr.shape
        num_detected = len(recognition_result.hand_landmarks)

        for i in range(num_detected):
            landmarks = recognition_result.hand_landmarks[i]
            
            # Extract Gesture info
            gesture_name = "Unrecognized"
            gesture_score = 0.0
            if i < len(recognition_result.gestures) and recognition_result.gestures[i]:
                top_gesture = recognition_result.gestures[i][0]
                gesture_name = top_gesture.category_name
                gesture_score = float(top_gesture.score)

            # Extract Handedness (Left / Right)
            handedness = "Hand"
            handedness_score = 0.0
            if i < len(recognition_result.handedness) and recognition_result.handedness[i]:
                top_hand = recognition_result.handedness[i][0]
                handedness = top_hand.category_name
                handedness_score = float(top_hand.score)

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

            # Record telemetry for GUI cards
            index_tip = pixel_pts[8] if len(pixel_pts) > 8 else (0, 0)
            telemetry.append({
                "hand_idx": i,
                "handedness": handedness,
                "handedness_score": handedness_score,
                "gesture": gesture_name,
                "gesture_score": gesture_score,
                "index_tip": index_tip,
                "landmark_count": len(landmarks),
            })

            # 1. Draw Skeleton Bones
            if show_bones and len(pixel_pts) == 21:
                for connections, color, thickness in FINGER_CONNECTIONS:
                    for start_idx, end_idx in connections:
                        pt1 = pixel_pts[start_idx]
                        pt2 = pixel_pts[end_idx]
                        cv2.line(img_bgr, pt1, pt2, color, thickness, cv2.LINE_AA)

            # 2. Draw Landmark Joints with Glow
            if len(pixel_pts) == 21:
                for idx, (px, py) in enumerate(pixel_pts):
                    if idx in FINGERTIP_INDICES:
                        tip_idx = FINGERTIP_INDICES.index(idx)
                        tip_color = FINGERTIP_COLORS[tip_idx]
                        if show_glow:
                            cv2.circle(img_bgr, (px, py), 9, tip_color, 2, cv2.LINE_AA)
                            cv2.circle(img_bgr, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(img_bgr, (px, py), 3, tip_color, -1, cv2.LINE_AA)
                        else:
                            cv2.circle(img_bgr, (px, py), 6, tip_color, -1, cv2.LINE_AA)
                    else:
                        if show_glow:
                            cv2.circle(img_bgr, (px, py), 4, (120, 140, 180), -1, cv2.LINE_AA)
                            cv2.circle(img_bgr, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
                        else:
                            cv2.circle(img_bgr, (px, py), 3, (200, 200, 220), -1, cv2.LINE_AA)

            # 3. Draw Bounding Box & HUD Label on Video
            if show_hud and xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                pad_x = int((max_x - min_x) * 0.15) + 10
                pad_y = int((max_y - min_y) * 0.15) + 10
                box_x1 = max(0, min_x - pad_x)
                box_y1 = max(0, min_y - pad_y)
                box_x2 = min(w - 1, max_x + pad_x)
                box_y2 = min(h - 1, max_y + pad_y)

                bracket_len = min(25, (box_x2 - box_x1) // 4, (box_y2 - box_y1) // 4)
                bracket_color = (0, 220, 255) if handedness == "Right" else (255, 120, 0)
                
                # Top-Left
                cv2.line(img_bgr, (box_x1, box_y1), (box_x1 + bracket_len, box_y1), bracket_color, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x1, box_y1), (box_x1, box_y1 + bracket_len), bracket_color, 2, cv2.LINE_AA)
                # Top-Right
                cv2.line(img_bgr, (box_x2, box_y1), (box_x2 - bracket_len, box_y1), bracket_color, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y1), (box_x2, box_y1 + bracket_len), bracket_color, 2, cv2.LINE_AA)
                # Bottom-Left
                cv2.line(img_bgr, (box_x1, box_y2), (box_x1 + bracket_len, box_y2), bracket_color, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x1, box_y2), (box_x1, box_y2 - bracket_len), bracket_color, 2, cv2.LINE_AA)
                # Bottom-Right
                cv2.line(img_bgr, (box_x2, box_y2), (box_x2 - bracket_len, box_y2), bracket_color, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (box_x2, box_y2), (box_x2 - bracket_len, box_y2), bracket_color, 2, cv2.LINE_AA)

                # Floating label above bounding box
                _, friendly_name = GESTURE_EMOJIS.get(gesture_name, ("🖐️", gesture_name))
                label_text = f"{handedness} Hand | {friendly_name} ({int(gesture_score*100)}%)"
                
                label_y = max(24, box_y1 - 10)
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                pill_x1 = box_x1
                pill_y1 = label_y - text_h - 6
                pill_x2 = box_x1 + text_w + 14
                pill_y2 = label_y + baseline + 2
                
                overlay = img_bgr.copy()
                cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 24, 32), -1)
                cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
                cv2.rectangle(img_bgr, (pill_x1, pill_y1), (pill_x2, pill_y2), bracket_color, 1, cv2.LINE_AA)
                
                cv2.putText(
                    img_bgr,
                    label_text,
                    (pill_x1 + 7, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return telemetry


# ---------------------------------------------------------------------------
# Modern Tkinter GUI Application
# ---------------------------------------------------------------------------
class HandVisionStudioApp:
    """Main Tkinter Desktop UI Application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hand Vision Studio — MediaPipe Tasks API")
        self.root.geometry("1240x820")
        self.root.minsize(1050, 720)
        self.root.configure(bg="#0e1117")

        # Application state
        self.is_running = False
        self.camera_index = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Settings variables
        self.var_mirror = tk.BooleanVar(value=True)
        self.var_bones = tk.BooleanVar(value=True)
        self.var_glow = tk.BooleanVar(value=True)
        self.var_hud = tk.BooleanVar(value=True)
        self.var_confidence = tk.DoubleVar(value=0.5)
        self.var_num_hands = tk.IntVar(value=2)
        self.var_cam_idx = tk.IntVar(value=0)

        # Performance counters
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Thread-safe frame queue
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.telemetry_data: List[Dict[str, Any]] = []

        # Setup MediaPipe Model & Engine
        self._init_mediapipe_engine()

        # Build GUI
        self._build_ui()

        # Start Camera Stream
        self.start_camera()

        # Periodic UI update loop
        self.root.after(15, self._gui_update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_mediapipe_engine(self):
        """Initializes model downloader and MediaPipe Tasks engine."""
        try:
            model_path = ensure_model_file(MODEL_FILENAME)
            self.engine = MediaPipeGestureEngine(
                model_path=model_path,
                num_hands=self.var_num_hands.get(),
                min_confidence=self.var_confidence.get(),
            )
            self.model_status_text = "Model: gesture_recognizer.task (Tasks API)"
        except Exception as e:
            messagebox.showerror("Model Init Error", f"Failed to load MediaPipe model:\n{e}")
            sys.exit(1)

    def _build_ui(self):
        """Constructs modern styled dark theme widgets."""
        header_frame = tk.Frame(self.root, bg="#161b22", height=60, bd=0, highlightthickness=1, highlightbackground="#30363d")
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="🖐️ HAND VISION STUDIO",
            font=("Segoe UI", 16, "bold"),
            fg="#58a6ff",
            bg="#161b22",
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="• MediaPipe Tasks API (Python 3.14 Ready)",
            font=("Segoe UI", 10),
            fg="#8b949e",
            bg="#161b22",
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=0, pady=14)

        self.lbl_header_fps = tk.Label(
            header_frame,
            text="⚡ FPS: -- | Latency: -- ms",
            font=("Consolas", 11, "bold"),
            fg="#3fb950",
            bg="#21262d",
            padx=12,
            pady=4,
            relief=tk.FLAT,
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

        body_frame = tk.Frame(self.root, bg="#0e1117")
        body_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Left Column: Video Viewport
        video_col = tk.Frame(body_frame, bg="#161b22", highlightthickness=1, highlightbackground="#30363d")
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        video_header = tk.Frame(video_col, bg="#1c2128", height=36)
        video_header.pack(fill=tk.X, side=tk.TOP)
        video_header.pack_propagate(False)

        tk.Label(
            video_header,
            text="📹 LIVE SKELETON FEED",
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

        # Video Canvas / Label
        self.lbl_video = tk.Label(video_col, bg="#0b0e14", text="Starting camera...", fg="#8b949e", font=("Segoe UI", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Right Column: Sidebar (Telemetry & Controls)
        sidebar_col = tk.Frame(body_frame, bg="#0e1117", width=420)
        sidebar_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        sidebar_col.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_col, bg="#0e1117", highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar_col, orient="vertical", command=sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(sidebar_canvas, bg="#0e1117")

        self.sidebar_content.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", width=400)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_telemetry_section()
        self._build_controls_section()
        self._build_gesture_legend_section()

    def _build_telemetry_section(self):
        """Builds live cards for detected hand skeleton telemetry."""
        sec_frame = tk.LabelFrame(
            self.sidebar_content,
            text="  🖐️ DETECTED HAND TELEMETRY  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        sec_frame.pack(fill=tk.X, pady=(0, 12), padx=4)

        self.cards_frame = tk.Frame(sec_frame, bg="#161b22")
        self.cards_frame.pack(fill=tk.X, padx=10, pady=10)

        self.hand_card_1 = self._create_hand_card(self.cards_frame, "Hand 1")
        self.hand_card_1["frame"].pack(fill=tk.X, pady=(0, 8))

        self.hand_card_2 = self._create_hand_card(self.cards_frame, "Hand 2")
        self.hand_card_2["frame"].pack(fill=tk.X)

    def _create_hand_card(self, parent: tk.Widget, title: str) -> Dict[str, Any]:
        """Creates a single visual telemetry card for a hand."""
        cframe = tk.Frame(parent, bg="#21262d", bd=0, highlightthickness=1, highlightbackground="#30363d")
        
        header = tk.Frame(cframe, bg="#262c36", height=28)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        lbl_title = tk.Label(header, text=title, font=("Segoe UI", 9, "bold"), fg="#c9d1d9", bg="#262c36")
        lbl_title.pack(side=tk.LEFT, padx=8, pady=4)

        lbl_side = tk.Label(header, text="Inactive", font=("Segoe UI", 8, "bold"), fg="#8b949e", bg="#1f242c", padx=6, pady=1)
        lbl_side.pack(side=tk.RIGHT, padx=8, pady=4)

        body = tk.Frame(cframe, bg="#21262d", padx=10, pady=8)
        body.pack(fill=tk.X)

        row1 = tk.Frame(body, bg="#21262d")
        row1.pack(fill=tk.X, pady=(0, 4))
        
        lbl_gesture_icon = tk.Label(row1, text="❓", font=("Segoe UI Emoji", 18), bg="#21262d")
        lbl_gesture_icon.pack(side=tk.LEFT, padx=(0, 8))

        gesture_meta = tk.Frame(row1, bg="#21262d")
        gesture_meta.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl_gesture_name = tk.Label(gesture_meta, text="No Hand in Frame", font=("Segoe UI", 11, "bold"), fg="#8b949e", bg="#21262d", anchor="w")
        lbl_gesture_name.pack(fill=tk.X)

        lbl_conf = tk.Label(gesture_meta, text="Confidence: 0%", font=("Segoe UI", 8), fg="#8b949e", bg="#21262d", anchor="w")
        lbl_conf.pack(fill=tk.X)

        row2 = tk.Frame(body, bg="#21262d")
        row2.pack(fill=tk.X, pady=(4, 0))

        lbl_coords = tk.Label(row2, text="Index Tip: (-, -) | 21 Joints", font=("Consolas", 8), fg="#8b949e", bg="#21262d", anchor="w")
        lbl_coords.pack(side=tk.LEFT)

        return {
            "frame": cframe,
            "title": lbl_title,
            "side_badge": lbl_side,
            "gesture_icon": lbl_gesture_icon,
            "gesture_name": lbl_gesture_name,
            "conf_lbl": lbl_conf,
            "coords_lbl": lbl_coords,
        }

    def _build_controls_section(self):
        """Builds controls for camera, visualization toggles, and sensitivity sliders."""
        ctrl_frame = tk.LabelFrame(
            self.sidebar_content,
            text="  ⚙️ CONTROL PANEL  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        ctrl_frame.pack(fill=tk.X, pady=(0, 12), padx=4)

        pad_frame = tk.Frame(ctrl_frame, bg="#161b22", padx=10, pady=10)
        pad_frame.pack(fill=tk.X)

        btn_row = tk.Frame(pad_frame, bg="#161b22")
        btn_row.pack(fill=tk.X, pady=(0, 10))

        self.btn_toggle_cam = tk.Button(
            btn_row,
            text="⏹ Stop Camera",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#da3633",
            activebackground="#b62324",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6,
            command=self.toggle_camera,
            cursor="hand2",
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        cam_sel_frame = tk.Frame(btn_row, bg="#21262d", padx=4, pady=2)
        cam_sel_frame.pack(side=tk.RIGHT)
        
        tk.Label(cam_sel_frame, text="Cam #", font=("Segoe UI", 8), fg="#c9d1d9", bg="#21262d").pack(side=tk.LEFT, padx=(2, 2))
        cam_spin = tk.Spinbox(
            cam_sel_frame,
            from_=0,
            to=5,
            width=2,
            textvariable=self.var_cam_idx,
            font=("Segoe UI", 9, "bold"),
            bg="#161b22",
            fg="#58a6ff",
            buttonbackground="#30363d",
            command=self.switch_camera_device,
        )
        cam_spin.pack(side=tk.LEFT, padx=2)

        toggles_grid = tk.Frame(pad_frame, bg="#161b22")
        toggles_grid.pack(fill=tk.X, pady=(0, 10))

        cb_mirror = tk.Checkbutton(
            toggles_grid,
            text="Mirror Video",
            variable=self.var_mirror,
            font=("Segoe UI", 9),
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#21262d",
            activebackground="#161b22",
            activeforeground="#ffffff",
        )
        cb_mirror.grid(row=0, column=0, sticky="w", pady=2)

        cb_bones = tk.Checkbutton(
            toggles_grid,
            text="Skeleton Bones",
            variable=self.var_bones,
            font=("Segoe UI", 9),
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#21262d",
            activebackground="#161b22",
            activeforeground="#ffffff",
        )
        cb_bones.grid(row=0, column=1, sticky="w", pady=2, padx=10)

        cb_glow = tk.Checkbutton(
            toggles_grid,
            text="Joint Glows",
            variable=self.var_glow,
            font=("Segoe UI", 9),
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#21262d",
            activebackground="#161b22",
            activeforeground="#ffffff",
        )
        cb_glow.grid(row=1, column=0, sticky="w", pady=2)

        cb_hud = tk.Checkbutton(
            toggles_grid,
            text="Video HUD Labels",
            variable=self.var_hud,
            font=("Segoe UI", 9),
            fg="#c9d1d9",
            bg="#161b22",
            selectcolor="#21262d",
            activebackground="#161b22",
            activeforeground="#ffffff",
        )
        cb_hud.grid(row=1, column=1, sticky="w", pady=2, padx=10)

        sl_frame = tk.Frame(pad_frame, bg="#161b22")
        sl_frame.pack(fill=tk.X, pady=(2, 4))

        lbl_sl = tk.Label(sl_frame, text="Min Detection Confidence: 0.50", font=("Segoe UI", 8), fg="#8b949e", bg="#161b22")
        lbl_sl.pack(anchor="w")

        def _on_conf_change(val):
            v = float(val)
            lbl_sl.config(text=f"Min Detection Confidence: {v:.2f}")
            self.engine.rebuild_recognizer(min_confidence=v)

        conf_scale = tk.Scale(
            sl_frame,
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
        )
        conf_scale.pack(fill=tk.X, pady=(2, 0))

    def _build_gesture_legend_section(self):
        """Displays gesture recognition guide and supported classes."""
        legend_frame = tk.LabelFrame(
            self.sidebar_content,
            text="  📖 GESTURE CHEAT SHEET  ",
            font=("Segoe UI", 10, "bold"),
            fg="#58a6ff",
            bg="#161b22",
            bd=1,
            relief=tk.SOLID,
            highlightbackground="#30363d",
        )
        legend_frame.pack(fill=tk.X, pady=(0, 10), padx=4)

        grid_box = tk.Frame(legend_frame, bg="#161b22", padx=8, pady=8)
        grid_box.pack(fill=tk.X)

        gestures_list = [
            ("🖐️", "Open Palm", "Open hand with fingers extended"),
            ("✊", "Closed Fist", "All fingers curled inward"),
            ("☝️", "Pointing Up", "Index finger raised"),
            ("✌️", "Victory / Peace", "Index and Middle fingers in V"),
            ("👍", "Thumb Up", "Approval / Thumb facing up"),
            ("👎", "Thumb Down", "Disapproval / Thumb facing down"),
            ("🤟", "I Love You", "Thumb, Index, and Pinky up"),
        ]

        for idx, (emoji, name, desc) in enumerate(gestures_list):
            row_f = tk.Frame(grid_box, bg="#21262d" if idx % 2 == 0 else "#161b22", padx=6, pady=4)
            row_f.pack(fill=tk.X, pady=1)

            tk.Label(row_f, text=emoji, font=("Segoe UI Emoji", 12), bg=row_f["bg"]).pack(side=tk.LEFT, padx=(2, 6))
            tk.Label(row_f, text=name, font=("Segoe UI", 9, "bold"), fg="#c9d1d9", bg=row_f["bg"]).pack(side=tk.LEFT)
            tk.Label(row_f, text=f"- {desc}", font=("Segoe UI", 8), fg="#8b949e", bg=row_f["bg"]).pack(side=tk.LEFT, padx=(6, 0))

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
        self._reset_telemetry_cards()

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
        """Background thread for grabbing frames, MediaPipe inference, and rendering."""
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

        print("[Camera] VideoCapture opened successfully. Beginning inference loop.")
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

            # Perform Gesture Recognition & Hand Landmark Detection
            try:
                recognition_result = self.engine.process_rgb_frame(rgb_frame)
            except Exception as e:
                print(f"[MediaPipe Tasks] Inference exception: {e}")
                recognition_result = None

            t_infer = (time.perf_counter() - t0) * 1000.0

            # Render Skeletons & HUD directly onto the frame (BGR)
            telemetry = HandSkeletonRenderer.draw_skeleton(
                frame,
                recognition_result,
                show_bones=self.var_bones.get(),
                show_glow=self.var_glow.get(),
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

    def _gui_update_loop(self):
        """Pulls latest processed frame from queue and updates Tkinter widgets."""
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

                self._update_telemetry_cards(telemetry)

        except Exception:
            pass

        self.root.after(16, self._gui_update_loop)

    def _update_telemetry_cards(self, telemetry: List[Dict[str, Any]]):
        """Updates the visual cards with detected hand gestures."""
        cards = [self.hand_card_1, self.hand_card_2]

        for idx, card in enumerate(cards):
            if idx < len(telemetry):
                data = telemetry[idx]
                handedness = data["handedness"]
                side_score = data["handedness_score"]
                gesture = data["gesture"]
                gesture_score = data["gesture_score"]
                tip_x, tip_y = data["index_tip"]

                emoji, friendly_name = GESTURE_EMOJIS.get(gesture, ("🖐️", gesture))

                card["side_badge"].config(
                    text=f"{handedness} ({int(side_score * 100)}%)",
                    fg="#58a6ff" if handedness == "Right" else "#ff7b72",
                    bg="#1f242c",
                )

                card["gesture_icon"].config(text=emoji)
                card["gesture_name"].config(
                    text=friendly_name,
                    fg="#3fb950" if gesture_score > 0.6 else "#d29922",
                )
                card["conf_lbl"].config(
                    text=f"Confidence: {int(gesture_score * 100)}%",
                    fg="#c9d1d9",
                )
                card["coords_lbl"].config(
                    text=f"Index Tip: ({tip_x}, {tip_y}) | 21 Joints Tracked",
                    fg="#8b949e",
                )
                card["frame"].config(highlightbackground="#58a6ff" if handedness == "Right" else "#ff7b72")
            else:
                card["side_badge"].config(text="Inactive", fg="#8b949e", bg="#1f242c")
                card["gesture_icon"].config(text="❓")
                card["gesture_name"].config(text="No Hand in Frame", fg="#8b949e")
                card["conf_lbl"].config(text="Confidence: 0%", fg="#8b949e")
                card["coords_lbl"].config(text="Index Tip: (-, -) | 0 Joints", fg="#8b949e")
                card["frame"].config(highlightbackground="#30363d")

    def _reset_telemetry_cards(self):
        """Resets all cards to inactive state."""
        self._update_telemetry_cards([])

    def on_closing(self):
        """Gracefully closes resources, background threads, and window."""
        print("[App] Closing application...")
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
    print("[HAND VISION STUDIO] Modern MediaPipe Tasks API & Tkinter")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"MediaPipe Version: {mp.__version__}")
    print("Tasks API Vision: GestureRecognizer & 21-Landmark Skeleton")
    print("=" * 65)

    root = tk.Tk()
    app = HandVisionStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
