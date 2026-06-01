"""Shared helpers for GestureFlow.

The project uses MediaPipe hand landmarks instead of raw images. A single hand
has 21 landmarks, and every landmark has x, y, and z coordinates. That gives us
63 numeric features per frame, which is small, fast, and beginner-friendly.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gesture_model.joblib"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

STATIC_GESTURES = ["open_palm", "fist", "peace"]
MOTION_GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]
ALL_GESTURES = MOTION_GESTURES + STATIC_GESTURES

LANDMARK_COUNT = 21
FEATURES_PER_HAND = LANDMARK_COUNT * 3
SEQUENCE_LENGTH = 18

GESTURE_ACTIONS = {
    "swipe_left": "Previous filter / previous slide",
    "swipe_right": "Next filter / next slide",
    "swipe_up": "Scroll up",
    "swipe_down": "Scroll down",
    "open_palm": "Pause",
    "fist": "Select",
    "peace": "Take screenshot",
}

PASTEL = {
    "mint": (187, 247, 208),
    "pink": (244, 194, 194),
    "lavender": (221, 214, 254),
    "sky": (186, 230, 253),
    "cream": (254, 243, 199),
    "ink": (45, 55, 72),
    "white": (255, 255, 255),
}


@dataclass
class GesturePrediction:
    """Small container used by the live app."""

    name: str
    confidence: float
    action: str


def ensure_project_folders() -> None:
    """Create data/model/output folders if they do not already exist."""

    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    for gesture in ALL_GESTURES:
        (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)


def landmark_list_to_array(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to a flat 63-value NumPy array."""

    coords = []
    for landmark in hand_landmarks.landmark:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """Normalize landmarks so hand position and scale affect the model less.

    The wrist is used as the origin. Coordinates are then divided by the hand's
    largest distance from the wrist. This makes a sample more about hand shape
    and less about where the hand appears in the camera frame.
    """

    points = features.reshape(LANDMARK_COUNT, 3).copy()
    wrist = points[0].copy()
    points -= wrist
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points.flatten()


def hand_center_from_landmarks(hand_landmarks) -> tuple[float, float]:
    """Return the average x/y center for one detected hand."""

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return float(np.mean(xs)), float(np.mean(ys))


def wrist_and_index_tip(hand_landmarks) -> tuple[float, float, float, float]:
    """Return wrist and index fingertip x/y values for motion tracking."""

    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    return wrist.x, wrist.y, index_tip.x, index_tip.y


def append_landmark_sample(csv_path: Path, gesture: str, features: Iterable[float]) -> None:
    """Append one landmark row to a gesture CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    header = ["label"] + [f"feature_{i}" for i in range(FEATURES_PER_HAND)]

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([gesture, *features])


def draw_rounded_box(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.84,
) -> None:
    """Draw a soft filled rectangle using alpha blending."""

    overlay = frame.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = PASTEL["ink"],
    thickness: int = 2,
) -> None:
    """Draw readable text with a subtle white shadow."""

    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, PASTEL["white"], thickness + 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_ui_overlay(
    frame: np.ndarray,
    prediction: GesturePrediction | None,
    recording: bool = False,
    sample_count: int = 0,
    message: str = "",
) -> None:
    """Draw the pastel GestureFlow interface on top of the webcam feed."""

    height, width = frame.shape[:2]
    draw_rounded_box(frame, (18, 18), (min(width - 18, 470), 160), PASTEL["lavender"])
    put_label(frame, "GestureFlow", (34, 52), 0.9)

    if prediction:
        gesture = prediction.name.replace("_", " ").title()
        put_label(frame, f"Gesture: {gesture}", (34, 88), 0.68)
        put_label(frame, f"Confidence: {prediction.confidence:.2f}", (34, 118), 0.62)
        put_label(frame, f"Action: {prediction.action}", (34, 148), 0.56)
    else:
        put_label(frame, message or "No hand detected.", (34, 95), 0.68)

    if recording:
        draw_rounded_box(frame, (18, height - 92), (360, height - 24), PASTEL["pink"])
        put_label(frame, f"Recording... samples: {sample_count}", (34, height - 50), 0.7)
    else:
        draw_rounded_box(frame, (18, height - 72), (470, height - 24), PASTEL["cream"], alpha=0.78)
        put_label(frame, "Q quit   R record   S save/stop", (34, height - 42), 0.58)


def classify_motion(history: Deque[tuple[float, float, float, float]]) -> GesturePrediction | None:
    """Classify swipe direction from wrist and index fingertip movement.

    This heuristic is intentionally lightweight: motion gestures are often
    easier to recognize from movement direction than with a static classifier.
    It averages wrist and index fingertip displacement across recent frames.
    """

    if len(history) < SEQUENCE_LENGTH:
        return None

    start = np.array(history[0], dtype=np.float32)
    end = np.array(history[-1], dtype=np.float32)
    dx = float(np.mean([end[0] - start[0], end[2] - start[2]]))
    dy = float(np.mean([end[1] - start[1], end[3] - start[3]]))

    min_movement = 0.18
    if abs(dx) < min_movement and abs(dy) < min_movement:
        return None

    if abs(dx) > abs(dy):
        name = "swipe_right" if dx > 0 else "swipe_left"
        confidence = min(abs(dx) / 0.45, 1.0)
    else:
        # Camera y coordinates increase downward, so negative dy means up.
        name = "swipe_down" if dy > 0 else "swipe_up"
        confidence = min(abs(dy) / 0.45, 1.0)

    return GesturePrediction(name=name, confidence=confidence, action=GESTURE_ACTIONS[name])


def create_hands_detector(max_num_hands: int = 2):
    """Create a MediaPipe Hands detector with friendly defaults."""

    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    )


def draw_hand_landmarks(frame: np.ndarray, results) -> None:
    """Draw MediaPipe hand landmarks on the frame."""

    if not results.multi_hand_landmarks:
        return
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )


def newest_csv_for_gesture(gesture: str) -> Path:
    """Return the standard CSV path for one gesture."""

    return DATA_DIR / gesture / f"{gesture}_landmarks.csv"


def make_motion_history() -> Deque[tuple[float, float, float, float]]:
    """Create the fixed-length history used for swipe detection."""

    return deque(maxlen=SEQUENCE_LENGTH)
