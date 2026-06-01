"""Run GestureFlow in live prediction mode."""

from __future__ import annotations

from datetime import datetime

import cv2
import joblib
import numpy as np

from utils import (
    GESTURE_ACTIONS,
    MODEL_PATH,
    SCREENSHOT_DIR,
    GesturePrediction,
    classify_motion,
    create_hands_detector,
    draw_hand_landmarks,
    draw_ui_overlay,
    ensure_project_folders,
    landmark_list_to_array,
    make_motion_history,
    normalize_landmarks,
    wrist_and_index_tip,
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file missing: {MODEL_PATH}\n"
            "Collect static gesture data, then run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_static(model, features: np.ndarray) -> GesturePrediction:
    """Predict a static gesture and return its confidence."""

    probabilities = model.predict_proba([features])[0]
    best_index = int(np.argmax(probabilities))
    name = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return GesturePrediction(name=name, confidence=confidence, action=GESTURE_ACTIONS.get(name, "Show response"))


def save_screenshot(frame) -> str:
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    filename = SCREENSHOT_DIR / f"gestureflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(filename), frame)
    return str(filename)


def main() -> None:
    ensure_project_folders()

    try:
        model = load_model()
    except FileNotFoundError as error:
        print(error)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check that your camera is connected and not used by another app.")
        return

    motion_history = make_motion_history()
    last_action = "Waiting"
    cooldown = 0

    with create_hands_detector(max_num_hands=2) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read from webcam. Try restarting the camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            draw_hand_landmarks(frame, results)

            prediction = None
            message = "No hand detected."

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                motion_history.append(wrist_and_index_tip(hand_landmarks))
                motion_prediction = classify_motion(motion_history)

                features = landmark_list_to_array(hand_landmarks)
                features = normalize_landmarks(features)
                static_prediction = predict_static(model, features)

                # Prefer confident swipes over static hand-shape predictions.
                prediction = motion_prediction if motion_prediction and motion_prediction.confidence >= 0.50 else static_prediction

                if cooldown > 0:
                    cooldown -= 1
                elif prediction.confidence >= 0.55:
                    last_action = prediction.action
                    cooldown = 18
                    if prediction.name == "peace":
                        saved_path = save_screenshot(frame)
                        last_action = f"Screenshot saved: {saved_path}"
            else:
                motion_history.clear()

            if prediction:
                prediction.action = last_action
            draw_ui_overlay(frame, prediction, recording=False, message=message)

            cv2.imshow("GestureFlow - Live Prediction", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("Live mode does not record samples. Use collect_data.py to record training data.")
            if key == ord("s"):
                saved_path = save_screenshot(frame)
                print(f"Screenshot saved to {saved_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
