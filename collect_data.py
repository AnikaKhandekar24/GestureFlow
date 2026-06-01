"""Collect landmark samples for GestureFlow.

Usage:
    python collect_data.py --gesture open_palm
    python collect_data.py --gesture fist --camera 1

Keyboard shortcuts:
    R - start/stop recording
    S - save/stop the current recording session
    Q - quit
"""

from __future__ import annotations

import argparse

import cv2

from utils import (
    ALL_GESTURES,
    GesturePrediction,
    append_landmark_sample,
    create_hands_detector,
    draw_hand_landmarks,
    draw_ui_overlay,
    ensure_project_folders,
    landmark_list_to_array,
    newest_csv_for_gesture,
    normalize_landmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MediaPipe landmark samples for a gesture.")
    parser.add_argument("--gesture", choices=ALL_GESTURES, required=True, help="Gesture label to record.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Most laptops use 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_folders()

    csv_path = newest_csv_for_gesture(args.gesture)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam. Check that your camera is connected and not used by another app.")
        return

    recording = False
    sample_count = 0

    with create_hands_detector(max_num_hands=2) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read from webcam. Try restarting the camera or changing --camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            draw_hand_landmarks(frame, results)

            message = "No hand detected."
            if results.multi_hand_landmarks:
                message = f"Ready to collect: {args.gesture}"
                if recording:
                    for hand_landmarks in results.multi_hand_landmarks:
                        features = landmark_list_to_array(hand_landmarks)
                        features = normalize_landmarks(features)
                        append_landmark_sample(csv_path, args.gesture, features)
                        sample_count += 1

            prediction = GesturePrediction(
                name=args.gesture,
                confidence=1.0 if recording else 0.0,
                action=f"Saving to {csv_path.name}",
            )
            draw_ui_overlay(frame, prediction, recording=recording, sample_count=sample_count, message=message)
            cv2.imshow("GestureFlow - Collect Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recording = not recording
            if key == ord("s"):
                recording = False
                print(f"Saved {sample_count} samples for {args.gesture} to {csv_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Total samples collected this session: {sample_count}")


if __name__ == "__main__":
    main()
