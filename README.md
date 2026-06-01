# GestureFlow

GestureFlow is a real-time hand gesture recognition project built with Python, OpenCV, MediaPipe Hands, NumPy, Pandas, and scikit-learn. It uses webcam input to detect hand landmarks, recognize static gestures, detect swipe motion, and show a pastel on-screen response.

This project is designed to be understandable enough for a student portfolio while still using a practical computer vision pipeline.

## Features

- Opens a webcam feed in real time.
- Detects one or two hands with MediaPipe Hands.
- Draws hand landmarks clearly on the video.
- Stores 21 hand landmarks as x, y, z coordinates.
- Recognizes static gestures with a trained Random Forest model:
  - open palm
  - fist
  - peace sign
- Recognizes motion gestures from recent wrist and index fingertip movement:
  - swipe left
  - swipe right
  - swipe up
  - swipe down
- Displays the predicted gesture and confidence score.
- Shows the triggered action on screen.
- Includes keyboard controls:
  - `Q` to quit
  - `R` to start or stop recording samples in collection mode
  - `S` to save/stop samples in collection mode, or save a screenshot in live mode
- Uses a pastel UI overlay for readable labels and status messages.
- Handles missing webcam, missing hand, and missing model errors.

## Project Structure

```text
GestureFlow/
  collect_data.py
  train_model.py
  predict_live.py
  utils.py
  requirements.txt
  README.md
  data/
    swipe_left/
    swipe_right/
    swipe_up/
    swipe_down/
    open_palm/
    fist/
    peace/
  models/
  screenshots/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Web Preview

GestureFlow also includes a browser version:

```text
index.html
styles.css
app.js
```

Run a local web server from the project folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

The website uses your browser webcam and MediaPipe Hands in JavaScript. It supports live landmarks, open palm, fist, peace sign, swipe detection, screenshot download, and CSV sample download. Webcam access usually requires `localhost` or HTTPS, so opening `index.html` directly may not allow camera permissions.

If you are on macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Collect Custom Gesture Data

Collect samples for each static gesture. Keep your hand visible, vary the position slightly, and collect at least 100-200 samples per class.

```bash
python collect_data.py --gesture open_palm
python collect_data.py --gesture fist
python collect_data.py --gesture peace
```

In the webcam window:

- Press `R` to start recording.
- Move your hand slightly while keeping the same gesture.
- Press `R` again or `S` to stop.
- Press `Q` to quit.

Samples are saved as CSV files:

```text
data/open_palm/open_palm_landmarks.csv
data/fist/fist_landmarks.csv
data/peace/peace_landmarks.csv
```

The swipe folders are included for a clean dataset structure. In this version, swipes are detected using motion direction instead of a trained model, because directional movement is reliable and easy to explain for a student project.

## Train the Model

After collecting at least two static gesture classes, train the Random Forest model:

```bash
python train_model.py
```

The trained model is saved to:

```text
models/gesture_model.joblib
```

## Run Live Gesture Recognition

```bash
python predict_live.py
```

The app will show:

- webcam feed
- detected hand landmarks
- current gesture prediction
- confidence score
- action triggered

If the model file is missing, the app will ask you to run `train_model.py` first.

## Gesture Actions

| Gesture | Action |
| --- | --- |
| Swipe left | Previous filter / previous slide |
| Swipe right | Next filter / next slide |
| Swipe up | Scroll up |
| Swipe down | Scroll down |
| Open palm | Pause |
| Fist | Select |
| Peace sign | Take screenshot |

## How the Model Works

GestureFlow does not train on raw webcam images. Instead, it uses MediaPipe Hands to find 21 landmark points on the hand. Each point has three coordinates:

- `x`
- `y`
- `z`

That creates 63 features for one hand.

Before saving or predicting, GestureFlow normalizes the landmarks:

1. The wrist becomes the origin point.
2. All landmarks are shifted relative to the wrist.
3. The hand is scaled so the model is less affected by hand size or distance from the camera.

For static gestures, `train_model.py` trains a Random Forest classifier using the landmark CSV files.

For motion gestures, `predict_live.py` tracks the wrist and index fingertip across the last 18 frames. If the movement is mostly horizontal, it becomes a left or right swipe. If the movement is mostly vertical, it becomes an up or down swipe.

## Tips for Better Accuracy

- Collect samples in different lighting conditions.
- Record each gesture from slightly different positions.
- Keep your hand clearly inside the camera frame.
- Use a plain background if possible.
- Collect a balanced number of samples for each class.
- Add more gestures only after the first three static classes work well.
- Increase `n_estimators` in `train_model.py` for a stronger Random Forest.
- Try an SVM or small neural network after the Random Forest baseline works.
- For advanced motion recognition, save 15-20 frame landmark sequences and train a sequence model.

## Portfolio Ideas

- Add a mode that controls PowerPoint slides.
- Add browser scrolling with `pyautogui`.
- Show a small gesture history timeline.
- Add a settings panel for confidence thresholds.
- Export training metrics and a confusion matrix.
- Train a separate model for motion gesture sequences.
