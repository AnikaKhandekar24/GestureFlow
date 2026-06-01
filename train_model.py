"""Train the static gesture classifier for GestureFlow.

The model learns static hand shapes such as open palm, fist, and peace sign.
Swipe gestures are recognized live from movement direction in predict_live.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import DATA_DIR, MODEL_PATH, STATIC_GESTURES, ensure_project_folders


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    for gesture in STATIC_GESTURES:
        csv_path = DATA_DIR / gesture / f"{gesture}_landmarks.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(
            "No static gesture CSV files were found. Collect data first, for example: "
            "python collect_data.py --gesture open_palm"
        )

    data = pd.concat(frames, ignore_index=True)
    if "label" not in data.columns:
        raise ValueError("Training CSV files must contain a 'label' column.")

    x = data.drop(columns=["label"])
    y = data["label"]
    return x, y


def main() -> None:
    ensure_project_folders()
    x, y = load_training_data()

    if y.nunique() < 2:
        raise ValueError("Please collect at least two different static gesture classes before training.")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("GestureFlow static gesture report:")
    print(classification_report(y_test, predictions, zero_division=0))

    Path(MODEL_PATH).parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
