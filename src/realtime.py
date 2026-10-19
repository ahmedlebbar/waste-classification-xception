"""Real-time classification from a webcam or an IP camera stream.

Usage:
    python -m src.realtime                      # default webcam (index 0)
    python -m src.realtime --source 1           # another webcam
    python -m src.realtime --source http://192.168.1.10:8080/video   # phone (IP Webcam app)

Press "q" to quit.
"""
import argparse

import cv2

from src.config import CAMERA_SOURCE, IMAGE_SIZE, parse_camera_source
from src.model import load_trained_model, predict_array


def classify_frame(model, frame_bgr):
    """OpenCV frames are BGR: convert to RGB (as in training) before predicting."""
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
    return predict_array(model, rgb)


def run(model, source=CAMERA_SOURCE):
    cap = cv2.VideoCapture(parse_camera_source(source))
    if not cap.isOpened():
        print(f"Error: could not open camera source '{source}'.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        class_name, confidence = classify_frame(model, frame)
        cv2.putText(frame, f"{class_name} ({confidence:.0%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Waste classification - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time waste classification.")
    parser.add_argument("--source", default=CAMERA_SOURCE,
                        help="Webcam index or stream URL (default: $CAMERA_SOURCE or 0)")
    args = parser.parse_args()
    run(load_trained_model(), args.source)


if __name__ == "__main__":
    main()
