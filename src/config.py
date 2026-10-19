"""Shared configuration: paths, image size, class names and camera source."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Paths can be overridden with environment variables
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "garbage_classification"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "model.h5"))
RESULTS_DIR = ROOT_DIR / "results"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Image size used for training and inference
IMAGE_SIZE = (320, 320)
IMAGE_CHANNELS = 3

# Training hyper-parameters
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

# The 12 classes, in alphabetical order. This is the order assigned by Keras'
# flow_from_dataframe, i.e. the index of each class in the model's output.
CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
    "metal", "paper", "plastic", "shoes", "trash", "white-glass",
]

# Camera used for live detection: a webcam index ("0") or a stream URL,
# e.g. the "IP Webcam" Android app: http://<phone-ip>:8080/video
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")


def parse_camera_source(source: str):
    """Return an int for a webcam index, otherwise the URL string."""
    return int(source) if str(source).isdigit() else source
