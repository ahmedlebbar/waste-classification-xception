"""Predict the class of one or more images.

Usage:
    python -m src.predict data/samples/plastic.jpg [more images...]
"""
import argparse

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.config import IMAGE_SIZE
from src.model import load_trained_model, predict_array


def predict_image(model, image_path):
    """Return (class_name, confidence) for an image file."""
    image = load_img(image_path, target_size=IMAGE_SIZE)  # RGB
    return predict_array(model, img_to_array(image))


def main():
    parser = argparse.ArgumentParser(description="Classify waste images.")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    args = parser.parse_args()

    model = load_trained_model()
    for image_path in args.images:
        class_name, confidence = predict_image(model, image_path)
        print(f"{image_path}: {class_name} ({confidence:.1%})")


if __name__ == "__main__":
    main()
