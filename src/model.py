"""Model definition: frozen Xception backbone (ImageNet) + softmax classifier."""
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import xception

from src.config import CLASS_NAMES, IMAGE_CHANNELS, IMAGE_SIZE, MODEL_PATH


def build_model(weights="imagenet", num_classes=len(CLASS_NAMES)):
    """Build the transfer-learning model.

    Architecture: Xception preprocessing -> frozen Xception (no top)
    -> GlobalAveragePooling2D -> Dense(num_classes, softmax).
    Only the last Dense layer is trained (24,588 parameters for 12 classes).
    """
    input_shape = (*IMAGE_SIZE, IMAGE_CHANNELS)

    backbone = xception.Xception(include_top=False, weights=weights, input_shape=input_shape)
    backbone.trainable = False

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Lambda(xception.preprocess_input, name="xception_preprocessing"),
        backbone,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["categorical_accuracy"])
    return model


def load_trained_model(path: Path = MODEL_PATH):
    """Rebuild the architecture and load the trained weights.

    Loading weights (instead of the full serialized model) avoids
    deserializing the Lambda layer, which is fragile across Keras versions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Download it from the GitHub release "
            "(see models/README.md) or train it with `python -m src.train`."
        )
    model = build_model(weights=None)
    model.load_weights(path)
    return model


def predict_array(model, rgb_image: np.ndarray):
    """Predict the class of an RGB image already resized to IMAGE_SIZE.

    Returns (class_name, confidence).
    """
    batch = np.expand_dims(rgb_image.astype("float32"), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index])
