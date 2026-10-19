"""Train the classifier.

Usage:
    python -m src.train [--epochs 20] [--no-augment]
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from src.config import EPOCHS, MODEL_PATH, RESULTS_DIR
from src.data import list_images, make_generator, split_dataset
from src.model import build_model


def plot_history(history, output_path):
    """Save the loss / accuracy curves (training vs validation)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    ax1.plot(history["loss"], color="b", label="Training loss")
    ax1.plot(history["val_loss"], color="r", label="Validation loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(history["categorical_accuracy"], color="b", label="Training accuracy")
    ax2.plot(history["val_categorical_accuracy"], color="r", label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Train the waste classifier.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    train_df, val_df, test_df = split_dataset(list_images())
    print(f"train = {len(train_df)}, validation = {len(val_df)}, test = {len(test_df)}")

    train_gen = make_generator(train_df, augment=not args.no_augment)
    val_gen = make_generator(val_df, shuffle=False)

    model = build_model()
    model.summary()

    # Stop when validation accuracy stops improving and keep the best weights
    early_stop = EarlyStopping(monitor="val_categorical_accuracy", mode="max", patience=2,
                               min_delta=0.001, restore_best_weights=True, verbose=1)

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs,
                        callbacks=[early_stop])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(history.history).to_csv(RESULTS_DIR / "training_history.csv", index_label="epoch")
    plot_history(history.history, RESULTS_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
