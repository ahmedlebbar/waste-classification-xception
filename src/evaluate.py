"""Evaluate the trained model on the held-out test set.

Writes results/classification_report.txt and results/confusion_matrix.png.

Usage:
    python -m src.evaluate
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import CLASS_NAMES, RESULTS_DIR
from src.data import list_images, make_generator, split_dataset
from src.model import load_trained_model


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Row-normalized confusion matrix (each row sums to 1 = recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (normalized by true class)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    _, _, test_df = split_dataset(list_images())
    test_gen = make_generator(test_df, shuffle=False)

    model = load_trained_model()
    probabilities = model.predict(test_gen, verbose=1)

    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_gen.classes

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print(f"Test accuracy: {accuracy:.2%} on {len(y_true)} images\n")
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {accuracy:.4f} ({len(y_true)} images)\n\n{report}", encoding="utf-8")
    plot_confusion_matrix(y_true, y_pred, RESULTS_DIR / "confusion_matrix.png")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
