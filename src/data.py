"""Dataset loading and reproducible train / validation / test split (80/10/10)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMAGE_SIZE, SEED


def list_images(data_dir=DATA_DIR):
    """Return a DataFrame with one row per image: relative filename + class name."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. See data/README.md for download instructions."
        )
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            rows.append({"filename": f"{class_name}/{image_path.name}", "category": class_name})
    return pd.DataFrame(rows)


def split_dataset(df):
    """Stratified 80% train / 10% validation / 10% test split."""
    train_df, rest_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["category"])
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=SEED, stratify=rest_df["category"])
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def make_generator(df, augment=False, shuffle=True, batch_size=BATCH_SIZE):
    """Keras generator that reads the images from disk."""
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=30, shear_range=0.1, zoom_range=0.3,
            horizontal_flip=True, vertical_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
        )
    else:
        datagen = ImageDataGenerator()

    return datagen.flow_from_dataframe(
        df, directory=str(DATA_DIR), x_col="filename", y_col="category",
        target_size=IMAGE_SIZE, class_mode="categorical", classes=CLASS_NAMES,
        batch_size=batch_size, shuffle=shuffle, seed=SEED,
    )
