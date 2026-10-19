# Data

## Dataset

The model is trained on the public **Garbage Classification (12 classes)** dataset by Mostafa Mohamed
(15,515 images): <https://www.kaggle.com/datasets/mostafaabla/garbage-classification>

It is not stored in this repository. To download it:

```bash
# Option 1: Kaggle CLI (requires a Kaggle API token in ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d mostafaabla/garbage-classification -p data/ --unzip

# Option 2: download the zip from the Kaggle page and extract it into data/
```

Expected layout (one folder per class):

```
data/garbage_classification/
├── battery/      ├── clothes/      ├── paper/      ├── trash/
├── biological/   ├── green-glass/  ├── plastic/    └── white-glass/
├── brown-glass/  ├── metal/        ├── shoes/
├── cardboard/
```

A different location can be used with the `DATA_DIR` environment variable.

## Samples

`samples/` contains a few images to quickly test the model:

```bash
python -m src.predict data/samples/*.jpg
```
