# Model weights

The trained model (`model.h5`, ~84 MB) is too large for the repository. It is attached to the
[v1.0 GitHub release](https://github.com/ahmedlebbar/waste-classification-xception/releases/tag/v1.0).

```bash
# Download it into models/
curl -L -o models/model.h5 https://github.com/ahmedlebbar/waste-classification-xception/releases/download/v1.0/model.h5
```

Alternatively, retrain it with `python -m src.train` (the ImageNet Xception weights are downloaded
automatically by Keras).

A different location can be used with the `MODEL_PATH` environment variable.
