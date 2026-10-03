# Handwritten Digit Recognizer (CNN)

A Convolutional Neural Network that classifies handwritten digits (0-9),
built with TensorFlow/Keras and trained on the scikit-learn `digits`
dataset (1,797 8x8 grayscale images — a bundled, MNIST-style corpus that
needs no download, so the project runs anywhere with zero setup).

## Result

**99.17% test accuracy** on a held-out set of 360 images.

| | |
|---|---|
| Test accuracy | 0.9917 |
| Test loss | 0.0208 |
| Architecture | 2x Conv2D + BatchNorm → MaxPool → Dropout → Dense(128) → Dense(10, softmax) |

See `outputs/classification_report.txt` and `outputs/metrics.json` for the
full breakdown.

## Project structure

```
digit-recognizer/
├── digit_recognizer.py       # data loading, model, training, evaluation
├── requirements.txt
├── model/
│   └── digit_cnn.keras       # trained model weights
└── outputs/
    ├── training_curves.png       # loss/accuracy over epochs
    ├── confusion_matrix.png      # per-class performance
    ├── sample_predictions.png    # example predictions (green=correct, red=wrong)
    ├── classification_report.txt
    └── metrics.json
```

## How it works

1. **Data** — `sklearn.datasets.load_digits()` loads 1,797 images (8x8 px,
   values 0-16), which are normalized to `[0, 1]` and reshaped to
   `(8, 8, 1)` for the CNN's input.
2. **Split** — 72% train / 8% validation / 20% test, stratified by class.
3. **Model** — a small CNN (two convolutional blocks, batch norm, max
   pooling, and dropout for regularization) trained with the Adam
   optimizer and early stopping on validation loss.
4. **Evaluation** — accuracy, a full classification report (precision /
   recall / F1 per digit), and a confusion matrix to see exactly which
   digits get confused for which.

## Running it

```bash
pip install -r requirements.txt
python digit_recognizer.py
```

This retrains the model from scratch (takes under a minute on CPU) and
regenerates everything in `outputs/`.

## Using the real 28x28 MNIST dataset instead

The pipeline is written so the dataset is the only thing that needs to
change. Swap the `load_data()` function for:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

and adjust `input_shape` in `build_model()` to `(28, 28, 1)`. Everything
else (architecture, training loop, evaluation, plots) works unchanged.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn.
