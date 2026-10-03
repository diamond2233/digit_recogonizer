"""
Handwritten Digit Recognizer
-----------------------------
Trains a Convolutional Neural Network (CNN) to classify handwritten digits
(0-9) using the scikit-learn `digits` dataset (a bundled, offline version of
the classic MNIST-style handwritten digit corpus: 1,797 images, 8x8 pixels,
10 classes).

The pipeline (load -> preprocess -> build CNN -> train -> evaluate -> save)
is fully self-contained and works with no internet access, which also makes
it trivial to re-run in CI or on any machine.

Swapping in the full 70,000-image, 28x28 MNIST dataset (e.g. via
`tensorflow.keras.datasets.mnist.load_data()` or a downloaded CSV) requires
no changes beyond the `load_data()` function below and the input shape.

Usage:
    python digit_recognizer.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_data():
    """Load and preprocess the digits dataset into CNN-ready tensors."""
    digits = load_digits()
    X = digits.images.astype("float32") / 16.0  # pixel values are 0-16
    X = X[..., np.newaxis]  # add channel dimension -> (N, 8, 8, 1)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_SEED, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape=(8, 8, 1), num_classes=10):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=140)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Test Set")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=140)
    plt.close(fig)


def plot_sample_predictions(X_test, y_test, y_pred, n=12):
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    idx = np.random.choice(len(X_test), n, replace=False)
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        color = "green" if y_pred[i] == y_test[i] else "red"
        ax.set_title(f"true={y_test[i]} pred={y_pred[i]}", color=color, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=140)
    plt.close(fig)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    plot_training_curves(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(X_test, y_test, y_pred)

    model.save(os.path.join(MODEL_DIR, "digit_cnn.keras"))

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\nSaved model, metrics, and plots to their respective folders.")


if __name__ == "__main__":
    main()
