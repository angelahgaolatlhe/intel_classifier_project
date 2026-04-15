import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "seg_test")
MODEL_PATH = os.path.join(BASE_DIR, "angelah_model.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = (150, 150)
BATCH_SIZE = 32


# ── Load dataset ───────────────────────────────────────────────────────────────
def load_data():
    return tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)


# ── Confusion matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)

    plt.title("TensorFlow Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(OUTPUT_DIR, "tensorflow_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Saved: {save_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Device: {'GPU' if gpus else 'CPU'}")

    test_ds = load_data()

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nAccuracy: {acc * 100:.2f}%")

    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    evaluate()