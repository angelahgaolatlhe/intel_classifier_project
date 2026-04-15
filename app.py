import os
import io
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 150
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

PYTORCH_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "angelah_model.pth")
TENSORFLOW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "angelah_model.keras")


# ── PyTorch model definition ───────────────────────────────────────────────────
# Must be defined here so we can load the saved state_dict
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ── Load models at startup ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Loading PyTorch model from: {PYTORCH_MODEL_PATH}")
pytorch_model = CNNModel().to(device)
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
pytorch_model.eval()
print("[INFO] PyTorch model loaded successfully.")

print(f"[INFO] Loading TensorFlow model from: {TENSORFLOW_MODEL_PATH}")
tf_model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)
print("[INFO] TensorFlow model loaded successfully.")


# ── Preprocessing ──────────────────────────────────────────────────────────────
pytorch_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_for_pytorch(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised PyTorch tensor (1, 3, H, W)."""
    image = image.convert("RGB")
    tensor = pytorch_transform(image)
    return tensor.unsqueeze(0).to(device)      # add batch dimension


def preprocess_for_tensorflow(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NumPy array (1, H, W, 3) for TensorFlow.
    The model's Rescaling layer handles /255 normalisation internally."""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)    # shape: (H, W, 3)
    return np.expand_dims(arr, axis=0)         # shape: (1, H, W, 3)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- Validate request ---
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file      = request.files["image"]
    framework = request.form.get("framework", "pytorch").lower()

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if framework not in ("pytorch", "tensorflow"):
        return jsonify({"error": "Invalid framework. Choose 'pytorch' or 'tensorflow'."}), 400

    # --- Read image ---
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    # --- Run inference ---
    try:
        if framework == "pytorch":
            tensor = preprocess_for_pytorch(image)
            with torch.no_grad():
                outputs = pytorch_model(tensor)
                predicted_idx = torch.argmax(outputs, dim=1).item()

        else:  # tensorflow
            arr = preprocess_for_tensorflow(image)
            predictions = tf_model.predict(arr, verbose=0)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])

        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = get_confidence(framework, predicted_idx, image)

        return jsonify({
            "class":      predicted_class,
            "confidence": confidence,
            "framework":  framework
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


def get_confidence(framework: str, predicted_idx: int, image: Image.Image) -> str:
    """Return the confidence score as a formatted percentage string."""
    if framework == "pytorch":
        tensor = preprocess_for_pytorch(image)
        with torch.no_grad():
            outputs = pytorch_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0][predicted_idx].item() * 100
    else:
        arr = preprocess_for_tensorflow(image)
        predictions = tf_model.predict(arr, verbose=0)
        confidence = float(predictions[0][predicted_idx]) * 100

    return f"{confidence:.1f}%"


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False)
