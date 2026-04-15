import os
import io
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── Safe PyTorch import ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 150
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "angelah_model.pth")
TENSORFLOW_MODEL_PATH = os.path.join(BASE_DIR, "angelah_model.keras")

# ── Global model placeholders (IMPORTANT for lazy loading) ────────────────────
pytorch_model = None
tf_model = None

# ── PyTorch model definition ───────────────────────────────────────────────────
if TORCH_AVAILABLE:
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

# ── Load models lazily (THIS FIXES RENDER LOOP) ───────────────────────────────
def load_models():
    global pytorch_model, tf_model

    # Load TensorFlow model once
    if tf_model is None:
        print("[INFO] Loading TensorFlow model...")
        tf_model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)
        print("[INFO] TensorFlow model loaded.")

    # Load PyTorch model only if available
    if TORCH_AVAILABLE and pytorch_model is None:
        print("[INFO] Loading PyTorch model...")
        device = torch.device("cpu")

        model = CNNModel().to(device)
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
        model.eval()

        pytorch_model = model
        print("[INFO] PyTorch model loaded.")

# ── TensorFlow preprocessing ──────────────────────────────────────────────────
def preprocess_tf(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ── PyTorch preprocessing ─────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    pytorch_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def preprocess_torch(image):
    image = image.convert("RGB")
    tensor = pytorch_transform(image)
    return tensor.unsqueeze(0)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    load_models()  

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    framework = request.form.get("framework", "pytorch").lower()

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        # ── PyTorch prediction ───────────────────────────────────────────────
        if framework == "pytorch":
            if not TORCH_AVAILABLE:
                return jsonify({"error": "PyTorch not available"}), 500

            tensor = preprocess_torch(image)

            with torch.no_grad():
                outputs = pytorch_model(tensor)
                predicted_idx = torch.argmax(outputs, dim=1).item()

        # ── TensorFlow prediction ───────────────────────────────────────────
        else:
            arr = preprocess_tf(image)
            predictions = tf_model.predict(arr, verbose=0)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])

        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = get_confidence(framework, predicted_idx, image)

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "framework": framework
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Confidence calculation ────────────────────────────────────────────────────
def get_confidence(framework, predicted_idx, image):
    if framework == "pytorch":
        tensor = preprocess_torch(image)
        with torch.no_grad():
            outputs = pytorch_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0][predicted_idx].item() * 100
    else:
        arr = preprocess_tf(image)
        predictions = tf_model.predict(arr, verbose=0)
        confidence = float(predictions[0][predicted_idx]) * 100

    return f"{confidence:.1f}%"

if __name__ == "__main__":
    app.run(debug=False)
