import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "seg_test")
MODEL_PATH = os.path.join(BASE_DIR, "angelah_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE    = 150
BATCH_SIZE  = 32


# ── Model definition ───────────────────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
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


# ── Confusion Matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)

    plt.title("PyTorch Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(OUTPUT_DIR, "pytorch_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Saved: {save_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_data = datasets.ImageFolder(DATA_PATH, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Samples: {len(test_data)}")
    print(f"Classes: {test_data.classes}")

    model = CNNModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(f"\nAccuracy: {100 * correct / total:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    evaluate()