import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

import random
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"
REPORT_PATH = "evaluation_report.txt"
CONFUSION_MATRIX_PNG = "confusion_matrix.png"
CONFUSION_MATRIX_NORM_PNG = "confusion_matrix_normalized.png"
SAVE_NORMALIZED_CONFUSION_MATRIX = True
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# Support either a raw state_dict or a checkpoint dict containing it
state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# Overall Accuracy
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ==========================
# Confusion Matrix + Class-wise Accuracy
# ==========================
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
per_class_total = cm.sum(axis=1)
per_class_correct = np.diag(cm)
per_class_acc = np.divide(
    per_class_correct,
    per_class_total,
    out=np.zeros_like(per_class_correct, dtype=float),
    where=per_class_total != 0,
)

print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

print("\nClass-wise Accuracy:")
for i, name in enumerate(class_names):
    print(f"- {name}: {per_class_acc[i] * 100:.2f}% ({per_class_correct[i]}/{per_class_total[i]})")

def _plot_confusion_matrix(cm_to_plot, out_path, title, normalize=False):
    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm_to_plot.astype(float), row_sums, out=np.zeros_like(cm_to_plot, dtype=float), where=row_sums != 0)
    else:
        cm_disp = cm_to_plot

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_disp, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = float(np.max(cm_disp)) / 2.0 if cm_disp.size else 0.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

_plot_confusion_matrix(cm, CONFUSION_MATRIX_PNG, title="Confusion Matrix", normalize=False)
print(f"\nSaved confusion matrix plot to: {CONFUSION_MATRIX_PNG}")

if SAVE_NORMALIZED_CONFUSION_MATRIX:
    _plot_confusion_matrix(cm, CONFUSION_MATRIX_NORM_PNG, title="Confusion Matrix (Normalized)", normalize=True)
    print(f"Saved normalized confusion matrix plot to: {CONFUSION_MATRIX_NORM_PNG}")

# ==========================
# F1 Score
# ==========================
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

report_str = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:")
print(report_str)

# ==========================
# Save Report
# ==========================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Data: {DATA_DIR}\n")
    f.write(f"Device: {DEVICE}\n\n")
    f.write(f"Overall Accuracy: {overall_acc * 100:.2f}%\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write("Confusion Matrix (rows=true, cols=pred):\n")
    f.write(np.array2string(cm))
    f.write("\n\nClass-wise Accuracy:\n")
    for i, name in enumerate(class_names):
        f.write(f"- {name}: {per_class_acc[i] * 100:.2f}% ({per_class_correct[i]}/{per_class_total[i]})\n")
    f.write("\nClassification Report:\n")
    f.write(report_str)

print(f"\nSaved evaluation report to: {REPORT_PATH}")


def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")



test_image_path = os.path.join(DATA_DIR, "5/340.png")
predict_single_image(test_image_path)


# Pick random image from dataset
random_image_path, _ = random.choice(dataset.samples)
predict_single_image(random_image_path)