import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
DATA_DIR = "data/train/"
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick-run / debugging options
SEED = 42
USE_SMALL_SAMPLE = True
MAX_SAMPLES = 2000          # used only when USE_SMALL_SAMPLE=True
VAL_SPLIT = 0.1             # fraction of (possibly sampled) data used for validation

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = base_dataset.classes

# Optionally train on a small random subset (faster iteration)
g = torch.Generator().manual_seed(SEED)
if USE_SMALL_SAMPLE:
    n = min(MAX_SAMPLES, len(base_dataset))
    indices = torch.randperm(len(base_dataset), generator=g)[:n].tolist()
    dataset = Subset(base_dataset, indices)
else:
    dataset = base_dataset

# Train/Val split for curves
val_len = max(1, int(len(dataset) * VAL_SPLIT))
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", class_names)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(weights=None)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Training Loop
# ==========================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100 * correct / total

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / max(1, val_total)
    val_acc = 100 * val_correct / max(1, val_total)

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
    )

print("Training Complete!")

# ==========================
# Training Curves
# ==========================
epochs = list(range(1, EPOCHS + 1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(epochs, train_losses, label="Train")
ax1.plot(epochs, val_losses, label="Val")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(epochs, train_accs, label="Train")
ax2.plot(epochs, val_accs, label="Val")
ax2.set_title("Accuracy (%)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.suptitle(f"Training Curves (n={len(train_ds)} train, {len(val_ds)} val)")
fig.tight_layout()
fig.savefig("training_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved training curves to: training_curves.png")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved!")
