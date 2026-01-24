import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

IMG_SIZE = 224

def get_data_loaders(dataset_name, batch_size, pin_memory=False):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_name == 'MNIST':
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    total_size = len(full_train)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)

    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=2)

    return train_loader, val_loader, test_loader

def get_model(model_name, num_classes=10):
    if model_name == 'ResNet-18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'ResNet-50':
        model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, use_amp=True):
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

    train_time = (time.time() - start_time) * 1000
    return model, train_losses, val_losses, train_accs, val_accs, train_time

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title(f'{title} - Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(train_accs, label='Train'); ax2.plot(val_accs, label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title(f'{title} - Accuracy'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    experiment_configs = [
        {'batch_size': 16, 'optimizer': 'SGD', 'lr': 0.001},
        {'batch_size': 16, 'optimizer': 'SGD', 'lr': 0.0001},
        {'batch_size': 16, 'optimizer': 'Adam', 'lr': 0.001},
        {'batch_size': 16, 'optimizer': 'Adam', 'lr': 0.0001},
        {'batch_size': 32, 'optimizer': 'SGD', 'lr': 0.001},
        {'batch_size': 32, 'optimizer': 'SGD', 'lr': 0.0001},
        {'batch_size': 32, 'optimizer': 'Adam', 'lr': 0.001},
        {'batch_size': 32, 'optimizer': 'Adam', 'lr': 0.0001},
    ]

    models_list = ['ResNet-18', 'ResNet-50']
    epochs_list = [1, 2]
    pin_memory_list = [False, True]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = []
    best_models = {}
    dataset_name = 'MNIST'

    print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

    for model_name in models_list:
        print(f"\n--- Model: {model_name} ---")

        for num_epochs in epochs_list:
            for pin_memory in pin_memory_list:
                for config in experiment_configs:
                    batch_size = config['batch_size']
                    opt_name = config['optimizer']
                    lr = config['lr']

                    print(f"\nConfig: BS={batch_size}, Opt={opt_name}, LR={lr}, Epochs={num_epochs}, pin_memory={pin_memory}")

                    train_loader, val_loader, test_loader = get_data_loaders(dataset_name, batch_size, pin_memory=pin_memory)
                    model = get_model(model_name)
                    criterion = nn.CrossEntropyLoss()

                    if opt_name == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                    model, train_losses, val_losses, train_accs, val_accs, train_time = train_model(
                        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, use_amp=True
                    )

                    test_acc = evaluate_model(model, test_loader, device)
                    print(f"Test Accuracy: {test_acc:.2f}%")

                    results.append({
                        'Dataset': dataset_name, 'Model': model_name, 'Batch Size': batch_size,
                        'Optimizer': opt_name, 'Learning Rate': lr, 'Epochs': num_epochs,
                        'pin_memory': pin_memory, 'Test Accuracy (%)': test_acc, 'Train Time (ms)': train_time
                    })

                    exp_key = f"{dataset_name}_{model_name}"
                    if exp_key not in best_models or test_acc > best_models[exp_key]['accuracy']:
                        best_models[exp_key] = {'model': model, 'accuracy': test_acc, 'config': config}

                    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        f"{dataset_name}_{model_name}_BS{batch_size}_{opt_name}_LR{lr}_E{num_epochs}_PM{pin_memory}")

    df_results = pd.DataFrame(results)
    print("\n" + "="*80 + "\nMNIST Results Summary\n" + "="*80)
    print(df_results.to_string())
    df_results.to_csv('part1_mnist_results.csv', index=False)

    for key, data in best_models.items():
        print(f"\nBest {key}: {data['accuracy']:.2f}%")
        torch.save(data['model'].state_dict(), f"best_model_{key}.pth")

    print("\nResults saved to part1_mnist_results.csv")
