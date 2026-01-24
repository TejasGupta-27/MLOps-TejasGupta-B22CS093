import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

IMG_SIZE = 224

# ==================== Q1(b): SVM ====================

def get_flat_data(dataset_name, max_samples=10000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if dataset_name == 'MNIST':
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_data.data.numpy().reshape(-1, 28*28)[:max_samples] / 255.0
    y_train = train_data.targets.numpy()[:max_samples]
    X_test = test_data.data.numpy().reshape(-1, 28*28) / 255.0
    y_test = test_data.targets.numpy()

    return X_train, y_train, X_test, y_test

def run_svm_experiments():
    svm_configs = [
        {'kernel': 'poly', 'degree': 2, 'C': 1.0},
        {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        {'kernel': 'poly', 'degree': 4, 'C': 1.0},
        {'kernel': 'rbf', 'gamma': 'scale', 'C': 1.0},
        {'kernel': 'rbf', 'gamma': 'auto', 'C': 1.0},
        {'kernel': 'rbf', 'gamma': 0.01, 'C': 1.0},
        {'kernel': 'rbf', 'gamma': 'scale', 'C': 10.0},
    ]

    svm_results = []

    for dataset_name in ['MNIST', 'FashionMNIST']:
        print(f"\n{'='*60}\nSVM on {dataset_name}\n{'='*60}")

        X_train, y_train, X_test, y_test = get_flat_data(dataset_name, max_samples=10000)
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        for config in svm_configs:
            kernel = config['kernel']

            if kernel == 'poly':
                svm = SVC(kernel=kernel, degree=config['degree'], C=config['C'])
                config_str = f"kernel={kernel}, degree={config['degree']}, C={config['C']}"
            else:
                svm = SVC(kernel=kernel, gamma=config['gamma'], C=config['C'])
                config_str = f"kernel={kernel}, gamma={config['gamma']}, C={config['C']}"

            print(f"\nTraining: {config_str}")

            start_time = time.time()
            svm.fit(X_train, y_train)
            train_time = (time.time() - start_time) * 1000

            y_pred = svm.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred) * 100

            print(f"Accuracy: {test_acc:.2f}%, Time: {train_time:.2f}ms")

            svm_results.append({
                'Dataset': dataset_name, 'Kernel': kernel, 'Config': config_str,
                'Test Accuracy (%)': test_acc, 'Train Time (ms)': train_time
            })

    df_svm = pd.DataFrame(svm_results)
    print("\n" + "="*80 + "\nQ1(b) SVM Results Summary\n" + "="*80)
    print(df_svm.to_string())
    df_svm.to_csv('part3_svm_results.csv', index=False)
    return df_svm

# ==================== Q2: CPU vs GPU ====================

def get_data_loaders(batch_size, pin_memory=False):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1, use_amp=True):
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
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

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

def count_flops(model):
    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, 224, 224)
        flops, params = profile(model.cpu(), inputs=(input_tensor,), verbose=False)
        return flops
    except:
        return "N/A"

def run_cpu_gpu_experiments():
    q2_models = ['ResNet-18', 'ResNet-50']

    flops_dict = {}
    for model_name in q2_models:
        model = get_model(model_name)
        flops = count_flops(model)
        flops_dict[model_name] = flops
        print(f"{model_name} FLOPs: {flops:,.0f}" if isinstance(flops, (int, float)) else f"{model_name} FLOPs: {flops}")

    q2_configs = [
        {'compute': 'CPU', 'batch_size': 16, 'optimizer': 'SGD', 'lr': 0.001},
        {'compute': 'CPU', 'batch_size': 16, 'optimizer': 'Adam', 'lr': 0.001},
        {'compute': 'GPU', 'batch_size': 16, 'optimizer': 'SGD', 'lr': 0.001},
        {'compute': 'GPU', 'batch_size': 16, 'optimizer': 'Adam', 'lr': 0.001},
    ]

    q2_results = []
    num_epochs_q2 = 1

    for config in q2_configs:
        compute = config['compute']
        batch_size = config['batch_size']
        opt_name = config['optimizer']
        lr = config['lr']

        if compute == 'GPU' and not torch.cuda.is_available():
            print(f"Skipping GPU - CUDA not available")
            continue

        device = torch.device('cuda' if compute == 'GPU' else 'cpu')
        use_amp = (compute == 'GPU')

        print(f"\n{'='*60}\nCompute: {compute}, BS: {batch_size}, Opt: {opt_name}, LR: {lr}\n{'='*60}")

        for model_name in q2_models:
            print(f"\n--- {model_name} ---")

            train_loader, val_loader, test_loader = get_data_loaders(batch_size, pin_memory=(compute == 'GPU'))
            model = get_model(model_name)
            criterion = nn.CrossEntropyLoss()

            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)

            model, train_losses, val_losses, train_accs, val_accs, train_time = train_model(
                model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs_q2, use_amp=use_amp
            )

            test_acc = evaluate_model(model, test_loader, device)

            q2_results.append({
                'Compute': compute, 'Batch Size': batch_size, 'Optimizer': opt_name,
                'Learning Rate': lr, 'Model': model_name, 'Test Accuracy (%)': test_acc,
                'Train Time (ms)': train_time, 'FLOPs': flops_dict[model_name]
            })

            print(f"Test Acc: {test_acc:.2f}%, Train Time: {train_time:.2f}ms")

    df_q2 = pd.DataFrame(q2_results)
    print("\n" + "="*80 + "\nQ2 CPU vs GPU Results Summary\n" + "="*80)
    print(df_q2.to_string())
    df_q2.to_csv('part3_cpu_gpu_results.csv', index=False)

    if len(df_q2) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = []
        cpu_times, gpu_times = [], []
        cpu_accs, gpu_accs = [], []

        for model_name in q2_models:
            model_data = df_q2[df_q2['Model'] == model_name]
            labels.append(model_name)

            cpu_data = model_data[model_data['Compute'] == 'CPU']
            gpu_data = model_data[model_data['Compute'] == 'GPU']

            cpu_times.append(cpu_data['Train Time (ms)'].mean() if len(cpu_data) > 0 else 0)
            gpu_times.append(gpu_data['Train Time (ms)'].mean() if len(gpu_data) > 0 else 0)
            cpu_accs.append(cpu_data['Test Accuracy (%)'].mean() if len(cpu_data) > 0 else 0)
            gpu_accs.append(gpu_data['Test Accuracy (%)'].mean() if len(gpu_data) > 0 else 0)

        x = np.arange(len(labels))
        width = 0.35

        axes[0].bar(x - width/2, cpu_times, width, label='CPU')
        axes[0].bar(x + width/2, gpu_times, width, label='GPU')
        axes[0].set_ylabel('Train Time (ms)')
        axes[0].set_title('Training Time: CPU vs GPU')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].legend()
        axes[0].grid(True, axis='y')

        axes[1].bar(x - width/2, cpu_accs, width, label='CPU')
        axes[1].bar(x + width/2, gpu_accs, width, label='GPU')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy: CPU vs GPU')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        axes[1].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('Q2_CPU_vs_GPU_comparison.png', dpi=150)
        plt.close()

    return df_q2

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PART 3: SVM + CPU vs GPU Experiments")
    print("="*80)

    print("\n>>> Running Q1(b) SVM Experiments...")
    df_svm = run_svm_experiments()

    print("\n>>> Running Q2 CPU vs GPU Experiments...")
    df_q2 = run_cpu_gpu_experiments()

    print("\n" + "="*80 + "\nPart 3 Summary\n" + "="*80)
    print(f"\nQ1(b) SVM:")
    print(f"  Best MNIST SVM: {df_svm[df_svm['Dataset']=='MNIST']['Test Accuracy (%)'].max():.2f}%")
    print(f"  Best FashionMNIST SVM: {df_svm[df_svm['Dataset']=='FashionMNIST']['Test Accuracy (%)'].max():.2f}%")

    print(f"\nQ2 CPU vs GPU:")
    if len(df_q2) > 0:
        cpu_results = df_q2[df_q2['Compute'] == 'CPU']
        gpu_results = df_q2[df_q2['Compute'] == 'GPU']

        if len(cpu_results) > 0:
            print(f"  Avg CPU Train Time: {cpu_results['Train Time (ms)'].mean():.2f}ms")
        if len(gpu_results) > 0:
            print(f"  Avg GPU Train Time: {gpu_results['Train Time (ms)'].mean():.2f}ms")
            if len(cpu_results) > 0:
                speedup = cpu_results['Train Time (ms)'].mean() / gpu_results['Train Time (ms)'].mean()
                print(f"  GPU Speedup: {speedup:.2f}x")

    print("\nFiles saved:")
    print("  - part3_svm_results.csv")
    print("  - part3_cpu_gpu_results.csv")
    print("  - Q2_CPU_vs_GPU_comparison.png")
