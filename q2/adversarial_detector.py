import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
from dotenv import load_dotenv
from train_resnet import create_resnet18

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_cifar10_numpy():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_images, train_labels = [], []
    for img, label in trainset:
        train_images.append(img.numpy())
        train_labels.append(label)

    test_images, test_labels = [], []
    for img, label in testset:
        test_images.append(img.numpy())
        test_labels.append(label)

    return (np.array(train_images), np.array(train_labels),
            np.array(test_images), np.array(test_labels))


def create_resnet34_detector():
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 2)
    return model


def generate_adversarial(classifier, images, attack_type, eps=0.15, eps_step=0.02, max_iter=20, batch_size=512):
    if attack_type == 'pgd':
        attack = ProjectedGradientDescent(
            estimator=classifier, eps=eps, eps_step=eps_step,
            max_iter=max_iter, batch_size=batch_size,
        )
    elif attack_type == 'bim':
        attack = BasicIterativeMethod(
            estimator=classifier, eps=eps, eps_step=eps_step,
            max_iter=max_iter, batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_type}")

    adv_images = attack.generate(x=images)
    return adv_images


def prepare_detector_data(clean_images, adv_images):
    n = len(clean_images)
    X = np.concatenate([clean_images, adv_images], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)], axis=0)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    return (DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True))


def train_detector(model, train_loader, test_loader, device, epochs=25, lr=0.001, attack_name=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Detector [{attack_name}] Epoch {epoch+1}"):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100. * correct / total

        wandb.log({
            f"{attack_name}_det_train_loss": running_loss / len(train_loader.dataset),
            f"{attack_name}_det_train_acc": train_acc,
            f"{attack_name}_det_test_acc": test_acc,
            "epoch": epoch + 1,
        })
        print(f"  Epoch {epoch+1} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    return test_acc


def denormalize_np(images):
    mean = np.array(CIFAR10_MEAN).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD).reshape(1, 3, 1, 1)
    return np.clip(images * std + mean, 0, 1)


def log_samples(clean_np, adv_np, labels, attack_name, n=10):
    clean_vis = denormalize_np(clean_np[:n])
    adv_vis = denormalize_np(adv_np[:n])

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0][i].imshow(clean_vis[i].transpose(1, 2, 0))
        axes[0][i].set_title(CLASSES[labels[i]], fontsize=8)
        axes[0][i].axis('off')
        axes[1][i].imshow(adv_vis[i].transpose(1, 2, 0))
        axes[1][i].set_title(f"{attack_name}", fontsize=8)
        axes[1][i].axis('off')
    axes[0][0].set_ylabel("Clean", fontsize=10)
    axes[1][0].set_ylabel("Adversarial", fontsize=10)
    plt.tight_layout()
    path = f"samples_{attack_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    wandb.log({f"{attack_name}_samples": wandb.Image(path)})

    wandb.log({f"{attack_name}_10_clean": [
        wandb.Image(clean_vis[i].transpose(1, 2, 0), caption=f"Clean: {CLASSES[labels[i]]}")
        for i in range(n)
    ]})
    wandb.log({f"{attack_name}_10_adv": [
        wandb.Image(adv_vis[i].transpose(1, 2, 0), caption=f"Adv ({attack_name}): {CLASSES[labels[i]]}")
        for i in range(n)
    ]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/resnet18_cifar10.pth')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--project', type=str, default='adversarial-cifar10')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    wandb.init(project=args.project, name='adversarial_detector', config=vars(args))

    source_model = create_resnet18().to(device)
    source_model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    source_model.eval()

    classifier = PyTorchClassifier(
        model=source_model, loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32), nb_classes=10, optimizer=None,
    )

    print("Loading CIFAR-10 data...")
    train_images, train_labels, test_images, test_labels = get_cifar10_numpy()
    n = min(args.n_samples, len(train_images))
    subset_images = train_images[:n]
    subset_labels = train_labels[:n]

    results = {}
    for attack_name in ['pgd', 'bim']:
        print(f"\n{'='*50}")
        print(f"Generating {attack_name.upper()} adversarial examples...")
        adv_images = generate_adversarial(classifier, subset_images, attack_name)

        log_samples(subset_images, adv_images, subset_labels, attack_name)

        print(f"Training {attack_name.upper()} detector...")
        detector = create_resnet34_detector().to(device)
        train_loader, test_loader = prepare_detector_data(subset_images, adv_images)
        det_acc = train_detector(detector, train_loader, test_loader, device,
                                 epochs=args.epochs, attack_name=attack_name)
        results[attack_name] = det_acc

        os.makedirs('weights', exist_ok=True)
        torch.save(detector.state_dict(), f"weights/{attack_name}_detector.pth")
        print(f"{attack_name.upper()} Detector Accuracy: {det_acc:.2f}%")

    print(f"\n{'='*50}")
    print("Detection Results Comparison:")
    print(f"{'Attack':<10} {'Detection Accuracy':<20}")
    for attack, acc in results.items():
        print(f"{attack.upper():<10} {acc:.2f}%")

    wandb.log({"pgd_detection_acc": results.get('pgd', 0), "bim_detection_acc": results.get('bim', 0)})
    wandb.finish()


if __name__ == '__main__':
    main()

