import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_dataloaders(batch_size=256):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader


def create_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def train(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
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
    return running_loss / total, 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Eval"):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    wandb.init(project='adversarial-cifar10', name='resnet18_train', config=vars(args))

    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = create_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })
        print(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'resnet18_cifar10.pth'))

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    wandb.finish()


if __name__ == '__main__':
    main()
