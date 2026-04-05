import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import timm
from peft import LoraConfig, get_peft_model
import wandb
import os
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    valset.dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader


def create_model(num_classes=100, use_lora=True, rank=4, alpha=4, dropout=0.1):
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

    if not use_lora:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_lora_gradients(model, step):
    grad_norms = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.grad is not None:
            grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
    if grad_norms:
        wandb.log(grad_norms, step=step)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_lora, global_step, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        if use_lora:
            log_lora_gradients(model, global_step)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        global_step += 1

        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    return running_loss / total, 100. * correct / total, global_step


def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
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
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--project', type=str, default='vit-lora-cifar100')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    use_lora = not args.no_lora

    run_name = "no_lora" if not use_lora else f"lora_r{args.rank}_a{args.alpha}_d{args.dropout}"
    wandb.init(project=args.project, name=run_name, config=vars(args))

    train_loader, val_loader, _ = get_dataloaders(args.batch_size)
    model = create_model(num_classes=100, use_lora=use_lora, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model = model.to(device)

    trainable_params = count_trainable_params(model)
    wandb.log({"trainable_parameters": trainable_params})
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_lora, global_step, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{run_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
