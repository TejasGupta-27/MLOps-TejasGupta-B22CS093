import argparse
import torch
import torch.nn as nn
from torch.amp import autocast
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True), testset.classes


def create_model(num_classes=100, use_lora=True, rank=4, alpha=4, dropout=0.1):
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    if not use_lora:
        return model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["qkv"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def test_model(model, test_loader, device, num_classes=100):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1

    overall_acc = 100. * correct / total
    class_acc = [100. * class_correct[i] / max(class_total[i], 1) for i in range(num_classes)]
    return overall_acc, class_acc


def plot_class_accuracy(class_acc, class_names, save_path, run_name):
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(class_acc)), class_acc, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Class-wise Test Accuracy - {run_name}')
    plt.xticks(range(0, len(class_acc), 5), [class_names[i] for i in range(0, len(class_acc), 5)], rotation=45, ha='right', fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--project', type=str, default='vit-lora-cifar100')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    use_lora = not args.no_lora
    run_name = "no_lora" if not use_lora else f"lora_r{args.rank}_a{args.alpha}_d{args.dropout}"

    wandb.init(project=args.project, name=f"test_{run_name}", config=vars(args))

    test_loader, class_names = get_test_loader(args.batch_size)
    model = create_model(num_classes=100, use_lora=use_lora, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model = model.to(device)

    overall_acc, class_acc = test_model(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"Config: {run_name}")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"{'='*50}")

    wandb.log({"test_accuracy": overall_acc})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"| {'LoRA' if use_lora else 'No LoRA'} | {args.rank} | {args.alpha} | {args.dropout} | {overall_acc:.2f}% | {trainable:,} |")

    hist_path = plot_class_accuracy(class_acc, class_names, f"class_acc_{run_name}.png", run_name)
    wandb.log({"class_accuracy_histogram": wandb.Image(hist_path)})

    wandb.finish()


if __name__ == '__main__':
    main()
