import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
from dotenv import load_dotenv
from train_resnet import create_resnet18

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_test_data(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True), testset


def fgsm_attack_scratch(model, images, labels, epsilon, device):
    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return perturbed.detach()


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def evaluate_adversarial_scratch(model, loader, epsilon, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc=f"FGSM scratch eps={epsilon}"):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        adv_inputs = fgsm_attack_scratch(model, inputs, targets, epsilon, device)
        with torch.no_grad():
            outputs = model(adv_inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def evaluate_adversarial_art(classifier, test_data_np, test_labels_np, epsilon):
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, batch_size=512)
    adv_images = attack.generate(x=test_data_np)
    predictions = classifier.predict(adv_images, batch_size=512)
    acc = 100. * np.mean(np.argmax(predictions, axis=1) == test_labels_np)
    return acc, adv_images


def denormalize(tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def visualize_samples(model, testset, epsilon, device, classifier, test_np, save_path):
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))

    for i in range(10):
        img, label = testset[i]
        img_t = img.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)

        orig = denormalize(img)
        axes[0][i].imshow(orig.permute(1, 2, 0).numpy())
        axes[0][i].set_title(CLASSES[label], fontsize=8)
        axes[0][i].axis('off')

        adv_scratch = fgsm_attack_scratch(model, img_t, label_t, epsilon, device)
        adv_scratch_img = denormalize(adv_scratch.squeeze(0))
        with torch.no_grad():
            pred_scratch = model(adv_scratch).argmax(1).item()
        axes[1][i].imshow(adv_scratch_img.permute(1, 2, 0).numpy())
        axes[1][i].set_title(f"{CLASSES[pred_scratch]}", fontsize=8, color='red' if pred_scratch != label else 'green')
        axes[1][i].axis('off')

        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        img_np = test_np[i:i+1]
        adv_art_np = attack.generate(x=img_np)
        pred_art = np.argmax(classifier.predict(adv_art_np), axis=1)[0]
        adv_art_t = torch.tensor(adv_art_np[0])
        adv_art_img = denormalize(adv_art_t)
        axes[2][i].imshow(adv_art_img.permute(1, 2, 0).numpy())
        axes[2][i].set_title(f"{CLASSES[pred_art]}", fontsize=8, color='red' if pred_art != label else 'green')
        axes[2][i].axis('off')

    axes[0][0].set_ylabel("Original", fontsize=10)
    axes[1][0].set_ylabel("FGSM Scratch", fontsize=10)
    axes[2][0].set_ylabel("FGSM ART", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/resnet18_cifar10.pth')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--project', type=str, default='adversarial-cifar10')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    wandb.init(project=args.project, name='fgsm_attack', config=vars(args))

    model = create_resnet18().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    test_loader, testset = get_test_data(args.batch_size)

    clean_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model, loss=criterion, input_shape=(3, 32, 32), nb_classes=10,
        optimizer=None, clip_values=None,
    )

    test_images, test_labels = [], []
    for img, label in testset:
        test_images.append(img.numpy())
        test_labels.append(label)
    test_np = np.array(test_images)
    labels_np = np.array(test_labels)

    epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    results = []

    for eps in epsilons:
        if eps == 0.0:
            acc_scratch = clean_acc
            acc_art = clean_acc
        else:
            acc_scratch = evaluate_adversarial_scratch(model, test_loader, eps, device)
            acc_art, _ = evaluate_adversarial_art(classifier, test_np, labels_np, eps)

        results.append({'epsilon': eps, 'clean': clean_acc, 'scratch': acc_scratch, 'art': acc_art})
        wandb.log({'epsilon': eps, 'acc_clean': clean_acc, 'acc_fgsm_scratch': acc_scratch, 'acc_fgsm_art': acc_art})
        print(f"eps={eps:.2f} | Clean: {clean_acc:.2f}% | Scratch: {acc_scratch:.2f}% | ART: {acc_art:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot([r['epsilon'] for r in results], [r['clean'] for r in results], 'g-o', label='Clean')
    plt.plot([r['epsilon'] for r in results], [r['scratch'] for r in results], 'r-s', label='FGSM Scratch')
    plt.plot([r['epsilon'] for r in results], [r['art'] for r in results], 'b-^', label='FGSM ART')
    plt.xlabel('Perturbation Strength (epsilon)')
    plt.ylabel('Accuracy (%)')
    plt.title('FGSM Attack: Perturbation Strength vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('fgsm_eps_comparison.png', dpi=150)
    plt.close()
    wandb.log({"eps_comparison": wandb.Image('fgsm_eps_comparison.png')})

    vis_path = visualize_samples(model, testset, 0.1, device, classifier, test_np, 'fgsm_visual_comparison.png')
    wandb.log({"visual_comparison": wandb.Image(vis_path)})

    wandb.log({"fgsm_10_samples": [
        wandb.Image(denormalize(testset[i][0]).permute(1, 2, 0).numpy(), caption=f"Clean: {CLASSES[testset[i][1]]}")
        for i in range(10)
    ]})

    wandb.finish()
    print("\nResults Table:")
    print(f"{'Epsilon':<10} {'Clean':<10} {'Scratch':<10} {'ART':<10}")
    for r in results:
        print(f"{r['epsilon']:<10.2f} {r['clean']:<10.2f} {r['scratch']:<10.2f} {r['art']:<10.2f}")


if __name__ == '__main__':
    main()
