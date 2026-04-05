#!/bin/bash
set -e

GPU=0

echo "===== Q1: ViT-S + LoRA on CIFAR-100 ====="

echo "--- Training without LoRA ---"
python q1/train.py --no_lora --epochs 10 --gpu $GPU

echo "--- Training with LoRA (all rank x alpha combinations) ---"
for rank in 2 4 8; do
    for alpha in 2 4 8; do
        echo "--- LoRA: rank=$rank, alpha=$alpha, dropout=0.1 ---"
        python q1/train.py --rank $rank --alpha $alpha --dropout 0.1 --epochs 10 --gpu $GPU
    done
done

echo "--- Testing all models ---"
python q1/test.py --no_lora --weights weights/no_lora_best.pth --gpu $GPU

for rank in 2 4 8; do
    for alpha in 2 4 8; do
        python q1/test.py --rank $rank --alpha $alpha --dropout 0.1 \
            --weights weights/lora_r${rank}_a${alpha}_d0.1_best.pth --gpu $GPU
    done
done

echo "--- Optuna Search ---"
python q1/optuna_search.py --n_trials 20 --gpu $GPU

echo ""
echo "===== Q2: Adversarial Attacks ====="

echo "--- Training ResNet18 on CIFAR-10 ---"
python q2/train_resnet.py --epochs 50 --gpu $GPU

echo "--- FGSM Attack ---"
python q2/fgsm_attack.py --weights weights/resnet18_cifar10.pth --gpu $GPU

echo "--- Adversarial Detector (PGD + BIM) ---"
python q2/adversarial_detector.py --weights weights/resnet18_cifar10.pth --gpu $GPU

echo ""
echo "===== All experiments complete! ====="
