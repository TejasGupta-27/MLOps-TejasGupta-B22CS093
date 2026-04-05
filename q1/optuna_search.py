import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import optuna
import wandb
import os
from dotenv import load_dotenv
from train import get_dataloaders, create_model, train_one_epoch, evaluate, count_trainable_params

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

GPU_ID = 0


def objective(trial):
    rank = trial.suggest_categorical('rank', [2, 4, 8])
    alpha = trial.suggest_categorical('alpha', [2, 4, 8])
    dropout = 0.1
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    device = torch.device(f'cuda:{GPU_ID}')
    train_loader, val_loader, _ = get_dataloaders(batch_size=128)

    model = create_model(num_classes=100, use_lora=True, rank=rank, alpha=alpha, dropout=dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    run_name = f"optuna_r{rank}_a{alpha}_lr{lr:.6f}"
    wandb.init(project='vit-lora-cifar100-optuna', name=run_name, config={
        'rank': rank, 'alpha': alpha, 'dropout': dropout, 'lr': lr,
    }, reinit=True)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(10):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, True, global_step, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        best_val_acc = max(best_val_acc, val_acc)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.finish()
    return best_val_acc


def main():
    global GPU_ID
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    GPU_ID = args.gpu

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "="*50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Val Acc): {trial.value:.2f}%")
    print(f"  Params: {trial.params}")

    best = trial.params
    device = torch.device(f'cuda:{GPU_ID}')
    train_loader, val_loader, _ = get_dataloaders(batch_size=128)
    model = create_model(num_classes=100, use_lora=True, rank=best['rank'], alpha=best['alpha'], dropout=0.1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=best['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    wandb.init(project='vit-lora-cifar100', name=f"best_optuna_r{best['rank']}_a{best['alpha']}", config=best)

    global_step = 0
    for epoch in range(10):
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, True, global_step, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc})

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_optuna_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")
    wandb.finish()


if __name__ == '__main__':
    main()
