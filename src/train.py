import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.data import get_loaders
from src.models import build_model
from src.utils import set_seed, get_device

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    pbar.set_postfix(loss=f"{total_loss / total:.3f}", acc=f"{100 * correct / total:.1f}%")
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="simple_cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.num_workers)
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} | {n_params:,} parameters | {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = f"models/{args.model}_fp32.pt"
    best_val = 0.0

    with open(f"results/train_{args.model}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "epoch_time_s"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            dt = time.perf_counter() - t0

            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_acc:.2f}", f"{lr:.2e}", f"{dt:.1f}"])
            mark = ""
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), ckpt_path)
                mark = " *"   # stjärna = ny bästa modell
            print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.3f} | "
                  f"train {train_acc:5.2f} % | val {val_acc:5.2f} % | {dt:.0f} s{mark}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    size_mb = os.path.getsize(ckpt_path) / 2**20
    print(f"\nBest val {best_val:.2f} % | TEST {test_acc:.2f} % | {size_mb:.2f} MB")


if __name__ == "__main__":
    main()