import argparse
import torch
import torch.nn as nn

from mltoolkit.utils.seed import set_seed
from mltoolkit.data.cv import make_cifar10_loaders
from mltoolkit.models.cv_resnet import ResNet18Cifar
from mltoolkit.optim.schedulers import make_optimizer, make_scheduler
from mltoolkit.privacy.dp import maybe_make_private
from mltoolkit.training.trainer import train_epoch, eval_epoch
from mltoolkit.training.early_stopping import EarlyStopping
from mltoolkit.utils.logging import JsonlLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--epsilon", type=float, default=8.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = make_cifar10_loaders(batch_size=args.batch_size)
    model = ResNet18Cifar(num_classes=10).to(device)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, epochs=args.epochs)

    # optional DP
    model, optimizer, train_loader, dp_engine = maybe_make_private(
        model, optimizer, train_loader,
        enabled=args.dp,
        epsilon=args.epsilon,
        delta=1e-5,
        max_grad_norm=1.0,
    )

    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=5, min_delta=0.0)
    logger = JsonlLogger(out_path=torch.path.Path("outputs/cv_metrics.jsonl") if hasattr(torch, "path") else __import__("pathlib").Path("outputs/cv_metrics.jsonl"))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip=1.0, task="cv")
        va = eval_epoch(model, test_loader, loss_fn, device, task="cv")
        scheduler.step()

        best = max(best, va["acc"])
        print(f"[epoch {epoch}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f}")

        logger.log({
            "task": "cv",
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "dp": bool(args.dp),
            "epsilon_target": args.epsilon if args.dp else None,
        })

        if stopper.step(va["acc"]):
            print("Early stopping triggered.")
            break

    print(f"Best val acc: {best:.4f}")

if __name__ == "__main__":
    main()
