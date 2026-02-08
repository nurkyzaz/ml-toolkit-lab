import argparse
from pathlib import Path
import torch
import torch.nn as nn

from mltoolkit.utils.seed import set_seed
from mltoolkit.data.nlp import make_sentiment_loaders
from mltoolkit.models.nlp_lstm import BiLSTMSentiment
from mltoolkit.optim.schedulers import make_optimizer, make_scheduler
from mltoolkit.training.trainer import train_epoch, eval_epoch
from mltoolkit.training.early_stopping import EarlyStopping
from mltoolkit.utils.logging import JsonlLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, vocab = make_sentiment_loaders(args.csv, batch_size=args.batch_size)
    model = BiLSTMSentiment(vocab_size=len(vocab.itos), pad_id=vocab.pad_id).to(device)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, epochs=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    stopper = EarlyStopping(patience=5, min_delta=0.0)
    logger = JsonlLogger(out_path=Path("outputs/nlp_metrics.jsonl"))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, loader, optimizer, loss_fn, device, grad_clip=1.0, task="nlp")
        va = eval_epoch(model, loader, loss_fn, device, task="nlp")  # demo uses same loader
        scheduler.step()

        best = max(best, va["acc"])
        print(f"[epoch {epoch}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | eval loss={va['loss']:.4f} acc={va['acc']:.4f}")

        logger.log({
            "task": "nlp",
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "eval_loss": va["loss"],
            "eval_acc": va["acc"],
            "vocab_size": len(vocab.itos),
        })

        if stopper.step(va["acc"]):
            print("Early stopping triggered.")
            break

    print(f"Best acc: {best:.4f}")

if __name__ == "__main__":
    main()
