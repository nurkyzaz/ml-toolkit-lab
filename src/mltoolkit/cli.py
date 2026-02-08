import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ml-toolkit-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    cv = sub.add_parser("cv", help="train CIFAR-10 model")
    cv.add_argument("--epochs", type=int, default=2)
    cv.add_argument("--batch-size", type=int, default=128)
    cv.add_argument("--lr", type=float, default=3e-4)
    cv.add_argument("--dp", action="store_true")
    cv.add_argument("--epsilon", type=float, default=8.0)

    nlp = sub.add_parser("nlp", help="train sentiment model from CSV")
    nlp.add_argument("--csv", type=str, required=True)
    nlp.add_argument("--epochs", type=int, default=5)
    nlp.add_argument("--batch-size", type=int, default=32)
    nlp.add_argument("--lr", type=float, default=3e-4)

    return p
