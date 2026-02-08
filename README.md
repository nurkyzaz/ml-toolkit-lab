# ml-toolkit-lab
ML research toolkit with modular training engine (PyTorch), CV + NLP pipelines, differential privacy (DP-SGD), and optimization utilities

This repo is a compact ML research toolkit demonstrating:
- **Computer Vision**: CIFAR-10 classification with ResNet18
- **NLP / NLU**: Sentiment classification from CSV with BiLSTM
- **Optimization**: modular optimizer + scheduler + gradient clipping + early stopping
- **Privacy**: optional Differential Privacy (DP-SGD) using Opacus
- **C++**: a small SGD demo implementation

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

