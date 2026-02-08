from dataclasses import dataclass
from typing import List, Tuple
import csv
import re
import torch
from torch.utils.data import Dataset, DataLoader

def basic_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return [t for t in text.split() if t]

@dataclass
class Vocab:
    stoi: dict
    itos: list
    pad_id: int
    unk_id: int

def build_vocab(texts: List[str], min_freq: int = 1) -> Vocab:
    freq = {}
    for t in texts:
        for tok in basic_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    itos = ["<pad>", "<unk>"]
    for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            itos.append(tok)

    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])

class SentimentCsvDataset(Dataset):
    def __init__(self, csv_path: str, vocab: Vocab, max_len: int = 64):
        self.samples: List[Tuple[List[int], int]] = []
        self.vocab = vocab
        self.max_len = max_len

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                label = int(row["label"])
                ids = [vocab.stoi.get(tok, vocab.unk_id) for tok in basic_tokenize(text)]
                ids = ids[:max_len]
                self.samples.append((ids, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_pad(batch, pad_id: int):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : x.numel()] = x
    return padded, lengths, torch.stack(ys)

def make_sentiment_loaders(csv_path: str, batch_size: int = 32, num_workers: int = 0):
    # build vocab from the same file (simple demo)
    texts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])

    vocab = build_vocab(texts, min_freq=1)
    ds = SentimentCsvDataset(csv_path=csv_path, vocab=vocab, max_len=64)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, vocab.pad_id),
    )
    return loader, vocab
