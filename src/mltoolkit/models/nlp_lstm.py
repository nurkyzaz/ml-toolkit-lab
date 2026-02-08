import torch
import torch.nn as nn

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hid_dim: int = 128, num_classes: int = 2, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hid_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        emb = self.embedding(x)  # [B, T, E]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)

        # h: [2, B, H] because bidirectional
        h_fwd = h[0]
        h_bwd = h[1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2H]
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)  # [B, C]
        return logits
