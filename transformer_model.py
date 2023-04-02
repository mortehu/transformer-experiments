import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional encoding for the Transformer model."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for infix to postfix conversion."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, max_position=5000):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_position, d_model) 
        self.transformer = nn.Transformer(d_model, nhead, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)

        src = self.embedding(src) * math.sqrt(self.d_model) + self.pos_embedding(src_pos)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_embedding(tgt_pos)
        out = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc(out)