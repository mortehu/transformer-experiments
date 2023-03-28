import itertools
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from random import choice, randint
import string

# Data generation
def generate_expression(max_len=10):
    operands = string.digits
    operators = "+-*/"
    n = randint(1, max_len // 2)
    expr = [choice(operands)]

    for _ in range(n):
        op = choice(operators)
        num = choice(operands)
        expr.extend([op, num])

    return "".join(expr)

def infix_to_postfix(expr):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    stack = []
    postfix = []

    for c in expr:
        if c in string.digits:
            postfix.append(c)
        elif c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()
        else:
            while stack and precedence[stack[-1]] >= precedence[c]:
                postfix.append(stack.pop())
            stack.append(c)

    while stack:
        postfix.append(stack.pop())

    return "".join(postfix) + "E"

def output_to_text(output):
    # Get the most probable character index (dimension: batch_size x seq_length)
    char_indices = torch.argmax(output, dim=-1)

    # Convert the character index to ASCII character
    text_batch = []
    for char_idx_seq in char_indices:
        text = "".join([chr(idx) for idx in char_idx_seq])
        text_batch.append(text)

    return text_batch

# Dataset and DataLoader
class InfixPostfixDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        infix = generate_expression()
        postfix = infix_to_postfix(infix)
        return infix, postfix

class BatchIterator:
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = [self.dataset[i % len(self.dataset)] for i in range(self.batch_size)]
            yield collate_fn(batch)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(src) for src in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(tgt) for tgt in tgts], dtype=torch.long)
    return srcs, src_lens, tgts, tgt_lens


class PositionalEncoding(nn.Module):
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

# Model
class TransformerModel(nn.Module):
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

# Training and evaluation
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for srcs, src_lens, tgts, tgt_lens in dataloader:
        # Encoding
        srcs = [torch.tensor([ord(c) for c in src]) for src in srcs]
        tgts_in = [torch.tensor([ord('<')] + [ord(c) for c in tgt]) for tgt in tgts]
        tgts_out = [torch.tensor([ord(c) for c in tgt] + [ord('>')]) for tgt in tgts]

        srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True).to(device)
        tgts_in = nn.utils.rnn.pad_sequence(tgts_in, batch_first=True).to(device)
        tgts_out = nn.utils.rnn.pad_sequence(tgts_out, batch_first=True).to(device)


        # Masks
        src_key_padding_mask = (srcs == 0)
        tgt_key_padding_mask = (tgts_in == 0)

        # Forward pass
        optimizer.zero_grad()
        output = model(srcs, tgts_in, src_key_padding_mask, tgt_key_padding_mask)

        for target, actual in zip(tgts, output_to_text(output)):
            try:
                actual_end = actual.index('E')
                actual = actual[:actual_end]
            except ValueError:
                pass
            print(target.rjust(20), actual)
            break

        # Loss calculation and backpropagation
        output = output.view(-1, output.shape[-1])
        tgts_out = tgts_out.view(-1)
        loss = criterion(output, tgts_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n

def evaluate(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    n = 0

    with torch.no_grad():
        for srcs, src_lens, tgts, tgt_lens in iterator:
            # Encoding
            srcs = [torch.tensor([ord(c) for c in src]) for src in srcs]
            tgts_in = [torch.tensor([ord('<')] + [ord(c) for c in tgt]) for tgt in tgts]
            tgts_out = [torch.tensor([ord(c) for c in tgt] + [ord('>')]) for tgt in tgts]

            srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True).to(device)
            tgts_in = nn.utils.rnn.pad_sequence(tgts_in, batch_first=True).to(device)
            tgts_out = nn.utils.rnn.pad_sequence(tgts_out, batch_first=True).to(device)

            # Masks
            src_key_padding_mask = (srcs == 0)
            tgt_key_padding_mask = (tgts_in == 0)

            # Forward pass
            output = model(srcs, tgts_in, src_key_padding_mask, tgt_key_padding_mask)

            # Loss calculation
            output = output.view(-1, output.shape[-1])

            tgts_out = tgts_out.view(-1)
            loss = criterion(output, tgts_out)

            total_loss += loss.item()
            n += 1

    return total_loss / n

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    batch_size = 128
    d_model = 128
    nhead = 16
    num_layers = 2

    # Data
    train_dataset = InfixPostfixDataset(1)
    train_iterator = BatchIterator(train_dataset, batch_size, num_batches=64)

    # Model, optimizer, and loss
    vocab_size = 128  # ASCII characters
    model = TransformerModel(vocab_size, d_model, nhead, num_layers).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

    # Training loop
    for epoch in itertools.count():
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, device)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")

if __name__ == "__main__":
    main()
