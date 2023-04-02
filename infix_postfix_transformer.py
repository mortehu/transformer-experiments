import collections
import itertools
import math
import random
import sys

import lark
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from random import choice, randint
import string

# Data generation
grammar = '''
    start: expr
    expr: expr "+" term -> add
        | expr "-" term -> sub
        | term
    term: term "*" factor -> mul
        | term "/" factor -> div
        | factor
    factor: "(" expr ")" -> parens
          | NUMBER
    %import common.NUMBER
    %import common.WS
    %ignore WS
'''

parser = lark.Lark(grammar, parser='lalr')

class GrammarRandomGenerator:
    def __init__(self, parser, max_depth=2):
        self.rules = collections.defaultdict(list)
        for rule in parser.rules:
            self.rules[rule.origin].append(rule.expansion)
        self.max_depth = max_depth

    def terminal_to_value(self, terminal_name):
        if terminal_name == "PLUS":
            return "+"
        elif terminal_name == "MINUS":
            return "-"
        elif terminal_name == "STAR":
            return "*"
        elif terminal_name == "SLASH":
            return "/"
        elif terminal_name == "LPAR":
            return "("
        elif terminal_name == "RPAR":
            return ")"
        elif terminal_name == "NUMBER":
            return str(random.randint(0, 9))
        else:
            return terminal_name

    def generate(self, symbol=lark.grammar.NonTerminal('start'), depth=0):
        if symbol not in self.rules:
            return symbol

        if depth >= self.max_depth:
            alternatives = []
            for rule in self.rules[symbol]:
                if all(child.is_term for child in rule):
                    alternatives.append(rule)
            if not alternatives:
                alternatives = self.rules[symbol]
                alternatives.sort(key=len)
                alternatives = alternatives[:1]
        else:
            alternatives = self.rules[symbol]
        chosen_alternative = random.choice(alternatives)


        generated = ''
        for child in chosen_alternative:
            if child.is_term:
                generated += self.terminal_to_value(child.name)
            else:
                generated += self.generate(symbol=child, depth=depth+1)

        return generated


class ToPostfix(lark.visitors.Interpreter):
    def add(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' +'

    def sub(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' -'

    def mul(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' *'

    def div(self, tree):
        return self.visit(tree.children[0]) + ' ' + self.visit(tree.children[1]) + ' /'

    def parens(self, tree):
        return self.visit(tree.children[0])

    def expr(self, tree):
        return self.visit(tree.children[0])

    def term(self, tree):
        return self.visit(tree.children[0])

    def factor(self, tree):
        return self.visit(tree.children[0])

    def NUMBER(self, tree):
        return tree.value

    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            return super().visit(tree)
        elif isinstance(tree, lark.Token) and tree.type == 'NUMBER':
            return self.NUMBER(tree)
        else:
            return tree

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
        self.generator = GrammarRandomGenerator(parser)
        self.to_postfix = ToPostfix()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        infix = self.generator.generate()
        postfix = self.to_postfix.visit(parser.parse(infix))[0]
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
            print(f'\033[31;3m{target}\033[m')
            print(actual)
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
    batch_size = 32
    d_model = 128
    nhead = 16
    num_layers = 2

    # Data
    train_dataset = InfixPostfixDataset(1)
    train_iterator = BatchIterator(train_dataset, batch_size, num_batches=64)

    # Model, optimizer, and loss
    vocab_size = 128  # ASCII characters
    model = TransformerModel(vocab_size, d_model, nhead, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

    # Training loop
    for epoch in itertools.count():
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, device)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")

if __name__ == "__main__":
    main()
