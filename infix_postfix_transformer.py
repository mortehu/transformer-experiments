import itertools
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sqlite3
import subprocess

from transformer_model import TransformerModel
from infix_postfix_dataset import InfixPostfixDataset, BatchIterator
from train_eval import train_epoch, evaluate

def get_git_info():
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    uncommitted_changes = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    return git_hash, uncommitted_changes

def create_database():
    conn = sqlite3.connect('runs.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  git_hash TEXT,
                  uncommitted_changes BOOLEAN)''')

    c.execute('''CREATE TABLE IF NOT EXISTS epochs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id INTEGER,
                  epoch INTEGER,
                  train_loss REAL,
                  learning_rate REAL,
                  FOREIGN KEY(run_id) REFERENCES runs(id))''')

    conn.commit()
    return conn

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

    # Create database and log run information
    conn = create_database()
    c = conn.cursor()
    git_hash, uncommitted_changes = get_git_info()
    c.execute("INSERT INTO runs (git_hash, uncommitted_changes) VALUES (?, ?)", (git_hash, uncommitted_changes))
    run_id = c.lastrowid
    conn.commit()

    # Training loop
    for epoch in itertools.count():
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, device)

        # Log epoch metrics
        learning_rate = optimizer.param_groups[0]['lr']
        c.execute("INSERT INTO epochs (run_id, epoch, train_loss, learning_rate) VALUES (?, ?, ?, ?)",
                  (run_id, epoch, train_loss, learning_rate))
        conn.commit()

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Learning Rate: {learning_rate:.6f}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
