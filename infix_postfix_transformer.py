import itertools
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformer_model import TransformerModel
from infix_postfix_dataset import InfixPostfixDataset, BatchIterator
from train_eval import train_epoch, evaluate

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
        sys.stdout.flush()

if __name__ == "__main__":
    main()
