import torch
import torch.nn as nn

def output_to_text(output):
    # Get the most probable character index (dimension: batch_size x seq_length)
    char_indices = torch.argmax(output, dim=-1)

    # Convert the character index to ASCII character
    text_batch = []
    for char_idx_seq in char_indices:
        text = "".join([chr(idx) for idx in char_idx_seq])
        text_batch.append(text)

    return text_batch

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
                actual = actual[:(actual_end + 1)]
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
