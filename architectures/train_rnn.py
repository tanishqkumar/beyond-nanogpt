'''
Recurrent Neural Networks (RNNs) are the OG sequence model, and while they've been largely supplanted by transformers,
they remain instructive for a few key reasons:
1) They are the simplest possible way to process sequences, just iteratively updating a fixed-size state vector
2) They demonstrate key concepts like vanishing/exploding gradients that plague all sequence models
3) They show why normalization (here LayerNorm) is critical for stable training of deep nets
4) The residual connections (0.1 * input) show an early example of skip connections that are now ubiquitous

The architecture here is a simple multi-layer RNN with LayerNorm and residual connections. While basic,
it demonstrates core sequence modeling concepts that carry over directly to transformers and other modern architectures. Also, 
we'll see in Linear Attention that there is a interpretation of Transfomers as RNNs if you modify their 
architecture a little bit. 
'''

import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import string
import math
import os
import wandb
from datasets import load_dataset
from tqdm import tqdm

# notice how simple an RNN is to implement because of parameter sharing, compared to a transformer 
class RNN(nn.Module):  # [b, s] -> [b, s, v] with a classification head on the final hidden state
    def __init__(self, embed_dim=512, state_size=512, V=100, nlayers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.nlayers = nlayers
        self.embeddings = nn.Embedding(V, embed_dim)
        self.state_size = state_size
        self.Wxs = nn.Linear(embed_dim, state_size)
        self.Wss = nn.ModuleList([
            nn.Linear(state_size, state_size) for _ in range(nlayers)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(state_size) for _ in range(nlayers)
        ])
        self.Wsy = nn.Linear(state_size, V)
        self.act = nn.GELU()

    def forward(self, x):  # x: [b, s] of token ids -> returns list of logits [b, s, v] and last token logits [b, V]
        b, s = x.shape
        h = [torch.zeros(b, self.state_size, device=x.device) for _ in range(self.nlayers)]
        embeds = self.embeddings(x)  # [b, s, embed_dim]
        outputs = []
        for i in range(s):
            input = self.act(self.Wxs(embeds[:, i]))
            for layer in range(self.nlayers):
                h[layer] = self.lns[layer](
                    self.act(self.Wss[layer](input if layer == 0 else h[layer-1]) + 0.1 * (input if layer == 0 else h[layer]))
                )
            outputs.append(self.Wsy(h[-1]))
        return torch.stack(outputs, dim=1), self.Wsy(h[-1])  # [b, s, V], [b, V]

# char level vocab for simplicity
chars = string.printable
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Custom collate function for character encoding, not too important
def collate_chars(batch):
    texts = [item['text'] for item in batch]
    encoded = [[char_to_idx[c] for c in text if c in chars] for text in texts]
    max_len = max(len(seq) for seq in encoded)
    padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
    return torch.tensor(padded)

def main():
    parser = argparse.ArgumentParser(description="Train RNN on TinyStories dataset")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--state_size", type=int, default=1024, help="RNN state size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=4001, help="Optional limit on training steps")
    parser.add_argument("--nlayers", type=int, default=4, help="Number of RNN layers")
    parser.add_argument("--verbose", action="store_true", help="Print additional training details")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()


    if args.wandb:
        # you may have to login with your wandb credential here 
        wandb.init(project="train_rnn", config=vars(args))

    if args.verbose:
        print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if args.verbose:
        print("Creating dataloader...")
    train_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_chars)

    if args.verbose:
        print("Initializing model and training components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(embed_dim=args.embed_dim, state_size=args.state_size, V=vocab_size, nlayers=args.nlayers).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lr)
    
    # we use a cosine lr schedule since important for RNN 
    total_steps = len(train_dataloader) if args.num_steps is None else args.num_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (step / warmup_steps)
            if step < warmup_steps
            else (0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
                  if step < total_steps else 0)
    )
    
    criterion = nn.CrossEntropyLoss()

    # define BOS token as the last vocab index
    bos_token = vocab_size - 1

    total_loss = 0.0
    steps = 0

    # Create directory for periodic checkpoints
    os.makedirs('rnn_checkpoints', exist_ok=True)

    if args.verbose:
        print("Starting training loop...")
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        x = batch.to(device)
        
        # Add BOS token at the start of each sequence
        bos = torch.full((x.shape[0], 1), bos_token, device=device)
        x_with_bos = torch.cat([bos, x], dim=1)
        
        # x_with_bos shape: [batch_size, seq_len+1]
        batch_logits, _ = model(x_with_bos)  # batch_logits shape: [batch_size, seq_len+1, vocab_size]
        targets = x_with_bos[:, 1:]  # shape: [batch_size, seq_len]
        # Reshape logits to [batch_size*seq_len, vocab_size] and targets to [batch_size*seq_len]
        loss = criterion(batch_logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

        # log to wandb 
        if args.wandb:
            wandb.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}, step=steps)

        if steps % 10 == 0:
            print(f"Step {steps}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
        if steps % 1000 == 0:
            checkpoint_path = f"hypers/{steps}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            
        if args.num_steps is not None and steps >= args.num_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else float('nan')
    print(f"Average Loss: {avg_loss:.4f}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()