'''
[https://www.bioinf.jku.at/publications/older/2604.pdf] Long Short-Term Memory 

This is basically the same as train_rnn.py but with the LSTM architecture, since you can think of that 
as an RNN++. It was built to address the fact that a naive RNN will be hard to train because the product of layers 
term in the closed form RNN gradient will lead to exploding/vanishing gradients. 

The approach it takes is pretty interesting to me, coming from a Transformer/LLM prior: it adds a shit-ton of inductive bias, 
basically it introduces a notion of a "cell" c_t that is meant to be the source of information moving through the model 
-- you can think of this as a primitive "residual stream" except it doesn't explicitly have resiudal connections 
(that came in the ResNet paper in 2016, whereas this was published in 1992). Instead of using residuals for training stability, 
it basically hardcodes the "allowable read/write" structure in every state update. And now instead of h_t the state being
used to both transmit information across time as well as store information to be read out in self.Wsy,
 the two roles are separated where h_t does now only the latter (act as a readout) and the "cell" that is introduced
c_t is focused on information transmission and stable training. 

A penultimate comment is how some parts strike me as primitive transformer-like computations. For instance, 
having i * g "feels like" a qkv computation, where i=qk, and g=v. Also, the fact that all f, i, g, o gatings 
are "different views" of a `ss`-size vector feels much like how q, k, v are all "different views" of the input, x_t. 
So you're essentially taking "different views" and having them talk to each other, and letting the new input/state 
be a function of their "discussion." But this is a very informal analogy -- I wonder if someone has made this 
relationship mathematically precise, I'm sure there is one (of some form), though it's not clear how revealing
that would even be. 

Overall, I find the amount of inductive bias personally quite amazing, since the notion of the different gates and how the 
update rules are structured means we're essentially telling the model how to use its subspaces, which seems quite different
from the modern "less inductive bias is better" philosophy. Maybe all of classical ML is a story of clever inductive bias!


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

class LSTM(nn.Module): # let's now implement this which introduces a c_t that comes before h_t which is now for outputs
        def __init__(self, ss=64, D=256, V=10_000, act=nn.GELU()):
                super().__init__()
                self.ss = ss
                self.D = D
                self.V = V
                self.act = act

                # params
                self.Wxs = nn.Linear(D, 4 * ss) 
                self.Wss = nn.Linear(ss, 4 * ss) # we'll chunk this and combine the outputs into next state
                self.Wsy = nn.Linear(ss, V)
                
                # move to device
                self.to(device)

        @staticmethod
        def act_figo(f, i, g, o): 
                sig = nn.Sigmoid(); tanh = nn.Tanh()

                f = sig(f); i = sig(i); o = sig(o)
                g = tanh(g)

                return f, i, g, o
        
        def forward(self, x):
                x = x.to(device)
                
                b, s, _ = x.shape
                
                h = torch.zeros(b, self.ss, device=device)
                c = torch.zeros(b, self.ss, device=device)
                                
                out = []
                for i in range(s): 
                        # c_new = c_prev * f + i * g where f, i, g all have sigmoid of a copy of h_prev
                        # h_new = o * tanh(c_new) 

                        comb = self.Wxs(x[:, i, :]) + self.Wss(h) # h is [b, ss]
                        f, i, g, o = comb.chunk(4, dim=-1) # each is [b, ss]
                        f, i, g, o = self.act_figo(f, i, g, o)
                        
                        c = c * f + i * g # this is just forcing the "residual stream" to stick to a 
                        h = o * c # certain prior structure on how subspaces therein should be used 
                        # incredible that this architecture has so much inductive bias! 

                        y_i = self.Wsy(h) # y_i is [b, V]
                        out.append(y_i.unsqueeze(1)) # reshape to [b, 1, V]

                # out at this point is an s-list of [b, 1, V] so we want to cat along dim=1
                return torch.cat(out, dim=1) # return [b, s, v], in principle also h for decoding

# char level vocab for simplicity
chars = string.printable
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# not important detail, focus on lstm 
def collate_chars(batch):
    texts = [item['text'] for item in batch]
    encoded = [[char_to_idx[c] for c in text if c in chars] for text in texts]
    max_len = max(len(seq) for seq in encoded)
    padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
    return torch.tensor(padded)

def main():
    parser = argparse.ArgumentParser(description="Train LSTM on TinyStories dataset")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--state_size", type=int, default=1024, help="LSTM state size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=4001, help="Optional limit on training steps")
    parser.add_argument("--verbose", action="store_true", help="Print additional training details")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        # you may have to login with your wandb credential here 
        wandb.init(project="train_lstm", config=vars(args))

    if args.verbose:
        print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if args.verbose:
        print("Creating dataloader...")
    train_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_chars)

    if args.verbose:
        print("Initializing model and training components...")
    
    embeddings = nn.Embedding(vocab_size, args.embed_dim).to(device)
    
    # this is the important bit 
    model = LSTM(ss=args.state_size, D=args.embed_dim, V=vocab_size)
    
    param_count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in embeddings.parameters())
    print(f"Model has {param_count:,} parameters")
    
    # combine parameters from both model and embeddings
    optimizer = torch.optim.AdamW([*model.parameters(), *embeddings.parameters()], lr=args.lr, weight_decay=args.lr)
    
    # cosine lr schedule
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
    os.makedirs('lstm_checkpoints', exist_ok=True)

    if args.verbose:
        print("Starting training loop...")
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        x = batch.to(device)
        
        # add bos 
        bos = torch.full((x.shape[0], 1), bos_token, device=device)
        x_with_bos = torch.cat([bos, x], dim=1)
        
        embedded = embeddings(x_with_bos).to(device)  # [batch_size, seq_len+1, embed_dim]
        batch_logits = model(embedded)  # [batch_size, seq_len+1, vocab_size]
        
        targets = x_with_bos[:, 1:]  # shape: [batch_size, seq_len]
        loss = criterion(batch_logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *embeddings.parameters()], max_norm=0.2)
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
            checkpoint_path = f"lstm_checkpoints/{steps}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'model': model.state_dict(),
                'embeddings': embeddings.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'steps': steps
            }, checkpoint_path)
            
        if args.num_steps is not None and steps >= args.num_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else float('nan')
    print(f"Average Loss: {avg_loss:.4f}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
