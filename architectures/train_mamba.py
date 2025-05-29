'''
TODO, fill out my internal notes below. Lots of intuition pumps needed, this is a conceptually confusing arch. 

- Discretization
    - Why discretize and what to discretize 
    - ZOH 
    - Euler
- Eigenvalues
- Intuition for shapes and typical sizes 
- Comms vs comp 
    - Conv = local token mixing of adjacent $$k$$ tokens, `selective_scan()` mixes globally by moving relevant global information selectively into state and then using state to make the next token prediction. 
    - Linear projections (where most of the params in the model are) compute along single tokens, like MLPs in Transformers. 
- Where are nonlinearities? 
- Comparison to Transformer/RNN 
    - Complexity Analysis 
- Scan -- what is elementwise vs matmul and why 
    - $$A(t) h(t)$$ is actually an elementwise product because $$A$$ is assumed diagonal -- if it were not, it'd have $$DN \times DN$$ params (untenably large). This is OK because micro-states and channels are mixed in other places, such as in $$y(t) = C(t) h(t)$$ mixes micro states as it's `bdn,bn->bd`. 
        - Then MLP layers mix channels $$D$$. 
    - 

'''
from __future__ import annotations 
import argparse
import torch, torch.nn as nn
import torch.utils.data as data
import string
import math
import os
import wandb
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

@dataclass 
class MambaArgs: 
    D: int
    depth: int = 12
    V: int = 10_000 # vocab size
    mult: int = 2
    N: int = 8 # d_h = dn, ie. 'expansion factor' of hidden dim 
    D_h: Optional[int] = None
    k: int = 3 # kernel size for depthwise conv1d
    act: nn.Module = nn.GELU

class Mamba(nn.Module): # embed, mambablocks
    def __init__(self, args: MambaArgs): 
        super().__init__()
        self.embed = nn.Embedding(args.V, args.D)
        self.unembed = nn.Linear(args.D, args.V) 
        self.final_ln = RMSNorm()
        self.blocks = nn.ModuleList([MambaBlock(args) for _ in range(args.depth)])

    # takes in a tensor of bl tokens and outputs logits of bld
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        h = self.embed(x) # bl -> bld
        for block in self.blocks: 
            h = block(h)
        h = self.final_ln(h)
        return self.unembed(h) # bld -> blv logits

# normalize each token, d-vector, but the rmsnorm of that vector
    # essentially layernorm without the centering/mean (and rmsnorm instead of sdev norm)
class RMSNorm(nn.Module): 
    def __init__(self, eps: float = 1e-5): 
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        norms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True)) # bld -> bl1, one norm for each bl token
        return (x * self.gamma)/(norms + self.eps)
        

class MambaBlock(nn.Module): 
    def __init__(self, args: MambaArgs): 
        super().__init__()
        self.D = args.D 
        self.N = args.N # expansion factor from d -> state size dn
        self.mult = args.mult
        self.D_h = args.D * args.mult if args.D_h is None else args.D_h # "true hidden dim"
        self.k = args.k
        
        # wrapper params (non ssm state)
        self.ln_in = RMSNorm()
        self.ln_out = RMSNorm()
        self.W_in = nn.Linear(args.D, 2 * self.D_h) # d -> 2 * args.d_h so we can chunk into [g, u]
        self.conv = nn.Conv1d(
            in_channels=self.D_h, 
            out_channels=self.D_h, 
            kernel_size=args.k,
            groups=self.D_h,
            padding=args.k//2 # ensures we preserve dim 
            ) # [d_h, k] params, conv for local mixing of tokens
        self.W_g = nn.Linear(self.D_h, self.D_h) 
        self.W_out = nn.Linear(self.D_h, args.D)
        
        self.act = args.act()

        # ssm params (state, core) as one projection
            # this projects x up to [b, c, delta], the data-dependent ssm activations 
            # (conceptually activations like q and k in attn, data dependent projections)
                # and thus analogously, you can x_proj of this like self.proj_qkv: d -> 3d in a transformer
        self.x_proj = nn.Linear(self.D_h, (2 * self.N + 1) * self.D_h) # b and c are [d_h, n] and delta is [d_h]
        
        # this is a learnable coefficient to scale delta, which is token dependent
        self.A = nn.Parameter(torch.ones(self.D_h, self.N)/self.D_h)
        self.D = nn.Parameter(torch.ones(self.D_h)/self.D_h)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # bld -> bld, same as both transformers/rnns
        B = x.shape[0]

        self.state = torch.zeros((B, self.D_h, self.N), dtype=x.dtype, device=x.device)

        x = self.ln_in(x)
        z = self.W_in(x)

        # g is residual used in output gating, u goes through block 
            # g is bld_h
        u, g = torch.chunk(z, 2, dim=-1) 

        # u -> h goes through local mixing then global mixing via conv -> ssm 
        u = u.transpose(1, 2) # bld_h -> bd_hl for conv1d
        u = torch.sigmoid(self.conv(u))
        u = u.transpose(1, 2) # bd_hl -> bld_h

        ssm_proj = self.x_proj(u)
        h = self.SSM(u, ssm_proj)

        # h contains all info from fwd, we want now to gate and add residual 
        g = torch.sigmoid(self.W_g(g)) # bld_h
        y = self.act(g * h) # both are bld_h
        o = self.W_out(y) # bld_h -> bld for output, as usual in any such sequence model 
        return self.ln_out(x + o)


    # this does preprocessing on ssm-related params/activations, and self.scan does the actual ssm recurrance in parallel over t
    def SSM(self, u: torch.Tensor, ssm_proj: torch.Tensor) -> torch.Tensor: # bld -> bld, global sequence mixing akin to attn 
        # ssm_proj is [b, l, (2n+1)d_h]
        bsz, seqlen, _ = u.shape
        B_raw, C_raw, delta = torch.split(ssm_proj, [self.N * self.D_h, self.N * self.D_h, self.D_h], dim=-1)
        
        B_raw = B_raw.reshape(bsz, seqlen, self.D_h, self.N)
        C_raw = C_raw.reshape(bsz, seqlen, self.D_h, self.N)

        log_A_bar = torch.einsum('b l d, d n -> b l d n', delta, self.A.to(dtype=torch.float32))
        A_bar = torch.exp(-log_A_bar)
        
        # use euler discretization for b rather than full since a is more important
            # this is really b_bar @ u as we'll need this product for state update in self.scan 
        B_bar_u = torch.einsum('b l d, b l d n, b l d -> b l d n', delta, B_raw, u)
        
        return self.scan(A_bar, B_bar_u, C_raw, self.D, u) # abc are [b, l, d_h, n]


    def scan(self, A: torch.Tensor, Bu: torch.Tensor, C: torch.Tensor, D: torch.Tensor, u: torch.Tensor) -> torch.Tensor: 
        # return y[t] = bonk(x, self.state[t]) and update self.state in the process
        seqlen = u.shape[1]
        ys = [] 

        # self.state is [b, d_h, n], a_bar and b_bar_u are [b,l,d_h,n]
        for t in range(seqlen): # slow rnn-like linear scan, paper does hardware-aware parallel scan, but math is equiv
            self.state = A[:, t] * self.state + Bu[:, t]
            # d is [d_h] and u is [b, l, d_h], mixing across microstates happens here within each channel 
            y_t = torch.einsum('b d n, b d n -> b d', C[:, t], self.state) 
            ys.append(y_t)
        
        out = torch.stack(ys, dim=1) + D.unsqueeze(0).unsqueeze(0) * u # [b, l, d_h] + [b, l, d_h]

        return out

# char level vocab for simplicity
chars = string.printable
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# custom collate function for character encoding, not too important
def collate_chars(batch):
    texts = [item['text'] for item in batch]
    encoded = [[char_to_idx[c] for c in text if c in chars] for text in texts]
    # limit sequence length for faster training
    encoded = [seq[:256] for seq in encoded]  # limit to 128 tokens
    max_len = max(len(seq) for seq in encoded)
    padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
    return torch.tensor(padded)

def main():
    parser = argparse.ArgumentParser(description="Train Mamba on TinyStories dataset")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension") 
    parser.add_argument("--depth", type=int, default=4, help="Number of Mamba layers") 
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")  
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size") 
    parser.add_argument("--num_steps", type=int, default=1_000, help="Optional limit on training steps")  # fewer steps
    parser.add_argument("--verbose", action="store_true", help="Print additional training details")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    if args.wandb:
        # you may have to login with your wandb credential here 
        wandb.init(project="train_mamba", config=vars(args))

    if args.verbose:
        print("Loading dataset...")
    # use a smaller subset of the dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")  # only use 5% of the data

    if args.verbose:
        print("Creating dataloader...")
    train_dataloader = data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_chars,
    )

    if args.verbose:
        print("Initializing model and training components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set autocast for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    mamba_args = MambaArgs(
        D=args.embed_dim,
        depth=args.depth,
        V=vocab_size
    )
    
    model = Mamba(mamba_args).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lr/10)
    
    # simplified lr schedule for demo
    total_steps = min(1_000, len(train_dataloader))
    warmup_steps = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )
    
    criterion = nn.CrossEntropyLoss()

    # define bos token as the last vocab index
    bos_token = vocab_size - 1

    total_loss = 0.0
    steps = 0

    os.makedirs('mamba_checkpoints', exist_ok=True)

    if args.verbose:
        print("Starting training loop...")
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        x = batch.to(device)
        
        # add bos token at the start of each sequence
        bos = torch.full((x.shape[0], 1), bos_token, device=device)
        x_with_bos = torch.cat([bos, x], dim=1)
        
        # use mixed precision for faster training
        with torch.amp.autocast('cuda'):
            # x_with_bos shape: [batch_size, seq_len+1]
            logits = model(x_with_bos)  # logits shape: [batch_size, seq_len+1, vocab_size]
            targets = x_with_bos[:, 1:]  # shape: [batch_size, seq_len]
            # reshape logits to [batch_size*seq_len, vocab_size] and targets to [batch_size*seq_len]
            loss = criterion(logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))
        
        # scale the loss and backprop
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # increased for faster convergence
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

        # log to wandb 
        if args.wandb:
            wandb.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}, step=steps)

        if steps % 50 == 0:
            print(f"Step {steps}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
        if steps % 250 == 0:  # save less frequently
            checkpoint_path = f"mamba_checkpoints/{steps}.pt"
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