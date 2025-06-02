'''
(https://arxiv.org/abs/2312.00752) Mamba: Linear-Time Sequence Modeling with Selective State Spaces

This is a conceptually subtle architecture if you're coming from Transformer-land, 
and requires understanding a few different things 
(basic ODEs, notions of linearity/time-invariance in sequence models, convolutions vs recurrences, etc),
so I'll spend a bit more time explaining this than usual. 

- What even is an SSM 
    --> State space models are sequence models that come from a branch of math called linear dynamical systems. 
    This means the state of the system evolves linearly over time like 
        d[s(t)]/dt := As(t) + Bx(t)
        y(t) = C(t)s(t) + D(t)

    Where x(t) is the input sequence and y(t) is the sequence being modeled. The key is that we're hypothesizing
    a generative process where the target sequence is a pure function of the current input token, and all past 
    x(t) affect it only through a latent (small) hidden state. 
    
    In language, the "state" is memory of all the important context from before we'll need to predict the next token. 
    Notationally, s(t) is the state (tensor), x(t) is the input (tokens in an LLM where t is discretized), and y(t) is the output
    (eg. predicted next token in a sequence model). 

    --> There's two important terms that help distinguish between SSMs/RNNs/archs like this: "linear" and "time invariant." 
    Linear just means there's no nonlinearities in state update. So RNNs which have something like s(t+1) = tanh(A(t)s(t) + B(t)x(t)) 
    are not linear, but obviously you can set the nonlinearity to a linear function so they become linear RNNs. Then
    "time invariant" means you can "unfold" the update over time. Examples: 
        
        RNN: Nonlinear, time-invariant
        GRU/LSTM: nonlinear, time-variant
        Linear RNN/attention: Linear, time-invariant
        S4 SSM: linear, time invariant (prev work by authors)
        Mamba: linear, *not* time-invariant (that's the point! selectivity = data dependent SSM params = time-variance of parameters)
        
        "SSM" generally refers to linear systems like linear attention, linear RNN, or Mamba. 

- Discretization
    - Why discretize and what to discretize 
        --> While a linear dynamical system like above is written with t being continuous, in language 
        there are finite number of tokens equally spaced, tokens aren't in "continuous time." So we have to derive 
        the same recurrences, but in discrete time over token space. 
        --> We do this by learning the continuous params, then applying mathematical transformations to discretize. 
        --> The transformations are A_disc = exp(-A_cont * delta), B_disc = 
        --> While these may seem strange, they are mathematically derived starting with the SSM equations: 
                d[s(t)]/dt := As(t) + Bx(t)
                y(t) = C(t)s(t) + D(t)

        We get these equations as follows. 
            First integrate the first equation wrt t, assuming that the input x(t) is constant in every interval 
        between discrete inputs n, n+1, ie. that x(t) is constant in between tokens [t * delta, (t+1) * delta]. 
            This assumption is called "Zero-th order hold discretization" (ZOH). 
            Using the method of integrating factors to solve the SSM ODE within a given interval, 
            we get that (if you've ever taken diffeqs you should know this)
                d[s(t)]/dt := As(t) + Bx(t)
                becomes 
                d/dt[exp(-At)h(t)] = \int_0^delta exp(-Atau)Bu(tau) dtau 
                where by ZOH, u(tau) = u is independent within each interval 
                so we have 
                d/dt[exp(-At)h(t)] = u * \int_0^delta exp(-Atau)B dtau
                where we can integrate both sides wrt t and rearrange to get the solution to the ODE: 
                h(t) = exp(-At)h(t_0) + \int_0^delta exp(t-tau) Bu
                in ODEs, the first term is the "free response" and second term the "forced response" driving the 
                dynamics. 
                Then, we can observe that for invertible A, we have that in general 
                \int_0^delta exp(t-tau)B = A^{-1}(exp(delta * A) - I)B
                which you can see by differentiating the RHS to get the left. 
                This means that the co-efficient A_d in front of h(t_0) is 
                A_d = exp(A delta) and in front of u is 
                B_d = A^{-1}(exp(delta * A) - I)B
                and these are the equations you see in the discretization in the paper. 

                Because A is diagonal (stronger than invertible), (see below section on intuition behind matrix A
                and the significance of it being assumed diagonal), this means computing A_d and B_d actually 
                only involves elementwise products (not matmuls). 
            
        --> Can't you just learn the discretized A_d(t) and B_d(t) directly? Why do we learn the continuous ones then 
        discretize using this annoying math? 
            --> You can, but there are some constraints these matrices need to satisfy that discretizing after learning forces, 
            eg. A must be a contraction ||A|| <= 1 (so that we don't blow up over time upon continuously applying it to hidden state), etc. 
            Learning A in continuous space then discretizing "algorithmically" by setting A_d = exp(-A * delta)
            makes sure these constraints are satisfied, and that the computation has a clear, principled, interpretation.
            --> And a side note: math like this (signal processing) has actually been the workhorse behind machine/statistical learning 
            for decades before neural networks ate everyone's lunch. It was enough to get man on the moon (Kalman filters, etc)!
            So it's not "annoying" -- it's absolutely fundamentals for any AI researcher!

The following matrices are the learned state space parameters. 

- The matrix A
    --> Interpretation: 
            In principle, this should be ND x ND to "mix the entire state" at every step. However, for any reasonable state size, 
            that's too large. So we implicitly parameterize it as diagonal, and learn only diag(a), which is just ND parameters. 
            This means it computes over each state element separately. Importantly, it is *not parameter dependent*, ie. A(t) = A
            is constant wrt tokens. However, this is fine because we are not learning A directly as it appears in the SSM update -- 
            the update has bar_A = exp(-delta * A), so delta being token-dependent is enough to make the state mixing 
            token dependent. 
    --> Eigenvalues 
            These have to be negative real parts to ensure stability - if Re(eigenvalues) > 0, the system would explode over time.
            For diagonal A, eigenvalues are just the diagonal elements themselves, so we need diag(A) < 0. This constraint
            is naturally satisfied in practice through the exponential discretization bar_A = exp(-delta * A), which is always
            a contraction (||bar_A|| <= 1) regardless of A's values.
    --> Elementwise vs matrix product 
        Because A is diagonal, all operations involving A become elementwise multiplications rather than expensive matrix
        multiplications. This is crucial for efficiency - a full ND x ND matrix multiply would be O((ND)^2) = O(N^2 D^2),
        but diagonal A makes it O(ND). The diagonal constraint trades off expressiveness for computational tractability,
        similar to how [tridiagonal parametrizations](https://www.diva-portal.org/smash/get/diva2:316016/FULLTEXT02) 
        reduce parameter count while maintaining system representability.

- The matrix B 
    --> Interpretation: 
        Projects input tokens x(t) ∈ R^D into the state space h(t) ∈ R^(ND). Shape is ND x D, so it's a "wide" matrix
        that expands the D-dimensional input into ND state dimensions. Unlike A, B is token-dependent B(t) through the 
        selective mechanism, allowing the model to choose which inputs to incorporate into the hidden state based on context.
        The selectivity comes from B being computed as a function of the input: B(t) = Linear_B(x(t)), making it adaptive.

- The matrix C
    --> Interpretation: mixes micro-states N -> N, where a "micro-state" for a channel d in D is just eg. N=16 
    length vector storing the information for that channel. 
    
- The matrix Delta
    --> Interpretation: 
    --> Critical, "learned discretization" 
        --> This allows selectivity by modifying state to remember new tokens less/more 

- Intuition pump 
    --> Full arch takes BLD -> BLD like any other sequence model. 
    --> Computation within tokens vs communication between tokens 
        --> For any arch, a good question to ask is: where are things mixed? 
            - For example, a transformer mixes tokens with attention, and channels/features within tokens in MLPs
            - Mamba mixes tokens in two places: conv1d (local mixing of k adjacent tokens) and .selective_scan() which is 
            global mixing, a kind of "Attention mechanism" of fixed state wrt sequence length. 
                - A computes on each micro-state (of which there are DN) independently. 
                - delta selectively updates state, large delta(x_t) means token x_t shouldn't enter state h_t and vice versa. 
                - y(t) = C(t)h(t) where C(t) is selective and mixes micro-states in h, ie. C: N -> N
            - Linear projections mix across D, ie. within tokens, like MLPs in Transformers
        Thus each of (N, D, t) are mixed at some point in the model, allowing it to learn sequences. 
    --> State size needs to be some N>>1 multiple of D because it needs to store "a running memory" of the past, 
    and D is just enough information for a single token. So for instance if channel k in [D] represents the 
    "tone" of the piece of writing (formal vs casual vs legal vs emoji-like etc), then the N microstates in 
    the state corresponding to channel k will store information about the running history of the tone throughout the document
    (eg. character X in this play talks with tone Y but character A with tone B). 
    

- What's the .ssm() and .selective_scan()? 
    --> These are the heart of Mamba. SSM() is just a wrapper around .selective_scan that reshapes everything and preprocesses SSM params 
    (eg. discretize) to pass into .selective_scan(). This function basically implements the state space recurrence. 
    --> While we write this recurrence as a simple for loop that takes O(L), the actual Mamba paper uses a O(logL) time algorithm that 
    is hardware aware. Using a simple for loop over seqlen for each forward pass is actually one of the key disadvantages of RNN training, 
    which is why it's important that a production Mamba implementation *does not* do that. The two are mathematically equivalent, though. 
    They call it "hardware aware parallel scan." 

- Where are nonlinearities that give expressive power? 
    - Sigmoid after conv in each block 
    - GELU during gating in each block
    --> See Fig 3 in Paper 
    
- Time complexity comparison to Transformer/RNN 
    Training 
        Transformers: O(L^2d)
            --> But they are parallelizble over L, whereas RNNs are not, so they they 
            "train in parallel" leading to very high throughput in practice, even though, yes, 
            the amount of raw computation is quadratic in seqlen. 
        RNNs: O(Ld^2) -- not parallelizable though, so you "feel" the O(L) term. 
        Mamba: O(Ld^2), same reason as RNN, we have a fixed size state. 
    Inference
        Transformers: O(Ld) with KV cache since each new token's query needs to attend to all prev keys. 
        RNNs: O(d) independent of seqlen by just bonking new token embedding against h_t state. 
        Mamba: O(d), same as RNN. 
Upshot of Mamba is token-dependent parameters, a key feature of attention, and porting it to 
RNN-land which allows for blazing fast inference. 

'''
from __future__ import annotations 
import argparse
import torch, torch.nn as nn
import torch.utils.data as data
import string
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
    mult: int = 2 # expansion factor for mlps
    # state size is DN, N is the num of "micro channels" each channel has to store info in state
    N: int = 8 
    D_h: Optional[int] = None # this is "the true hidden dim D", using an expansion factor D_h = D * mult
    k: int = 3 # kernel size for depthwise conv1d -- this just means we mix adjacent tokens [x_{t-1}, x_t, x_{t+1}] in the conv 
    act: nn.Module = nn.GELU

class Mamba(nn.Module): 
    def __init__(self, args: MambaArgs): 
        super().__init__()
        self.embed = nn.Embedding(args.V, args.D)
        self.unembed = nn.Linear(args.D, args.V) 
        self.final_ln = RMSNorm()
        self.blocks = nn.ModuleList([MambaBlock(args) for _ in range(args.depth)])

    # takes in a tensor of [B, L] tokens (ints) and outputs logits of [B, L, D] as per usual 
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        h = self.embed(x) # bl -> bld
        for block in self.blocks: 
            h = block(h)
        h = self.final_ln(h)
        return self.unembed(h) # bld -> blv logits

# RMSNorm normalizes each token, ie. d-vector, by the rmsnorm of that vector
    # essentially layernorm without centering and with rmsnorm instead of stdev as normalization
class RMSNorm(nn.Module): 
    def __init__(self, eps: float = 1e-5): 
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        norms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True)) # bld -> bl1, one norm for each bl token
        return (x * self.gamma)/(norms + self.eps)
        
## Key class to conceptually understand in this file -- esp the role of 1) the conv and 2) the selective scan 
class MambaBlock(nn.Module): 
    def __init__(self, args: MambaArgs): 
        super().__init__()
        self.D = args.D 
        self.N = args.N 
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
            groups=self.D_h, # this makes the conv "depthwise," ie. the same conv across channel
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

## BELOW IS JUST VANILLA TRAINING CODE TAKEN FROM train_rnn.py, important Mamba definitions are above ##
chars = string.printable
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# custom collate function for character encoding, not too important
def collate_chars(batch):
    texts = [item['text'] for item in batch]
    encoded = [[char_to_idx[c] for c in text if c in chars] for text in texts]
    # limit sequence length for faster training
    encoded = [seq[:256] for seq in encoded] 
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
    

    dataset = load_dataset("roneneldan/TinyStories", split="train")  

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
    
    # mixed precision training
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
    
    # simplified lr schedule 
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        scaler.step(optimizer)
        scheduler.step()  
        scaler.update()

        total_loss += loss.item()
        steps += 1

        if args.wandb:
            wandb.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}, step=steps)

        if steps % 10 == 0:
            print(f"Step {steps}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
        if args.num_steps is not None and steps >= args.num_steps:
            break

    avg_loss = total_loss / steps if steps > 0 else float('nan')
    print(f"Average Loss: {avg_loss:.4f}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()