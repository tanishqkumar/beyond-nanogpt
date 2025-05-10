'''
[https://arxiv.org/pdf/2101.03961] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
[https://arxiv.org/pdf/2006.16668] GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

...and many papers since then have built on the MoE idea (eg. DeepSeekMoE), adding bells and whistles of their own. 

Here we implement an MoE, ie. an MoEMLP layer and plug that into a Transformer training loop. This is just the standard 
transformer but with MLP replaced with an MoEMLP, which splits b*s tokens up and batches them to each be computed by an expert
out of m experts. This involves a scatter-gather, with some subtle broadcasting/indexing to make sure tokens are scattered to the correct 
expert then gathered back into [b, s, d] for next layer processing. 
    We use a router MLP to choose which expert each token goes to, so Router: [b, s, d] -> [b, s, m] and 
    other hypers like top_k determine how many experts a token is seen by. Since we want to do a batched matmul, the matrices for all experts
    have to be the same size. So we preallocate a tensor allowing each expert slightly more than its fair share of tokens, ie. 
    (b*s//m)*c where c is "overflow capacity." Tokens that overflow are dropped, ie. not computed on by any expert, and put in the output 
    as they were in the input. 

The key thing to understand in this code is the expert_scatter and expert_gather implementations, including the dims of each object. 
Many demo MoE implementations include for loops, eg. over experts or top_k. Our implementation is subtle because it's much closer 
to a production implementation in that it involves *no for loops whatsoever* and everything is batched/vectorized. IMO, 
writing scatter_expert and gather_expert from scratch is one of the most instructive pytorch tasks I've ever done, and I encourage you 
to do the same. 
'''
from __future__ import annotations 
import argparse
import json
import time
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

# helper 
ACT2FN = {
    'relu': torch.nn.functional.relu,
    'gelu': torch.nn.functional.gelu,
    'silu': torch.nn.functional.silu,
    'swish': torch.nn.functional.silu,
}

class TransformerLayer(nn.Module): 
    def __init__(self, D=512, mult=4, head_dim=64, m=16, top_k=2, c=1.25, act=nn.GELU(), causal=True, device=None): 
        super().__init__()
        self.D = D
        self.mult = mult
        self.m = m
        self.top_k = top_k
        self.c = c
        self.head_dim = head_dim 
        assert self.D % head_dim == 0 
        self.nheads = int(self.D/head_dim)
        self.act = act 
        self.causal = causal 
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qkv_proj = nn.Linear(D, 3 * D)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=self.nheads, batch_first=True, bias=True)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

        self.mlp = MoEMLP(d=D, m=m, c=c, k=top_k, mult=mult, act=act, device=self.device)

    @staticmethod
    def get_causal_mask(S, device):
        # create a mat full of -inf, then zero out the lower triangle + diagonal
        mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x, cos=None, sin=None): 
        _, s, _ = x.shape 
        device = x.device
        
        residual = x
        x = self.ln1(x)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # apply rotary embeddings
        if cos is not None and sin is not None:
            q_heads = q.view(q.shape[0], q.shape[1], self.nheads, self.head_dim)
            k_heads = k.view(k.shape[0], k.shape[1], self.nheads, self.head_dim)
            q = (q_heads * cos) + (self.rotate_half(q_heads) * sin)
            k = (k_heads * cos) + (self.rotate_half(k_heads) * sin)
            q = q.view(q.shape[0], q.shape[1], -1)
            k = k.view(k.shape[0], k.shape[1], -1)
            
        mask = self.get_causal_mask(s, device=device) if self.causal else torch.ones(s, s, device=device) 
        attn_out, _ = self.attn(query=q, key=k, value=v, attn_mask=mask) # outputs [b, s, d] = wout(sm(q@k)v)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x, router_logits = self.mlp(x)
        x = residual + x

        return x, router_logits

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)


class Transformer(nn.Module): 
    def __init__(self, depth=6, D=512, V=10_000, mult=4, m=16, top_k=2, c=1.25, act=nn.GELU(), head_dim=64, causal=True, device=None): 
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = nn.Embedding(V, D)
        self.unemb = nn.Linear(D, V)
        
        self.layers = nn.ModuleList([
            TransformerLayer(D=D, m=m, top_k=top_k, c=c, mult=mult, act=act, head_dim=head_dim, causal=causal, device=self.device) 
            for _ in range(depth)
        ])

        # rope embeddings, not that important for this implementation, focus on moe details instead 
        self.max_seq_len = 2048  
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_rotary_embeddings(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, :]
        sin = emb.sin()[:, None, :]
        return cos, sin

    def forward(self, x): 
        all_router_logits = []
        h = self.emb(x)
        
        cos, sin = self.get_rotary_embeddings(x)
        
        for layer in self.layers: 
            h, logits = layer(h, cos=cos, sin=sin)  
            all_router_logits.append(logits)
        return self.unemb(h), torch.stack(all_router_logits)

class Router(nn.Module): # [b, s, d] -> [b, s, M] learns to assign each token an expert 
    def __init__(self, d: int = 512, act: nn.Module = nn.GELU(), mult: int = 4, m: int = 16): 
        super().__init__()
        self.w1 = nn.Linear(d, mult * d)
        self.act = act 
        self.w2 = nn.Linear(mult * d, m)

    # [b, s, d] -> [b, s, m] automatically over last time of inputs
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.w2(self.act(self.w1(x)))  # [b, s, d] -> [b, s, m]

class MoEMLP(nn.Module): 
    def __init__(self, d: int = 512, m: int = 16, c: float = 1.25, k: int = 2, mult: int = 4, act: nn.Module = nn.GELU(), device=None): 
        super().__init__()
        self.m = m
        self.c = c
        self.k = k
        self.d = d
        self.mult = mult
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mlp1 = nn.Parameter(torch.randn(m, d, d*mult) / (d ** 0.5)) # make sure scaled by fan_in so matmul outputs are O(1) indep of d
        self.mlp2 = nn.Parameter(torch.randn(m, d*mult, d) / ((d*mult) ** 0.5)) 
        self.act = act 

        self.router = Router(d=d, act=act, mult=mult, m=m)

    # [b, s, d] -> ([m, cap, d], [N, d], [N], [N], [b, s, m])
    def expert_scatter(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        # we return the index information about how we "scatter" to experts so that we can reassemble everything 
        # correctly in expert_gather 
        b, s, d = x.shape 
        x_flat = x.view(-1, d) # [b*s, d]
        c, m, k = self.c, self.m, self.k
        B = b*s

        cap = int(B*c*k//m) # token capacity for a single token 
        expert_inputs = torch.zeros(m, cap, d, device=x.device)  # [m, cap, d]
        
        logits = self.router(x)  # [b, s, m]
        vals, inds = torch.topk(logits, k, dim=-1) # [b, s, m] -> [b, s, k]
        probs = F.softmax(vals, dim=-1).view(-1) # [b*s, k] -> sm over each k -> [b*s*k]
        inds = inds.view(-1) # [b*s*k]

        one_hot = F.one_hot(inds, num_classes=m) # [b*s*k, m], expanded version of above
        counts = torch.cumsum(one_hot, dim=0) # [b*s*k, m]
        pos = counts[torch.arange(B*k, device=x.device), inds] - 1  
        # [B*k], for each token its position in its expert's batch from [0, cap-1]

        mask = pos < cap # [B*k] of bools, of which N are true 
        selected_experts = inds[mask] # [N]
        selected_probs = probs[mask] # [N], but now N can be >B
        selected_pos = pos[mask] # [N] where N is number in mask 
        selected_src = x_flat.repeat_interleave(k, dim=0)[mask]  # [N, d]

        flat_inputs = expert_inputs.view(m * cap, -1)  # [m*cap, d]
        flat_idx = selected_experts * cap + selected_pos  # [N]
        flat_idx_expanded = flat_idx.unsqueeze(-1).expand(-1, d) # [N] -> [N, d]

        flat_inputs.scatter_(0, flat_idx_expanded, selected_src)  # [m*cap, d]
        
        expert_inputs = flat_inputs.view(m, cap, -1)  # [m, cap, d]

        return (expert_inputs, flat_idx_expanded, mask, selected_probs, logits)

    def expert_gather(self, 
                      outputs: torch.Tensor,
                      mask: torch.Tensor, 
                      flat_idx: torch.Tensor,
                      x: torch.Tensor,
                      selected_probs: torch.Tensor) -> torch.Tensor: 
        
        b, s, d = x.shape
        c, m, k = self.c, self.m, self.k
        B = b * s
        cap = int(B * c * k // m)

        # Step 1: Flatten expert outputs to [m*cap, d]
        flat_outputs = outputs.view(m * cap, d)

        # Step 2: Gather and weight contributions from each expert
        # flat_idx: [N, d] selects rows in flat_outputs
        # selected_probs: [N] is the weight for each contribution
        gathered = torch.gather(flat_outputs, 0, flat_idx) * selected_probs.unsqueeze(-1)  # [N, d]
        
        # Initialize result with a copy of the input (flattened)
        result_accum = x.view(B, d).clone()
        
        # Get indices of tokens that were processed by experts
        token_idx = torch.arange(B, device=x.device).repeat_interleave(k)[mask]  # [N]
        
        # Zero out the positions that will receive expert outputs
        # (to avoid adding to the original values)
        result_accum.index_fill_(0, token_idx, 0)
        
        # Add weighted expert outputs to the result
        result_accum.index_add_(0, token_idx, gathered)
        
        return result_accum.view(b, s, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        expert_inputs, flat_idx, mask, selected_probs, logits = self.expert_scatter(x)  # [m, cap, d], [N, d], [N], [N], [b, s, m]

        expert_outputs = torch.bmm(
            self.act(
                torch.bmm(expert_inputs, self.mlp1)
                )
            , self.mlp2) # [m, cap, d] @ [m, d, mult * d] -> [m, cap, mult * d] @ [m, mult * d, d] -> [m, cap, d] mlp 

        out = self.expert_gather(expert_outputs, mask, flat_idx, x, selected_probs) # [b, s, d]
        
        return out, logits  # [b, s, d], [b, s, m], stack logits over layers to create tensor we'll use for loss comp 

def loss_fn(logits, target, router_logits, z_coeff=0.1, balance_coeff=0.01): # router_logits are [layers, b, s, m]
    ntp_loss = nn.CrossEntropyLoss()(logits, target)
    
    # Z-loss: penalize router logits for being too large for any individual class
    z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)

    # Balance loss: encourage equal expert utilization across *tokens*
        # the key diff w/ above is encouraging uniformity across tokens vs classes
        # we want to do both, but with indep coefficients, hence two diff loss terms
    m = router_logits.shape[-1]
    # we want the experts to be used uniformly, so we penalize distance of 
        # our distb over experts from Unif[m] ~ 1/m
    unif_balance = torch.ones(m, device=router_logits.device)/m
    logits_flat = router_logits.view(-1, m)
    route_probs = torch.softmax(logits_flat, dim=-1)
    actual_balance = torch.mean(route_probs, dim=0)
    balance_loss = F.mse_loss(actual_balance, unif_balance)

    return ntp_loss + z_coeff * z_loss + balance_coeff * balance_loss

def collate_batch(batch, tokenizer, device='cuda', maxlen=512):
    # minimal collate function to create batches from TinyStories dataset 
    texts = [item['text'] for item in batch]
    
    # tokenize and pad sequences
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
    return encoded['input_ids'].to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MoE Transformer model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # in production we'd include grad_accum_steps so we can scale up batch size independent of gpu memory constraints
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in MoEMLP')
    parser.add_argument('--top_k', type=int, default=1, help='Top-K experts per token')
    parser.add_argument('--capacity_factor', type=float, default=1.25, help='Capacity factor for experts')
    parser.add_argument('--mlp_mult', type=int, default=4, help='Multiplier for MLP hidden dimension')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--compile', action='store_true', help='Run compiler to optimize perf')
    parser.add_argument('--profile', action='store_true', help='Profile data loading and compute time')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--z_coeff', type=float, default=0.1, help='Z-loss coefficient for penalizing large router logits')
    parser.add_argument('--balance_coeff', type=float, default=0.01, help='Balance loss coefficient for encouraging equal expert utilization')
    parser.add_argument('--aux_loss', action='store_true', help='Enable auxiliary losses (z-loss and balance loss)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stream the dataset to avoid OOM for now since this is not the focus 
    if args.verbose:
        print('Loading TinyStories dataset...')
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    dataset = dataset['train'].shuffle(buffer_size=1000)

    # load tokenizer
    if args.verbose:
        print('Loading GPT-2 tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    B, S, D, L = args.batch_size, args.seq_len, args.hidden_dim, args.nlayers

    if args.verbose:
        print(f'Initializing model with {L} layers, hidden dim {D}, '
              f'{args.num_experts} experts, top_k={args.top_k}, vocab size {vocab_size}...')
    model = Transformer(
        L, D, vocab_size, 
        device=device, 
        m=args.num_experts, 
        top_k=args.top_k, 
        c=args.capacity_factor,
        mult=args.mlp_mult
    ).to(device)

    if args.compile:
        if args.verbose: print('Running torch.compile on the model...')
        model = torch.compile(model)

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # we use this for active param computation 
        params_one_expert_mlp = (D * D * args.mlp_mult) + \
                                (D * args.mlp_mult) + \
                                (D * args.mlp_mult * D) + \
                                D
        
        if args.num_experts > 1 and args.top_k > 0 and args.top_k < args.num_experts:
            param_reduction_actual_count = L * params_one_expert_mlp * (args.num_experts - args.top_k)
            param_reduction_M = param_reduction_actual_count / 1e6
            
            active_params_M = total_params - param_reduction_M
            print(f'Active parameters: {active_params_M:.2f}M ({args.top_k}/{args.num_experts} experts active per MoE layer)')
        else:
            active_params_M = total_params
            
            if args.num_experts <= 1:
                print(f'Active parameters: {active_params_M:.2f}M (Dense model or num_experts <= 1)')
            elif args.top_k == 0:
                print(f'Active parameters: {active_params_M:.2f}M (top_k=0, experts are part of model but not selected)')
            else:
                print(f'Active parameters: {active_params_M:.2f}M (All {args.num_experts} experts selected as top_k >= num_experts)')
        
        print(f'Initializing dataloader with batch size {B}, sequence length {S} '
              f'(~{round(B * S / 1e3,2)}K tokens per batch)')

    dataloader = DataLoader(
        dataset.take(args.steps * args.batch_size),
        batch_size=B,
        collate_fn=lambda batch: collate_batch(batch, tokenizer)
    )

    if args.verbose:
        print('Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data_time = 0.0
    compute_time = 0.0
    dataloader_iter = iter(dataloader)

    for step in range(args.steps):
        if args.profile:
            data_start = time.time()

        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # create labels by shifting inputs
        labels = torch.empty_like(batch)
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = 0

        if args.profile:
            data_time += time.time() - data_start
            compute_start = time.time()

        outputs, router_logits = model(batch.to(device))

        B_out, S_out, V = outputs.shape
        logits = outputs.reshape(-1, V).contiguous()
        tgt = labels.reshape(-1).contiguous().to(device)

        loss = loss_fn(logits, tgt, router_logits, z_coeff=args.z_coeff, balance_coeff=args.balance_coeff)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if step % 10 == 0 and args.verbose:
            print(f'[Step {step}/{args.steps}] loss: {loss.item():.4f}')

        opt.step()
        opt.zero_grad(set_to_none=True)
        del outputs, logits, tgt, loss

        if args.profile:
            compute_time += time.time() - compute_start

    if args.profile:
        total = data_time + compute_time
        print('\nProfiling results:')
        print(f'  Data loading time:   {data_time:.2f}s ({100*data_time/total:.1f}%)')
        print(f'  Compute time:        {compute_time:.2f}s ({100*compute_time/total:.1f}%)')

    if args.save:
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        model_name = f'moe_tinystories_lr{args.lr}_bs{args.batch_size}_seq{args.seq_len}.pt'
        save_path = save_dir / model_name
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), str(save_path))

