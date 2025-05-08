'''
TODO: 
# I don't always annotate types because it's often obvious, but this is a relatively subtle 
# implementation where the index book-keeping etc is a bit involved, so I figured it'd be useful here 
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

class Attention(torch.nn.Module): # BSD -> BSD
    def __init__(self, D=768, layer_idx=None, head_dim=64, causal=True, device="cuda", gqa=False): 
        super().__init__()
        self.D = D 
        self.head_dim = head_dim
        self.gqa = gqa 
        assert D % head_dim == 0
        self.nheads = D//head_dim
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.causal = causal 
        self.Wo = torch.nn.Linear(D, D)
        self.device = device
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, kv_cache): # input is [B, S, D] 
        B, S, D = x.shape
        # this multi-head now, ie. make each QKV [B, S, D] --> [B, nh, S, hd]

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x) # all [B, S, D]

        Q = Q.view(B, S, self.nheads, self.head_dim).transpose(1,2) # [B, nh, S, hd]
        K = K.view(B, S, self.nheads, self.head_dim).transpose(1,2) 
        V = V.view(B, S, self.nheads, self.head_dim).transpose(1,2)

        # update kv cache 
        layer_idx = self.layer_idx 
        if kv_cache is not None and layer_idx is not None: 
            # its preallocated, just write to the memory of the cache using state of current_length 
            kv_cache.update(layer_idx, K, V)
            K = kv_cache.keys[layer_idx][:, :, :kv_cache.current_length, :]
            V = kv_cache.values[layer_idx][:, :, :kv_cache.current_length, :]

        # [B, nh, S, hd] @ [B, nh, hd, S] -> [B, nh, S, S]
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=Q.dtype, device=self.device))
        logits = (Q @ K.transpose(-2, -1)) / scale
        if self.causal:
            mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
            logits_masked = logits.masked_fill(mask, float('-inf'))
        else:
            logits_masked = logits

        A = torch.nn.functional.softmax(logits_masked, dim=-1) # [B, nh, S, S]
        
        preout = torch.einsum('bnxy,bnyd->bnxd', A, V) # [B, nh, S, S] @ [B, nh, S, hd] -> [B, nh, S, hd]
        preout = preout.transpose(1, 2).reshape(B, S, -1) # [B, nh, S, hd] -> [B, S, nh * hd]
        
        out = self.Wo(preout) # [B, S, D]
        return out # [B, S, D]

class LN(torch.nn.Module): 
    def __init__(self, D, eps=1e-9, device=None): 
        super().__init__()
        self.D = D 
        self.eps = eps
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_scale = torch.nn.Parameter(torch.zeros(D))
        self.std_scale = torch.nn.Parameter(torch.ones(D))

    def forward(self, x): # x is [B, S, D]
        mean = x.mean(dim=-1, keepdim=True) # [B, S, 1]
        std = (x.var(dim=-1, keepdim=True) + self.eps)**0.5 # [B, S, 1]
        x_norm = (x - mean)/(std) 
        return x_norm * self.std_scale + self.mean_scale

class TransformerLayer(torch.nn.Module): 
    def __init__(self, D, gqa=False, device=None, num_experts=16, top_k=2, capacity_factor=1.25, mult=4): 
        super().__init__()
        self.D = D 
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn = Attention(D, gqa=gqa, device=self.device)
        self.mlp = MoEMLP(D, device=self.device, m=num_experts, k=top_k, c=capacity_factor, mult=mult)
        self.ln1 = LN(D, device=self.device)
        self.ln2 = LN(D, device=self.device)  
    
    def forward(self, x, kv_cache=None): # x is BSD
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, kv_cache=kv_cache)
        x = x + attn_out
        ln2_out = self.ln2(x)
        mlp_out, router_logits = self.mlp(ln2_out)
        x = x + mlp_out
        return x, router_logits

# simple pos_embeddings, leave RoPE etc to future implementation
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, D, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = torch.nn.Parameter(torch.randn(max_seq_len, D))
    
    def forward(self, x): # x is [B, S, D]
        B, S, D = x.shape
        return x + self.pos_embedding[:S] # Broadcasting handles batch dimension

class EmbeddingLayer(torch.nn.Module): 
    # this is just a lookup table, gradients flow only to the entry we found in the lookup, not the same as matmul 
    def __init__(self, vocab_size, D, device=None): 
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = torch.nn.Parameter(torch.randn(vocab_size, D))

    def forward(self, x): 
        return self.embedding[x]

class UnembeddingLayer(torch.nn.Module): 
    # similar to above, this is just a lookup table that maps embeddings back to logits
    def __init__(self, vocab_size, D, device=None): 
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unembedding = torch.nn.Linear(D, vocab_size)

    def forward(self, x): # x is [B, S, D]
        # Return logits of shape [B, S, vocab_size]
        return self.unembedding(x)

class Transformer(torch.nn.Module): 
    def __init__(self, depth, hidden_dim, vocab_size, max_seq_len=16384, device=None, gqa=False, 
                 num_experts=16, top_k=2, capacity_factor=1.25, mult=4): 
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.emb = EmbeddingLayer(vocab_size, hidden_dim, device=device)
        self.pos_emb = PositionalEmbedding(max_seq_len, hidden_dim, device=device)
        self.unemb = UnembeddingLayer(vocab_size, hidden_dim, device=device)
        self.gqa = gqa 
        self.layers = torch.nn.ModuleList([
            TransformerLayer(
                hidden_dim, 
                gqa, 
                device=device, 
                num_experts=num_experts, 
                top_k=top_k, 
                capacity_factor=capacity_factor,
                mult=mult
            ) for _ in range(depth)
        ])
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i  
        self.device = device

    # todo: rewrite to include all layer logits 
    def forward(self, x, kv_cache=None):
        x = self.emb(x)
        if kv_cache is not None:
            # When decoding, only add positional embeddings for the new tokens.
            pos_offset = kv_cache.current_length
            pos_emb = self.pos_emb.pos_embedding[pos_offset: pos_offset + x.size(1)].unsqueeze(0)
            x = x + pos_emb
        else:
            x = self.pos_emb(x)
        
        all_router_logits = []
        for i, layer in enumerate(self.layers):
            x, router_logits = layer(x, kv_cache=kv_cache)
            all_router_logits.append(router_logits)
        
        logits = self.unemb(x)
        
        # Combine router logits from all layers as a tensor including layer dimension
        combined_router_logits = torch.stack(all_router_logits, dim=0).mean(dim=0)
        return logits, combined_router_logits

class Router(nn.Module): # [b, s, d] -> [b, s, M] learns to assign each token an expert 
    def __init__(self, d: int = 512, act: nn.Module = nn.GELU(), mult: int = 4, m: int = 16): 
        super().__init__()
        self.w1 = nn.Linear(d, mult * d)
        self.act = act 
        self.w2 = nn.Linear(mult * d, m)

    # [b, s, d] -> [b, s, m] automatically over last time of inputs
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.w2(self.act(self.w1(x)))

# TODO: add top_k>1 support after tracing the current implementation
class MoEMLP(nn.Module): 
    def __init__(self, d: int = 512, m: int = 16, c: float = 1.25, k: int = 2, mult: int = 4, act: nn.Module = nn.GELU(), device=None): 
        super().__init__()
        self.m = m
        self.c = c
        self.k = k
        self.d = d
        self.mult = mult
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # we use nn.Parameter and not nn.Linear because we want a 3-tensor of params across all experts     
            # you could also use an nn.ModuleList to track the nn.Linear across the m experts 
        self.mlp1 = nn.Parameter(torch.randn(m, d, d*mult) / (d ** 0.5)) # make sure scaled by fan_in so matmul outputs are O(1) indep of d
        self.mlp2 = nn.Parameter(torch.randn(m, d*mult, d) / ((d*mult) ** 0.5)) 
        self.act = act 

        self.router = Router(d=d, act=act, mult=mult, m=m)

    # [b, s, d] -> ([m, cap, d], [N, d], [N], [b, s, k])
        # first object is input to MLP computation along last dim 
        # second two tell us how the scatter was done so in gather() we can invert it back to [b, s, d]
    def expert_scatter(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        b, s, d = x.shape 
        x_flat = x.view(-1, d) # [b*s, d]
        c, m, k = self.c, self.m, self.k
        B = b*s

        cap = int(B*c//m) # token capacity for a single token 
        expert_inputs = torch.zeros(m, cap, d, device=x.device) # this is the object we seek to construct using our input x 
        
        # add argmax here 
        logits = self.router(x)
        indices = torch.argmax(logits, dim=-1).view(-1) # [b, s, m] -> [b, s] -> [b*s]

        one_hot = F.one_hot(indices, num_classes=m) # [b*s, m], we'll call b*s=B to represent "effective batch size over tokens"
        counts = torch.cumsum(one_hot, dim=0) # [B, m] tells us how many tokens assigned to that expert seen so far (vertically)
        
        # this pos computation is one of the most subtle lines, make sure to understand it
            # it uses clever indexing/broadcasting to take each index to the its index within [cap]
            # which is precisely the count above less one because we're going from a count to a 0-idx
            # here's how the indexing works
                # an important torch rule: indexing tensor X with tensors of shape
                    #  Y outputs a tensor of shape Y
                # eg. X[1, 2, 3] the tensors Y are scalars, so we get a scalar, ie. the 1-2-3th entry in X

                # here we want, for each row in range(b*s), the indices[row] element
                # we want it to be [B] dim, so lets use range(B) and indices as our indices
        pos = counts[torch.arange(B), indices] - 1

        mask = pos < cap # these are the tokens we'll process at all, this is a [B] tensor of bools
        selected_experts = indices[mask] # [N], where N is the number of indices that are true 
                                                # in the mask (data dependent)
        
        selected_pos = pos[mask] # [N]
        selected_src = x_flat[mask]

        flat_inputs = expert_inputs.view(m * cap, -1)
        flat_idx = selected_experts * cap + selected_pos # for each selected idx, get its index in the target tensor
        flat_idx = flat_idx.unsqueeze(-1).expand(-1, d) # [N] -> [N, d] so it has same shape as flat_inputs for .scatter_()
        flat_inputs.scatter_(0, flat_idx, selected_src) # in place scatter of rows, everything was building up to be able to do this
        expert_inputs = flat_inputs.view(m, cap, -1) 

        return (expert_inputs, flat_idx, mask, logits) 

    # take scatter indices and the expert-wise outputs [m, cap, d] and 
    def expert_gather(self, outputs: torch.Tensor, mask: torch.Tensor, flat_idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor: 
        b, s, d = x.shape
        c, m, k = self.c, self.m, self.k
        cap = int(b*s*c//m)

        ## now we have outputs, gather them back into [b, s, d] 
        # gather has a similar api to scatter() and so we can re-use flat_idx to invert scatter()
        flat_outputs = outputs.view(m*cap, d) # [m*cap, d]
        gathered = torch.gather(flat_outputs, 0, flat_idx) # [N, d]
        result = x.clone().reshape(b*s, d) # [b*s, d], we clone so that unprocessed tokens are pushed through 
        result[mask] = gathered # processed tokens overwritten by flat_outputs, which is output of MLPs
        result = result.view(b, s, d) # want output to be [b, s, d], like input in an MLP layer 
        
        return result 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # assign each token to an expert and arrange tokens ready for expert computation
        expert_inputs, flat_idx, mask, logits = self.expert_scatter(x)

        ## now do mlp computation on the inputs with a bmm along the num_experts axis (first, so treated as batch by default)
        expert_outputs = torch.bmm(self.act(torch.bmm(expert_inputs, self.mlp1)), self.mlp2) # [m, cap, d] @ [m, d, d] -> [m, cap, d]

        # use our scatter() indices to gather everything back into place into [b, s, d]
        out = self.expert_gather(expert_outputs, mask, flat_idx, x) # [b, s, d]
        
        return out, logits # will need logits for z-loss computation


# TODO: rewrite from scratch 
def loss_fn(logits, target, router_logits, z_coeff=0.1, balance_coeff=0.1): # router_logits is [b, s, m]
    ntp_loss = nn.CrossEntropyLoss()(logits, target)
    
    # Z-loss: penalize router logits for being too large
    # encourages per-token logits over experts to not be overconfident
    z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
    
    # Balance loss: encourage equal expert utilization
    # Calculate the fraction of tokens routed to each expert
    route_probs = torch.softmax(router_logits, dim=-1)
    # Mean probability of routing to each expert, averaged over the batch
    mean_probs = route_probs.mean(dim=0)
    # We want this to be uniform (1/num_experts for each expert)
    num_experts = route_probs.size(-1)
    target_probs = torch.ones_like(mean_probs) / num_experts
    balance_loss = torch.sum((mean_probs - target_probs)**2)

    return ntp_loss + z_coeff * z_loss + balance_coeff * balance_loss


# TODO: rewrite 
def collate_batch(batch, tokenizer):
    """Collate function to create batches from TinyStories dataset"""
    # get the text 
    texts = [item['text'] for item in batch]
    
    # tokenize and pad sequences
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded['input_ids'].cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MoE Transformer model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stream the dataset to avoid OOM
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
        num_experts=args.num_experts, 
        top_k=args.top_k, 
        capacity_factor=args.capacity_factor,
        mult=args.mlp_mult
    ).to(device)

    if args.compile:
        if args.verbose: print('Running torch.compile on the model...')
        model = torch.compile(model)

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        # Calculate parameters for a single MLP expert.
        # Assumes the MLP structure is: Linear(D, D*mlp_mult) -> Act -> Linear(D*mlp_mult, D)
        # D is args.hidden_dim (aliased as D in the script), mlp_mult is args.mlp_mult.
        
        # Parameters for one MLP expert:
        # Layer 1 (up_proj): (D * (D * args.mlp_mult)) weights + (D * args.mlp_mult) biases
        # Layer 2 (down_proj): ((D * args.mlp_mult) * D) weights + D biases
        params_one_expert_mlp = (D * D * args.mlp_mult) + \
                                (D * args.mlp_mult) + \
                                (D * args.mlp_mult * D) + \
                                D
        
        # total_params is already in Millions.
        # Active parameters calculation depends on MoE configuration.
        # L is args.nlayers.
        if args.num_experts > 1 and args.top_k > 0 and args.top_k < args.num_experts:
            # This is the typical MoE case where a subset of experts is chosen.
            # Calculate the "saved" parameters compared to a model where all experts are always active.
            # This reduction is for (num_experts - top_k) experts per layer, across all L layers.
            param_reduction_actual_count = L * params_one_expert_mlp * (args.num_experts - args.top_k)
            param_reduction_M = param_reduction_actual_count / 1e6
            
            active_params_M = total_params - param_reduction_M
            print(f'Active parameters: {active_params_M:.2f}M ({args.top_k}/{args.num_experts} experts active per MoE layer)')
        else:
            # This covers several cases:
            # 1. Dense model (num_experts <= 1).
            # 2. MoE where all experts are active (top_k >= num_experts).
            # 3. Edge cases like top_k = 0 (assuming model still has all expert params).
            active_params_M = total_params # All parameters in the model are considered active.
            
            if args.num_experts <= 1:
                # This includes num_experts = 0 or 1. If num_experts = 1, it's a dense MLP model.
                # If num_experts = 0, model structure might be different; total_params reflects it.
                print(f'Active parameters: {active_params_M:.2f}M (Dense model or num_experts <= 1)')
            elif args.top_k == 0:
                # All expert parameters exist but none are selected.
                # "Active" here means parameters part of the model definition.
                print(f'Active parameters: {active_params_M:.2f}M (top_k=0, experts are part of model but not selected)')
            else: # top_k >= num_experts (and num_experts > 1)
                # All existing experts are selected.
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
