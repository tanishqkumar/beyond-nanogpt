''' 
(https://arxiv.org/pdf/1909.08053) 
Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

This code implements tensor parallelism training for a transformer. 
This means we split each tensor in each layer (for both MLP/attn) across 
gpus. I wrote/ran this code on a node of 4 H100s, but it'll work with any 
GPU node, just resize the model so it first in memory. For instance, I can train the 
model up to a few billion parameters no issue. 

The key reason this implementation is tricky is because we use all_reduce to 
synchronize activations/gradients in the forward/backward. Since comms are 
in-place operations, this means we have to write our own backward pass, which 
always requires careful tracking of dimensions and keeping paper nearby on which 
to do math. I found my understanding of softmax much better after having to 
write the autodiff through it. 

The key background you need before you read this file is this: 
    - The matmuls are what is sharded across GPUs, so we basically have to 
    reimplement the MLP and attn layers with multi-GPU in mind. 
    - MLP is not too hard, but only non-obvious fact is why we do row-parallel for 
    first matmul and column for second. Reason is row-parallel lets us avoid 
    syncing after first gemm, but then since we have to sync after second gemm, 
    we do column parallel. 
    - Attention is trickier, esp writing gradients through softmax. I would say 
    there's no "intuition" to be had here, you just have to implement it yourself 
    with pencil and paper in hand. 
        - Again, we do row-parallel for QKV as in the first MLP gemm. 
    - If you're implementing an nn.Module with custom backward, as we are here 
    (because our tp forward involves in-place, non differentiable all_reduce of 
    activations on fwd/bwd), then one standard way to do this is to 
        1) write a functional fwd/bwd wrapping torch.autograd.Function to tell 
        torch you're writing a functional transformation that involves forward
        AND backward. This has a standard API where it'll expect the backward 
        to output a gradient tensor for each object in the forward. 
        2) Wrap the functional (eg. my tp_mlp_f) in a more standard 
        torch module wrapper, which you can then 
        3) Use these custom nn Modules to define a Transformer/model class 
        as usual 
    - Attn admits natural parallelization across attention heads, so that 
    each GPU stores only D/wsz params in Q, K, V instead of (D, D). This 
    D/wsz quantity I call the "chunk size" C. 

    --> Subtle: study the matrix calculus in softmax derivative, ie. 
    ```
    pg = (dA * A).sum(dim=-1, keepdim=True)
    dlogits_r = A * (dA - pg)`
    ```
    in the backward pass of tp_attn_f; this comes from a Jacobian-vector product 
    which is worth getting mathematical intuition 
    for but is beyond the scope of me explaining here (see Wikipedia entry on 
    softmax, for instance). 

Overall, I'd say this is one of the most involved but instructive from-scratch 
implementations I've done, highly recommend! 
''' 

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist 
from typing import Tuple 

def init(): 
    dist.init_process_group('nccl')
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r)
    return r, w

def close(): 
    dist.destroy_process_group() 

class tp_mlp(nn.Module): 
    def __init__(self, rank: int, wsz: int, D: int = 512, mult: int = 4, base_seed: int = 42): 
        super().__init__()
        
        self.rank = rank 
        self.wsz = wsz 
        self.D = D 
        self.mlp_width = mult * D 
        self.C = self.mlp_width // wsz # chunk_sz
        
        torch.manual_seed(base_seed + rank)
        self.W1_chunk = nn.Linear(D, self.C, bias = False, dtype=torch.bfloat16)
        self.W2_chunk = nn.Linear(self.C, D, bias = False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return tp_mlp_f.apply(x, self.W1_chunk.weight.T, self.W2_chunk.weight.T)

# autograd.Function needs to be functional, can wrap this in a module for easy idiomatic usage
class tp_mlp_f(torch.autograd.Function): 

    # need to pass in tensors, not modules, so that autograd can track 
    @staticmethod 
    def forward(ctx, X: torch.Tensor, W1_weight: torch.Tensor, W2_weight: torch.Tensor) -> torch.Tensor:     
        # Convert to bf16 for computation
        X_bf16 = X.to(torch.bfloat16)
        Z_local = X_bf16 @ W1_weight  # BSD @ DC -> BSC
        h_local = F.silu(Z_local)  # BS(chunk_sz) output
        out_local = h_local @ W2_weight # BSC -> BSD
        dist.all_reduce(out_local, op=dist.ReduceOp.SUM)
        ctx.save_for_backward(X_bf16, Z_local, h_local, W1_weight, W2_weight)
        return out_local 

    @staticmethod 
    def backward(ctx, dX: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        # X.shape = grad_output.shape = BSD
        # want dW2_chunk.weight.shape = CD
        # dW1_chunk.weight.shape = DC

        X, Z_local, h_local, W1_weight, W2_weight = ctx.saved_tensors 
        C = W2_weight.shape[0]
        B, S, _ = h_local.shape 
        D = X.shape[-1]

        dX_bf16 = dX.to(torch.bfloat16)
        
        dW2_chunk = h_local.reshape(-1, C).T @ dX_bf16.reshape(-1, D) # [B*S, D] @ [B*S, C] -> [C, D] grads 
        
        dh_local = dX_bf16 @ W2_weight.T  # BSD @ DC -> BSC
        dZ_local = dh_local * (Z_local.sigmoid() * (1 + Z_local * (1 - Z_local.sigmoid())))  # BSC

        dW1_chunk = dZ_local.reshape(-1, C).T @ X.reshape(-1, D) # [BS]C @ [BS]D -> [C, D] grads
        # allreduce dX_local to give to the next layer 
        dX_next = (dZ_local.reshape(-1, C) @ W1_weight.T).reshape(B, S, D) # [BS]C @ CD -> [BS]D -> BSD
        dist.all_reduce(dX_next, op=dist.ReduceOp.SUM)

        return dX_next, dW1_chunk.T, dW2_chunk.T
    
class tp_attn_f(torch.autograd.Function): 
    
    @staticmethod 
    def forward(
        ctx, 
        X: torch.Tensor, 
        qkv_weight: torch.Tensor, 
        out_weight: torch.Tensor, 
        d_h: int, 
        causal: bool, 
    ) -> torch.Tensor:
        # X is BSD, qkv is [D, 3 * C], out is [C, D] where C = nhpr * d_h
            # ie. we compute attn for a few heads in parallel 
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        B, S, D = X.shape 
        
        X_bf16 = X.to(torch.bfloat16) # bf16 means we can handle bigger gemms on same compute 
        qkv = X_bf16 @ qkv_weight # BSD @ [D, 3 * nhpr * d_h] -> BS[3 * nhpr * d_h]
        nhpr = qkv.shape[-1] // (3 * d_h)

        q, k, v = torch.chunk(qkv, 3, dim=-1) # each is [B, S, nhpr * d_h]
        q = q.reshape(B, S, nhpr, d_h).transpose(1, 2)
        k = k.reshape(B, S, nhpr, d_h).transpose(1, 2)
        v = v.reshape(B, S, nhpr, d_h).transpose(1, 2) # [B, nhpr, S, d_h]

        attn_logits = torch.einsum('bnid,bnjd->bnij', q, k)
        
        if causal: 
            mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
            attn_logits = torch.where(mask, attn_logits, float('-inf'))
        
        A = F.softmax(attn_logits / (d_h ** 0.5), dim=-1) # [B, nhpr, S, S]
        Av = torch.einsum('bnij,bnjd->bnid', A, v) # [B, nhpr, S, S] @ [B, nhpr, S, d_h] -> [B, nhpr, S, d_h]
        ctx.Av = Av # save to avoid recomputing 
        # out is [B, nhpr, S, d_h]
        out = Av.transpose(1, 2).reshape(B, S, -1) # [B, S, nhpr * d_h], note nhpr * d_h = C
        final_out = out @ out_weight # [B, S, C] @ [C, D] -> [B, S, D]
        dist.all_reduce(final_out, op=dist.ReduceOp.SUM)
        # save all activations to ctx 
        ctx.save_for_backward(X_bf16, qkv_weight, out_weight, A, q, k, v) 
        ctx.d_h = d_h
        ctx.causal = causal
        ctx.nhpr = nhpr
        return final_out 

    @staticmethod 
    def backward(ctx, dX: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]: 
        X, qkv_weight, out_weight, A, q, k, v = ctx.saved_tensors

        d_h = ctx.d_h
        causal = ctx.causal
        nhpr = ctx.nhpr
        Av = ctx.Av
        
        B, S, D = X.shape 
        
        dX_bf16 = dX.to(torch.bfloat16)
        
        # get dW_out
        out = Av.transpose(1, 2).reshape(B, S, -1) # [B, S, C]
        C = out.shape[-1]
        dX_flat = dX_bf16.reshape(-1, D) # [BS, D]
        out_flat = out.reshape(-1, C) # [BS, C]
        d_out_weight = out_flat.T @ dX_flat  # [BS, C].T @ [BS, D] -> [C, D]

        # get dA and dv from dAv
        d_out = dX_bf16 @ out_weight.T # [B, S, D] @ [C, D].T -> [B, S, C]
        dAv = d_out.reshape(B, S, nhpr, d_h).transpose(1, 2) # d_out is [B, S, nhpr * d_h], want [B, nhpr, S, d_h]

        dA = torch.einsum('bnid,bnjd->bnij', dAv, v) # dAv @ v = [B, nhpr, S, d_h] @ [B, nhpr, S, d_h] -> [B, nhpr, S, S]
        dv = torch.einsum('bnid,bnij->bnjd', dAv, A) # dAv @ A = [B, nhpr, S, d_h] @ [B, nhpr, S, S] -> [B, nhpr, S, d_h]
        
        # review this derivation (ie. softmax matrix deriative and complexity) at some point in the future, subtle
        pg = (dA * A).sum(dim=-1, keepdim=True)
        dlogits_r = A * (dA - pg)

        # dlogits_r = torch.einsum('bnij,bnkj->bnik', dA, sm_d) # [B, nhpr, S, S] @ [B, nhpr, S, S] -> [B, nhpr, S, S]
        dlogits = dlogits_r / (d_h ** 0.5)
        
        if causal: 
            mask = torch.tril(torch.ones(S, S)).to(torch.bool).cuda()
            dlogits = torch.where(mask, dlogits, 0)

        dq = torch.einsum('bnij,bnjd->bnid', dlogits, k) # [B, nhpr, S, S] @ [B, nhpr, S, d_h] -> [B, nhpr, S, d_h] 
        dk = torch.einsum('bnij,bnid->bnjd', dlogits, q) # [B, nhpr, S, S] @ [B, nhpr, S, d_h] -> [B, nhpr, S, d_h]

        dq = dq.transpose(1, 2).reshape(B, S, C)
        dk = dk.transpose(1, 2).reshape(B, S, C) 
        dv = dv.transpose(1, 2).reshape(B, S, C)

        # stack dqkv from dq, dk, dv
        dqkv = torch.cat([dq, dk, dv], dim=-1) # [B, S, 3C]
        dqkv_flat = dqkv.reshape(-1, 3 * C)
        X_flat = X.reshape(-1, D)
        dqkv_weight = X_flat.T @ dqkv_flat # [BS, D].T @ [BS, 3C] -> [D, 3C]
        dX_next = dqkv @ qkv_weight.T #  [B, S, 3C] @ [D, 3C].T -> [B, S, D]

        dist.all_reduce(dX_next, op=dist.ReduceOp.SUM)

        return dX_next, dqkv_weight, d_out_weight, None, None

class tp_attn(nn.Module): 
    def __init__(self, rank: int, wsz: int, D: int = 512, d_h: int = 64, causal: bool = True): 
        super().__init__()
        self.rank = rank 
        self.wsz = wsz 
        self.D = D 
        self.d_h = d_h 
        self.n_h = self.D // self.d_h 
        # split along num heads, each rank does n_h_per_rank's worth of computations 
        self.n_h_per_rank = self.n_h // wsz 
        self.C = D // wsz 
        
        self.causal = causal 

        self.qkv_proj = nn.Linear(D, 3 * self.C, bias = False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.C, D, bias = False, dtype=torch.bfloat16) 


    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        return tp_attn_f.apply(
            X, 
            self.qkv_proj.weight.T, 
            self.out_proj.weight.T, 
            self.d_h,
            self.causal, 
        )
    

class Transformer(nn.Module): # BSD -> BSD for now 
    def __init__(self, rank: int, wsz: int, D: int, depth: int, mult: int = 4): 
        super().__init__()
        self.rank = rank 
        self.wsz = wsz 
        self.D = D
        self.depth = depth 
        self.mlp_width = mult * D

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mlp': tp_mlp(rank, wsz, D, mult),
                'ln1': nn.LayerNorm(D, dtype=torch.bfloat16), 
                'attn': tp_attn(rank, wsz, D), 
                'ln2': nn.LayerNorm(D, dtype=torch.bfloat16), 
            }) for _ in range(depth)
        ])

        # omit embeddings, LM head, etc since we're just learning BSD -> BSD to test this TP implementation

    def forward(self, x: torch.Tensor): 
        h = x.to(torch.bfloat16)  
        for layer in self.layers: 
            h = layer['mlp'](layer['ln1'](h)) + h
            h = layer['attn'](layer['ln2'](h)) + h
        
        return h

if __name__ == "__main__": 
    rank, wsz = init() 

    B, S, D = 8, 1024, 8192
    depth = 12
    lr = 1e-4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(42)
    X = torch.randn(B, S, D, dtype=torch.bfloat16).to(device)  
    Y = torch.randn(B, S, D, dtype=torch.bfloat16).to(device)  # some dummy target data for us to check correctness 
                                        # (loss should vanish since labels can be memorized, they are input independent)

    tp_model = Transformer(rank, wsz, D, depth).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in tp_model.parameters())
        print(f"Total parameters: {total_params/1e6:.3f}M, {B*S} tokens per batch")
    
    optimizer = torch.optim.Adam(tp_model.parameters(), lr=lr)
    
    # smol training loop
    for step in range(10_000):
        optimizer.zero_grad()
        output = tp_model(X)
        loss = F.mse_loss(output.float(), Y.float())  
        loss.backward()
        optimizer.step()
        
        if rank == 0 and step % 10 == 0:  
            print(f"Step {step}, Loss: {loss.item():.6f}")

    close()