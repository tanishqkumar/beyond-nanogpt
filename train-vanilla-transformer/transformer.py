import torch 
from typing import List, Dict, Optional, Tuple 

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

# most important fact about MLP: it operates on each token independently, ie. D --> D
class MLP(torch.nn.Module): 
    def __init__(self, D, hidden_multiplier=4, act='swish', device=None): 
        super().__init__()
        self.D = D
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_proj = torch.nn.Linear(D, D*hidden_multiplier)
        self.down_proj = torch.nn.Linear(D*hidden_multiplier, D)
        self.act = ACT2FN[act]

    def forward(self, x): # BSD -> BSD automatically on last dim 
        return self.down_proj(self.act(self.up_proj(x)))

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
    def __init__(self, D, gqa=False, device=None): 
        super().__init__()
        self.D = D 
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn = Attention(D, gqa=gqa, device=self.device)
        self.mlp = MLP(D, device=self.device)
        self.ln1 = LN(D, device=self.device)
        self.ln2 = LN(D, device=self.device)  
    
    def forward(self, x, kv_cache=None): # x is BSD
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, kv_cache=kv_cache)
        x = x + attn_out
        ln2_out = self.ln2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out
        return x 

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
    def __init__(self, depth, hidden_dim, vocab_size, max_seq_len=16384, device=None, gqa=False): 
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.emb = EmbeddingLayer(vocab_size, hidden_dim, device=device)
        self.pos_emb = PositionalEmbedding(max_seq_len, hidden_dim, device=device)
        self.unemb = UnembeddingLayer(vocab_size, hidden_dim, device=device)
        self.gqa = gqa 
        self.layers = torch.nn.ModuleList([TransformerLayer(hidden_dim, gqa, device=device) for _ in range(depth)])
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i  
        self.device = device

    def forward(self, x, kv_cache=None):
        x = self.emb(x)
        if kv_cache is not None:
            # When decoding, only add positional embeddings for the new tokens.
            pos_offset = kv_cache.current_length
            pos_emb = self.pos_emb.pos_embedding[pos_offset: pos_offset + x.size(1)].unsqueeze(0)
            x = x + pos_emb
        else:
            x = self.pos_emb(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, kv_cache=kv_cache)
        x = self.unemb(x)
        return x