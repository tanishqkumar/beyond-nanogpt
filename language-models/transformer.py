import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass

# helper 
ACT2FN = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'swish': F.silu,
}

@dataclass
class AttentionConfig:
    D: int = 768
    layer_idx: Optional[int] = None
    head_dim: int = 64
    causal: bool = True
    device: str = "cuda"

class Attention(nn.Module): # BSD -> BSD
    def __init__(self, 
                 config: AttentionConfig): 
        super().__init__()
        self.D = config.D 
        self.head_dim = config.head_dim
        assert self.D % self.head_dim == 0
        self.nheads = self.D//self.head_dim
        self.Wq = nn.Linear(self.D, self.D)
        self.Wk = nn.Linear(self.D, self.D)
        self.Wv = nn.Linear(self.D, self.D)
        self.causal = config.causal 
        self.Wo = nn.Linear(self.D, self.D)
        self.device = config.device
        self.layer_idx = config.layer_idx

    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor: # input is [B, S, D] 
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

        A = F.softmax(logits_masked, dim=-1) # [B, nh, S, S]
        
        preout = torch.einsum('bnxy,bnyd->bnxd', A, V) # [B, nh, S, S] @ [B, nh, S, hd] -> [B, nh, S, hd]
        preout = preout.transpose(1, 2).reshape(B, S, -1) # [B, nh, S, hd] -> [B, S, nh * hd]
        
        out = self.Wo(preout) # [B, S, D]
        return out # [B, S, D]

@dataclass
class MLPConfig:
    D: int
    hidden_multiplier: int = 4
    act: str = 'swish'
    device: Optional[torch.device] = None

# most important fact about MLP: it operates on each token independently, ie. D --> D
class MLP(nn.Module): 
    def __init__(self, 
                 config: MLPConfig): 
        super().__init__()
        self.D = config.D
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_proj = nn.Linear(self.D, self.D*config.hidden_multiplier)
        self.down_proj = nn.Linear(self.D*config.hidden_multiplier, self.D)
        self.act = ACT2FN[config.act]

    def forward(self, x: torch.Tensor) -> torch.Tensor: # BSD -> BSD automatically on last dim 
        return self.down_proj(self.act(self.up_proj(x)))

@dataclass
class LNConfig:
    D: int
    eps: float = 1e-9
    device: Optional[torch.device] = None

class LN(nn.Module): 
    def __init__(self, 
                 config: LNConfig): 
        super().__init__()
        self.D = config.D 
        self.eps = config.eps
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_scale = nn.Parameter(torch.zeros(self.D))
        self.std_scale = nn.Parameter(torch.ones(self.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is [B, S, D]
        mean = x.mean(dim=-1, keepdim=True) # [B, S, 1]
        std = (x.var(dim=-1, keepdim=True) + self.eps)**0.5 # [B, S, 1]
        x_norm = (x - mean)/(std) 
        return x_norm * self.std_scale + self.mean_scale

@dataclass
class TransformerLayerConfig:
    D: int
    device: Optional[torch.device] = None

class TransformerLayer(nn.Module): 
    def __init__(self, 
                 config: TransformerLayerConfig): 
        super().__init__()
        self.D = config.D 
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        attn_config = AttentionConfig(D=self.D, device=self.device)
        mlp_config = MLPConfig(D=self.D, device=self.device)
        ln_config = LNConfig(D=self.D, device=self.device)
        
        self.attn = Attention(attn_config)
        self.mlp = MLP(mlp_config)
        self.ln1 = LN(ln_config)
        self.ln2 = LN(ln_config)  
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor: # x is BSD
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, kv_cache=kv_cache)
        x = x + attn_out
        ln2_out = self.ln2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out
        return x 

@dataclass
class PositionalEmbeddingConfig:
    max_seq_len: int
    D: int
    device: Optional[torch.device] = None

# simple pos_embeddings, leave RoPE etc to future implementation
class PositionalEmbedding(nn.Module):
    def __init__(self, 
                 config: PositionalEmbeddingConfig):
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.D))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is [B, S, D]
        B, S, D = x.shape
        return x + self.pos_embedding[:S] # Broadcasting handles batch dimension

@dataclass
class EmbeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None

class EmbeddingLayer(nn.Module): 
    # this is just a lookup table, gradients flow only to the entry we found in the lookup, not the same as matmul 
    def __init__(self, 
                 config: EmbeddingLayerConfig): 
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Parameter(torch.randn(config.vocab_size, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.embedding[x]

@dataclass
class UnembeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None

class UnembeddingLayer(nn.Module): 
    # similar to above, this is just a lookup table that maps embeddings back to logits
    def __init__(self, 
                 config: UnembeddingLayerConfig): 
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.V = config.vocab_size
        self.unembedding = nn.Linear(config.D, self.V) # MTP = 4 token prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is [B, S, D]
        return self.unembedding(x)

@dataclass
class TransformerConfig:
    vocab_size: int
    depth: int = 8
    hidden_dim: int = 512
    max_seq_len: int = 16384
    device: Optional[torch.device] = None
    mtp: bool = False

class Transformer(nn.Module): 
    def __init__(self, 
                 config: TransformerConfig): 
        super().__init__()
        self.depth = config.depth
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        
        emb_config = EmbeddingLayerConfig(vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device)
        pos_emb_config = PositionalEmbeddingConfig(max_seq_len=config.max_seq_len, D=config.hidden_dim, device=config.device)
        unemb_config = UnembeddingLayerConfig(vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device)
        
        self.emb = EmbeddingLayer(emb_config)
        self.pos_emb = PositionalEmbedding(pos_emb_config)
        self.unemb = UnembeddingLayer(unemb_config)
        self.mtp = config.mtp 
        
        layer_config = TransformerLayerConfig(D=config.hidden_dim, device=config.device)
        self.layers = nn.ModuleList([TransformerLayer(layer_config) for _ in range(config.depth)])
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i  

        if config.mtp: # mtp_heads[i] predictions token t+i using {x_k}_{k=1}^t as conditioning
            self.mtp_heads = nn.ModuleList([nn.Linear(self.hidden_dim, self.vocab_size) for _ in range(4)]) # num_to_predict = 4

        self.device = config.device

    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor:
        x = self.emb(x)
        if kv_cache is not None:
            # When decoding, only add positional embeddings for the new tokens.
            pos_offset = kv_cache.current_length
            pos_emb = self.pos_emb.pos_embedding[pos_offset: pos_offset + x.size(1)].unsqueeze(0)
            x = x + pos_emb
        else:
            x = self.pos_emb(x)
        for _, layer in enumerate(self.layers):
            x = layer(x, kv_cache=kv_cache)
        
        return x if self.mtp else self.unemb(x) # give trunk back to do head unembedding by hand if mtp 