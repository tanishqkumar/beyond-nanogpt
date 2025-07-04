'''
(https://arxiv.org/pdf/2006.15704) PyTorch Distributed: Experiences on Accelerating Data Parallel Training

This file assumes only dist.comms primitives like dist.broadcast and dist.all_reduce, and implements
DDP. Torch DDP is nice because all you have to do is 1) pass in a DistributedSampler into your dataloader 
when init'ing it with your HF dataset, and 2) wrap model -> DDP(model), and voila -- if you run the resulting
code with torchrun on a machine with multiple GPUs, they will automatically run data parallel. 

BTW, "data parallel" just means each GPU stores a copy of the model, gets a separate (disjoint) batch 
from the other GPUs, computes the grad, and the *final grad applied to the model* is an average of the 
empirical batch grad over all GPUs, so you effectively increase batch size by world_size. This speeds up 
training by a linear factor (ie. reduces number of required steps to reach a certain loss) if you're 
not close to the critical batch size (which is when your bsz is so large you get diminishing returns in 
terms of reduction in grad variance). 

The key objects and how they work are as follows. 
    A HF dataset is just a lookup table that stores data and implements __getitem__ so you can 
    read from it, eg. dataset[10]. 
    A pytorch dataloader wraps a dataset (taking it as its first arg), and implements batching, shuffling, 
    seeding, iterating, etc. 
    A sampler is something a dataloader uses under the hood to sample indices from the dataset to batch. 
    So for instance with normal single-process dataloading, if you set Shuffle=True, a RandomSampler 
    will be used under the hood that returns a permutation of data indices to the dataset, and then 
    the dataset will create batches in that new permuted order. 

    In our code, the two key objects we create are: 
        - DistributedSampler: gives each rank indices for its entire epoch (of size len_dataset//world_sz)
          making sure each GPU sees a disjoint subset of the full dataset. 
        - DDP: this wraps the model, supporting distributed data parallel training:
          * Ensures all ranks start with same parameters
          * All-reduces gradients after local backward pass
          * Uses gradient bucketing for efficiency
          * Overlaps communication with computation
          * Exposes underlying model functions for training compatibility
          * Provides cleanup logic for registered hooks
            
    Intuition for this code: 
        --> We create and use two key abstractions: buckets and hooks. 
            
            --> Buckets group parameters into disjoint 
        subsets to avoid per-parameter allreduce calls (wasteful for small data). We fire 
        allreduce only when all parameters in a bucket have computed gradients, tracked via 
        per-bucket counters that parameters increment when their .grad is populated.

            --> Hooks are PyTorch functions triggered after key operations (fwd/bwd/grad accumulation). 
        We use "register_post_accumulate_grad_hook" to allreduce after gradients populate. 
        Using async_op=True prevents blocking the backward pass - we share/average layer N 
        gradients while layers N-1, N-2 are still computing. "Overlapping comms/comp" is 
        simply setting async_op=True and storing work handles for synchronization.

'''


import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist 
from datasets import load_dataset
from torch.utils.data import DataLoader 
import argparse
from typing import List, Dict 


### KEY NEW DDP ADDITIONS HERE ###
def init(): 
    dist.init_process_group(backend="nccl")
    r, wsz = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r)

    return r, wsz

def shutdown(): 
    dist.destroy_process_group()




# we don't support grad accumulation here
class DDP(nn.Module):
    def __init__(self, model: nn.Module, rank: int, world_size: int, bucket_sz: int = 4): 
        super().__init__()
        self.model = model 
        self.rank = rank 
        self.world_size = world_size 
        self.bucket_sz = bucket_sz
        self.num_buckets = 0
        self.pending_works = []
        self.hooks_to_clean = []

        self._sync_params()  
        buckets, param2bucket = self._make_buckets() # updates self.num_buckets in place 
        self.bucket_counter = [0] * self.num_buckets 
        self.last_bucket_sz = len(buckets[-1])
        self._register_hooks(buckets, param2bucket)
        
        # sync params to start 
    def _sync_params(self, src: int = 0): 
        for p in self.model.parameters(): 
            dist.broadcast(p.data, src=src)

    def _make_buckets(self): # put params in buckets 
        params_list = list(self.model.parameters())
        num_params = len(params_list)
        buckets = []
        bucket = []
        param2bucket = {} # idx to idx 
        for i in range(num_params): 
            p = params_list[i]
            bucket.append(p)
            param2bucket[id(p)] = len(buckets)
            if len(bucket) == self.bucket_sz: 
                buckets.append(bucket)
                self.num_buckets += 1
                bucket = []
        
        # leftover bucket
        if bucket: 
            for p in bucket: 
                param2bucket[id(p)] = len(buckets) 
            buckets.append(bucket)
            self.num_buckets += 1

        return buckets, param2bucket

    # call this after bwd, before opt.step(), so we make sure that 
        # all the all_reduces finish (ie. grads are sync'd)
    def synchronize(self): 
        for w in self.pending_works: 
            w.wait()
        self.pending_works = []

    # [bucket params, register allreduce hook per bucket]
    def _register_hooks(self, buckets: List, param2bucket: Dict): 
        def bucket_allreduce_hook(param: torch.Tensor): # runs on param.
            # find the bucket this param is in 
            bucket_idx = param2bucket[id(param)]
            self.bucket_counter[bucket_idx] += 1 # all params in this bucket ready, ie. have p.grad populated 

            # Check if this is the last bucket and all parameters in it are ready
            is_last_bucket = (bucket_idx == self.num_buckets - 1)
            last_bucket_ready = is_last_bucket and self.bucket_counter[bucket_idx] == self.last_bucket_sz
            regular_bucket_ready = not is_last_bucket and self.bucket_counter[bucket_idx] == self.bucket_sz

            if regular_bucket_ready or last_bucket_ready: 
                # fire allreduce for all params in this bucket 
                for param in buckets[bucket_idx]:
                    work = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                    self.pending_works.append(work)
                self.bucket_counter[bucket_idx] = 0 # reset for next bwd 

        # actually register hooks now that we have buckets and hook fn 
        for p in self.model.parameters(): 
            handle = p.register_post_accumulate_grad_hook(bucket_allreduce_hook)
            self.hooks_to_clean.append(handle)
        
    def forward(self, *args, **kwargs): 
        return self.model(*args, **kwargs)

    def __del__(self): 
        self.synchronize()
        for hk in self.hooks_to_clean: 
            hk.remove() # cleanup to help gc
        self.hooks_to_clean = []


# want to return bsz indices based on (r, wsz)
class DistributedSampler(torch.utils.data.Sampler): 
    # now let's support shuffle, drop_last and 
    def __init__(
        self, 
        len_dataset: int, 
        rank: int, 
        world_size: int, 
        shuffle: bool = False, 
        drop_last: bool = False, # forces len of __iter__ output to be same across ranks 
    ):
        super().__init__()
        self.len_dataset = len_dataset - (len_dataset % world_size) if drop_last else len_dataset
        self.rank = rank 
        self.world_size = world_size
        self.shuffle = shuffle 
        self.epoch = 0
        self.drop_last = drop_last  

    def __iter__(self): 
        if self.shuffle: 
            g = torch.Generator()
            # universal so we shuffle the same way, but diff over epochs
            g.manual_seed(69 + self.epoch) 
            full_dataset = torch.randperm(self.len_dataset, generator=g)
        else: 
            full_dataset = torch.arange(self.len_dataset)

        all_chunks = torch.chunk(full_dataset, self.world_size, dim=0)
        # sampler should return an iterator that DataLoader can go through 
        self.epoch += 1 # we're called once per epoch 
        return iter(all_chunks[self.rank].tolist())

    # 10 -> [4, 4, 2], asks for local rank len 
    def __len__(self): 
        q, r = divmod(self.len_dataset, self.world_size)
        return r if self.rank == self.world_size - 1 else q # 2 if last rank else 4

    # only DDP dataloaders need this, not normal ones since they just reseed iter(dataloader)
    def set_epoch(self, epoch: int): 
        self.epoch = epoch

### END NEW DDP ADDITIONS ###

## SETTING UP A SIMPLE TRANSFORMER TRAINING LOOP TO TEST DDP, BELOW IS NOT IMPORTANT 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import argparse

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim, nlayers, nheads, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, nheads, dropout) 
            for _ in range(nlayers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test distributed training with DDP')
    parser.add_argument('--len_dataset', type=int, default=50_000, help='Length of dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Shuffle dataset')
    parser.add_argument('--clip-grad-norm', action='store_true', default=True, help='Clip grad norm')
    parser.add_argument('--drop_last', action='store_true', default=True, help='Drop last incomplete batch')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    len_dataset = args.len_dataset
    batch_size = args.batch_size
    epochs = args.epochs
    seq_len = args.seq_len
    hidden_dim = args.hidden_dim
    nlayers = args.nlayers
    lr = args.lr
    shuffle = args.shuffle
    drop_last = args.drop_last
    verbose = args.verbose
    clip_grad = args.clip_grad_norm
    
    r, wsz = init()
    device = torch.device(f'cuda:{r}')
    
    if verbose:
        print(f'Rank {r}: Initialized distributed training with world size {wsz}')
    
    if verbose:
        print(f'Rank {r}: Loading TinyStories dataset (taking {len_dataset} samples)...')
    dataset = load_dataset("roneneldan/TinyStories")["train"].take(len_dataset)
    
    if verbose:
        print(f'Rank {r}: Loading GPT-2 tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def collate_batch(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
        return encoded['input_ids']
    
    if verbose:
        print(f'Rank {r}: Creating distributed sampler...')
    dsampler = DistributedSampler(len_dataset, r, wsz, shuffle=shuffle, drop_last=drop_last)
    
    if verbose:
        print(f'Rank {r}: Creating dataloader with batch size {batch_size}...')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=dsampler,
        collate_fn=collate_batch
    )

    if verbose:
        print(f'Rank {r}: Creating transformer model with {nlayers} layers, {hidden_dim} hidden dim...')
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        nlayers=nlayers,
        nheads=4,
        dropout=0.1
    ).to(device)
    
    if verbose:
        print(f'Rank {r}: Wrapping model in custom DDP...')
    ddp_model = DDP(model, r, wsz)
    
    # some telemetry 
    total_params = sum(p.numel() for p in ddp_model.parameters())
    param_millions = total_params / 1_000_000
    print(f'Rank {r}: Model has {param_millions:.2f}M parameters')
    sample_batch = next(iter(dataloader))
    tokens_per_batch = sample_batch.numel()
    print(f'Rank {r}: Each batch contains {tokens_per_batch} tokens')
    
    if verbose:
        print(f'Rank {r}: Creating AdamW optimizer with lr={lr}...')
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr)

    print(f'Rank {r}: Starting DDP training for {epochs} epochs')
    
    for epoch in range(epochs):
        dsampler.set_epoch(epoch)
        ddp_model.train()
        
        print(f'Rank {r}: Starting epoch {epoch}')
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            optimizer.zero_grad()
            logits = ddp_model(inputs)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss.backward()
            ddp_model.synchronize() 
            
            if clip_grad: torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose and batch_idx % 5 == 0:
                print(f'Rank {r}, Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}')
            elif not verbose and batch_idx % 20 == 0:
                print(f'Rank {r}, Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f'Rank {r}: Completed epoch {epoch}, Average Loss = {avg_loss:.4f}')
    
    print(f'Rank {r}: DDP training completed')
    shutdown()