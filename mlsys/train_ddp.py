'''
(https://arxiv.org/pdf/2006.15704) PyTorch Distributed: Experiences on Accelerating Data Parallel Training
TODO, this is a work in progress

'''

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist 
from datasets import load_dataset
from torch.utils.data import DataLoader 
import argparse
from typing import List, Dict, Optional 


def init(): 
    dist.init_process_group(backend="nccl")
    r, wsz = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(f"cuda:{r}")

    return r, wsz

def shutdown(): 
    dist.destroy_process_group()


# this should wrap model, and implement
    # ensure all ranks start with same params
    # allreduce grads after local backward pass
    # use grad bucketing 
    # overlap comms / comp 
    # expose any key functions from the underlying model that training will assume 
        # eg. if training assume model.x we want to set ddp.x(args, kwargs) = ddp.model.x(args, kwargs)
    # cleanup logic for hooks 

# we don't support grad accumulation here
class DDP(nn.Module): 
    def __init__(self, model: nn.Module, rank: int, world_size: int, bucket_sz: int = 4, clip_grad: bool = True): 
        super().__init__()
        self.model = model 
        self.rank = rank 
        self.world_size = world_size 
        self.bucket_sz = bucket_sz
        self.num_buckets = 0
        self.last_bucket_sz = None
        self.clip_grad = clip_grad 

        self._sync_params()  
        buckets, param2bucket_idx = self._make_buckets() # updates self.num_buckets in place 
        self.bucket_counter = [0] * self.num_buckets 
        # self.last_bucket_sz exists by this point 
        self._register_hooks(buckets, param2bucket_idx)
        
        # sync params to start 
    def _sync_params(self, src: int = 0): 
        for p in self.model.parameters(): 
            dist.broadcast(p.data, src=src)

    def _make_buckets(self): # put params in buckets 
        params_list = list(self.model.parameters())
        num_params = len(params_list)
        buckets = []
        bucket = []
        param2bucket_idx = {}
        for i in range(num_params): 
            p = params_list[i]
            bucket.append(p)
            param2bucket_idx[p] = len(buckets)
            if len(bucket) == self.bucket_sz: 
                buckets.append(bucket)
                self.num_buckets += 1
                bucket = []
        
        # leftover bucket
        if bucket: 
            for p in bucket: 
                param2bucket_idx[p] = len(buckets) 
            buckets.append(bucket)
            self.num_buckets += 1
            self.last_bucket_sz = len(bucket)
            

        return buckets, param2bucket_idx

    # [bucket params, register allreduce hook per bucket]
    def _register_hooks(self, buckets: List, param2bucket_idx: Dict): 
        
        def bucket_allreduce_hook(p: torch.Tensor): # runs on the param object, not on param.data tensor 
            # find the bucket this param is in 
            bucket_idx = param2bucket_idx[p]
            self.bucket_counter[bucket_idx] += 1 # all params in this bucket ready, ie. have p.grad populated 

            last_bucket_check = bool(bucket_idx == self.num_buckets - 1 and self.bucket_counter[bucket_idx] == self.last_bucket_sz) 

            # is this fine or should we account for two params hitting this at the same time? 
            if self.bucket_counter[bucket_idx] == self.bucket_sz or last_bucket_check: 
                # fire allreduce for all params in this bucket 
                for param in buckets[bucket_idx]:
                    # clip grads 
                    if self.clip_grad: nn.utils.clip_grad_norm_(param, 1.)
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
                self.bucket_counter[bucket_idx] = 0 # reset for next bwd 
            
            
        # actually register hooks now that we have buckets and hook fn 
        for p in self.model.parameters(): 
            p.register_post_accumulate_grad_hook(bucket_allreduce_hook)

    def forward(self, *args, **kwargs): 
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try: 
            return super().__getattr__(name)
        except AttributeError: 
            return getattr(self.model, name)



# want to return bsz indices based on (r, wsz)
    # TODO, do we need a set_epoch() method? are we handling drop_last correctly? 
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

    def __len__(self): 
        return self.len_dataset//self.world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test distributed sampler')
    parser.add_argument('--len_dataset', type=int, default=20, help='Length of dataset')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Shuffle dataset')
    parser.add_argument('--drop_last', action='store_true', default=True, help='Drop last incomplete batch')
    args = parser.parse_args()
    
    len_dataset = args.len_dataset
    shuffle = args.shuffle
    drop_last = args.drop_last


    dataset = load_dataset("roneneldan/TinyStories")["train"].take(len_dataset)

    r, wsz = init()
    dsampler = DistributedSampler(len_dataset, r, wsz, shuffle=shuffle, drop_last=drop_last)
    
    dataloader = DataLoader(
        dataset,
        drop_last=drop_last,
        sampler=dsampler, 
    )

    print(f'In rank {r}, next batch indices are {next(iter(dataloader))}')
    shutdown()