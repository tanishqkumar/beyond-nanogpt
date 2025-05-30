'''
This file implements communication protocols between GPUs. Most of the difficulty in large scale 
LLM training (whether that's pretraining or RL) is infra wrangling to do with operating at 
thousand+ GPU scale. Here, we get a taste of the most important bit of that -- sending data 
between GPUs efficiently. Pytorch offers several "communication collective primitives" that allow 
fast data movement between your GPUs if you're on a multi-GPU machine. 
    Examples are torch.distributed.scatter, which takes in a tensor, chops it up, and sends chunks to different GPUs. 
    Another is torch.distributed.allreduce, a fundamental primitive that applies an operator across the chunked data stored 
    across lots of GPUs (eg. operator = SUM or MULT), reducing the entire state to one tensor, and then 


Some notes: 
    - each process inits/destroys its process group because it's a local connection - 
    if we're in an allreduce and waiting on process N and it has called destroy_process_group
    it will hang indefinitely. 
    - wrap outer functions in main() in try/except rather than inside every fn 
    - think of a process group more as a protocol than a data structure/object under the hood 
        - init process group is crucial, no notion of distb state without it 
        - destroy less important, cleanup happens automatically 
    - we don't need to .barrier() after comms because it's built into dist.comm, the fn 
    won't return for one process until it's done across all processes
        - this is obviously not true if you manually use dist.comm(async_op=True), in which 
        case you'll have to bookkeep things yourself 
        - barrier is important for non collective work, like IO done only on one device, etc
    - never allocate inside, just preallocate based on rank and do conditional computation inside based on rank 
    - everything is in place, collectives don't return anything 
'''

import torch, torch.nn as nn, torch.nn.functional as F 
import torch.distributed as dist 
from typing import List, Union, Optional, Tuple

def init(): 
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size

def shutdown(): 
    dist.destroy_process_group()

def test_send(
        rank: int,
        src: int,
        dest: int,
        msg: str,
        device: torch.device = 'cuda'
    ): 
    s = torch.tensor([0])  

    # send from src to dest 
    if rank == src: 
        msg_t = torch.tensor([ord(c) for c in msg], dtype=torch.int64, device=device)
        dist.send(msg_t, dest)
    elif rank == dest: 
        # receive and convert back to string
        msg_t = torch.zeros(len(msg), dtype=torch.int64, device=device)
        dist.recv(msg_t, src)
        s = ''.join([chr(int(x)) for x in msg_t])

    print(f'[TEST SEND] In rank {rank}, my local string is {s}.')


def scatter(
    src: int, 
    rank: int, 
    world_size: int, 
    data: List[torch.Tensor], # assumes all the same shape 
    shape: Tuple[int, int], # size, dims
    device: torch.device = torch.device('cuda'),
): 
    # send data[i] chunk to rank i
    if rank == src: 
        data_t = torch.stack(data, dim=0).to(device)
        for dest in range(world_size): 
            if dest == src: 
                store_t = data_t[src]  # src also needs to store its own data
            else: 
                dist.send(data_t[dest], dest)
        
    else: 
        store_t = torch.empty(*shape, dtype=torch.float32, device=device)
        # recv 
        dist.recv(store_t, src)
    
    dist.barrier() # torch.dist comm collective wait at end automatically 
    print(f'[SCATTER] In rank {rank}, stored tensor sums to {store_t.sum():.2f}')
        

def gather(
    dest_rank: int, # rank to write into 
    out: Optional[torch.Tensor], # [world_size, *local_tensor.shape]
    local_tensor: torch.Tensor, # local tensor to send 
    rank: int, 
    world_size: int,
): 

    if rank != dest_rank: # send
        # send local tensor to dest_rank if rank != dest_rank
        dist.send(local_tensor, dest_rank)
    else: # on dest_rank, recv
        if out is None: raise Exception("In gather, but destination rank \
                                        not passed an out tensor to write gather into!")
        for sender in range(world_size):
            if sender == dest_rank: 
                # copy in place
                out[dest_rank].copy_(local_tensor)
            else: # recv 
                dist.recv(out[sender], sender)

        # we're on dest_rank 
        print(f'[GATHER] In dest_rank rank {dest_rank}, result sums to {out.sum()}!')
    
    dist.barrier()
    
# broadcast local_tensor from src into all 
def broadcast(
    src: int, 
    rank: int, 
    world_size: int, 
    local_tensor: torch.Tensor, 
    # no need for device since we don't allocate anything internally
): 
        
    if rank == src: 
        for dest in range(world_size): 
            if dest != src: 
                dist.send(local_tensor, dest)
    else: 
        dist.recv(local_tensor, src)
    
    dist.barrier()
    print(f'[BROADCAST] In rank {rank}, local tensor sum is {local_tensor.sum()}')


# reduce all tensors to a scalar on dest, op=SUM by default
def naive_reduce( 
    dest: int, 
    local_tensor: torch.Tensor, 
    rank: int, 
    world_size: int, 
): 
    
    if rank == dest: 
        out = local_tensor.sum() # init with local data 
        for sender in range(world_size): 
            if sender != dest: 
                temp = torch.empty_like(local_tensor)
                dist.recv(temp, sender)
                out += temp.sum()

    else: 
        # send 
        dist.send(local_tensor, dest)

    dist.barrier()
    if rank == dest: 
        print(f'[REDUCE] In rank {rank}, got reduced out = {out}')

# takes a single tensor from each process, every process should have 
    # caller expects the gathered [world_size, *local_tensor.shape] to be written to local_out
    # take our local tensor, gather into root's local_tensor 
    # broadcast root's local tensor to everyone else's local tensor, 
def allgather(
        local_out: torch.Tensor, # empty [world_size, *local_tensor.shape] for all processes 
        local_tensor: torch.Tensor, # torch.ones(10,10) * rank
        rank: int, 
        world_size: int, 
        gather_root: int = 0, 
    ): 

    assert local_out.shape == (world_size, *local_tensor.shape), \
        "Error: local_out in allgather should be (world_size, *local_tensor.shape)!"
    
    # take each local_tensor chunk -> collect into root's local_out 
    gather(gather_root,
           local_out,
           local_tensor, 
           rank,
           world_size
           )

    # at this point, root's local_out is [world_size, *local_tensor.shape] 
    broadcast(gather_root, rank, world_size, local_out) # local_out populated for all processes 
    
    dist.barrier()
    # should be sum(100 * i for i in range(world_size)) and same across all ranks
    expected_out = sum(100 * i for i in range(world_size))
    print(f'[ALLGATHER] In rank {rank}, sum is {local_out.sum()}, expected {expected_out}') 
    

# takes in local objects (eg. grads) from each device, aggregates using a reduce op (eg. take mean of grads) to get a single tensor
# then broadcasts this tensor to all devices. a naive implementation is just gather -> reduce -> broadcast 
# structurally similar to allgather but with a reduction op in the middle 
# so in some sense this is more general since allgather is just allreduce with reduction=identity 
    # we'll assume the reduce operation is mean() across batch (eg. averaging batch grads in DDP)
def naive_allreduce(
    local_tensor: torch.Tensor, # data_shape
    collate_tensor: Optional[torch.Tensor],# [world_size, *data_shape] to collate all local_tensors, only root needs this now
    local_out: torch.tensor, 
    rank: int, 
    world_size: int, 
    root: int = 0, 
): 

    # gather to root into local_out, other local_outs are None
    if rank != root:
        assert collate_tensor is None, "Error! Only root should have local_out defined in naive_allreduce"
    
    gather(
        root, 
        collate_tensor, 
        local_tensor, 
        rank,
        world_size
    )

    if rank == root: 
        local_out.copy_(collate_tensor.mean(axis=0))
    
    broadcast(root, rank, world_size, local_out)
    
    dist.barrier()
    if rank == root: 
        expected_out = collate_tensor.mean()
        print(f'[NAIVE ALLREDUCE] In rank {rank}, got output {local_out.mean()}, expected {expected_out:.1f}')

# we no longer need a collate tensor since we don't materialize the gathered local_tensors at any point 
# this alg is mathematically equiv to naive, just a faster algorithm -- most often used in production 
def ring_allreduce(
    local_tensor: torch.Tensor, 
    local_out: torch.tensor, 
    rank: int, 
    world_size: int, 
): 

    # this is just scatter -> reduce -> allgather 

    pass

def tree_allreduce(): 
    pass 


# modularize this and include assertions inside 
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    rank, world_size = init()

    ## init things all tests will use 
    data_shape = (10, 10)
    data = [torch.ones(*data_shape) * i for i in range(world_size)]
    ## 

    print(f'In rank {rank}, world size is {world_size}..')
    
    # test send/recv point to point primitives
    # print(f'--' * 20)
    # print(f'In test_send...')
    # TEST_FROM, TEST_TO = 0, 2
    # test_send(rank, TEST_FROM, TEST_TO, "Sending data between GPUs is working! w00t")
    # dist.barrier()
    # print(f'--' * 20)

    # # test naive scatter 
    # print(f'In scatter()...')
    # SCATTER_SRC = 0
    # scatter(SCATTER_SRC, rank, world_size, data, data_shape)
    # print(f'--' * 20)

    # # test gather
    # print(f'In gather()...')
    # GATHER_DEST = 0
    # local_data = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
    # out = None
    # if rank == GATHER_DEST: 
    #     out = torch.tensor((world_size, *data_shape), dtype=torch.float32, device=device) 
    # gather(GATHER_DEST, local_data, rank, world_size)
    # print(f'--' * 20)

    # print(f'In broadcaster()....')
    # BROADCAST_SRC = 2
    # local_data = torch.ones(data_shape, dtype=torch.float32, device=device) * int(rank == BROADCAST_SRC)
    # broadcast(BROADCAST_SRC, out, rank, world_size, local_data)
    # print(f'--' * 20)

    # print(f'In reduce()....')
    # REDUCE_DEST = 0
    # local_data = torch.ones(data_shape, dtype=torch.float32, device=device)
    # naive_reduce(REDUCE_DEST, local_data, rank, world_size)
    # print(f'--' * 20)

    # print(f'In allgather()...')
    # local_out = torch.empty((world_size, *data_shape), dtype=torch.float32, device=device)
    # local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
    # allgather(local_out, local_tensor, rank, world_size)
    # print(f'--' * 20)

    print(f'In allreduce()...')
    REDUCE_ROOT = 0
    local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
    local_out = torch.empty_like(local_tensor) # where output averaging all local tensors will be written
    collate_tensor = None
    if rank == REDUCE_ROOT: 
        collate_tensor = torch.empty((world_size, *data_shape), dtype=torch.float32, device=device)
    naive_allreduce(local_tensor, collate_tensor, local_out, rank, world_size, REDUCE_ROOT)
    print(f'--'*20)








    shutdown()