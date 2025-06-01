'''
This file implements communication protocols between GPUs. Most of the difficulty in large scale 
LLM training (whether that's pretraining or RL) is infra wrangling to do with operating at 
thousand+ GPU scale. Here, we get a taste of the most important bit of that -- sending data 
between GPUs efficiently. 

PyTorch offers several "communication collective primitives" that allow fast data movement between 
your GPUs if you're on a multi-GPU machine. Examples are torch.distributed.scatter, which takes 
in a tensor, chops it up, and sends chunks to different GPUs. Another is torch.distributed.allreduce, 
a fundamental primitive that applies an operator across the chunked data stored across lots of GPUs 
(e.g. operator = SUM or MULT), reducing the entire state to one tensor, and then broadcasting that 
result back to all processes.

Some notes on gotchas/things to know before about how the torch.dist API works:
    - Process group lifecycle: Each process inits/destroys its process group because it's a local 
      connection. If we're in an allreduce and waiting on process N and it has called 
      destroy_process_group, it will hang indefinitely. Think of a process group more as a protocol 
      than a data structure/object under the hood -- init process group is crucial (no notion of 
      distributed state without it), destroy less important since cleanup happens automatically.
    
    - Error handling: Wrap outer functions in main() in try/except rather than inside every function 
      to avoid masking distributed communication errors.
    
    - Synchronization: We don't need to .barrier() after comms because it's built into dist.comm -- 
      the function won't return for one process until it's done across all processes. This is 
      obviously not true if you manually use dist.comm(async_op=True), in which case you'll have to 
      bookkeep things yourself. Barriers are important for non-collective work, like IO done only 
      on one device, etc.
    
    - Memory management: Never allocate inside collective operations, just preallocate based on rank 
      and do conditional computation inside based on rank. Everything is in-place -- collectives 
      don't return anything, they modify tensors in place.
    
    - Deadlock prevention: Need to handle send/recv order carefully to avoid deadlock, especially 
      in ring topologies where even/odd rank ordering matters for breaking symmetry. See the parity trick in 
      ring_allreduce for an example. In tree_allreduce this takes care of itself if you read the code closely
      (ie. leaves/root send first, triggering recv for internal nodes which then go forward and send, etc). 
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

# we no longer need a collate tensor since we don't materialize the gathered local_tensors at any point 
# this alg is mathematically equiv to naive, just a faster algorithm -- most often used in production 
def ring_allreduce(
    local_tensor: torch.Tensor, # local_grads, 
    rank: int, 
    world_size: int, 
): 

    nrows, ncols = local_tensor.shape
    numel = local_tensor.numel()
    assert numel % world_size == 0 # for simplicity
    chunk_sz = numel // world_size # 144 // 4 = 36 

    local_tensor = local_tensor.reshape(world_size, chunk_sz) # [12, 12] -> [4, 36]
    # scatter-reduce
    for s in range(world_size - 1): 
        # send (s%wsz), recv into local_out[(s-1)%wsz]
        from_idx = (rank - s - 1) % world_size 
        to_idx = (rank - s) % world_size

        temp = torch.empty_like(local_tensor[from_idx])
        left_idx, right_idx = (rank - 1) % world_size, (rank + 1) % world_size
        
        # Need to handle send/recv order to avoid deadlock
        if rank % 2 == 0:
            dist.send(local_tensor[to_idx], right_idx)
            dist.recv(temp, left_idx)
        else:
            dist.recv(temp, left_idx)
            dist.send(local_tensor[to_idx], right_idx)
            
        local_tensor[from_idx] += temp # add into local_tensor[from_idx]

    # allgather has from/to +1 compared to scatter/gather, don't have good intuition on
    for s in range(world_size - 1):
        from_idx, to_idx = (rank - s) % world_size, (rank - s + 1) % world_size
        left_idx, right_idx = (rank - 1) % world_size, (rank + 1) % world_size
        
        # Need to handle send/recv order to avoid deadlock 
            # this implies convergence is bounded 

        if rank % 2 == 0:
            dist.send(local_tensor[to_idx], right_idx)
            dist.recv(local_tensor[from_idx], left_idx)
        else:
            dist.recv(local_tensor[from_idx], left_idx)
            dist.send(local_tensor[to_idx], right_idx)

    local_tensor = local_tensor.reshape(nrows, ncols)
    dist.barrier()

# parent i -> [2i+1, 2i+2] children implicitly defines the tree (0-indexed)
def tree_allreduce(
    local_tensor: torch.Tensor, 
    rank: int, 
    world_size: int, 
    root: int = 0, 
): 
    
    # helpers
    def get_left_idx(r: int, wsz: int): 
        return 2*r + 1
    
    def get_right_idx(r: int, wsz: int): 
        return 2*r + 2
    
    def has_left_child(r: int, wsz: int): 
        return bool(2*r+1<wsz)
    
    def has_right_child(r: int, wsz: int): 
        return bool(2*r+2<wsz)

    def get_parent(r: int, wsz: int): 
        return (r-1)//2 

    
    # REDUCE
    temp1, temp2 = None, None
    if has_left_child(rank, world_size): 
        left_child = get_left_idx(rank, world_size)
        temp1 = torch.empty_like(local_tensor)
        dist.recv(temp1, left_child)
        local_tensor += temp1 
    if has_right_child(rank, world_size): 
        right_child = get_right_idx(rank, world_size)
        temp2 = torch.empty_like(local_tensor)
        dist.recv(temp2, right_child)
        local_tensor += temp2 
    if rank != root: 
        # send up to parent, internal node 
        dist.send(local_tensor, get_parent(rank, world_size))

    dist.barrier()

    # BROADCAST 
    if rank != root: 
        dist.recv(local_tensor, get_parent(rank, world_size))
    if has_left_child(rank, world_size): 
        dist.send(local_tensor, get_left_idx(rank, world_size))
    if has_right_child(rank, world_size): 
        dist.send(local_tensor, get_right_idx(rank, world_size))

    dist.barrier()


def test_point_to_point(rank: int, world_size: int, device: torch.device):
    try:
        TEST_FROM, TEST_TO = 0, min(2, world_size - 1)
        test_send(rank, TEST_FROM, TEST_TO, "Sending data between GPUs is working! w00t")
        dist.barrier()
        print(f'[RANK {rank}] TEST PASSED: Point-to-point communication')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Point-to-point communication - {e}')
        return False

def test_scatter_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        data = [torch.ones(*data_shape) * i for i in range(world_size)]
        SCATTER_SRC = 0
        scatter(SCATTER_SRC, rank, world_size, data, data_shape)
        print(f'[RANK {rank}] TEST PASSED: Scatter')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Scatter - {e}')
        return False

def test_gather_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        GATHER_DEST = 0
        local_data = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
        out = None
        if rank == GATHER_DEST: 
            out = torch.empty((world_size, *data_shape), dtype=torch.float32, device=device)
        gather(GATHER_DEST, out, local_data, rank, world_size)
        
        # verify result on destination rank
        if rank == GATHER_DEST:
            expected_sum = sum(i * data_shape[0] * data_shape[1] for i in range(world_size))
            actual_sum = out.sum().item()
            assert abs(actual_sum - expected_sum) < 1e-6, f"Expected {expected_sum}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Gather')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Gather - {e}')
        return False

def test_broadcast_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        BROADCAST_SRC = min(2, world_size - 1)
        local_data = torch.ones(data_shape, dtype=torch.float32, device=device) * int(rank == BROADCAST_SRC)
        broadcast(BROADCAST_SRC, rank, world_size, local_data)
        
        # verify all ranks have the same data
        expected_sum = data_shape[0] * data_shape[1]  # should be 144 on all ranks
        actual_sum = local_data.sum().item()
        assert abs(actual_sum - expected_sum) < 1e-6, f"Expected {expected_sum}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Broadcast')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Broadcast - {e}')
        return False

def test_reduce_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        REDUCE_DEST = 0
        local_data = torch.ones(data_shape, dtype=torch.float32, device=device)
        naive_reduce(REDUCE_DEST, local_data, rank, world_size)
        print(f'[RANK {rank}] TEST PASSED: Reduce')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Reduce - {e}')
        return False

def test_allgather_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        local_out = torch.empty((world_size, *data_shape), dtype=torch.float32, device=device)
        local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
        allgather(local_out, local_tensor, rank, world_size)
        
        # verify result
        expected_sum = sum(i * data_shape[0] * data_shape[1] for i in range(world_size))
        actual_sum = local_out.sum().item()
        assert abs(actual_sum - expected_sum) < 1e-6, f"Expected {expected_sum}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Allgather')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Allgather - {e}')
        return False

def test_naive_allreduce_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        REDUCE_ROOT = 0
        local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
        local_out = torch.empty_like(local_tensor)
        collate_tensor = None
        if rank == REDUCE_ROOT: 
            collate_tensor = torch.empty((world_size, *data_shape), dtype=torch.float32, device=device)
        naive_allreduce(local_tensor, collate_tensor, local_out, rank, world_size, REDUCE_ROOT)
        
        # verify result - should be mean of all ranks
        expected_mean = sum(range(world_size)) / world_size
        expected_sum = expected_mean * data_shape[0] * data_shape[1]
        actual_sum = local_out.sum().item()
        assert abs(actual_sum - expected_sum) < 1e-6, f"Expected {expected_sum}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Naive allreduce')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Naive allreduce - {e}')
        return False

def test_ring_allreduce_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
        original_sum = local_tensor.sum().item()
        print(f'In rank {rank}, sum of local_tensor is {original_sum}')
        ring_allreduce(local_tensor, rank, world_size)
        
        # verify result - should be sum of all ranks across all elements
        expected_total = sum(range(world_size)) * data_shape[0] * data_shape[1]
        actual_sum = local_tensor.sum().item()
        assert abs(actual_sum - expected_total) < 1e-6, f"Expected {expected_total}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Ring allreduce')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Ring allreduce - {e}')
        return False

def test_tree_allreduce_primitive(rank: int, world_size: int, device: torch.device):
    try:
        data_shape = (12, 12)
        local_tensor = torch.ones(data_shape, dtype=torch.float32, device=device) * rank
        tree_allreduce(local_tensor, rank, world_size)
        
        # verify result - should be sum of all ranks across all elements
        expected_total = sum(range(world_size)) * data_shape[0] * data_shape[1]
        actual_sum = local_tensor.sum().item()
        assert abs(actual_sum - expected_total) < 1e-6, f"Expected {expected_total}, got {actual_sum}"
        
        print(f'[RANK {rank}] TEST PASSED: Tree allreduce')
        return True
    except Exception as e:
        print(f'[RANK {rank}] TEST FAILED: Tree allreduce - {e}')
        return False

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    rank, world_size = init()

    print(f'In rank {rank}, world size is {world_size}..')
    
    # run all tests
    tests = [
        test_point_to_point,
        test_scatter_primitive,
        test_gather_primitive,
        test_broadcast_primitive,
        test_reduce_primitive,
        test_allgather_primitive,
        test_naive_allreduce_primitive,
        test_ring_allreduce_primitive,
        test_tree_allreduce_primitive,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f'--' * 20)
        if test_func(rank, world_size, device):
            passed += 1
        print(f'--' * 20)
    
    print(f'[RANK {rank}] SUMMARY: {passed}/{total} tests passed')
    
    shutdown()