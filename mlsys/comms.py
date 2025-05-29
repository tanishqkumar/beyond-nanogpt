# Demo of PyTorch distributed functionality
import torch
import torch.distributed as dist
import os
import time
import socket

def initialize_process_group():
    """Initialize the distributed process group."""
    print(f"Initializing process group on {socket.gethostname()}")
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")  # Use NCCL for GPU communication
    
    # Print process group info
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Process group initialized: Rank {rank} of {world_size}")
    return rank, world_size

def cleanup():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Process group destroyed")

def demo_point_to_point():
    """Demonstrate point-to-point communication."""
    rank, world_size = initialize_process_group()
    
    try:
        if world_size < 2:
            print("Need at least 2 processes for point-to-point demo")
            return
        
        # Simple send/recv
        if rank == 0:
            # Create tensor on GPU
            tensor = torch.ones(10, 10, device=f"cuda:{rank}")
            # Send to rank 1
            dist.send(tensor, dst=1)
            print(f"Rank {rank}: Sent tensor to rank 1")
        elif rank == 1:
            # Create tensor on GPU to receive data
            tensor = torch.zeros(10, 10, device=f"cuda:{rank}")
            # Receive from rank 0
            dist.recv(tensor, src=0)
            print(f"Rank {rank}: Received tensor from rank 0, sum: {tensor.sum().item()}")
    finally:
        cleanup()


# scatter, gather, broadcast, allgather, reduce-scatter, ring-allreduce, tree-allreduce 

def demo_collective_ops():
    """Demonstrate collective operations."""
    rank, world_size = initialize_process_group()
    
    try:
        # Create tensor on current GPU
        tensor = torch.ones(10, 10, device=f"cuda:{rank}") * (rank + 1)
        print(f"Rank {rank}: Initial tensor sum = {tensor.sum().item()}")
        
        # All-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum([(i + 1) * 100 for i in range(world_size)])
        print(f"Rank {rank}: After all_reduce, tensor sum = {tensor.sum().item()}, expected = {expected_sum}")
        
        # Broadcast
        if world_size >= 2:
            tensor = torch.ones(10, 10, device=f"cuda:{rank}") * (rank + 1)
            dist.broadcast(tensor, src=0)
            expected = 100 if rank == 0 else 100  # All ranks should have rank 0's value
            print(f"Rank {rank}: After broadcast, tensor sum = {tensor.sum().item()}, expected = {expected}")
        
        # All-gather
        tensor = torch.ones(10, 10, device=f"cuda:{rank}") * (rank + 1)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        print(f"Rank {rank}: After all_gather, received {len(gathered)} tensors")
        for i, t in enumerate(gathered):
            print(f"Rank {rank}: Gathered tensor {i} sum = {t.sum().item()}, expected = {(i+1) * 100}")
    finally:
        cleanup()

def benchmark_bandwidth():
    """Benchmark communication bandwidth between GPUs."""
    rank, world_size = initialize_process_group()
    
    try:
        if world_size < 2:
            print("Need at least 2 processes for bandwidth benchmark")
            return
            
        sizes = [2**i for i in range(18, 28)]  # From 256KB to 256MB
        
        if rank == 0:
            print("\nBandwidth Benchmark:")
            print(f"{'Size (MB)':<10} {'Time (ms)':<10} {'Bandwidth (GB/s)':<15}")
        
        for size in sizes:
            num_elements = size
            tensor = torch.ones(num_elements, dtype=torch.float32, device=f"cuda:{rank}")
            
            # Warm up
            for _ in range(5):
                dist.all_reduce(tensor)
            
            # Synchronize before timing
            torch.cuda.synchronize()
            
            # Time the operation
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                dist.all_reduce(tensor)
                torch.cuda.synchronize()
            end = time.time()
            
            # Calculate bandwidth
            elapsed = (end - start) / iterations * 1000  # Convert to milliseconds
            size_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
            bandwidth = size_mb / (elapsed / 1000) / 1024  # GB/s
            
            if rank == 0:
                print(f"{size_mb:<10.2f} {elapsed:<10.2f} {bandwidth:<15.2f}")
    finally:
        cleanup()

def run_demos():
    """Run all demos."""
    # print("\n1. Running point-to-point communication demo...")
    # demo_point_to_point()
    
    print("\n2. Running collective operations demo...")
    demo_collective_ops()
    
    # print("\n3. Running bandwidth benchmark...")
    # benchmark_bandwidth()

# To run these demos with multiple processes, you need to launch this script with torchrun:
# torchrun --standalone --nproc_per_node=4 script.py
#
# For demonstration, we'll just print instructions if not in a distributed environment
if not dist.is_available():
    print("PyTorch distributed is not available")
elif "RANK" not in os.environ:
    print("To properly run this file, use torchrun, eg: ")
    print("\ntorchrun --standalone --nproc_per_node=4 comms.py")
else:
    run_demos()
