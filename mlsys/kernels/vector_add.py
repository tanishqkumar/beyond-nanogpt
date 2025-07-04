import torch 
import triton 
import triton.language as tl 
import time
import matplotlib.pyplot as plt
import numpy as np

@triton.jit 
def vector_add_kernel(
    x_ptr, y_ptr, 
    out_ptr, 
    n_el, 
    BLOCK_SIZE: tl.constexpr, 
):  
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_el 
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector addition kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    vec_size = 1024 
    BLOCK_SIZE = 256 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x, y = torch.randn(vec_size, device=device), torch.randn(vec_size, device=device)
    out = torch.empty(vec_size, device=device)
    grid = (triton.cdiv(vec_size, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, out, vec_size, BLOCK_SIZE) 

    assert torch.allclose(out, x + y), "Test failed."
    print(f'Test passed.')
    
    if args.bench:
        # asymptotically 
        vec_sizes = [2**i for i in range(10, 31, 2)]
        triton_times = []
        torch_times = []
        
        print("Benchmarking...")
        for size in vec_sizes:
            print(f"Testing size: {size}")
            
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            out = torch.empty(size, device=device)
            
            for _ in range(10):
                torch.add(x, y, out=out)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                torch.add(x, y, out=out)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_time = (time.time() - start_time) / 100
            torch_times.append(torch_time * 1000)
            
            grid = (triton.cdiv(size, BLOCK_SIZE),)
            for _ in range(10):
                vector_add_kernel[grid](x, y, out, size, BLOCK_SIZE)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                vector_add_kernel[grid](x, y, out, size, BLOCK_SIZE)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100
            triton_times.append(triton_time * 1000)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(vec_sizes, torch_times, 'o-', label='PyTorch', linewidth=2, markersize=6)
        plt.loglog(vec_sizes, triton_times, 's-', label='Triton', linewidth=2, markersize=6)
        plt.xlabel('Vector Size')
        plt.ylabel('Time (ms)')
        plt.title('Vector Addition Benchmark: PyTorch vs Triton')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('vector_add_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'vector_add_bench.png'")
        
        print("\nSpeedup Summary:")
        for i, size in enumerate(vec_sizes):
            speedup = torch_times[i] / triton_times[i]
            print(f"Size {size:8d}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
