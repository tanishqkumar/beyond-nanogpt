import torch 
import triton 
import triton.language as tl 
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse

def naive_softmax(
    x: torch.Tensor # [m, n]
) -> torch.Tensor: 
    x = x - x.max(dim=-1, keepdim=True)[0]
    num = torch.exp(x)
    denom = num.sum(dim=-1, keepdim=True)
    return num / denom

@triton.jit 
def fused_softmax_fwd(
    input_ptr, 
    input_row_stride, 
    output_ptr, 
    output_row_stride,
    n_rows, 
    n_cols,  
    BLOCK_SIZE: tl.constexpr, 
): 
    pid = tl.program_id(axis=0)
    
    row_start_ptr = input_ptr + pid * input_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x_row, axis=0)
    x_norm = x_row - x_max
    num = tl.exp(x_norm)
    denom = tl.sum(num, axis=0)

    out = num / denom 
    row_out_ptr = output_ptr + pid * output_row_stride
    tl.store(row_out_ptr + offsets, out, mask=mask)

# toDO: add comments and a bwd 

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Fused softmax kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    n_rows, n_cols = 256, 512
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.randn((n_rows, n_cols), device=device)

    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    fused_softmax_fwd[grid](
        x, x.stride(0), 
        y, y.stride(0), 
        n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE, 
    )
    
    y_ref = naive_softmax(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Softmax kernel test failed"
    print("Softmax kernel test passed!")
    
    if args.bench:
        matrix_sizes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
        triton_times = []
        torch_times = []
        
        print("Benchmarking...")
        for n_rows, n_cols in matrix_sizes:
            print(f"Testing size: {n_rows}x{n_cols}")
            
            x = torch.randn((n_rows, n_cols), device=device)
            y_torch = torch.empty_like(x)
            y_triton = torch.empty_like(x)
            
            for _ in range(10):
                torch.softmax(x, dim=-1, out=y_torch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                torch.softmax(x, dim=-1, out=y_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_time = (time.time() - start_time) / 100
            torch_times.append(torch_time * 1000)
            
            col_pow2 = triton.next_power_of_2(n_cols)
            BLOCK_SIZE = col_pow2
            grid = (n_rows,)
            
            for _ in range(10):
                fused_softmax_fwd[grid](
                    x, x.stride(0),
                    y_triton, y_triton.stride(0),
                    n_rows, n_cols,
                    BLOCK_SIZE=BLOCK_SIZE
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                fused_softmax_fwd[grid](
                    x, x.stride(0),
                    y_triton, y_triton.stride(0),
                    n_rows, n_cols,
                    BLOCK_SIZE=BLOCK_SIZE
                )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 100
            triton_times.append(triton_time * 1000)
        
        size_labels = [f"{r}x{c}" for r, c in matrix_sizes]
        x_pos = range(len(matrix_sizes))
        
        plt.figure(figsize=(12, 6))
        plt.bar([x - 0.2 for x in x_pos], torch_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([x + 0.2 for x in x_pos], triton_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Matrix Size (rows x cols)')
        plt.ylabel('Time (ms)')
        plt.title('Softmax Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, size_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('softmax_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'softmax_bench.png'")
        
        print("\nSpeedup Summary:")
        for i, (n_rows, n_cols) in enumerate(matrix_sizes):
            speedup = torch_times[i] / triton_times[i]
            print(f"Size {n_rows:4d}x{n_cols:4d}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
