import torch, torch.nn as nn, torch.nn.functional as F 
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
    input_ptr, input_stride, 
    output_ptr, output_stride,
    n_rows, n_cols,  
    BLOCK_SIZE: tl.constexpr, 
    ): 

    pid = tl.program_id(axis=0) # axis=0 bc each row is a vector with only one dim
    row_ptr = input_ptr + pid * input_stride 
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    ## HBM -> SRAM read 
    row_data = tl.load(row_ptr + offsets, mask=mask, other=-float('inf')) # setting masked logits to -inf important

    ## IN SRAM since we handle just one row 
    row_max = tl.max(row_data, axis=0)
    row_data = row_data - row_max # subtract max *before* exponentiation 
    num = tl.exp(row_data)
    denom = tl.sum(num, axis=0)
    row_softmaxed = num / denom 
    ## IN SRAM

    out_row_ptr = output_ptr + pid * output_stride 
    ## SRAM -> HBM write
    tl.store(out_row_ptr + offsets, row_softmaxed, mask=mask)

@triton.jit 
def fused_softmax_bwd(
    ds_ptr, ds_stride, # grad outputs
    s_ptr, s_stride, # softmax outputs s_i 
    out_ptr, out_stride, 
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr, 
    ):   
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols 

    # x_i -> s_i = X
    dX_row_ptr = ds_ptr + pid * ds_stride 
    s_row_ptr = s_ptr + pid * s_stride 
    ds_data = tl.load(dX_row_ptr + offsets, mask=mask)
    s_data = tl.load(s_row_ptr + offsets, mask=mask)

    # one pid per row, so pid = i for delta, need one-hot vector 
    dot = tl.sum(s_data * ds_data, axis=0)
    dx_data = s_data * (ds_data - dot) 

    out_row_ptr = out_ptr + pid * out_stride
    tl.store(out_row_ptr + offsets, dx_data, mask=mask)


if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='Fused softmax kernel')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    n_rows, n_cols = 523, 777
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(69)
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

    y_naive = naive_softmax(x)
    ## TEST FWD ## 
    assert torch.allclose(y, y_naive, atol=1e-5), "Forward: FAILED"
    print("Forward: PASSED")
    ## TEST FWD ## 

    ## TEST BWD ## 
    labels = torch.ones_like(y)

    ds = torch.ones_like(y) # grad_output, ie. dL_ds = dL
    dx_kernel = torch.empty_like(ds)

    fused_softmax_bwd[grid](
        ds, ds.stride(0), 
        y, y.stride(0), 
        dx_kernel, dx_kernel.stride(0), 
        n_rows, n_cols, BLOCK_SIZE
    )

    # compute dx in pure torch 
    x_torch = x.clone().detach().requires_grad_(True)
    y_torch = naive_softmax(x_torch)
    loss_naive = (y_torch * labels).sum()
    loss_naive.backward()
    assert torch.allclose(x_torch.grad, dx_kernel, atol=1e-5), "Backward: FAILED"
    print('Backward: PASSED')
    ## TEST BWD ## 
    
    
    if args.bench:
        matrix_sizes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
        triton_fwd_times = []
        torch_fwd_times = []
        triton_bwd_times = []
        torch_bwd_times = []
        
        print("Benchmarking...")
        for n_rows, n_cols in matrix_sizes:
            print(f"Testing size: {n_rows}x{n_cols}")
            
            x = torch.randn((n_rows, n_cols), device=device)
            y_torch = torch.empty_like(x)
            y_triton = torch.empty_like(x)
            
            # forward benchmark
            for _ in range(10):
                torch.softmax(x, dim=-1, out=y_torch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                torch.softmax(x, dim=-1, out=y_torch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_fwd_time = (time.time() - start_time) / 100
            torch_fwd_times.append(torch_fwd_time * 1000)
            
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
            triton_fwd_time = (time.time() - start_time) / 100
            triton_fwd_times.append(triton_fwd_time * 1000)
            
            # backward benchmark
            labels = torch.ones((n_rows, n_cols), device=device)
            
            # warmup for PyTorch backward
            for _ in range(10):
                x_torch = x.clone().detach().requires_grad_(True)
                y_torch = torch.softmax(x_torch, dim=-1)
                loss = (y_torch * labels).sum()
                loss.backward()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                x_torch = x.clone().detach().requires_grad_(True)
                y_torch = torch.softmax(x_torch, dim=-1)
                loss = (y_torch * labels).sum()
                loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_bwd_time = (time.time() - start_time) / 100
            torch_bwd_times.append(torch_bwd_time * 1000)
            
            ds = torch.ones_like(y_triton)
            dx_kernel = torch.empty_like(ds)
            
            for _ in range(10):
                fused_softmax_bwd[grid](
                    ds, ds.stride(0),
                    y_triton, y_triton.stride(0),
                    dx_kernel, dx_kernel.stride(0),
                    n_rows, n_cols, BLOCK_SIZE
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                fused_softmax_bwd[grid](
                    ds, ds.stride(0),
                    y_triton, y_triton.stride(0),
                    dx_kernel, dx_kernel.stride(0),
                    n_rows, n_cols, BLOCK_SIZE
                )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            triton_bwd_time = (time.time() - start_time) / 100
            triton_bwd_times.append(triton_bwd_time * 1000)
        
        size_labels = [f"{r}x{c}" for r, c in matrix_sizes]
        x_pos = range(len(matrix_sizes))
        
        plt.figure(figsize=(15, 10))
        
        # forward benchmark plot
        plt.subplot(2, 1, 1)
        plt.bar([x - 0.2 for x in x_pos], torch_fwd_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([x + 0.2 for x in x_pos], triton_fwd_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Matrix Size (rows x cols)')
        plt.ylabel('Time (ms)')
        plt.title('Softmax Forward Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, size_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # backward benchmark plot
        plt.subplot(2, 1, 2)
        plt.bar([x - 0.2 for x in x_pos], torch_bwd_times, 0.4, label='PyTorch', alpha=0.8)
        plt.bar([x + 0.2 for x in x_pos], triton_bwd_times, 0.4, label='Triton', alpha=0.8)
        plt.xlabel('Matrix Size (rows x cols)')
        plt.ylabel('Time (ms)')
        plt.title('Softmax Backward Benchmark: PyTorch vs Triton')
        plt.xticks(x_pos, size_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('softmax_bench.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'softmax_bench.png'")
        
        print("\nForward Speedup Summary:")
        for i, (n_rows, n_cols) in enumerate(matrix_sizes):
            speedup = torch_fwd_times[i] / triton_fwd_times[i]
            print(f"Size {n_rows:4d}x{n_cols:4d}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        print("\nBackward Speedup Summary:")
        for i, (n_rows, n_cols) in enumerate(matrix_sizes):
            speedup = torch_bwd_times[i] / triton_bwd_times[i]
            print(f"Size {n_rows:4d}x{n_cols:4d}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
