import torch 
from datasets import load_dataset
from transformer import Transformer 
from tqdm import tqdm 
path = "/n/home11/tanishqkumar/gravity-chamber/fundamentals/llm/train-vanilla-transformer"
from transformers import GPT2Tokenizer
import argparse
import time
import json
from pathlib import Path
import gc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--compile', action='store_true', help='Run compiler to optimize perf')
    parser.add_argument('--profile', action='store_true', help='Profile data loading and compute time')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    args = parser.parse_args()
    
    data_path = "/n/netscratch/gershman_lab/Lab/tkumar/datasets/dclm/global-shard_01_of_10/newest_data/dclm_seed_2b"
    
    # Initialize GPT-2 tokenizer
    if args.verbose:
        print(f'Loading GPT-2 tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    B, S, D, nlayers = args.batch_size, args.seq_len, args.hidden_dim, args.nlayers
    
    if args.verbose:
        print(f'Initializing model with {nlayers} layers, hidden dim {D}, vocab size {vocab_size}...')
    model = Transformer(nlayers, D, vocab_size).to('cuda')

    if args.compile: 
        print(f'Running torch.compile...')
        model = torch.compile(model)
    
    if args.verbose:
        print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters...')
        print(f'Initializing dataloader with batch size {B}, sequence length {S}...')
        print(f'Using batch size of {round((args.batch_size * args.seq_len)/1e3, 2)}K tokens.')

    # Initialize dataloader
    dataloader = DataLoader(data_path, batch_size=B, seq_len=S)

    if args.verbose: print(f'Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_time = 0
    compute_time = 0

    for step in range(args.steps):
        if args.profile:
            data_start = time.time()
            
        # Clear CUDA cache periodically to reduce fragmentation
        if step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        batch = next(dataloader)  
        
        # Create labels in-place to avoid extra memory allocation
        labels = torch.empty_like(batch)
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = 0
        
        if args.profile:
            data_time += time.time() - data_start
            compute_start = time.time()
            
        outputs = model(batch)
        
        # Get shapes once to avoid repeated calculations
        B, S, V = outputs.shape 
        
        # Reshape with contiguous to ensure memory efficiency
        outputs = outputs.reshape(-1, V).contiguous()
        labels = labels.reshape(-1).contiguous()

        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Apply gradient clipping to prevent memory spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if step % 10 == 0: print(f'[Step {step}/{args.steps}]: Train loss {loss.item()} ...')
        opt.step()
        opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Explicitly free memory
        del outputs, labels, loss
        
        if args.profile:
            compute_time += time.time() - compute_start

    if args.profile:
        print(f"\nProfiling results:")
        print(f"Total data loading time: {data_time:.2f}s")
        print(f"Total compute time: {compute_time:.2f}s")
        print(f"Data loading percentage: {100 * data_time/(data_time + compute_time):.1f}%")
        print(f"Compute percentage: {100 * compute_time/(data_time + compute_time):.1f}%")

    # save model for inference in jupyter notebook 
    if args.save: 
        save_dir = "/n/netscratch/gershman_lab/Lab/tkumar/datasets/dclm/global-shard_01_of_10/newest_data/gravity-models"
        model_name = f'model_lr{args.lr}_bs{args.batch_size}_seq{args.seq_len}.pt'
        save_path = f'{save_dir}/{model_name}'
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), save_path)
