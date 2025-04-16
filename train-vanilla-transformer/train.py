import torch 
from datasets import load_dataset
from transformer import Transformer 
from tqdm import tqdm 
from transformers import GPT2Tokenizer
import argparse
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader
import gc

# this is a Karpathy-like training file, with arch in transformer.py. We use an off-the-shelf tokenizer and dataloader, 
# we will build both of our own in the other files/folders, this file is meant to be the starting point in this repo
# so that one should understand this code (ie. be at a NanoGPT level) before starting to work through the repo. 
def collate_batch(batch, tokenizer):
    """Collate function to create batches from TinyStories dataset"""
    # get the text 
    texts = [item['text'] for item in batch]
    
    # tokenize and pad sequences
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded['input_ids'].cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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
    
    # stream the dataset to avoid OOM if we try to take it all in one shot 
    if args.verbose:
        print(f'Loading TinyStories dataset...')
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    dataset = dataset['train'].shuffle(buffer_size=1000)
    
    # use a simple off-the-shelf tokenizer, we want to focus on training the arch
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

    # Use PyTorch's DataLoader with custom collate function
    dataloader = DataLoader(
        dataset.take(args.steps * args.batch_size),  # Only take what we need
        batch_size=B,
        collate_fn=lambda batch: collate_batch(batch, tokenizer)
    )

    if args.verbose: print(f'Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_time = 0
    compute_time = 0
    
    dataloader_iter = iter(dataloader)

    for step in range(args.steps):
        if args.profile:
            data_start = time.time()
            
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Create labels in-place to avoid extra memory allocation
        labels = torch.empty_like(batch)
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = 0
        
        if args.profile:
            data_time += time.time() - data_start
            compute_start = time.time()
            
        outputs = model(batch)
        
        B, S, V = outputs.shape 
        outputs = outputs.reshape(-1, V).contiguous()
        labels = labels.reshape(-1).contiguous()

        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # avoid training spikes 
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

    # save model for inference in jupyter notebook if needed 
    if args.save: 
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        model_name = f'tinystories_model_lr{args.lr}_bs{args.batch_size}_seq{args.seq_len}.pt'
        save_path = save_dir / model_name
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), str(save_path))
