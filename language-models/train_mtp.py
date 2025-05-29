import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformer import Transformer, TransformerConfig
from tqdm import tqdm 
from transformers import GPT2Tokenizer
import argparse
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader
import gc

def collate_batch(batch, tokenizer):
    texts = [item['text'] for item in batch]
    
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
    parser.add_argument('--mtp', action='store_true', help='Multi-token prediction')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save', action='store_true', help='Save the model') 
    parser.add_argument('--compile', action='store_true', help='Run compiler to optimize perf') 
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value') 
    args = parser.parse_args()
    
    if args.verbose:
        print(f'Loading TinyStories dataset...')
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    dataset = dataset['train'].shuffle(buffer_size=1000)
    
    if args.verbose:
        print(f'Loading GPT-2 tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    B, S, D, nlayers = args.batch_size, args.seq_len, args.hidden_dim, args.nlayers
    mtp = args.mtp
    
    if args.verbose:
        print(f'Initializing model with {nlayers} layers, hidden dim {D}, vocab size {vocab_size}...')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(
        depth=nlayers,
        hidden_dim=D,
        vocab_size=vocab_size,
        max_seq_len=S,
        device=device,
        mtp=mtp, 
    )
    model = Transformer(config).to(device)

    if args.compile: 
        print(f'Running torch.compile...')
        model = torch.compile(model)
    
    if args.verbose:
        print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters...')
        print(f'Initializing dataloader with batch size {B}, sequence length {S}...')
        print(f'Using batch size of {round((args.batch_size * args.seq_len)/1e3, 2)}K tokens.')

    dataloader = DataLoader(
        dataset.take(args.steps * args.batch_size),
        batch_size=B,
        collate_fn=lambda batch: collate_batch(batch, tokenizer)
    )

    if args.verbose: print(f'Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    dataloader_iter = iter(dataloader)

    for step in range(args.steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        

        ## MTP CHANGES HERE ##
        
        if mtp: # get trunk, run it throgh heads, accum loss for each
            B, S = batch.shape
            tokens_predicted_mtp = 4 # hardcode to best number found in MTP paper for now 
            trunk_out = model(batch)

            loss = 0

            for i, head in enumerate(model.mtp_heads): 
                outputs = head(trunk_out) # BSD -> BSV, predicting token t+i+1
                labels = torch.empty(B, S-(i+1), dtype=torch.long, device=device) 
                labels[:, :] = batch[:, i+1:]  # [B, S-(i+1)]
                
                outputs = outputs[:, :S-(i+1), :].reshape(-1, vocab_size).contiguous()
                labels = labels.view(-1).contiguous()
                
                loss += loss_fn(outputs, labels)

            loss /= tokens_predicted_mtp # observe tokens_predicted_mtp == len(model.mtp_heads)
            loss.backward()

        else: 
            B, S = batch.shape
            labels = torch.empty(B, S-1, dtype=torch.long, device=device) # batch is [B, S] of ints, labels used as indices in loss comp so need to be long
            labels[:, :] = batch[:, 1:] # labels should be [B, S-1], dropping last entry 
            
            outputs = model(batch)
            outputs = outputs[:, :-1, :].reshape(-1, vocab_size).contiguous()
            labels = labels.view(-1).contiguous()
            
            loss = loss_fn(outputs, labels)
            loss.backward()

        ## MTP CHANGES FINISH ##

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if step % 10 == 0: print(f'[Step {step}/{args.steps}]: Train loss {loss.item()} ...')
        opt.step()
        opt.zero_grad(set_to_none=True)

    if args.save: 
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        model_name = f'tinystories_model_lr{args.lr}_bs{args.batch_size}_seq{args.seq_len}.pt'
        save_path = save_dir / model_name
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), str(save_path))
