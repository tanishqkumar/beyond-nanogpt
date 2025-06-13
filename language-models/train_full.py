
import torch 
from transformer import Transformer 
from tqdm import tqdm 
from transformers import GPT2TokenizerFast
import argparse
import time
from pathlib import Path
from dataloader0 import DataLoader # default is slow, single-threaded dataloading
import os 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--seqlen', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--nworkers', type=int, default=os.cpu_count()//2, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save', action='store_true', help='Save the model') 
    parser.add_argument('--compile', action='store_true', help='Run compiler to optimize perf') 
    parser.add_argument('--profile', action='store_true', help='Profile data loading and compute time') 
    parser.add_argument('--fastload', action='store_true', help='Faster, multi-threaded data loading') # in case you want to try another dataloader 
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value') 
    parser.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps') 
    parser.add_argument('--data_path', type=str, default="/n/netscratch/gershman_lab/Lab/tkumar/datasets/dclm/global-shard_01_of_10/newest_data/tinystories.jsonl", help='Dataset path') 

    args = parser.parse_args()
    if args.fastload: 
        from dataloader2 import DataLoader # use whichever you want to try as an alternative
    
    # use a simple off-the-shelf tokenizer, we want to focus on training the arch
    if args.verbose:
        print(f'Loading GPT-2 tokenizer...')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.model_max_length = args.seqlen * 100 # suppress warning 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    B, S, D, nlayers = args.batch, args.seqlen, args.hidden_dim, args.nlayers
    
    if args.verbose:
        print(f'Initializing model with {nlayers} layers, hidden dim {D}, vocab size {vocab_size}...')
    model = Transformer(nlayers, D, vocab_size).to('cuda')

    if args.compile: 
        print(f'Running torch.compile...')
        model = torch.compile(model)
    
    if args.verbose:
        print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters...')
        print(f'Initializing dataloader with batch size {B}, sequence length {S}...')
        print(f'Using effective batch size of {round((args.batch * args.seqlen * args.accum)/1e3, 2)}K tokens.')
        print(f'Number of available CPUs: {os.cpu_count()}')


    # lets try our custom dataloader
    if args.verbose: print(f'Initializing Custom Dataloader!')
    if args.fastload: 
        dataloader = DataLoader(tokenizer, args.data_path, nworkers=args.nworkers, batch_size=args.batch, seqlen=args.seqlen, verbose=args.verbose)
    else: 
        dataloader = DataLoader(tokenizer, args.data_path, batch_size=args.batch, seqlen=args.seqlen, verbose=args.verbose)
    if args.verbose: print(f'Custom Dataloader initialized!')

    if args.verbose: print(f'Setting up training...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_time = 0
    compute_time = 0
    
    for step in tqdm(range(args.steps)):
            
        opt.zero_grad()  
        accumulated_loss = 0
        

        for _ in range(args.accum): # gradient accumulation so we don't oom in high batch training
            if args.profile:
                data_start = time.time()
            inputs, targets = dataloader.get_batch()
            if args.profile:
                data_time += time.time() - data_start
                compute_start = time.time()
            
            if inputs is None:
                if args.verbose:
                    print("Reached end of dataset, restarting dataloader")
                # our hand-written dataloaders are written for one pass, so this reinits for multi-epoch use
                dataloader.close()
                dataloader = DataLoader(tokenizer, args.data_path, nworkers=args.nworkers, batch_size=args.batch, seqlen=args.seqlen, verbose=args.verbose)
                inputs, targets = dataloader.get_batch()
            
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')
                
            outputs = model(inputs)
            
            B, S, V = outputs.shape 
            outputs = outputs.reshape(-1, V).contiguous()
            targets = targets.reshape(-1).contiguous()

            loss = loss_fn(outputs, targets) / args.accum 
            loss.backward()
            accumulated_loss += loss.item() * args.accum

            if args.profile:
                compute_time += time.time() - compute_start
        
        # avoid training spikes 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        if step % 10 == 0: 
            print(f'[Step {step}/{args.steps}]: Train loss {accumulated_loss/args.accum} ...')
        opt.step()
        
        
    # dataloader.close()
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
        model_name = f'tinystories_model_lr{args.lr}_bs{args.batch}_seq{args.seqlen}.pt'
        save_path = save_dir / model_name
        if args.verbose:
            print(f'Saving model to {save_path}...')
        torch.save(model.state_dict(), str(save_path))
