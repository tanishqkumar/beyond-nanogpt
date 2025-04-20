import torch 
from multiprocessing import Process, Queue
from typing import List
import orjson as json # optimized json IO library 
import os 
from functools import partial

'''
How this dataloader works:
- Uses multiprocessing to parallelize data loading and tokenization
- Each worker reads a portion of the dataset file and tokenizes text
- Workers maintain local token buffers that get filled with tokenized sequences
- When a local buffer has enough tokens for a batch, it's copied to a shared batch buffer
- The main process retrieves batches from the shared buffer via queues
- Batches are formatted as (input, target) pairs for language modeling
- Uses a producer-consumer pattern with free/filled slot queues for coordination
- Handles EOF gracefully by signaling completion to the main process
- Manages memory efficiently with pinned memory for GPU transfer
'''

class DataLoader:
    def __init__(self, tokenizer, path="/n/netscratch/gershman_lab/Lab/tkumar/datasets/dclm/global-shard_01_of_10/newest_data/tinystories.jsonl",
                 nworkers=16, batch_size=64, seqlen=512, prefetch=200, verbose=False):
        """
        Initialize the DataLoader with:
        - tokenizer: tokenizer to use for encoding text
        - path: path to the dataset files
        - nworkers: number of worker processes for parallel processing
        - batch_size: number of sequences per batch
        - prefetch: number of batches to prefetch 
        - seqlen: maximum sequence length
        - verbose: whether to print progress information
        """
        # Store tokenizer and configuration
        self.tokenizer = tokenizer 
        self.path = path 
        self.filesize = os.path.getsize(path)
        
        # Batch buffer stores properly formatted batches ready for model input
        self.batch_buffer = torch.empty([prefetch, batch_size, seqlen], dtype=torch.long)
        # get_batch is called on gpu, reads from this tensor on cpu, so make sure it has easy access 
        self.batch_buffer = self.batch_buffer.pin_memory() 
        self.batch_buffer.share_memory_() # in place, so do after pinning

        self.free_batch_slots = Queue()
        self.filled_batch_slots = Queue()

        for i in range(prefetch):
            self.free_batch_slots.put(i) 

        # Set up multiprocessing queues and processes
        self.nworkers = nworkers 
        self.bytes_per_worker = self.filesize//nworkers
        self.batch_size = batch_size 
        self.seqlen = seqlen
        self.verbose = verbose
        self.prefetch = prefetch 
         
        # Consider separating reader and tokenizer workers
        self.worker_procs = []

        # Create a tokenize function with the tokenizer already bound
        self.tokenize_text = partial(self._tokenize_text, self.tokenizer)
        
        for i in range(self.nworkers): 
            p = Process(target=self._worker, args=(i,))
            p.daemon = True 
            p.start() 
            self.worker_procs.append(p)
    
    @staticmethod  # review how static methods and dataclasses work 
    def _tokenize_text(tokenizer, text, **kwargs):
        return tokenizer(text, **kwargs)
        
    def _string_list_to_1d_tensor_with_eos(self, string_batch: List[str]) -> torch.Tensor: 
        # takes in List[str] and tokenizes with eos sprinkled in, then flattens 
        tokens = self.tokenize_text(string_batch,
                       return_tensors='pt',
                       padding=True,
                       truncation=False,
                       add_special_tokens=True)
        return tokens.input_ids.masked_select(tokens.attention_mask.bool())  # 1d tensor [seq1, eos, seq2, eos, ...]

    def _worker(self, i: int, tokenize_minibatch_sz: int = 64): 
        # keep a local local_token_buffer tensor without sharing of size B*S
        # keep doing the below loop of [read, tokenize, copy_] into it while its not full 
        # when full, get a free_idx from globalq and write to batch_buffer via copy_ a whole batch 
        try: 
            token_buffer_sz = 10 * self.batch_size * self.seqlen
            local_token_buffer = torch.empty(token_buffer_sz, dtype=torch.long)
            local_buffer_sz = 0
            with open(self.path, 'r', encoding='utf-8') as f:
                f.seek(i * self.bytes_per_worker)
                if i > 0: f.readline()  # skip partial line if we land there
                
                string_batch, string_batch_len = [], 0
                eof_encountered = False

                worker_end_pos = None if (i == self.nworkers - 1) else (i + 1) * self.bytes_per_worker

                while worker_end_pos is None or f.tell() < worker_end_pos:
                    line = f.readline()
                    if not line:  # eof 
                        if self.verbose: 
                            print(f'EOS found by worker {i}')
                        eof_encountered = True
                        break
                    try: 
                        string_batch.append(json.loads(line)["text"])
                        string_batch_len += 1
                    except: 
                        pass 

                    if string_batch_len == tokenize_minibatch_sz:
                        tokens_1d = self._string_list_to_1d_tensor_with_eos(string_batch)

                        # put in local token buffer 
                        L = tokens_1d.numel()
                        
                        # edge case 
                        if L > token_buffer_sz: 
                            new_buffer_sz = max(token_buffer_sz * 2, L)
                            new_token_buffer = torch.empty(new_buffer_sz, dtype=torch.long)
                            new_token_buffer[:local_buffer_sz].copy_(local_token_buffer[:local_buffer_sz])
                            local_token_buffer = new_token_buffer
                            token_buffer_sz = new_buffer_sz
                            if self.verbose:
                                print(f"Worker {i} resized token buffer to {token_buffer_sz}")
                        
                        # copy, usual case 
                        local_token_buffer[local_buffer_sz:local_buffer_sz+L].copy_(tokens_1d)
                        local_buffer_sz += L

                        string_batch = []
                        string_batch_len = 0

                    single_batch_len = self.batch_size * self.seqlen
                    
                    if local_buffer_sz >= single_batch_len: 
                        # write to global buffer, update queues, reset local buffer 
                        free_global_idx = self.free_batch_slots.get()
                        self.batch_buffer[free_global_idx].copy_(local_token_buffer[:single_batch_len].reshape(self.batch_size, self.seqlen))
                        self.filled_batch_slots.put(free_global_idx)
                        # move any leftover tokens to front of buffer so we don't waste them 
                        local_token_buffer[:local_buffer_sz-single_batch_len].copy_(local_token_buffer[single_batch_len:local_buffer_sz])
                        local_buffer_sz -= single_batch_len 

                # Only last worker hits this by construction 
                if eof_encountered:
                    print(f'Worker {i} found EOF, is putting None on filled_batch_slots')
                    self.filled_batch_slots.put(None)  # Signal end to tokenizers
            
            if self.verbose:
                print("Read+tokenizer finished")
        except Exception as e: 
            print(f"Error in _async_read: {str(e)}")
            raise Exception(f"Error reading data in async_read in reader thread: {str(e)}")    
    
    def get_batch(self):  # Returns: tuple: (inputs, targets) where each is a tensor of shape (batch_size, seqlen-1)
        batch_idx = self.filled_batch_slots.get()

        if batch_idx is None:
            if self.verbose:
                print(f'End of dataset reached, get_batch returning None')
            return None, None
            
        batch = self.batch_buffer[batch_idx]  # this is a tensor [B, S] that is sequence packed
        inputs = batch[:, :-1]  
        targets = batch[:, 1:]  

        self.free_batch_slots.put(batch_idx)
        
        return inputs, targets
    
    def close(self):
        """Clean up resources gracefully when done"""
                
        # Allow batch processes to finish if possible.
        for p in self.worker_procs:
            try:
                p.join(timeout=0.05)
                if p.is_alive():
                    if self.verbose:
                        print(f"Batch process {p.pid} did not terminate gracefully; forcing termination.")
                    p.terminate()
                    p.join()
            except:
                pass
            
        # Close all queues
        try:
            self.free_batch_slots.close()
            self.filled_batch_slots.close()
        except:
            pass