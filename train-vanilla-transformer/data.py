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

class NaiveDataLoader:
    def __init__(self, data_path, batch_size, seq_len, device='cuda'):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set pad token to eos token if pad token is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.current_idx = 0
        self.device = device
        self.current_file = None
        self.current_data = []
        self.files = list(self.data_path.glob("*.jsonl"))
        self.file_idx = 0
        
    def load_next_file(self):
        if self.file_idx >= len(self.files):
            self.file_idx = 0
            if len(self.files) == 0:
                raise RuntimeError("No .jsonl files found in data directory")
        
        with open(self.files[self.file_idx], 'r') as f:
            self.current_data = [json.loads(line) for line in f]
        self.file_idx += 1
        self.current_idx = 0
        
    def __iter__(self):
        self.current_idx = 0
        self.file_idx = 0
        self.load_next_file()
        return self
    
    def __next__(self):
        batch_texts = []
        batch_tokens = []
        
        # Collect batch_size examples
        while len(batch_tokens) < self.batch_size:
            if self.current_idx >= len(self.current_data):
                try:
                    self.load_next_file()
                except RuntimeError:
                    if len(batch_tokens) == 0:
                        raise StopIteration
                    break
                    
            text = self.current_data[self.current_idx]['text']
            # Truncate text if it's too long to avoid tokenizer errors
            if len(text) > 10000:
                text = text[:10000]
                
            try:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=self.seq_len)
                
                # Truncate or pad to seq_len
                if len(tokens) > self.seq_len:
                    tokens = tokens[:self.seq_len]
                elif len(tokens) < self.seq_len:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
                    
                batch_texts.append(text)
                batch_tokens.append(tokens)
            except Exception as e:
                print(f"Error tokenizing text: {e}")
                # Skip this example and move to the next
                
            self.current_idx += 1
        
        if len(batch_tokens) == 0:
            raise StopIteration
            
        # Convert to tensors and move to CUDA
        batch_tokens = torch.tensor(batch_tokens, device=self.device)
        return batch_tokens


'''
Changelog Naive -> Better -> Best 
    - don't load an entire file into memory at once, takes up too much space, just read line by line or in chunks 
'''


class BetterDataLoader:
    def __init__(self, data_path, batch_size, seq_len, device='cuda'):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set pad token to eos token if pad token is not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.current_idx = 0
        self.device = device
        self.current_file = None
        self.current_data = []
        self.files = list(self.data_path.glob("*.jsonl"))
        self.file_idx = 0
        
    def load_next_file(self):
        if self.file_idx >= len(self.files):
            self.file_idx = 0
            if len(self.files) == 0:
                raise RuntimeError("No .jsonl files found in data directory")
        
        with open(self.files[self.file_idx], 'r') as f:
            self.current_data = [json.loads(line) for line in f]
        self.file_idx += 1
        self.current_idx = 0
        
    def __iter__(self):
        self.current_idx = 0
        self.file_idx = 0
        self.load_next_file()
        return self
    
    def __next__(self):
        batch_texts = []
        batch_tokens = []
        
        # Collect batch_size examples
        while len(batch_tokens) < self.batch_size:
            if self.current_idx >= len(self.current_data):
                try:
                    self.load_next_file()
                except RuntimeError:
                    if len(batch_tokens) == 0:
                        raise StopIteration
                    break
                    
            text = self.current_data[self.current_idx]['text']
            # Truncate text if it's too long to avoid tokenizer errors
            if len(text) > 10000:
                text = text[:10000]
                
            try:
                tokens = self.tokenizer.encode(text, truncation=True, max_length=self.seq_len)
                
                # Truncate or pad to seq_len
                if len(tokens) > self.seq_len:
                    tokens = tokens[:self.seq_len]
                elif len(tokens) < self.seq_len:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
                    
                batch_texts.append(text)
                batch_tokens.append(tokens)
            except Exception as e:
                print(f"Error tokenizing text: {e}")
                # Skip this example and move to the next
                
            self.current_idx += 1
        
        if len(batch_tokens) == 0:
            raise StopIteration
            
        # Convert to tensors and move to CUDA
        batch_tokens = torch.tensor(batch_tokens, device=self.device)
        return batch_tokens

