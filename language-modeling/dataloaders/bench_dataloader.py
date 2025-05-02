import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import time
import traceback
from transformers import GPT2TokenizerFast

from dataloaders.dataloader0 import DataLoader as DataLoader0
from dataloaders.dataloader1 import DataLoader as DataLoader1
from dataloaders.dataloader2 import DataLoader as DataLoader2

def create_dummy_file(filename, n_lines):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i in range(n_lines):
                text_content = ("This is sample text line number {}. It contains various words and punctuation to test tokenization. ".format(i)) * (5 + i % 5)
                line_data = {"id": i, "text": text_content}
                f.write(json.dumps(line_data) + "\n")
    except Exception as e:
        print("Error creating dummy file '{}': {}".format(filename, e), flush=True)
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)
        sys.exit(1)

def remove_dummy_file(filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print("Error removing dummy file '{}': {}".format(filename, e), flush=True)

# tweaked for dataloader0's tokenization
if __name__ == '__main__':
    verbose_mode = '--verbose' in sys.argv
    if verbose_mode:
        print("Verbose mode enabled.")
    dummy_file = "dummy_data.jsonl"
    n_lines = 200000  # change this to profile different sizes

    if verbose_mode:
        print(f"Creating dummy file '{dummy_file}' with {n_lines} lines.")
    # make a test file for the dataloaders
    create_dummy_file(dummy_file, n_lines)
    if verbose_mode:
        print("Dummy file created successfully.")

    try:
        if verbose_mode:
            print("Loading tokenizer from pretrained 'gpt2'.")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # make sure dataloader0 has the right eos token
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if verbose_mode:
            print("Tokenizer loaded and configured successfully.")
    except Exception as e:
        print("Error initializing tokenizer:", e, flush=True)
        traceback.print_exc()
        remove_dummy_file(dummy_file)
        sys.exit(1)

    # settings for profiling
    batch_size = 64
    seq_len = 512   # how long each packed sequence should be
    num_workers = 48
    prefetch_batches = 200
    max_batches_to_test = 500  # how many batches to run per implementation
    if verbose_mode:
        print("Profiling configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Number of workers: {num_workers}")
        print(f"  Prefetch batches: {prefetch_batches}")
        print(f"  Max batches to test: {max_batches_to_test}")

    def run_profiling(DataLoaderClass, label):
        if verbose_mode:
            print(f"Starting profiling for {label}...")
        if label == "dataloader0":
            loader = DataLoaderClass(tokenizer=tokenizer,
                                     path=dummy_file,
                                     batch_size=batch_size,
                                     seqlen=seq_len,
                                     verbose=verbose_mode)
        else:
            loader = DataLoaderClass(tokenizer=tokenizer,
                                     path=dummy_file,
                                     nworkers=num_workers,
                                     batch_size=batch_size,
                                     seqlen=seq_len,
                                     prefetch=prefetch_batches,
                                     verbose=verbose_mode)
        batches_received = 0
        total_tokens = 0
        start_time = time.monotonic()
        while batches_received < max_batches_to_test:
            inputs, targets = loader.get_batch()
            if inputs is None or targets is None:
                if verbose_mode:
                    print(f"{label}: No more batches received. Ending profiling loop.")
                break
            batches_received += 1
            total_tokens += targets.numel()
            if verbose_mode and (batches_received % 100 == 0):
                print(f"{label}: Processed {batches_received} batches so far.")
        end_time = time.monotonic()
        loader.close()
        elapsed = end_time - start_time
        if verbose_mode:
            print(f"{label}: Profiling completed in {elapsed:.2f} seconds with {batches_received} batches processed.")
        return {
            "label": label,
            "batches": batches_received,
            "tokens": total_tokens,
            "total_time_sec": elapsed,
            "batches_per_sec": batches_received / elapsed if elapsed > 0 else float('inf'),
            "tokens_per_sec_M": (total_tokens / elapsed / 1e6) if elapsed > 0 else float('inf')
        }

    if verbose_mode:
        print("Running profiling for all DataLoader implementations...")
    # test all the dataloaders
    result0 = run_profiling(DataLoader0, "dataloader0")
    result1 = run_profiling(DataLoader1, "dataloader1") 
    result2 = run_profiling(DataLoader2, "dataloader2")

    # print the final stats
    print("Profiling Summary:")
    for res in (result0, result1, result2):
        print(f"{res['label']} - Batches: {res['batches']}, Tokens: {res['tokens']}, Total Time: {res['total_time_sec']:.2f} sec, "
              f"Batches/sec: {res['batches_per_sec']:.2f}, Tokens/sec: {res['tokens_per_sec_M']:.3f} M")

    remove_dummy_file(dummy_file)