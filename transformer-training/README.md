A summary of this subfolder and perf optimizations in `dataloaders`. 

`transformer.py` defines the decoder-only causal Transformer.

`train_naive.py` trains that using data from `TinyStories` on HuggingFace.
`train_full.py` instead uses our hand-written dataloader from `dataloaders`. We provide three hand-written dataloaders, 
in increasing sophistication/throughput (see bottom for profiling). We use `dataloaders/bench_dataloader.py` to profile 
throughput of all three. 

This codebase includes three different DataLoader implementations with increasing performance:

1. DataLoader v0 (dataloader0.py) - "Naive Implementation"
   A simple, single-process loader that reads JSONL line-by-line and tokenizes synchronously. 
   While straightforward, it's CPU-bound with ~28% of runtime spent on data loading.

2. DataLoader v1 (dataloader1.py) - "The Assembly Line" 
   Uses multiple worker processes to parallelize JSONL chunk reading and tokenization.
   Leverages shared memory tensors and prefetch queues for improved throughput.
   Reduces data loading overhead but requires monitoring CPU usage.

   Key optimizations vs v0:
     - Parallel file reading across multiple processes to leverage multi-core CPUs.
     - Shared-memory batch buffers to minimize Python serialization overhead.
     - Prefetch queues to decouple I/O from training, slashing load time from ~28% to near-zero.

3. DataLoader v2 (dataloader2.py) - "The Ring Buffer Beast"
   The most optimized implementation using pre-allocated shared ring buffers,
   pinned CPU memory, chunked file reads, and efficient slot management.
   Provides near-zero stall time for GPU training.

   Key optimizations vs v1:
     - Fully pre-allocated shared ring buffer pinned to CPU memory for fast host->device transfers.
     - Chunked file slicing per worker to avoid partial-line and boundary overhead.
     - Efficient free/full slot queue management to eliminate synchronization bottlenecks.
     - orjson for lightning-fast JSON parsing, boosting tokens/sec from ~0.87M to ~1.12M.

Use `--slowload` flag to enable DataLoader v0 (default is v2). The profiling stats are: 

Profile from `dataloaders/bench_dataloader.py`:

| Dataloader | Batches | Tokens    | Total Time (s) | Batches/sec | Tokens/sec |
|------------|---------|-----------|----------------|-------------|------------|
| v0         | 500     | 16,384,000| 23.75         | 21.05       | 0.690M     |
| v1         | 500     | 16,352,000| 18.80         | 26.60       | 0.870M     |
| v2         | 500     | 16,352,000| 14.64         | 34.15       | 1.117M     |

IMO, writing your own dataloader is one of the most instructive ways to learn a few things:

- **Multi-threading**: I had to debug a lot of deadlocks/race conditions when writing async dataloader
- **Systems Design**: I originally wrote v1 to have `reader_workers` that did IO and tokenization and `batch workers` that converted a token stream to batches, but the interprocess communication ended up making it much slower than hoped for
- **Torch Memory Management**: The latencies of key things like `.copy_` compared to `torch.tensor(x)`, etc.
- **Architecture**: I learned a lot about CPU vs GPU architecture and dependency in the process

I highly recommend you use this as guidance to write your own dataloader from scratch. 