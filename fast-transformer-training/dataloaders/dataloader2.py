'''
fast transformer dataloader
---------------------------
this is a high-performance data loading system designed for transformer training with the following features:
* uses multiprocessing to parallelize data loading and preprocessing
* implements a producer-consumer pattern with worker processes and a main process
* workers handle file reading, tokenization, and batch packing in parallel
* uses a ring buffer (Queue) to prefetch batches ahead of training
* avoids Python GIL limitations by offloading work to separate processes
* minimizes data transfer overhead by sending only ready-to-use batches
* handles large datasets efficiently by streaming from disk
* advantages:
  - custom-built for transformer sequence packing
  - more efficient tokenization in parallel
  - better prefetching with configurable buffer size
  - reduced memory overhead by processing only what's needed
  - no need for collate functions or samplers
  - direct control over the data pipeline (because who doesn't love being in control?)

'''
import torch
import torch.multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
import orjson as json
import os 
import traceback

class DataLoader:
    def __init__(
        self,
        tokenizer,
        path="/n/netscratch/gershman_lab/Lab/tkumar/datasets/dclm/global-shard_01_of_10/newest_data/tinystories.jsonl",
        nworkers=48,
        batch_size=64,
        seqlen=512,
        prefetch=200,
        verbose=False,
    ):
        """
        DataLoader using multiprocessing to read, tokenize and pack batches.

        Args:
            tokenizer: a tokenizer with a valid `eos_token_id`.
            path (str): path to the JSONL dataset (each line has a "text" key).
            nworkers (int): number of worker processes.
            batch_size (int): sequences per batch.
            seqlen (int): fixed sequence length (get_batch returns seqlen-1 slices).
            prefetch (int): number of batches in the ring buffer.
            verbose (bool): enable verbose logging.
        """
        # check that tokenizer has a valid eos_token_id
        if not hasattr(tokenizer, "eos_token_id") or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have a valid 'eos_token_id'.")

        # store initialization parameters
        self.tokenizer = tokenizer
        self.path = path
        self.nworkers = max(1, nworkers)
        self.batch_size = batch_size
        self.seqlen = seqlen  # stored sequence length (batch slices have seqlen-1 tokens)
        self.prefetch = max(1, prefetch)
        self.verbose = verbose

        print(f"[fast‑loader] workers={self.nworkers}, batch_size={self.batch_size}, prefetch={self.prefetch}", flush=True)

        self.procs = []
        self._finished_workers_count = 0

        # initialize the shared ring buffer for batches
        self._init_ring_buffer()
        # initialize queues for free and filled slots
        self._init_queues()

        if self.nworkers <= 0:
            print("warning: nworkers <= 0, setting to 1.", flush=True)
            self.nworkers = 1

        # compute file chunk size based on dataset file size and number of workers
        chunk = self._compute_chunk()

        # start worker processes for parallel data loading
        self._start_workers(chunk)

        if self.verbose and self.nworkers:
            print(f"started {self.nworkers} workers; ring buffer prefetch={self.prefetch}", flush=True)

    def _init_ring_buffer(self):
        # initialize a shared ring buffer for storing batches
        if self.prefetch <= 0 or self.batch_size <= 0 or self.seqlen <= 0:
            raise ValueError("prefetch, batch_size, and seqlen must be positive.")
        try:
            self.ring = torch.empty((self.prefetch, self.batch_size, self.seqlen), dtype=torch.long)
            self.ring.share_memory_()
        except Exception as e:
            print(f"error creating shared memory tensor: {e}", flush=True)
            raise

        try:
            # if cuda is available, pin the ring buffer to cpu memory for faster transfers
            if torch.cuda.is_available():
                self.ring = self.ring.pin_memory()
                if self.verbose:
                    print("ring buffer pinned to cpu memory.", flush=True)
            elif self.verbose:
                print("cuda not available, skipping pinning.", flush=True)
        except Exception as e:
            print(f"warning: failed to pin memory: {e}", flush=True)

    def _init_queues(self):
        # initialize free_slots and full_slots queues
        self.free_slots = Queue(maxsize=self.prefetch)
        self.full_slots = Queue(maxsize=self.prefetch)
        for slot_id in range(self.prefetch):
            self.free_slots.put(slot_id)

    def _compute_chunk(self):
        # compute the file chunk size for each worker
        try:
            total = os.path.getsize(self.path)
        except FileNotFoundError:
            print(f"error: dataset file not found at {self.path}", flush=True)
            raise
        except Exception as e:
            print(f"error getting file size for {self.path}: {e}", flush=True)
            raise

        if total == 0:
            print(f"warning: dataset file {self.path} is empty. no workers started.", flush=True)
            self.nworkers = 0
            return 0
        else:
            chunk = total // self.nworkers
            if chunk == 0:
                print(f"warning: file size ({total}) smaller than workers ({self.nworkers}). using minimal chunk.", flush=True)
                chunk = 1
            return chunk

    def _start_workers(self, chunk):
        # start worker processes, each processing a specific chunk of the file
        for wid in range(self.nworkers):
            start = wid * chunk
            end = (wid + 1) * chunk if wid < self.nworkers - 1 else None
            p = mp.Process(
                target=self._worker_entrypoint,
                args=(
                    wid,
                    start,
                    end,
                    self.path,
                    self.tokenizer,
                    self.batch_size,
                    self.seqlen,
                    self.ring,
                    self.free_slots,
                    self.full_slots,
                ),
                name=f"DataLoaderWorker-{wid}",
            )
            p.start()
            self.procs.append(p)

    @staticmethod
    def _worker_entrypoint(*args):
        # wrapper to catch and report worker exceptions
        worker_id = args[0]
        full_queue = args[-1]
        try:
            DataLoader._worker(*args)
        except Exception as e:
            print(f"worker {worker_id} crashed: {e}", flush=True)
            traceback.print_exc()
            try:
                full_queue.put(None)
            except Exception as qe:
                print(f"worker {worker_id}: failed to signal completion: {qe}", flush=True)

    @staticmethod
    def _worker(worker_id, byte_start, byte_end,
                path, tokenizer, batch_size, seqlen,
                ring_buffer, free_queue, full_queue):
        """
        worker process: reads a file chunk, tokenizes texts, and packs batches.

        this worker uses a ring buffer mechanism to store complete batches.
        each slot (from free_queue) holds a tensor of shape [batch_size, seqlen].
        when a batch is ready, the worker copies it into the ring buffer and signals via full_queue.
        """
        eos_id = tokenizer.eos_token_id
        flat_buffer = torch.empty(batch_size * seqlen, dtype=torch.long)  # temporary buffer to accumulate tokens
        ptr = 0  # pointer into flat_buffer
        mini_batch_size = 64
        texts = []
        lines_processed = 0
        batches_produced = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                # if not starting at the beginning, skip a potential partial line
                if byte_start > 0:
                    f.seek(byte_start)
                    try:
                        f.readline()  # skip partial line
                    except Exception as e:
                        print(f"worker {worker_id}: error skipping partial line: {e}", flush=True)

                # helper function to process and merge texts into a token stream
                def process_and_merge(texts_list, mode="regular"):
                    try:
                        encoded = tokenizer(texts_list, add_special_tokens=False, truncation=False)["input_ids"]
                    except Exception as exc:
                        if mode == "flush":
                            print(f"worker {worker_id}: tokenizer error during flush: {exc}. skipping remaining texts.", flush=True)
                        else:
                            print(f"worker {worker_id}: tokenizer error: {exc}. skipping batch.", flush=True)
                        return None
                    merged_local = []
                    for i, seq in enumerate(encoded):
                        if isinstance(seq, list):
                            seq.append(eos_id)
                            merged_local.extend(seq)
                        else:
                            print(f"worker {worker_id}: tokenizer produced non-list at index {i}", flush=True)
                    return merged_local

                # helper function to flush flat_buffer into the ring buffer when full
                def flush_flat_buffer():
                    nonlocal ptr, batches_produced
                    try:
                        slot = free_queue.get()
                    except Exception as q_e:
                        print(f"worker {worker_id}: error getting free slot: {q_e}", flush=True)
                        raise
                    ring_buffer[slot].copy_(flat_buffer.view(batch_size, seqlen))
                    try:
                        full_queue.put(slot)
                    except Exception as q_e:
                        print(f"worker {worker_id}: error putting full slot: {q_e}", flush=True)
                        raise
                    batches_produced += 1
                    ptr = 0

                # helper function to copy tokens from merged tensor into flat_buffer
                def copy_tokens(merged_t):
                    nonlocal ptr
                    offset = 0
                    total_elements = merged_t.numel()
                    while offset < total_elements:
                        room = flat_buffer.numel() - ptr
                        ncopy = min(room, total_elements - offset)
                        flat_buffer[ptr : ptr + ncopy].copy_(merged_t[offset : offset + ncopy])
                        ptr += ncopy
                        offset += ncopy
                        # if flat_buffer is full, flush it into the ring buffer
                        if ptr == flat_buffer.numel():
                            flush_flat_buffer()

                # read the file line by line and accumulate texts
                while True:
                    current_pos = f.tell()
                    # break if past designated byte_end and no pending texts
                    if byte_end is not None and current_pos > byte_end and not texts:
                        break
                    raw = f.readline()
                    if not raw:
                        break
                    is_last_line = (byte_end is not None and current_pos >= byte_end)
                    lines_processed += 1
                    try:
                        data = json.loads(raw)
                        text = data["text"]
                        if not isinstance(text, str):
                            print(f"worker {worker_id}: skipping line {lines_processed}, invalid text", flush=True)
                            continue
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"worker {worker_id}: skipping line {lines_processed} due to error: {e}", flush=True)
                        continue
                    texts.append(text)
                    # if enough texts are accumulated or if it's the end of the chunk, process them
                    if len(texts) >= mini_batch_size or (is_last_line and texts):
                        merged = process_and_merge(texts, mode="regular")
                        texts.clear()
                        if merged is None or not merged:
                            continue
                        merged_t = torch.as_tensor(merged, dtype=torch.long)
                        copy_tokens(merged_t)

                # flush any remaining texts after file processing is complete
                if texts:
                    merged = process_and_merge(texts, mode="flush")
                    texts.clear()
                    if merged:
                        merged_t = torch.as_tensor(merged, dtype=torch.long)
                        copy_tokens(merged_t)
        except Exception as e:
            print(f"worker {worker_id} critical error: {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            # signal completion to the consumer by placing a None in full_queue
            try:
                full_queue.put(None, timeout=0.1)
            except Exception as e:
                print(f"worker {worker_id}: failed to signal completion: {e}", flush=True)

    def get_batch(self):
        """Return the next (inputs, targets) batch or (None, None) if finished."""
        # if no workers were started, return end-of-file immediately
        if not self.procs and self.nworkers == 0:
            if self.verbose:
                print("no workers, returning eof.", flush=True)
            return None, None

        # loop until a valid batch is retrieved or all workers have finished
        while self._finished_workers_count < self.nworkers:
            try:
                slot = self.full_slots.get(timeout=30.0)
            except Empty:
                if not [p for p in self.procs if p.is_alive()]:
                    print("get_batch timeout; no workers alive.", flush=True)
                    self._finished_workers_count = self.nworkers
                    break
                else:
                    if self.verbose:
                        print("get_batch waiting...", flush=True)
                    continue
            except Exception as e:
                print(f"error getting from full_slots: {e}", flush=True)
                traceback.print_exc()
                self._finished_workers_count = self.nworkers
                break

            if slot is None:
                self._finished_workers_count += 1
                if self.verbose:
                    print(f"received eof signal ({self._finished_workers_count}/{self.nworkers})", flush=True)
                continue
            else:
                try:
                    if not isinstance(slot, int) or not (0 <= slot < self.prefetch):
                        print(f"invalid slot index: {slot}", flush=True)
                        continue
                    batch = self.ring[slot]
                    # if seqlen is 1 or less, no slicing is needed
                    if self.seqlen <= 1:
                        inputs = batch.clone()
                        targets = batch.clone()
                    else:
                        inputs = batch[:, :-1].clone()
                        targets = batch[:, 1:].clone()
                    # recycle the slot
                    self.free_slots.put(slot)
                    return inputs, targets
                except IndexError:
                    print(f"invalid slot index {slot} in ring buffer", flush=True)
                    continue
                except Exception as e:
                    print(f"error processing batch from slot {slot}: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        self.free_slots.put(slot)
                    except Exception as qe:
                        print(f"error recycling slot {slot}: {qe}", flush=True)
                    continue

        if self.verbose:
            print("all workers finished.", flush=True)
        try:
            while True:
                slot = self.full_slots.get_nowait()
                if slot is not None:
                    if self.verbose:
                        print(f"recycling leftover slot {slot}.", flush=True)
                    self.free_slots.put(slot)
                else:
                    self._finished_workers_count += 1
        except Empty:
            pass
        except Exception as e:
            print(f"error draining full_slots: {e}", flush=True)

        return None, None

    def close(self):
        """Clean up resources by joining workers and closing queues."""
        # if the dataloader is already closed, exit early
        if not hasattr(self, "procs") or self.procs is None:
            if self.verbose:
                print("dataLoader already closed.", flush=True)
            return

        print("closing dataLoader...", flush=True)

        if self.procs:
            print("joining workers...", flush=True)
            for p in self.procs:
                try:
                    if p.is_alive():
                        p.join(timeout=0.1)
                        if p.is_alive():
                            print(f"worker {p.name} did not join, terminating.", flush=True)
                            p.terminate()
                            p.join(timeout=0.1)
                    elif self.verbose:
                        print(f"worker {p.name} already exited.", flush=True)
                except Exception as e:
                    print(f"error joining worker {p.name}: {e}", flush=True)
                    traceback.print_exc()
            print("workers joined.", flush=True)

        print("closing queues...", flush=True)
        for q_name in ["free_slots", "full_slots"]:
            q = getattr(self, q_name, None)
            if q is None:
                continue
            try:
                while True:
                    q.get_nowait()
            except Empty:
                pass
            except Exception as e:
                print(f"error draining queue {q_name}: {e}", flush=True)
            try:
                q.close()
                q.join_thread()
            except Exception as e:
                print(f"error closing queue {q_name}: {e}", flush=True)
        print("queues closed.", flush=True)

        self.procs = None
        self.ring = None
        self.free_slots = None
        self.full_slots = None
        if self.verbose:
            print("dataLoader closed.", flush=True)

    