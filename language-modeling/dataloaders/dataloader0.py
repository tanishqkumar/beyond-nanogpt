# this is the silly dataloader that just uses one process to go line by line and retrieve batch_sz worth of sequences
# tokenizing as it goes along. it's the starting point for our optimization in the other two files 
# which use many cpus and processes to be able to asynchronously construct batches to keep GPUs fed!

class DataLoader:
    def __init__(self, tokenizer, path, batch_size=64, seqlen=512, verbose=False):
        """
        Naive, serial dataloader for language modeling.
        Reads a JSONL file (each line a JSON object with a "text" field) one line at a time,
        accumulates token ids until there are enough to form a batch, then returns a batch of inputs and targets.
        
        Args:
            tokenizer: An instance of GPT2TokenizerFast.
            path (str): Path to the JSONL file.
            batch_size (int): Number of sequences per batch.
            seqlen (int): Sequence length of each sequence.
            verbose (bool): If True, prints debug information.
        """
        self.tokenizer = tokenizer
        self.path = path
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.verbose = verbose
        try:
            self.file = open(self.path, "r", encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to open file {self.path}: {e}")
        self.buffer_tokens = []  # Buffer for accumulating token ids

    def get_batch(self):
        """
        Accumulates tokens by reading the file one line at a time until
        there are enough tokens to carve out a contiguous segment of length (batch_size * seqlen) + 1.
        Each line is parsed as JSON and tokenized individually.
        Creates input tokens and target tokens (shifted by one token).
        
        Returns:
            inputs (torch.Tensor): Tensor of shape (batch_size, seqlen)
            targets (torch.Tensor): Tensor of shape (batch_size, seqlen)
            If not enough tokens can be read (e.g. end-of-file), returns (None, None)
        """
        required = self.batch_size * self.seqlen + 1  # one extra token for shifting targets
        import json
        import torch
        while len(self.buffer_tokens) < required:
            line = self.file.readline()
            if not line:
                if self.verbose:
                    print("Reached end of file; not enough tokens for a full batch.")
                return None, None
            try:
                data = json.loads(line)
                text = data.get("text", "")
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing line: {e}")
                continue
            try:
                encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            except Exception as e:
                if self.verbose:
                    print(f"Tokenizer error: {e}. Skipping this line.")
                continue
            if isinstance(encoded, list):
                if self.tokenizer.eos_token_id is not None:
                    encoded.append(self.tokenizer.eos_token_id)  # Append EOS token
                self.buffer_tokens.extend(encoded)
            else:
                if self.verbose:
                    print(f"Warning: tokenizer produced non-list result of type {type(encoded)}. Skipping.")
        # Slice a segment exactly long enough to create inputs and targets
        segment = self.buffer_tokens[:required]
        self.buffer_tokens = self.buffer_tokens[required:]
        input_ids = segment[:-1]
        target_ids = segment[1:]
        inputs = torch.tensor(input_ids, dtype=torch.long).view(self.batch_size, self.seqlen)
        targets = torch.tensor(target_ids, dtype=torch.long).view(self.batch_size, self.seqlen)
        return inputs, targets

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.get_batch()
        if batch[0] is None:
            raise StopIteration
        return batch
