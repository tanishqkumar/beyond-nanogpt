# Lessons from the Trenches 

**I've written over ten thousand lines of pytorch by hand for this repo**, 
often relying on nothing more than `nn.Linear` to implement things from diffusion transformers to multi-latent attention and deep RL techniques like PPO and SAC. As someone who started with an LLM-only background in deep learning, and some theoretical training, **here are some of the things I've learned, or wish I knew before I started.** 

- The objective function is paramount. People often focus on the architecture, algorithmic 
innovations, even the data, but really most seminal papers in deep learning are just new objectives to train on that make neural models particularly 
useful or well-behaved. Work in generative modeling/vision makes this particularly clear -- if you come from an LLM background this is definitely 
not obvious because you're only ever meaningfully exposed to next token prediction, so you think there's no room for innovation/alpha on the objective 
front, but actually it's the most important thing to innovate on outside of LLM-land. 
    - Even in LLM land, when it comes to RL, reward shaping and model design are just different ways of designing the objective for an RL algorithm to 
    hillclimb, it's exactly the same thing. 
- Inter-CPU comms are easy because CPUs have a shared memory space that allows them to talk to each other.
 This is why `multiprocessing` is so easy to use but inter-GPU communications
are relatively painful (both to write code for and performance-wise), because GPUs have their own memory space. 
- When using multi-CPUs (eg. for async RL, or optimized dataloaders), make sure to put big objects on `torch.share_memory` instead of passing them through `mp.Queue` which has a comm 
overhead that share memory doesn't. Instead, use queues to pass around indices or slots sparingly, that tell functions where in shared 
memory to read/write. 
- Avoid repeated `torch.tensor(x)` calls; instead, preallocate tensors and index into them when possible.
- Fast implementations minimize loops by leveraging torch native functions that are highly optimized (e.g., `cumsum`, `bmm`). For instance, expressing convolutions as `F.unfold + bmm` rather than looping over kernel windows. Often you can get a 10x performance gain by just stepping back and asking which vanilla python functions 
in the code can be rewritten as native torch functions. 
- Many operations can be expressed differently for efficiency - convolution operations are particularly flexible. Operations that might seem to require loops can often be implemented as clever convolutions with specific matrices. With practice, recognizing when to express linear operations as convolutions becomes intuitive.
- Sometimes code provides more clarity than mathematical notation. For example, the reparameterization trick (which looks complex mathematically) is simply `z * sqrt(std) + mean` where we differentiate through mean/std parameters ("differentiating through sampling").
- Broadcasting is critically important. For instance, if `logits` is a `[b, s, t]` tensor and `classes` is a `[b, s]` tensor specifying which class each of the `[b, s]` entries is, you can get the logit for each of `[b, s]` examples by simply indexing `logits[arange(b)[:,None], arange(s), classes]`. This is because PyTorch knows to broadcast the ranges into a 2-tensor of `[b, s]` and so fills the output, also a `[b, s]` 2-tensor, with `logits[0, 0, classes[0, 0]], logits[0, 1, classes[0, 1]]` and so forth. 
    - This small PyTorch trick performing a common operation (extracting class logits) basically gives you optimal performance compared to eg. looping or even materializing a `[b, s]` tensor of indices to pass  in (which can consume a lot of memory) as an index. These details matter!
- Most neural architectures are inherently difficult to train. Small details like normalization techniques (LayerNorm, BatchNorm, GroupNorm) and residual connections aren't mere implementation details but breakthrough concepts that made previously untrainable models viable. Before 2012, the scientific consensus held that learning complex nonlinear functions was fundamentally untenable due to instability and training difficulties - it turned out these challenges were engineering problems with solutions, not fundamental limitations.
- Modern deep learning involves more mathematical sophistication than is apparent from studying LLMs alone. Reading RL papers reveals discussions about loss landscape geometry, Hessians, and information theory (proper scoring functions, natural policy gradients) - this mathematical foundation is essential, not merely academic.
    - Generative models evince this fact. When developing new methods to learn high-dimensional distributions, concepts like metrics in distributional space and optimal transport become unavoidable. Understanding things like probability flow ODEs
    and why denoising is equivalent to learning the score (which in turn determines the distribution) is crucial for research in this area, even if you can implement working models without this depth.
    - This mathematical sophistication makes me grateful for my strong undergraduate training in math and statistics. In comparison, LLMs are conceptually simpler - training a transformer on next-token prediction, with everything else (activation functions, multi-token prediction, etc.) being minor variations on the theme. 
- There are many common programming patterns you start to see repeated once you implement many things. A typical example is the consumer-producer model, 
where a bunch of "producer" processes construct objects, put them into a shared buffer, and a small number of "consumer" processes consume them 
and perform some computation. 
    - This appears in `dataloaders` in the form of many CPU workers doing pre-processing (tokenization, sequence packing, adding BOS/EOS) to raw 
      `.jsonl` files, then feeding the outputs to a GPU to do forward/backward on (pretraining itself). The goal is for the workers to keep the GPU 
      fed, i.e., fully utilized, throughout. 
    - This also appears in distributed RL! In `train_impala.py`, IMPALA is an algorithm that uses CPU workers (producers) to do rollouts (small batch, 
      forward only) and store the rollouts $\{(s_t, a_t, r_t, d_t)\}_{t=0}^T$ in a global central buffer (shared memory) where then a single GPU worker 
      (consumer) learns based on those rollouts (i.e., high-batch, forward + backward, hence the need for a GPU for the large matmuls). 
    - In other words, writing a SOTA distributed RL algorithm was easy once I had written an optimized dataloader -- two seemingly unrelated concepts!
- Loss doesn't always have to go down! In some RL settings, loss can increase with reward (which ultimately is what we care about). 
When reward increases as policy improves, the loss can counter-intuitively increase too as average episode length grows (since your policy, e.g., in cartpole, 
keeps the pole up longer). 
  - Another instance where loss increase can actually be fine is in implementations with multiple networks interacting to learn. GANs are the simplest example, where learning in one network 
can make it harder for the other network to decrease loss, but DDPG in RL is another example, where the Q-network (critic) and policy (actor) are learning 
in parallel and both appear in each loss function. 
- Exploration in RL is fundamentally different across algorithms: epsilon-greedy (DQN) gradually reducing randomness, additive noise at inference-time (DDPG), and entropy-based exploration (policy gradient methods) all have different characteristics and failure modes. 
  - One research direction I think is interesting in the future is "structured exploration" -- adding noise as a way to incentivize exploration in RL systems is fine, but when outlier humans explore they explore some orthogonal -- but structured -- space rather than just a noised version 
of their usual space. 
- The difference between on-policy and off-policy algorithms is fundamental to RL. Using off-policy data effectively is a major research question in pure RL
in a way that is surprising to LLM folks like myself, because in LLM land off-policy training (next token prediction-based pretraining) *just works!* The issues 
of distribution mismatch between inference/pretraining time are kind of non-issues in practice if you instruction-tune the base model at the end. 
  - But, as we move into training for more long-horizon (agentic) LLM tasks, this distinction is important for LLM folks to understand. I bet you techniques like IMPALA (except between GPUs instead of between GPU/CPU) where you have one sub-cluster for rollouts and another for grad updates with broadcasting and some sort of importance adjustment are being used in production in frontier labs!
