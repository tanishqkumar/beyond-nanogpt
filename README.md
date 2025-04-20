## Beyond NanoGPT: Go From LLM Beginner to AI Researcher!

![image](https://github.com/user-attachments/assets/b2943618-d5ed-468d-b792-d1cf4e0d6c6a)


**Beyond-NanoGPT** is the minimal and educational repo aiming to **bridge between nanoGPT and research-level deep learning.** 
This repo includes annotated and from-scratch implementations of tens of crucial modern techniques in frontier deep learning, aiming to help newcomers learn enough practical deep learning to start running experiments and thus contributing to modern research. 

It implements everything from inference techniques like KV caching and speculative decoding to 
architectures like vision and diffusion transformers to attention variants like linear or sparse attention. *Thousands of lines of 
self-contained and hand-written PyTorch to help you upskill your technical fundamentals.* The goal is for you to 
read and reimplement the techniques and systems in this repository to learn the nitty-gritty details more deeply. 

*Because everything is implemented by-hand, the code comments explain the especially subtle
details often glossed over both in papers and production codebases.*

## Quickstart
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/tanishqkumar/beyond-nanogpt.git
   ```
2. **Get Minimal Dependencies:**

   ```bash
   pip install torch numpy torchvision wandb tqdm transformers datasets diffusers matplotlib pillow jupyter
   ```

3. **Start learning!**
   The code is meant for you to read carefully, hack around with, then re-implement yourself from scratch and compare to. 
   You can just run `.py` files with vanilla Python in the following way. 
   ```bash
   cd train-vanilla-transformer/
   python train.py
   ``` 
   or for instance 
   ```bash 
   cd architectures/
   python train_dit.py
   ```
   Everything is written to be run on a single GPU. The code is self-documenting with comments for intuition and elaborating 
   on subtleties I found tricky to implement. 
   Arguments are specified at the bottom of each file. 
   Jupyter notebooks are meant to be stepped through.
   

## Current Implementations and Roadmap

### Key Deep Learning architectures
- ✅ Vanilla causal Transformer for language modeling (starting point) `train-vanilla-transformer/train.py`
- ✅ Vision Transformer (ViT) `architectures/train_vit.py`
- ✅ Diffusion Transformer (DiT) `architectures/train_dit.py`
- ✅ RNN for language modeling `architectures/train_rnn.py` 
- ✅ Residual Networks (ResNet) `architectures/train_resnet.py`
- ✅ MLP-Mixer `architectures/train_mlp_mixer.py`
- [Coming Soon]: MoE, Decision Transformers, LSTM, Mamba. 

### Key Attention Variants
- ✅ Vanilla Self-Attention `attention-variants/vanilla_attention.ipynb` 
- ✅ Multi-head Self-Attention `attention-variants/mhsa.ipynb` 
- ✅ Grouped-Query Attention `attention-variants/gqa.ipynb`
- ✅ Linear Attention `attention-variants/linear_attention.ipynb` 
- ✅ Sparse Attention `attention-variants/sparse_attention.ipynb`
- ✅ Cross Attention `attention-variants/cross_attention.ipynb`
- [Coming Soon]: Multi-Latent Attention. 

### Key Transformer++ Optimizations
- ✅ KV Caching `transformer++/KV_cache.ipynb` 
- ✅ Speculative Decoding `transformer++/speculative_decoding.ipynb`
- ✅ Optimized Dataloading `train-vanilla-transformer/` 
   - ✅ Producer-consumer asynchronous dataloading
   - ✅ Sequence packing
- ✅ Byte-Pair Encoding `transformer++/bpe.ipynb`
- [Coming Soon]: RoPE embeddings, continuous batching.

### Key RL Techniques [Coming Soon]
- Classical RL:
   - Cartpole with DQN
   - Atari Breakout with PPO
   - HalfCheetah‑v3 with SAC
   - CarRacing‑v2 at >900 score (model-based RL)
   - IMPALA (distributed RL)
   - Neural Chess Engine (self-play, MCTS)
- LLMs
   - RLHF a base model with UltraFeedback 
   - DPO a base model with UltraFeedback
   - GRPO for reasoning: outcome reward on MATH
   - GRPO for humor: RLAIF reward signal 
   - GRPO for tool-use: constitutional AI
   - GRPO on theorem proving in Lean

### Key MLSys Techniques [Coming Soon]
- Communication collectives (scatter, gather, ring/tree allreduce)
- Minimal FSDP re-implementation (data and tensor parallelism)
- Ring Attention
- Flash Attention
- Paged Attention 
- 4-bit weight-only inference quantization

### Key Agents Techniques [Coming Soon]
- Simulate a Society as in "Generative Simulcra"
- Implement ReAct + Function Calling on a base model so it can use tools to respond to queries
- Make an LLM agent that takes (instructions, GitHub credentials) and makes a simple PR in the repo
- Make an LLM agent that can add items to cart and screenshot the total on WebArena

### Key Diffusion Techniques [Coming Soon]
- Image generation with a GAN
- 32 × 32 DDPM on CIFAR‑10 with DDIM sampling
- Classifier-free guidance
- Implement a consistency model
   - Compare scaling laws to a standard diffusion transformer 
- Score‑distillation cross‑modal generation
- Conditional text generation with a diffusion-language model in latent space
   - Implement discrete diffusion for language modeling 

---

## Notes

- The codebase expects a GPU. It might work with CPU, but no guarantees. 
I recommend either a consumer laptop with GPU, paying for Colab/Runpod, 
or simply asking a compute provider or local university for a compute grant if those are out of 
budget (this works surprisingly well, people are very generous). 
- Most `.py` scripts take in `--verbose` and `--wandb` as command line arguments when you run them, to enable detailed logging and sending logs to wandb, respectively. Feel free to hack these to your needs. 
- Feel free to email me at [tanishq@stanford.edu](mailto:tanishq@stanford.edu) with feedback, implementation/feature requests, 
and to raise any bugs as GitHub issues. I am committing to implementing new techniques people want over the next month, and 
welcome contributions or bug fixes by others. 

**Happy coding, and may your gradients never vanish!**
