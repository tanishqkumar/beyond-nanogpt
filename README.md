## Beyond NanoGPT: Go From LLM Beginner to AI Researcher!

![image](https://github.com/user-attachments/assets/b2943618-d5ed-468d-b792-d1cf4e0d6c6a)


**Beyond-NanoGPT** is the minimal and educational repo aiming to **bridge between nanoGPT and research-level deep learning.** 
This repo includes annotated and from-scratch implementations of tens of crucial modern techniques in frontier deep learning, aiming to help newcomers learn enough practical deep learning to start running experiments and thus contributing to modern research. 

It implements everything from inference techniques like KV caching and speculative decoding to 
architectures like vision and diffusion transformers to attention variants like linear or sparse attention. *Thousands of lines of 
self-contained and hand-written PyTorch to help you upskill your technical fundamentals. Because everything is 
implemented by-hand, the code comments explain the especially subtle
details often glossed over both in papers and production codebases.*

Checked boxes denote currently implemented and ready to be run. Others are either coming soon or in progress.

## Quickstart
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/tanishqkumar/beyond-nanogpt.git
   ```
2. **Get Minimal Dependencies:**

   ```bash
   pip install torch numpy torchvision wandb tqdm transformers datasets diffusers matplotlib pillow jupyter gym 
   ```

3. **Start learning!**
   The code is meant for you to read carefully, hack around with, then re-implement yourself from scratch and compare to. 
   You can just run `.py` files with vanilla Python in the following way. 
   ```bash 
   cd architectures/
   python train_dit.py
   ```
   or for instance 
   ```bash 
   cd rl/cartpole/
   python train_reinforce.py --verbose 
   ```
   Everything is written to be run on a single GPU. The code is self-documenting with comments for intuition and elaborating 
   on subtleties I found tricky to implement. 
   Arguments are specified at the bottom of each file. 
   Jupyter notebooks are meant to be stepped through.
   

## Current Implementations and Roadmap

### Architectures
- [x] Vanilla causal Transformer for language modeling (starting point) `fast-transformer-training/train_naive.py`
- [x] Vision Transformer (ViT) `architectures/train_vit.py`
- [x] Diffusion Transformer (DiT) `architectures/train_dit.py`
- [x] RNN for language modeling `architectures/train_rnn.py` 
- [x] Residual Networks (ResNet) `architectures/train_resnet.py`
- [x] MLP-Mixer `architectures/train_mlp_mixer.py`
- [ ] LSTM
- [ ] MoE
- [ ] Decision Transformer

### Attention Variants
- [x] Vanilla Self-Attention `attention-variants/vanilla_attention.ipynb` 
- [x] Multi-head Self-Attention `attention-variants/mhsa.ipynb` 
- [x] Grouped-Query Attention `attention-variants/gqa.ipynb`
- [x] Linear Attention `attention-variants/linear_attention.ipynb` 
- [x] Sparse Attention `attention-variants/sparse_attention.ipynb`
- [x] Cross Attention `attention-variants/cross_attention.ipynb`
- [ ] Multi-Latent Attention

### Language Modeling

- [x] Optimized Dataloading `fast-transformer-training/dataloaders` 
   - [x] Producer-consumer asynchronous dataloading 
   - [x] Sequence packing 
- [x] Byte-Pair Encoding `fast-transformer-training/bpe.ipynb`
- [x] KV Caching `fast-transformer-inference/KV_cache.ipynb` 
- [x] Speculative Decoding `fast-transformer-inference/speculative_decoding.ipynb`
- [ ] RoPE embeddings
- [ ] Multi-token Prediction
- [ ] Continuous Batching 

### Reinforcement Learning
- Classical RL:
   - Fundamentals `rl/cartpole`
      - [x] DQN `train_dqn.py`
      - [x] REINFORCE `train_reinforce.py`
      - [x] PPO `train_ppo.py`
   - Actor-Critic and Distributed Variants `rl/pendulum`
      - [x] Vanilla Actor-Critic (A1C) `train_a1c.py`
      - [ ] Asynchronous Advantage Actor-Critic (A3C)
      - [ ] Synchronous Advantage Actor-Critic (A2C)
      - [ ] Soft Actor-Critic (off-policy)
      - [ ] IMPALA (distributed RL)
   - Model-based RL 
      - [ ] PETS 
      - [ ] MBPO 
   - [ ] Neural Chess Engine (self-play, MCTS)
- LLMs
   - [ ] RL finetune a small LM to do arithmetic against a calculator program 
   - [ ] RLHF a base model with UltraFeedback 
   - [ ] DPO a base model with UltraFeedback
   - [ ] GRPO for reasoning: outcome reward on MATH
   - [ ] Distributed RLAIF for multi-tool use

### Generative Modeling

- [x] Image generation with a GAN `diffusion/train_gan.py`
- [x] Image generation with a VAE `diffusion/train_vae.py`
   - [x] Train an autoencoder for reconstruction `diffusion/train_autoencoder.py` 
- [x] Image generation with a U‑Net via DDPM `diffusion/train_ddpm.py` 
- [ ] Classifier-based and classifier-free guidance
- [ ] Discrete diffusion for language modeling 

### MLSys 
- [ ] Communication collectives (scatter, gather, ring/tree allreduce)
- [ ] Minimal FSDP re-implementation (data and tensor parallelism)
- [ ] Ring Attention
- [ ] Flash Attention
- [ ] Paged Attention 
- [ ] 4-bit weight-only quantization

[Coming Soon]: RAG, Agents, Multimodality, Robotics. 

---

## Notes

- The codebase will generally work with either a CPU or GPU, but most implementations basically require 
a GPU as they will be untenably slow otherwise. I recommend either a consumer laptop with GPU, paying for Colab/Runpod, 
or simply asking a compute provider or local university for a compute grant if those are out of 
budget (this works surprisingly well, people are very generous). 
- Most `.py` scripts take in `--verbose` and `--wandb` as command line arguments when you run them, to enable detailed logging and sending logs to wandb, respectively. Feel free to hack these to your needs. 
- Feel free to email me at [tanishq@stanford.edu](mailto:tanishq@stanford.edu) with feedback, implementation/feature requests, 
and to raise any bugs as GitHub issues. I am committing to implementing new techniques people want over the next month, and 
welcome contributions or bug fixes by others. 

**Happy coding, and may your gradients never vanish!**
