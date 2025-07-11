## Beyond NanoGPT: Go From LLM Beginner to AI Researcher!

<p align="center">
  <span style="display: inline-block; text-align: center; margin: 0 10px;">
    <img src="https://github.com/user-attachments/assets/b2943618-d5ed-468d-b792-d1cf4e0d6c6a" style="width: 350px; height: auto;" />
  </span>
</p>

**Beyond-NanoGPT** is the minimal and educational repo aiming to **bridge between nanoGPT and research-level deep learning.** 
This repo includes annotated and from-scratch implementations of almost 100 crucial modern techniques in frontier deep learning, aiming to help newcomers learn enough to start running experiments of their own. 

The repo implements everything from KV caching and speculative decoding for LLMs to 
architectures like vision transformers and MLP-mixers; from attention variants like linear or multi-latent attention to generative models like denoising diffusion models and flow matching algorithms; from landmark RL papers like PPO, A3C, and AlphaZero to 
systems fundamentals like GPU communication algorithms and data/tensor parallelism. 

**Because everything is implemented by-hand, the code comments explain the especially subtle details often glossed over both in papers and production codebases.**

<p align="center">
  <span style="display: inline-block; text-align: center; margin: 0 10px;">
    <a href="https://github.com/user-attachments/assets/e49fad0a-f51b-4771-a59a-f5d6a969f8ed">
      <img src="https://github.com/user-attachments/assets/e49fad0a-f51b-4771-a59a-f5d6a969f8ed" />
    </a>
    <div style="text-align: center; max-width: 600px; margin-top: 8px;">
      <sub>
        A glimpse of some plots you can make! <br />
        <b>(Left)</b> Language model speedups from 
        <code>attention-variants/linear_attention.ipynb</code>,<br />
        <b>(Center)</b> Samples from a small denoising diffusion model trained on MNIST in 
        <code>generative-models/train_ddpm.py</code>,<br />
        <b>(Right)</b> Reward over time for a small MLP policy on CartPole in 
        <code>rl/fundamentals/train_ppo.py</code>.
      </sub>
    </div>
  </span>
</p>

`LESSONS.md` documents some of the things I've learned in the months spent writing this codebase. 

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
   cd rl/fundamentals/
   python train_reinforce.py --verbose --wandb 
   ```
   Everything is written to be run on a single GPU. The code is self-documenting with comments for intuition and elaborating 
   on subtleties I found tricky to implement. 
   Arguments are specified at the bottom of each file. 
   Jupyter notebooks are meant to be stepped through.



## Current Implementations and Roadmap

Asterisks (*) denote particularly tricky implementations. 

### Architectures
- [x] Basic Transformer `language-models/transformer.py` and `train_naive.py` [[paper]](https://arxiv.org/abs/1706.03762)
- [x] Vision Transformer (ViT) `architectures/train_vit.py` [[paper]](https://arxiv.org/abs/2010.11929)
- [x] Diffusion Transformer (DiT) `architectures/train_dit.py` [[paper]](https://arxiv.org/abs/2212.09748)
- [x] Recurrent Neural Network (RNN) `architectures/train_rnn.py` [[paper]](https://arxiv.org/abs/1506.00019)
- [x] Residual Networks (ResNet) `architectures/train_resnet.py` [[paper]](https://arxiv.org/abs/1512.03385)
- [x] MLP-Mixer `architectures/train_mlp_mixer.py` [[paper]](https://arxiv.org/abs/2105.01601)
- [x] LSTM `architectures/train_lstm.py` [[paper]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [x] Mixture-of-Experts* (MoE) `architectures/train_moe.py` [[paper]](https://arxiv.org/abs/2101.03961)
- [x] Mamba* `architectures/train_mamba.py` [[paper]](https://arxiv.org/abs/2312.00752)

### Attention Variants
- [x] Vanilla Self-Attention `attention-variants/vanilla_attention.ipynb` [[paper]](https://arxiv.org/abs/1706.03762)
- [x] Multi-head Self-Attention `attention-variants/mhsa.ipynb` [[paper]](https://arxiv.org/abs/1706.03762)
- [x] Grouped-Query Attention `attention-variants/gqa.ipynb` [[paper]](https://arxiv.org/abs/2305.13245)
- [x] Linear Attention* `attention-variants/linear_attention.ipynb` [[paper]](https://arxiv.org/abs/2006.16236)
- [x] Sparse Attention `attention-variants/sparse_attention.ipynb` [[paper]](https://arxiv.org/abs/1904.10509)
- [x] Cross Attention `attention-variants/cross_attention.ipynb` [[paper]](https://arxiv.org/abs/1706.03762)
- [x] Multi-Latent Attention* `attention-variants/mla.ipynb` [[paper]](https://arxiv.org/abs/2405.04434)

### Language Models

- [x] Optimized Dataloading `language-models/dataloaders` [[reference]](https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39)
   - [x] Producer-consumer asynchronous dataloading 
   - [x] Sequence packing 
- [x] Byte-Pair Encoding `language-models/bpe.ipynb` [[paper]](https://arxiv.org/abs/1508.07909)
- [x] KV Caching `language-models/KV_cache.ipynb` [[reference]](https://huggingface.co/blog/not-lain/kv-caching)
- [x] Speculative Decoding `language-models/speculative_decoding.ipynb` [[paper]](https://arxiv.org/abs/2211.17192)
- [x] RoPE embeddings* `language-models/rope.ipynb` [[paper]](https://arxiv.org/abs/2104.09864)
- [x] Multi-token Prediction `language-models/train_mtp.py` [[paper]](https://arxiv.org/abs/2404.19737)

### Reinforcement Learning
- Deep RL
   - Fundamentals `rl/fundamentals`
      - [x] DQN `train_dqn.py` [[paper]](https://arxiv.org/abs/1312.5602)
      - [x] REINFORCE `train_reinforce.py` [[paper]](https://link.springer.com/article/10.1007/BF00992696)
      - [x] PPO `train_ppo.py` [[paper]](https://arxiv.org/abs/1707.06347)
   - Actor-Critic and Key Variants `rl/actor-critic`
      - [x] Advantage Actor-Critic (A2C) `train_a2c.py` [[paper]](https://arxiv.org/abs/1602.01783)
      - [x] Asynchronous Advantage Actor-Critic (A3C) `train_a3c.py` [[paper]](https://arxiv.org/abs/1602.01783)
      - [x] IMPALA* (distributed RL) `train_impala.py` [[paper]](https://arxiv.org/abs/1802.01561)
      - [x] Deep Deterministic Policy Gradient (DDPG) `train_ddpg.py` [[paper]](https://arxiv.org/abs/1509.02971)
      - [x] Soft Actor-Critic* (SAC) `train_sac.py` [[paper]](https://arxiv.org/abs/1801.01290)
   - Model-based RL  `rl/model-based`
      - [x] Model Predictive Control (MPC) `train_mpc.py`[[reference]](https://en.wikipedia.org/wiki/Model_predictive_control)
      - [x] Expert Iteration (MCTS) `train_expert_iteration.py` [[paper]](https://arxiv.org/abs/1705.08439)
      - [ ] Probabilistic Ensembles with Trajectory Sampling (PETS)
   - [ ] Neural Chess Engine (AlphaZero) `rl/chess` [[paper]](https://arxiv.org/abs/1712.01815)
      - [x] Define the architecture and environment `model.py` and `env.py`
      - [x] MCTS for move search `mcts.py`
      - [x] Self-play `train.py`
      - [ ] Dynamic batching and multiprocessing `mcts.py`
- LLMs `rl/llms`
   - [ ] RLHF a base model with UltraFeedback 
   - [ ] DPO a base model with UltraFeedback
   - [x] GRPO for reasoning: outcome reward on math `train_grpo_gsm.py` [[paper]](https://arxiv.org/pdf/2402.03300)
   - [ ] GRPO to use a new API correctly 
   - [ ] GRPO to write good haikus with an LLM autograder 

### Generative Models

- [x] Generative Adversarial Networks (GAN) `generative-models/train_gan.py` [[paper]](https://arxiv.org/abs/1406.2661)
- [x] Pix2Pix (Conditional GANs) `generative-models/train_pix2pix.py` [[paper]](https://arxiv.org/abs/1611.07004)
- [x] Variational Autoencoders (VAE) `generative-models/train_vae.py` [[paper]](https://arxiv.org/abs/1312.6114)
   - [x] Train an autoencoder for reconstruction `generative-models/train_autoencoder.py` 
- [ ] Neural Radiance Fields (NeRF)
- [x] Denoising Diffusion Probablistic Models* (DDPM) `generative-models/train_ddpm.py` [[paper]](https://arxiv.org/abs/2006.11239)
- [x] Classifier-based diffusion guidance `generative-models/ddpm_classifier_guidance.py` [[paper]](https://arxiv.org/abs/2105.05233)
   - [x] Classifier-free diffusion guidance `generative-models/ddpm_classifier_free_guidance.py` [[paper]](https://arxiv.org/abs/2207.12598)
- [x] Flow matching `generative-models/train_flow_matching.py` [[paper]](https://arxiv.org/abs/2210.02747)

### MLSys 
- [x] GPU Communication Algorithms* (scatter, gather, ring/tree allreduce) `mlsys/comms.py` [[reference]](https://developer.nvidia.com/nccl)
- [x] Distributed Data Parallel `mlsys/train_ddp.py` [[paper]](https://arxiv.org/pdf/2006.15704)
- [x] Tensor Parallel* `mlsys/train_tp.py` [[paper]](https://arxiv.org/pdf/1909.08053)
- [ ] Ring Attention (Context Parallel)
- [ ] Paged Attention
- [ ] Continuous Batching 
- [ ] Triton Kernels `mlsys/kernels`
   - [x] Vector Addition `vector_add.py`
   - [x] Fused Softmax Forward + Backward `softmax.py`
   - [ ] Matrix Multiplication (GEMM) Forward + Backward 
   - [ ] Layer Normalization Forward + Backward 
   - [ ] FlashAttention Forward 
   - [ ] FlashAttention Backward 

### Evals

- [ ] BERT on SST-2 (old-school NLP)
- [x] GSM8k (generative) `evals/eval_gsm8k.py` [[paper]](https://arxiv.org/pdf/2110.14168)
- [x] MMLU (multiple-choice) `evals/eval_mmlu.py` [[paper]](https://arxiv.org/abs/2009.03300)
- [x] SimpleQA (LLM judge) `evals/eval_simpleqa.py` [[paper]](https://arxiv.org/pdf/2411.04368)
- [ ] Design our own eval ("good taste")

### RAG 
- [ ] Train Small Embedding and Reranking Models
- [x] RAG 101: Retrieval on Q&A Answers `rag/intro_rag.py` 
- [ ] Multi-Hop Decomposition RAG
- [ ] Sparse and Dense Retrieval 
- [ ] Graph RAG

### Agents 
- [x] Let an LLM use internet search for Q&A `agents/basic-search-use` 
- [x] Coding Agent `agents/coding-agent`
   - [x] Tool use (search, run code, read/write files) & sandboxing for powerful tools [`/tools`]
   - [x] ReAct (iterated CoT with tool use in between) `agent.py` 
   - [x] Memory/context management distinguishing short vs long term memory `memory.py` 
   - [x] Evaluate: can it make a correct PR end-to-end in reponse to a GitHub issue? [[demo]](https://x.com/tanishqkumar07/status/1931709892236116293)
- [ ] Simulate a society with language models
- [ ] Tree-of-Thoughts deep research agents 
- [ ] Parallel multi-agent deep research  

---

## Notes

- The codebase will generally work with either a CPU or GPU, but most implementations basically require 
a GPU as they will be untenably slow otherwise. I recommend either a consumer laptop with GPU, paying for Colab/Runpod, 
or simply asking a compute provider or local university for a compute grant if those are out of 
budget (this works surprisingly well, people are very generous). Obvious exceptions like data/tensor parallel require 
multi-GPU nodes. 
- All `.py` scripts take in `--verbose` and `--wandb` as command line arguments when you run them. Feel free to hack these to your needs. 
- Feel free to email me at [tanishq@stanford.edu](mailto:tanishq@stanford.edu) with feedback, implementation/feature requests, 
and to raise any bugs as GitHub issues.
If this codebase helped you, please share it and give it a star! You can cite the repository 
in your work as follows. 

```bibtex
@misc{kumar2025beyond,
  author = {Tanishq Kumar},
  title = {Beyond-NanoGPT: From LLM Beginner to AI Researcher},
  year = {2025},
  howpublished = {\url{https://github.com/tanishqkumar/beyond-nanogpt}},
  note = {Accessed: 2025-01-XX}
}
```

**Happy coding, and may your gradients never vanish!**

