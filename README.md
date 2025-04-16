## Beyond NanoGPT: From LLM Beginner to AI Researcher!

**Beyond-NanoGPT** is the minimal and educational repo aiming to **bridge between nanoGPT and research-level deep learning.** 
This repo includes annotated and from-scratch implementations of tens of crucial modern techniques in frontier deep learning, aiming to help newcomers learn enough practical deep learning to 
start running experiments and thus contributing to modern research. 

It implements everything from inference techniques like KV caching and speculative decoding to 
architectures like vision and diffusion transformers to attention variants like linear or sparse attention. *Thousands of lines of 
self-contained and hand-written PyTorch to help you upskill your technical fundamentals.* The goal is for you to 
read and reimplement the techniques and systems in this repository most relevant to your desired research area to learn 
about the nitty-gritty details more deeply. Get cooking!

## Quickstart
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/tanishqkumar/beyond-nanogpt.git
   ```
2. **Set Up Your Virtual Environment:**
   Using venv:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   pip install -r requirements.txt
   ```
   Or using conda:
   ```bash
   conda create -n beyond-nanogpt # any Python >=3.8
   conda activate beyond-nanogpt
   pip install -r requirements.txt
   ```
3. **Start learning!**
   The code is meant for you to read carefully, hack around with, then re-implement yourself from scratch and compare to. 
   You can just run `.py` files with vanilla Python in the following way. 
   ```bash
   cd transformer++/train-vanilla-transformer/
   python train.py
   ``` 
   or for instance 
   ```bash 
   cd architectures/
   python train_dit.py
   ```
   Everything is written to be run on a single GPU. 
   The code is self-documenting with lots of comments that give intuition, and the arguments are laid out at the bottom of each file. 
   Jupyter notebooks are meant to be stepped through.
   

## Current Implementations and Roadmap
### Key Deep Learning architectures
- ✅ Vanilla causal Transformer for language modeling (starting point) `transformer++/train-vanilla-transformer/`
- ✅ Vision Transformer (ViT) `architectures/train_vit.py`
- ✅ Diffusion Transformer (DiT) `architectures/train_dit.py`
- ✅ RNN for language modeling `architectures/train_rnn.py`
- ✅ Residual Networks for Image Recognition (ResNet) `architectures/train_resnet.py`
- [Coming Soon]: MoE, Decision Transformers, Mamba

### Key Attention Variants
- ✅ Vanilla Self-Attention `attention-variants/vanilla_attention.ipynb`
- ✅ Multi-head Self-Attention `attention-variants/mhsa.ipynb`
- ✅ Grouped-Query Attention `attention-variants/gqa.ipynb`
- ✅ Linear Attention `attention-variants/linear_attention.ipynb`
- ✅ Sparse Attention `attention-variants/sparse_attention.ipynb`
- [Coming Soon]: Multi-Latent Attention, Ring Attention, Flash Attention

### Key Transformer++ Optimizations
- ✅ KV Caching `transformer++/KV_cache.ipynb`
- ✅ Speculative Decoding `transformer++/speculative_decoding.ipynb`
- ✅ Fast Dataloding `transformer++/train-vanilla-transformer/`
- ✅ Byte-Pair Encoding `transformer++/bpe.ipynb`
- [Coming Soon]: RoPE embeddings, continuous batching, sequence packing.

### Key RL Techniques
- [Coming Soon]: neural chess engine (self-play), LLM-RLHF, GRPO for humour with RLAIF. 

---

## Notes

- The codebase runs on GPU. I recommend either a consumer laptop with GPU, paying for Colab/Runpod, 
or simply asking a compute provider or local university for a compute grant if those are out of 
budget (this works surprisingly well, people are very generous). 
- Most `.py` scripts take in `--verbose` and `--wandb` as command line arguments when you run them, to enable detailed logging and sending logs to wandb, respectively. Feel free to hack these to your needs. 
- The name of the repo is inspired by the wonderful NanoGPT repo by Andrej Karpathy, 
though this repo has no official association with it.  
- Feel free to email me at [tanishq@stanford.edu](mailto:tanishq@stanford.edu) with feedback, implementation/feature requests, 
and to raise any bugs as GitHub issues. I am committing to implementing new techniques people want over the next month, and 
welcome contributions or bug fixes by others. 

**2025 is a wild time to be alive, and we need 
all hands on deck on frontier AI research.**
**Happy coding, and may your gradients never vanish!**