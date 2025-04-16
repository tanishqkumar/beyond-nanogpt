# Beyond NanoGPT: Go from LLM noob to AI Researcher!

Welcome to **Beyond-NanoGPT** – the minimal and educational repo aiming to bridge between nanoGPT and research-level deep learning. 
This repo includes annotated and from-scratch implementations of tens of crucial modern techniques in frontier deep learning, aiming to technical newcomers learn enough to 
start running experiments and thus contributing to modern deep learning research. 

It implements everything from inference techniques like KV caching and speculative decoding to 
architectures like vision and diffusion transformers to attention variants like linear or sparse attention. Thousands of lines of 
self-contained and hand-written PyTorch. 

## Quickstart
1. **Clone the Repo:**
   ```bash
   git clone https://yourrepo.example.com/yourname/llm.git
   cd llm
   ```
2. **Set Up Your Virtual Cauldron (Environment):**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. **Install the Required Potion Ingredients (Dependencies):**
   ```bash
   pip install -r requirements.txt
   ```
4. **Summon the Code:**
   Get ready for a magical session of innovation. Run the provided scripts or experiments to see how deep learning is truly conjured from scratch!
   ```bash
   python your_script.py   # Replace with the specific runner if applicable.
   ```

## Current Implementations and Roadmap
### Key Deep Learning Architectures
- ✅ Vanilla causal Transformer for language modeling (starting point) {transformer++/train-vanilla-transformer/}
- ✅ Vision Transformer (ViT) {arch/train_vit.py}
- ✅ Diffusion Transformer (DiT) {arch/train_dit.py}
- ✅ RNN for language modeling {arch/train_rnn.py}
- ✅ Residual Networks for Image Recognition (ResNet) {arch/train_resnet.py}
- [Coming Soon]: MoE, Decision Transformers, Mamba

### Key Attention Variants
- ✅ Vanilla Self-Attention {attn/vanilla_attention.ipynb}
- ✅ Multi-head Self-Attention {attn/mhsa.ipynb}
- ✅ Grouped-Query Attention {attn/gqa.ipynb}
- ✅ Linear Attention {attn/linear_attention.ipynb}
- ✅ Sparse Attention {attn/sparse_attention.ipynb}
- [Coming Soon]: Multi-Latent Attention, Ring Attention, Flash Attention

### Key Transformer++ Optimizations
- ✅ KV Caching {transformer++/KV_cache.ipynb}
- ✅ Speculative Decoding {transformer++/speculative_decoding.ipynb}
- ✅ Byte-Pair Encoding {transformer++/bpe.ipynb}
- ✅ Fast Dataloding Optimizations {transformer++/train-vanilla-transformer/}
- [Coming Soon]: RoPE embeddings, continuous batching, sequence packing.

### Key RL Techniques
- [Coming Soon]: neural chess engine (self-play), LLM-RLHF, GRPO for humour with RLAIF. 

---

## Notes

- The codebase runs on a GPU. CPUs are fine for tasting the basics, but if you want to work with advanced techniques, you will need a GPU of some kind. 
- Most scripts take in `--verbose` and `--wandb` as command line arguments when you run them, to enable detailed logging and sending logs to wandb, respectively. Feel free to hack these to your needs. 
- The name of the repo is inspired by the wonderful NanoGPT repo by Andrej Karpathy, 
though this repo is not officially associated. 
- Feel free to email me at [tanishq@stanford.edu](mailto:tanishq@stanford.edu) with feedback, things you want implemented, 
and to raise any bugs as GitHub issues. I am committing to implementing new techniques people want over the next month!

**Happy coding, and may your gradients never vanish!**