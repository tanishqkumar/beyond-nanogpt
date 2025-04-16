import torch 
import torch.nn.functional as F
import wandb
from torchtyping import TensorType as TT 
import torchvision
import torchvision.transforms as transforms
import ssl
import torch
import math 
from diffusers import AutoencoderKL
from tqdm import tqdm
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from PIL import Image

# Set higher precision for matrix multiplication
torch.set_float32_matmul_precision('high')

# hack to get mnist data, don't do in production
ssl._create_default_https_context = ssl._create_unverified_context

class Attention(torch.nn.Module): 
    def __init__(self, D: int = 768, head_dim: int = 64):
        super().__init__()
        self.D = D
        self.wq = torch.nn.Linear(D, D)
        self.wk = torch.nn.Linear(D, D)
        self.wv = torch.nn.Linear(D, D)
        self.wo = torch.nn.Linear(D, D)
        self.head_dim = head_dim 
        assert self.D % self.head_dim == 0 
        self.nheads = self.D // self.head_dim 


    def forward(self, x: TT["b", "s", "d"]) -> TT["b", "s", "d"]: # [B, S, D] -> [B, S, D]
        B, S, D = x.shape 
        q, k, v = self.wq(x), self.wk(x), self.wv(x) # [B, S, D]
        q = q.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]
        k = k.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]
        v = v.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]
        
        scale = torch.sqrt(torch.tensor(self.head_dim))
        logits = torch.einsum('bnik,bnjk->bnij', q, k) / scale # [B, nh, S, S]

        A = F.softmax(logits, dim=-1)
        preout = torch.einsum('bnij,bnjd->bnid', A, v) # [B, nh, S, hd]
        preout = preout.transpose(1, 2).reshape(B, S, D) # [B, S, D]

        return self.wo(preout)

class MLP(torch.nn.Module):
    def __init__(self, D=768, multiplier=4): 
        super().__init__()
        self.multiplier = multiplier
        self.wup = torch.nn.Linear(D, multiplier * D)
        self.wdown = torch.nn.Linear(multiplier * D, D)
        self.act = torch.nn.GELU()
    
    def forward(self, x): 
        return self.wdown(self.act(self.wup(x)))

class LN(torch.nn.Module): # BSD -> BSD, just subtracts mean and rescales by std of tensor
    def __init__(self, D=512, eps=1e-9):
        super().__init__()
        self.D = D
        self.mean_scale = torch.nn.Parameter(torch.ones(D))
        self.std_scale = torch.nn.Parameter(torch.ones(D))
        self.eps = eps 

    def forward(self, x): # BSD -> BSD 
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.std_scale * ((x - mean)/(std + self.eps)) + self.mean_scale

class DiTEmbeddings(torch.nn.Module): # (b, ch, h, w) -> (b, np, d)
    def __init__(self, D=512, t=1, h=3, w=3, ch=4, num_classes=10): 
        super().__init__()
        self.t = t
        self.D = D 
        self.num_patches = (h//t) * (w//t)
        self.ch = ch 
        self.h = h 
        self.w = w 
        self.num_classes = num_classes
        self.patch_proj = torch.nn.Linear(self.t*self.t*ch, D)
        
        def get_freq_embed(t):
            # Create sinusoidal position embeddings
            # t: [b] tensor of timesteps
            half_dim = self.D // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb # [b, D]
            
        self.freq_embed = get_freq_embed

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(D, 4 * D), 
            torch.nn.GELU(), 
            torch.nn.Linear(4 * D, D)
        )
        
        self.class_embeds = torch.nn.Embedding(self.num_classes, self.D)
        self.pos_embeds = torch.nn.Parameter(torch.randn(1, self.num_patches, D)) # take [b] -> sinusoidal [b,d] -> mlp [b, d]

    def forward(self, x, t, c): # (b, ch, h, w) -> (b, np, d) and t and c are [b] vectors we will project up as conditioning info 
        B, _, _, _ = x.shape # batch of noised latents 
        x = x.permute(0, 2, 3, 1).reshape(B, self.num_patches, self.t*self.t*self.ch)  # (b, ch, h, w) -> (b, h, w, ch) -> (b, np, t*t*ch), ie. b*np patches
        x = self.patch_proj(x) # t*t*ch -> D on last dim of x, so we have [b, np, d]
        
        x = x + self.pos_embeds

        # now we want to construct (t,c) tokens for conditioning that we'll pass into first layer 
        class_embeds = self.class_embeds(c) # this is now [b, d]
        # convert t to freq embeds and add class label to get conditioning vector 
        time_embeds = self.time_mlp(self.freq_embed(t))
        conditioning = class_embeds + time_embeds 
        
        return x, conditioning

def rescale(x, m, a): 
    b, np, d = x.shape 
    return x*m.unsqueeze(1).expand(b, np, d) + a.unsqueeze(1).expand(b, np, d) # x is [b, np, d] and m and v are [b, d] from learned AdaLN

class DiTLayer(torch.nn.Module): # [b, np, d] -> [b, np, d] then we can write the final layer/head separately 
    def __init__(self, D=512):
        super().__init__()
        self.attn = Attention(D)
        self.mlp = MLP(D)
        self.ln1 = LN(D)
        self.ln2 = LN(D)
        NUM_CONSTANTS = 6
        self.constants_mlp = torch.nn.Sequential( # [b, D] -> upproj -> [b, 6] constants
            torch.nn.SiLU(),
            torch.nn.Linear(D, NUM_CONSTANTS * D, bias=True) # output is [b, 6d]? 
        ) 

    def forward(self, x, conditioning): # ([b, np, d] and [b, d]) --> [b, np, d]
        g1, b1, a1, g2, b2, a2 = self.constants_mlp(conditioning).chunk(6, dim=1) # each is [b, d]

        h = rescale(self.attn(rescale(self.ln1(x), 1.+g1, b1)), 1.+a1, torch.zeros_like(a1)) + x
        h = rescale(self.mlp(rescale(self.ln2(h), 1.+g2, b2)), 1.+a2, torch.zeros_like(a2)) + h
        return h

class DiTOutputLayer(torch.nn.Module): 
    def __init__(self, D=512, t=1, ch=4, h=3, w=3): 
        super().__init__()
        self.D = D
        self.t = t 
        self.ch = ch
        self.h = h 
        self.w = w 
        self.ln = LN(D)
        self.linear = torch.nn.Linear(D, t*t*ch)
    
    def forward(self, x):  # [b, np, d] -> [b, np, ch * t * t] -> [b, np, ch, t, t] -> [b, ch, h, w]
        x = self.linear(self.ln(x)) # outputs [b, np, ch * t * t]
        b, _, _ = x.shape
        x = x.reshape(b, -1) # outputs [b, np * ch * t * t] where np * ch * t * t = ch * h * w
        return x.reshape(b, self.ch, self.h, self.w)
    
class DiT(torch.nn.Module):
    def __init__(self, D=512, nlayers=6, t=1, ch=4, h=3, w=3): 
        super().__init__()
        self.D = D 
        self.nlayers = nlayers 
        self.embed = DiTEmbeddings(D)
        self.predict_err_vector = DiTOutputLayer(D, t=t, ch=ch, h=h, w=w)
        self.layers = torch.nn.ModuleList([DiTLayer(D) for _ in range(nlayers)])
        self._init_weights()

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)
                
        # Initialize all layers
        self.apply(init_weights)
        
        # Zero-initialize the output layer
        if isinstance(self.predict_err_vector.linear, torch.nn.Linear):
            torch.nn.init.constant_(self.predict_err_vector.linear.weight, 0.0)
            torch.nn.init.constant_(self.predict_err_vector.linear.bias, 0.0)

    def forward(self, x, t, c): # (b, ch, h, w) -> (b, num_classes) of logits for each batch element
        h, cond = self.embed(x, t, c)
        for layer in self.layers: 
            h = layer(h, cond) 
        return self.predict_err_vector(h) # unemeddings, returns [b, np, h*w*ch] in latent space 

# Load pretrained VAE model properly
print(f'Loading VAE...')
vae = AutoencoderKL.from_pretrained(
    "playgroundai/playground-v2-1024px-aesthetic",
    subfolder="vae"
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale so only 1 channel
])

print(f'Loading MNIST...')
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)

# Revised get_batch to enforce fixed, full batches.
# Reference: [PyTorch Forum discussion](https://discuss.pytorch.org/t/runtimeerror-the-size-of-tensor-a-5-must-match-the-size-of-tensor-b-32-at-non-singleton-dimension-3/127427)
def get_batch(betas, batch_size, T, dataiter):
    x0, c = next(dataiter)
    x0, c = x0.cuda(), c.cuda()

    # Convert grayscale to RGB by repeating channel
    x0 = x0.repeat(1, 3, 1, 1)  # Convert [batch_size, 1, 28, 28] to [batch_size, 3, 28, 28]

    # Move VAE to GPU and encode images to latent space
    vae.cuda()
    with torch.no_grad():
        latents = vae.encode(x0).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # sample a [batch_size] vector each in [T]
    t = torch.randint(0, T, (batch_size,)).cuda()

    alphas = 1 - betas # [T]
    prod = torch.cumprod(alphas, dim=0) # [T] dimensional
    sqrt_at = torch.sqrt(prod)
    sqrt_1mat = torch.sqrt(1 - prod)
    
    t_clamped = torch.clamp(t, 0, T-1)
    
    sqrt_at_batch = sqrt_at[t_clamped].reshape(batch_size, 1, 1, 1).cuda()
    sqrt_1mat_batch = sqrt_1mat[t_clamped].reshape(batch_size, 1, 1, 1).cuda()
    
    noise = torch.randn_like(latents).cuda()
    noised_latents = sqrt_at_batch * latents + sqrt_1mat_batch * noise

    return noised_latents, noise, t, c

def update_ema_model(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

def train(num_steps=50000, D=384, nlayers=8, batch_size=256, T=300, save=False, noise_schedule="cosine", use_wandb=False, sample_every=5000):
    print(f'Starting training with parameters:')
    print(f'  Number of steps: {num_steps}')
    print(f'  Model dimension: {D}')
    print(f'  Number of layers: {nlayers}')
    print(f'  Batch size: {batch_size}')
    print(f'  Diffusion steps (T): {T}')
    print(f'  Noise schedule: {noise_schedule}')
    print(f'  Save model: {save}')
    print(f'  Use wandb: {use_wandb}')
    print(f'  Sample every: {sample_every} steps')
    
    if use_wandb:
        run_name = f"DiT_D{D}_L{nlayers}_B{batch_size}_T{T}_{noise_schedule}_fixed_999ema"
        wandb.init(project="train_dit", name=run_name, config={
            "model_dimension": D,
            "num_layers": nlayers,
            "batch_size": batch_size,
            "diffusion_steps": T,
            "noise_schedule": noise_schedule,
            "num_steps": num_steps,
            "sample_every": sample_every
        })
    
    # Setup data loader with drop_last to always get full batches.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)
    
    # Use an infinite iterator to wrap around epochs seamlessly.
    from itertools import cycle
    train_iter = cycle(trainloader)
    
    if noise_schedule == "linear":
        betas = torch.linspace(1e-4, 1e-2, T).cuda()
    elif noise_schedule == "cosine":
        steps = torch.linspace(0, T, T + 1)
        alpha_cumprod = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999).cuda()
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")
    model = DiT(D=D, nlayers=nlayers).cuda()
    try:
        model = torch.compile(model)
    except (RuntimeError, TypeError) as e:
        print(f"Warning: Could not compile model: {e}")
        print("Continuing with uncompiled model")
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_losses = []
    progress_bar = tqdm(range(num_steps), desc='Training')

    for step in progress_bar:
        optimizer.zero_grad()
        
        noised_latents, noise, t, c = get_batch(betas, batch_size, T, train_iter)
        
        eps_preds = model(noised_latents, t, c)
        loss = F.mse_loss(eps_preds, noise)
        loss.backward()
        optimizer.step()
        
        update_ema_model(model, ema_model)
        
        with torch.no_grad():
            ema_eps_preds = ema_model(noised_latents, t, c)
            ema_loss = F.mse_loss(ema_eps_preds, noise)
            train_losses.append(ema_loss.item())
        
        progress_bar.set_postfix({'ema_loss': ema_loss.item()})
        
        if use_wandb and step % 100 == 0:
            wandb.log({
                "train_loss": loss.item(),
                "ema_loss": ema_loss.item(),
                "step": step
            })
            
        if sample_every > 0 and (step + 1) % sample_every == 0:
            print(f"\nSampling images at step {step+1}...")
            sample_classes = [0, 2, 4, 7, 9]
            for class_label in sample_classes:
                sample_images(model, ema_model, betas, T, [class_label], num_samples=10, save=True, step=step+1)
        
    if save:
        torch.save({
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }, 'dit_model_checkpoint.pt')
    
    if use_wandb:
        wandb.finish()
    return model, ema_model, train_losses, betas

# Sample images from the trained DiT model for given class labels
def sample_images(model, ema_model, betas, T, class_labels=None, num_samples=5, save=False, step=None):
    if class_labels is None:
        class_labels = [0, 2, 4, 7, 9]
    
    ema_model.eval()
    with torch.no_grad():
        for class_label in class_labels:
            latents = torch.randn(num_samples, 4, 3, 3).cuda()
            c = torch.tensor([class_label] * num_samples).cuda()
            
            for t in reversed(range(T)):
                timesteps = torch.tensor([t] * num_samples).cuda()
                alphas = 1 - betas
                alpha_t = alphas[t]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_1m_alpha_t = torch.sqrt(1 - alpha_t)
                
                if t > 0:
                    noise = torch.randn_like(latents)
                else:
                    noise = torch.zeros_like(latents)
                
                eps_theta = ema_model(latents, timesteps, c)
                latents = (1 / sqrt_alpha_t) * (latents - (sqrt_1m_alpha_t * eps_theta)) + noise * torch.sqrt(betas[t])
            
            vae.cuda()
            images = vae.decode(latents / vae.config.scaling_factor).sample
            
            plt.figure(figsize=(15, 5))
            for i in range(num_samples):
                plt.subplot(1, num_samples, i+1)
                plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.title(f'Class {class_label}')
            
            if save:
                step_str = f"_step_{step}" if step is not None else ""
                save_dir = f"samples_dit_mnist_T{args.T}_layers{args.nlayers}_dim{args.D}_steps{args.steps}{step_str}"
                os.makedirs(save_dir, exist_ok=True)
                
                for i in range(num_samples):
                    img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_pil.save(f"{save_dir}/class_{class_label}_sample_{i}{step_str}.jpg")
                    img_pil.save(f"{save_dir}/class_{class_label}_sample_{i}{step_str}.png")
            
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DiT model on MNIST')
    parser.add_argument('--nlayers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--D', type=int, default=384, help='Model dimension')
    parser.add_argument('--steps', type=int, default=50000, help='Number of training steps')
    parser.add_argument('--B', type=int, default=256, help='Batch size')
    parser.add_argument('--T', type=int, default=300, help='Number of diffusion steps')
    parser.add_argument('--save', action='store_true', help='Save model checkpoint')
    parser.add_argument('--nsample', type=int, default=5, help='Number of samples to generate per class')
    parser.add_argument('--noise_schedule', type=str, default="cosine", choices=["linear", "cosine"], 
                        help='Noise schedule to use (linear or cosine)')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--sample_every', type=int, default=5000, 
                        help='Sample images every N steps during training (0 to disable)')
    
    args = parser.parse_args()
    
    model, ema_model, train_losses, betas = train(
        num_steps=args.steps,
        D=args.D,
        nlayers=args.nlayers,
        batch_size=args.B,
        T=args.T,
        save=args.save,
        noise_schedule=args.noise_schedule,
        use_wandb=args.wandb,
        sample_every=args.sample_every
    )
    
    for class_label in range(10):
        sample_images(model, ema_model, betas, args.T, [class_label], num_samples=args.nsample, save=True)