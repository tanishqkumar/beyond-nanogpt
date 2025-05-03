'''
(https://arxiv.org/pdf/2207.12598) Classifier-Free Diffusion Guidance 

This is a fork of the train_ddpm.py script that supports classifier-free guidance for conditional sampling. 
Before working through this, understand that as well as classifier-based guidance in ddpm_classifier_guidance.py 
since that is the technique this method seeks to obviate. 
'''

from train_ddpm import Conv, Attention, GroupNorm, TimeEmbedding
import torchvision 
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import ssl
import argparse
import wandb

# hack to get mnist fast, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context

class UBlock(nn.Module):
    # 2x [GN, +embeddings, SiLU, Conv] with global residual 
    def __init__(self, ch_in, up=True, bottleneck=False, k=3): # gamma, beta are both [b, ch] vectors from TimeEmbeddings that we use to steer things 
        super().__init__()
        # bottleneck means 1x1 + self attn 
        self.up = up 
        self.gn1 = GroupNorm(ch=ch_in)
        self.silu1 = nn.SiLU()
        self.gn2 = GroupNorm(ch=ch_in)
        self.silu2 = nn.SiLU()

        self.bottleneck = bottleneck

        if self.up: # upsample by 2
            self.conv1 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=2, kernel_sz=k)
            self.conv2 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=1, kernel_sz=k)
        elif bottleneck: # preserve shape and use attn
            self.attn = Attention(D=ch_in)
            self.conv1 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=1, kernel_sz=k)
            self.conv2 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=1, kernel_sz=k)
        else: # downsample by 2
            self.conv1 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=0.5, kernel_sz=k)
            self.conv2 = Conv(ch_in=ch_in, ch_out=ch_in, upsample_ratio=1, kernel_sz=k)
            self.skip = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=k, stride=2, padding=k//2)

    def forward(self, x, time_scale, time_shift, class_scale, class_shift, residual=None): # conv, norm, and skip connection, with optional self-attn
        b, ch, h, w = x.shape 
        og_x = x.clone() # we can't naively add a residual since out main branch changes size
        if self.up: # so this makes sure our residual changes size accordingly to match the size of the main branch 
            og_x = F.interpolate(og_x, scale_factor=2)
        elif not self.up and not self.bottleneck: # self.down 
            og_x = self.skip(og_x)

        time_scale = time_scale.view(b, ch, 1, 1)
        time_shift = time_shift.view(b, ch, 1, 1)
        class_scale = class_scale.view(b, ch, 1, 1)
        class_shift = class_shift.view(b, ch, 1, 1)
        
        # apply both time and class conditioning
        h1 = self.gn1(x) * (time_scale + class_scale) + (time_shift + class_shift)
        h1 = self.silu1(h1)
        if self.bottleneck: 
            h1 = self.attn(h1) # attn preserves [b, ch, h, w] shape
        h1 = self.conv1(h1)

        h2 = self.gn2(h1) * (time_scale + class_scale) + (time_shift + class_shift)
        h2 = self.silu2(h2)
        if self.bottleneck: 
            h2 = self.attn(h2)
        h2 = self.conv2(h2)
        
        if self.up: 
            return h2 + og_x + residual
        else: 
            return h2 + og_x

class ClassEmbedding(nn.Module): 
    def __init__(self, dim, num_classes=10, mlp_mult=4):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult), 
            nn.SiLU() ,
            nn.Linear(dim * mlp_mult, dim), 
        )
    
    def forward(self, c):  # [b] -> [b, dim]
        return self.mlp(self.embedding(c.long()))

class UNet(nn.Module): # [b, ch, h, w] noised image -> [b, ch, h, w] of error (score-matching)
    def __init__(self, nblocks=11, time_embed_dim=64, ch_data=1, ch=32, h=32, w=32, k=3): # 5 down, 1 bottleneck, 5 up 
        super().__init__()
        self.nblocks = nblocks
        self.time_embed_dim = self.class_embed_dim = time_embed_dim
        
        self.ch = ch
        self.h = h
        self.w = w
        
        # using more channels throughout net is an easy way to increase capacity of model linearly w/ ch
        self.input_proj = Conv(ch_in=ch_data, ch_out=ch, upsample_ratio=1.0, kernel_sz=k)
        
        self.time_embeddings = TimeEmbedding(self.time_embed_dim)
        
        self.ups = nn.Sequential(
            *[UBlock(ch, up=True, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        self.bottleneck = UBlock(ch, up=False, bottleneck=True, k=k)
        self.downs = nn.Sequential(
            *[UBlock(ch, up=False, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        self.output_proj = Conv(ch_in=ch, ch_out=ch_data, upsample_ratio=1.0, kernel_sz=k)
        
        self.time_to_ss = nn.ModuleList([nn.Linear(time_embed_dim, 2 * ch) for _ in range(nblocks)])
        self.class_to_ss = nn.ModuleList([nn.Linear(self.class_embed_dim, 2 * ch) for _ in range(nblocks)])


    def forward(self, x, t, c): # ([b, ch, h, w], [b], [b]) -> [b, ch, h, w]
        b = x.shape[0]
        h = self.input_proj(x)

        t_embeds = self.time_embeddings(t) # add to every UBlock 
        c_embeds = self.time_embeddings(c) # class to condition on for each batch el
        layer_counter = 0 

        residuals = []

        for down_block in self.downs: 
            residuals.append(h.clone()) 
            
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            time_scale, time_shift = ss_output.chunk(2, dim=1)  # split into two [b, ch] tensors across cols

            cc_output = self.class_to_ss[layer_counter](c_embeds)  # [b, 2*ch]
            class_scale, class_shift = cc_output.chunk(2, dim=1)  # split into two [b, ch] tensors across cols
            
            # NEW: combine time and class conditioning
            scale = time_scale + class_scale
            shift = time_shift + class_shift
            
            h = down_block(h, scale, shift)
            layer_counter += 1
        
        ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
        time_scale, time_shift = ss_output.chunk(2, dim=1)
        
        cc_output = self.class_to_ss[layer_counter](c_embeds)  # [b, 2*ch] 
        class_scale, class_shift = cc_output.chunk(2, dim=1)
        
        scale = time_scale + class_scale
        shift = time_shift + class_shift
        
        h = self.bottleneck(h, scale, shift)
        layer_counter += 1

        for i, up_block in enumerate(self.ups): 
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            time_scale, time_shift = ss_output.chunk(2, dim=1)
            
            cc_output = self.class_to_ss[layer_counter](c_embeds)  # [b, 2*ch]
            class_scale, class_shift = cc_output.chunk(2, dim=1)
            
            # combine embedding effects for upsampling
            scale = time_scale + class_scale
            shift = time_shift + class_shift
            
            h = up_block(h, scale, shift, residuals[-(i+1)]) 
            # using just i not (i+1) breaks when i = 0 = -i, notice the last el is -1, hence -(i+1) not -i
            layer_counter += 1

        return self.output_proj(h) # [b, ch, h, w], include long-range skip connection

    pass # Ublock(down) x N -> Ublock(up) x N with middle layers having attn 

def train(model, dataloader, betas, b=256, ch_data=1, h=32, w=32, epochs=10, lr=3e-4, print_every_steps=10, verbose=False): 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if verbose:
        print(f'Model is: ')
        print(model)
        print(f'--' * 10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    
    betas = betas.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()

    
    T = betas.shape[0]
    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_cp = torch.sqrt(alphas_cumprod)  # [T] 
    om_sqrt_cp = torch.sqrt(1 - alphas_cumprod)  # [T] 
    
    model.train()
    step = 0
    for epoch_idx in range(epochs): 
        if verbose:
            print(f"Starting epoch {epoch_idx+1}/{epochs}")
            dataloader_iter = tqdm(dataloader) if verbose else dataloader
        else:
            dataloader_iter = dataloader
            
        for real_batch, _ in dataloader_iter: 
            real_batch = real_batch.to(device)
            b = real_batch.shape[0] # actual batch size in case batch is incomplete/not full 
            real_batch = F.pad(real_batch, (2, 2, 2, 2), mode='constant', value=0)  # [b, 1, 28, 28] -> [b, 1, 32, 32]
            
            
            t = torch.randint(0, T, (b,), device=device) # fresh batch of [t]
            # generate a [b] tensor of class labels when/where we want to condition and None otherwise 
            c = 


            true_noise = torch.randn(b, ch_data, h, w, device=device)
            noised_batch = sqrt_cp[t].reshape(b, 1, 1, 1) * real_batch + om_sqrt_cp[t].reshape(b, 1, 1, 1) * true_noise

            pred_noise = model(noised_batch, t) # needs t for time step embeddings 
            loss = F.mse_loss(true_noise, pred_noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

            step +=1
            if verbose:
                if step % print_every_steps == 0: 
                    print(f'Step {step}, epoch {epoch_idx+1}/{epochs}: Loss {loss.item():.6f}')
            elif step % print_every_steps == 0:
                print(f'Step {step}, epoch {epoch_idx+1}: Loss {loss.item()}')


def sample(model, betas, n_samples=10, ch_data=1, h=32, w=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    T = betas.shape[0]
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start with pure noise
    x = torch.randn(n_samples, ch_data, h, w, device=device)
    
    # Reverse diffusion process
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            
            if t > 0:
                noise = torch.randn_like(x)
                beta_t = betas[t]
                alpha_cumprod_prev = alphas_cumprod[t-1]
                variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                noise = torch.randn_like(x) * torch.sqrt(variance.clamp(min=1e-20))
            else:
                noise = 0
            
            # Update x using the reverse process formula
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + noise
    
    # Denormalize and clip to [0, 1]
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    
    # Remove padding (if needed)
    if h == 32 and w == 32:
        x = x[:, :, 2:-2, 2:-2]  # Remove padding to get back to 28x28
    
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM UNet on MNIST')
    parser.add_argument('--ch_data', type=int, default=1, help='Number of channels in data') # ch_data = 1 on MNIST
    parser.add_argument('--ch', type=int, default=64, help='Number of channels in our latent space') 
    parser.add_argument('--h', type=int, default=32, help='Image height')
    parser.add_argument('--w', type=int, default=32, help='Image width')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=2e-2, help='Ending beta value')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Noise schedule type')
    parser.add_argument('--print_every', type=int, default=100, help='Print loss every N steps')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--sample', action='store_true', help='Generate samples after training')
    parser.add_argument('--save', action='store_true', help='Save model after training')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = args.timesteps
    
    # Default to cosine noise schedule
    if args.schedule == 'linear':
        betas = torch.linspace(args.beta_start, args.beta_end, T)
    else:  # cosine schedule like in improved ddpm paper (Nichol & Dharival)
        s = 0.008  # small offset to prevent singularity
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=args.beta_start, max=args.beta_end)
    
    if args.wandb:
        wandb.init(project="ddpm-unet", config=vars(args))
    
    model = UNet(
        ch_data=args.ch_data, 
        ch=args.ch,
        h=args.h,
        w=args.w
    )
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    train(
        model=model, 
        dataloader=dataloader, 
        betas=betas, 
        b=args.batch_size, 
        ch_data=args.ch_data, 
        h=args.h, 
        w=args.w, 
        epochs=args.epochs, 
        lr=args.lr, 
        print_every_steps=args.print_every,
        verbose=args.verbose
    )
    
    if args.save:
        save_path = f'ddpm_cfg_trained_e{args.epochs}_ch{args.ch}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    if args.sample:
        print("Generating samples...")
        samples = sample(model, betas, n_samples=10, ch_data=args.ch_data, h=args.h, w=args.w)
        
        # Create a grid of images
        grid = torchvision.utils.make_grid(samples, nrow=5)
        
        # Convert to PIL image and save
        grid_img = torchvision.transforms.ToPILImage()(grid)
        grid_img.save('unet_samples.png')
        print("Samples saved to unet_samples.png")


'''
Key modifications needed in train_ddpm.py to support classifier-free guidance:

1. Modify the training loop to randomly drop class labels:
   - Add a dropout_prob parameter (e.g. 0.1) to randomly replace class labels with None
   - When class label is None, model learns unconditional generation
   - When class label is provided, model learns class-conditional generation
   - This creates a single model that can do both conditional and unconditional generation

2. Modify UNet architecture to accept class embeddings:
   - Add class embedding layer to convert class labels to embeddings
   - Project class embeddings to same dimension as time embeddings
   - Concatenate class and time embeddings before projecting to shifts/scales
   - When class label is None, use zero vector for class embedding

3. Modify sampling to support guidance scale:
   - Add guidance_scale parameter (e.g. 3.0-7.0) to control conditioning strength
   - Run model forward pass twice:
     1. With class label = None to get unconditional prediction
     2. With target class label to get conditional prediction
   - Combine predictions: pred = uncond + scale * (cond - uncond)
   - Larger scale = stronger class guidance but potentially lower quality

4. Update training config:
   - Add dropout_prob parameter (e.g. 0.1) for label dropout rate
   - Add class embedding dimension parameter
   - Add guidance scale parameter for sampling

5. Data changes:
   - Modify dataloader to return (images, labels) instead of just images
   - Handle None labels in forward pass
   - Add class embedding layer initialization

The key insight is that by randomly dropping class labels during training,
the model learns both conditional and unconditional generation modes.
At inference time, we can run both modes and interpolate between them
using the guidance scale to control how strongly we want to condition
on the class label.
'''
