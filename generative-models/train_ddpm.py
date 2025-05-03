'''
(https://arxiv.org/pdf/2006.11239) Denoising Diffusion Probabilistic Models (DDPM)

A brief history of some key papers in diffusion to set the scene for this implementation 
and section more broadly. Obviously I don't expect you to read any of these at all -- the whole point here 
is that my implementation and comments should be sufficient for you to understand what's going on and 
how things work. I myself have only read papers 1-3 here. 

(0.) Papers in classical ML like [Hyvarinen, 2005; Vincent, 2011] establish you can learn a data distribution, 
(ie. compute its pdf/cdf up to a noramlizing constant) by just estimating the score, - grad[log p] of the 
data distribution, and that this might be easy to do. It's not obvious (to me) 
this is a "sufficient statistic" for a distribution, but it can be shown and is true. The Vincent paper 
shows theoretically the connection b/w score matching and denoising, important for going from DPM -> DDPM below. 

1. DPM (Solh-Dickstein, ..., Ganguli, learning score directly w/ DNNs)

    This is the beginning of diffusion in deep learning. These guys introduce the notion of a forward noising process that 
corrupts the data, and use neural networks to directly learn the score function. Such a neural net takes 
[b, ch, h, w] -> [b, ch, h, w] where voxel i is simply the scalar -grad[log p(x_i)]

2. DDPM (Ho, what we do in this implementation, 
    using a denoising objective to IMPLICITLY learn score)

    This paper shows you can do something much easier, but secretly equivalent to learning the score. 
That is, simply learn to predict either the signal/noise in a corrupted image (notice they are equiv
as you can get one once you know the other). So they train a neural network to take 
noised_image -> noise, and this turns out to be much easier and stronger to train. 

3. Improved DDPM (Nichol & Dhariwal)

    These guys make improvements to the [2] training pipeline by tweaking the objective inspired by 
variational models as well as introducing tricks like new noise schedules that empirically improve training. 
Conceptually not too different from DDPMs. 

Note: I've omitted some of Yang Song's seminal work on ODE/SDE diffusion modeling, mostly because 
I haven't read it closely enough to understand it, and "diffusion models" generally just refer to vanilla 
DDPMs (in the sense of [3]) these days. 

Let's implement DDPM with a U-Net architecture using a score-matching (or equivalently, noise prediction) objective.
This follows a similar setup as in DiT (Diffusion Transformer) but now with a convolutional backbone instead of a transformer.
Recall, our model takes noised_img [b, ch, h, w] -> noise_pred [b, ch, h, w], with MSE loss between true and predicted noise:

L = (eps_true - eps_pred)^2.mean()

Key architectural components in this implementation:
1. GroupNorm for normalization (instead of LayerNorm used in transformers)
2. Sinusoidal time embeddings that are projected and injected into the network
3. U-Net with a bottleneck structure featuring skip connections between encoder and decoder
4. Downsampling and upsampling paths to capture multi-scale features

Relation to DiT implementation: While DiT uses transformer blocks with self-attention for modeling the corrupted image,
our DDPM implementation uses convolutional layers in a U-Net structure. Both approaches condition on timesteps and
predict noise, but they differ in how they process spatial information - transformers with global attention vs.
convolutions with local receptive fields that gradually expand through the network depth.

'''
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

# hack, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context

# we implement our own architectural components for instructional purposes, obviously you can 
# use nn.Conv2d, nn.Attention, etc out of the box if you wanted to be concise 
# but I think an end-to-end understanding is the goal, and reassuring to see 

# this conv is more involved than in eg. our resnet implementation because we will be supporting arbitrary up/downsampling of inputs 
# but the core idea of converting a conv into a batched matmul is the same, just with casework to interpolate upsample before conv or 
# add a stride to downsample during the conv, conceptually not very different
# and i'm sure i could've abstracted out some of the code to make this cleaner/shorter but nbd
class Conv(nn.Module): # [b, ch, h, w] -> [b, ch', h', w'] where ch', h', w' depend on down/upsampling ratio 
    def __init__(self, upsample_ratio=1.0, kernel_sz=3, ch_in=1, ch_out=1): 
        super().__init__() 
        self.upsample_ratio = upsample_ratio
        self.kernel_sz = kernel_sz
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_weights = torch.nn.Parameter(torch.randn(ch_out, ch_in, kernel_sz, kernel_sz))
        nn.init.kaiming_normal_(self.kernel_weights, nonlinearity='relu') 
        # conv kernels are basically all our params, so we make sure they're initialized correctly 
        

    def forward(self, x): # residual conv 
        if self.upsample_ratio > 1.0:
            # interpolate up then conv to maintain 
            x = F.interpolate(x, scale_factor=self.upsample_ratio)

            # conv to maintain 
            patches = F.unfold(x, kernel_size=self.kernel_sz, padding=self.kernel_sz//2) # [b, ch_in * k * k, np]
            kernel_flat = self.kernel_weights.reshape(self.ch_out, -1) # [ch_out, ch_in * k * k]
            out = torch.einsum('bmn,om->bon', patches, kernel_flat)# [b, ch_out, np]
            b, _, h, w = x.shape
            out = out.reshape(b, self.ch_out, h, w)
            return out 
        elif self.upsample_ratio < 1.0:
            # downsample by setting stride=1//upsample_ratio
            stride = int(1/self.upsample_ratio)
            padding = self.kernel_sz//2 

            patches = F.unfold(x, kernel_size=self.kernel_sz, padding=padding, stride=stride) # [b, ch_in * k * k, np]
            kernel_flat = self.kernel_weights.reshape(self.ch_out, -1) # [ch_out, ch_in * k * k]
            out = torch.einsum('bmn,om->bon', patches, kernel_flat)# [b, ch_out, np]
            b, _, h, w = x.shape

            h_out = (h + 2*padding - self.kernel_sz)//stride + 1
            w_out = (w + 2*padding - self.kernel_sz)//stride + 1
            out = out.reshape(b, self.ch_out, h_out, w_out)
            return out 
        else: # == 1 
            # this conv maintains dimensionality
            patches = F.unfold(x, kernel_size=self.kernel_sz, padding=self.kernel_sz//2) # [b, ch_in * k * k, np]
            kernel_flat = self.kernel_weights.reshape(self.ch_out, -1) # [ch_out, ch_in * k * k]
            out = torch.einsum('bmn,om->bon', patches, kernel_flat)# [b, ch_out, np]
            b, _, h, w = x.shape
            out = out.reshape(b, self.ch_out, h, w)
            return out 

class GroupNorm(nn.Module): 
    def __init__(self, ch=1, channels_per_group=4, eps = 1e-8): 
        super().__init__()
        if channels_per_group > ch: 
            channels_per_group = ch 
        self.channels_per_group = channels_per_group
        assert ch % channels_per_group == 0, \
            "GroupNorm requires number of channels to be a multiple of channels_per_group!"
        self.num_groups = int(ch/channels_per_group)
        self.shift = nn.Parameter(torch.zeros(self.num_groups))
        self.scale = nn.Parameter(torch.ones(self.num_groups))
        self.eps = eps 

    
    def forward(self, x): # [b, ch, h, w] -> [b, ch, h, w] 
        # reshape to [b, ch//g, g * h * w], normalize within the last, and reshape back 
        # g is number of groups 
        b, ch, h, w = x.shape
        out = x.reshape(b, self.num_groups, self.channels_per_group * h * w)
        mean = out.mean(dim=-1, keepdim=True)
        var = out.var(dim=-1, keepdim=True, unbiased=False)
        out = (out - mean)/(torch.sqrt(var + self.eps)) # LOL -- the self.eps being outside sqrt causes NaNs during training 
        # broadcast from [ng] to [1, ng, 1] to can mult with [b, ng, cpg * h * w]
        shift_reshaped = self.shift.view(1, self.num_groups, 1)
        scale_reshaped = self.scale.view(1, self.num_groups, 1)
        out = out * scale_reshaped + shift_reshaped   # apply to each group in each batch separately
        return out.reshape(b, ch, h, w)

class Attention(nn.Module): 
    def __init__(self, D=1): # ch = 1 for mnist, ch=1 for cifar-10
        super().__init__()
        self.D = D
        self.wq = nn.Linear(D, D)
        self.wk = nn.Linear(D, D)
        self.wv = nn.Linear(D, D)
        self.wo = nn.Linear(D, D)
    
    def forward(self, x): 
        # fwd is [b, ch, h, w] -> [b, ch, h * w] -> then 
        # self attn across last dim with b, ch as batch dims, reshape to [b, ch, h, w]
        b, ch, h, w = x.shape
        # [b, h*w, ch] like in a transformer now [b,s,d] with s = h*w tokens and ch = features (D)
        x = x.reshape(b, h * w, ch) # [b, h * w, ch] = [b, s, d]
        q, k, v = self.wq(x), self.wk(x), self.wv(x) # [b, s, d]

        scale = math.sqrt(self.D)
        A_logits = torch.bmm(q/scale, k.transpose(-1, -2))
        A = F.softmax(A_logits, dim=-1) # [b, s, s]

        out = torch.bmm(A, v) # [b, s, s] @ [b, s, d] -> [b, s, d]
        out = self.wo(out) # [b, s, d] = [b, h * w, ch]
        return out.transpose(-1, -2).reshape(b, ch, h, w) # output [b, ch, h, w]
        
# how time embeddings work in UNet 
# t is [b] vector representing what timestep we want to draw each noise [ch, h, w] from in the batch 
# where larger timesteps = more noise according to our predefined noise schedule
# freqs is [dim//2]
# t[:,None] * freqs[None, :] is [b, dim//2]
# we cat these two along dim=-1 to get [b, dim] output from TimeEmbeddings
# UNet class stores projections [b, dim] -> [b, 2*ch] for each UBlock 
# we compute t_embed = TimeEmbedding(t) once at the beginning
# then does shift_i, scale_i = projs[i](t_embeds) 
# and passes in shift_i and scale_i to UBlocks[i]
class TimeEmbedding(nn.Module): 
    def __init__(self, dim, max_period=10000, mlp_mult=4):
        super().__init__()
        self.dim = dim
        self.max_period = max_period # controls frequency of our sinusoidal embeddings
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult), 
            nn.SiLU() ,
            nn.Linear(dim * mlp_mult, dim), 
        )
    
    def forward(self, t):  # [b] -> [b, dim]
        half_dim = self.dim // 2
        # Ensure max_period is a float tensor on the same device and dtype as t
        max_period_tensor = torch.tensor(self.max_period, device=t.device, dtype=t.dtype)
        # Create the arange on the correct device/dtype as well
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device, dtype=t.dtype) * torch.log(max_period_tensor) / half_dim
        )
        
        args = t[:, None].to(t.dtype) * freqs[None, :]  # [b, half_dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [b, dim]
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return self.mlp(embedding)

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

    def forward(self, x, scale, shift, residual=None): # conv, norm, and skip connection, with optional self-attn
        b, ch, h, w = x.shape 
        og_x = x.clone() # we can't naively add a residual since out main branch changes size
        if self.up: # so this makes sure our residual changes size accordingly to match the size of the main branch 
            og_x = F.interpolate(og_x, scale_factor=2)
        elif not self.up and not self.bottleneck: # self.down 
            og_x = self.skip(og_x)

        scale = scale.view(b, ch, 1, 1)
        shift = shift.view(b, ch, 1, 1)
        
        h1 = self.gn1(x) * scale + shift
        h1 = self.silu1(h1)
        if self.bottleneck: 
            h1 = self.attn(h1) # attn preserves [b, ch, h, w] shape
        h1 = self.conv1(h1)

        h2 = self.gn2(h1) * scale + shift
        h2 = self.silu2(h2)
        if self.bottleneck: 
            h2 = self.attn(h2)
        h2 = self.conv2(h2)
        
        if self.up: 
            # breakpoint()
            return h2 + og_x + residual
        else: 
            return h2 + og_x
        

class UNet(nn.Module): # [b, ch, h, w] noised image -> [b, ch, h, w] of error (score-matching)
    def __init__(self, nblocks=11, time_embed_dim=64, ch_data=1, ch=32, h=32, w=32, k=3): # 5 down, 1 bottleneck, 5 up 
        super().__init__()
        self.nblocks = nblocks
        self.time_embed_dim = time_embed_dim
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

    # padding [b, 1, 28, 28] -> [b, 1, 32, 32] is done as batch augmentation before being passed into model 
    def forward(self, x, t): # [b, ch, h, w] -> [b, ch, h, w]
        b = x.shape[0]
        h = self.input_proj(x)

        t_embeds = self.time_embeddings(t) # add to every UBlock 
        layer_counter = 0 

        # Insturctional NOTE. this is an *extremely* critical part of the u-net architecture, 
        # allowing skip connetionsbetween down_block[i] and up_block[-i]. 
        # Initially, I included a skip connection within UBlock but not across 
        # the two ends of the UNet, and my loss wasn't moving at all and I spent a lot of time debugging
        # as soon as I added residuals between the two sides of the UNet (ie. up/down blocks) training 
        # became smooth and well behaved!
        residuals = []

        for down_block in self.downs: 
            # we want the INPUT to down_block[i] to be residual to be added to OUTPUT of up_block[-i] for dims to match 
            residuals.append(h.clone()) 
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            scale, shift = ss_output.chunk(2, dim=1)  # split into two [b, ch] tensors across cols
            h = down_block(h, scale, shift)
            layer_counter += 1
        
        ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
        scale, shift = ss_output.chunk(2, dim=1)  
        h = self.bottleneck(h, scale, shift)
        layer_counter += 1

        for i, up_block in enumerate(self.ups): 
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            scale, shift = ss_output.chunk(2, dim=1) 
            h = up_block(h, scale, shift, residuals[-(i+1)]) 
            # using just i not (i+1) breaks when i = 0 = -i, notice the last el is -1, hence -(i+1) not -i
            layer_counter += 1

        return self.output_proj(h) # [b, ch, h, w], include long-range skip connection 

    pass # Ublock(down) x N -> Ublock(up) x N with middle layers having attn 


# betas is a length T 1-tensor defining noise schedule, where betas[t] is incremental noise to add at timestep t
# in practice we can mathematically derive TOTAL noise to add to x_0 to get x_t 
# from this, and we precompupte that to be able to avoid incremental noising
# inputs are both [b, ch, h, w], latter is unet(real_batch + true_noise)
def train(model, dataloader, betas, b=256, ch_data=1, h=32, w=32, epochs=10, lr=3e-4, print_every_steps=10, verbose=False): 
    # use this if you are having nan issues on backward pass 
    # torch.autograd.set_detect_anomaly(True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # basic model telemetry one wants to see during training 
    if verbose:
        print(f'Model is: ')
        print(model)
        print(f'--' * 10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    betas = betas.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()

    # beta[t] is the incremental noise we add at step t to the signal 
    # the goal is to keep total variance the same but reduce signal/noise throughout t in fwd process
    # so we want x_t signal to have var = 1-betas[t], so we define alphas = 1-betas
    # and then want signal[t] = sqrt(prod_{i=0}^t alphas[i]) * x_0
    # and noise[t] = sqrt(prod_{i=0}^t (1-alphas[i])) * z where z~N(0,1) which thus gives 
    # x_t = torch.sqrt(alphas_cumprod[t]) * x_0 + torch.sqrt(1 - alphas_cumprod[t]) * noise
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
    # we project ch_data -> ch as the first thing in the fwd pass of our model 
    # increasing ch allows us to increase the capacity of our model fwd pass independent of data dimensions 
    # in the same was as changing hidden_dim allows you to control transformer capacity independent of 
    # the embed_dim you choose or the data distribution requires 
    parser.add_argument('--ch', type=int, default=64, help='Number of channels in our latent space') 
    
    # we use 32 because its a power of two, even though mnist is 28 
    # we get around this by padding mnist from 28 -> 32 in each batch of our dataloader 
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
        save_path = f'ddpm_trained_e{args.epochs}_ch{args.ch}.pt'
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
