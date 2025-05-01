'''
TODO: fill this out 

(https://arxiv.org/pdf/2006.11239) Denoising Diffusion Probabilistic Models (DDPM)

(0.) Papers introducing the notion of score matching in ML

1. DPM (Ganguli, learning score directly w/ DNNs)

2. DDPM (Ho, this implementation, using a denoising objective to IMPLICITLY learn score)

3. Improved DDPM 

let's implemet ddpm with a u-net with a score-matching (or equivalently, noise prediction) objective 
same setup as in DiT but now with a convolutional backbone. Recall, we take noised_img [b, ch, h, w] -> noise_pred [b, ch, h, w]
with (eps_true - eps_pred).mean() as loss 
new additions here: groupnorm, new type of timeEmbeddings, Ublock and Unet with bottleneck structure compared to DiT 

Relation to DiT implementation: ___ 

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

# this conv is more involved than in eg. our resnet implementation because we will be supporting arbitrary up/downsampling of inputs 
# but the core idea of converting a conv into a batched matmul is the same, just with casework to interpolate upsample before conv or 
# add a stride to downsample during the conv, conceptually not very different
# and i'm sure i could've abstracted out some of the code to make this cleaner/shorter but nbd
class Conv(nn.Module): # [b, ch, h, w] -> [b, ch', h', w'] where ch', h', w' depend on down/upsampling ratio 
    def __init__(self, upsample_ratio=1.0, kernel_sz=3, ch_in=1, ch_out=1): 
        super().__init__() # TODO: we should be able to clean this code up with a helper fn to avoid repeat in fwd()
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
            padding = self.kernel_sz//2 # TODO: understand stride/padding values well 

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

# TODO: want to normalize per channel, not group
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

# TODO: our down blocks should send residual connections to corresponding up blocks, key part of unet 
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

    def forward(self, x, scale, shift): # conv, norm, and skip connection, with optional self-attn
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
        return h2 + og_x
        

class UNet(nn.Module): # [b, ch, h, w] noised image -> [b, ch, h, w] of error (score-matching)
    def __init__(self, nblocks=11, time_embed_dim=64, input_channels=1, ch=32, h=32, w=32, k=3): # 5 down, 1 bottleneck, 5 up 
        super().__init__()
        self.nblocks = nblocks
        self.time_embed_dim = time_embed_dim
        self.ch = ch
        self.h = h
        self.w = w
        
        self.input_proj = Conv(ch_in=input_channels, ch_out=ch, upsample_ratio=1.0, kernel_sz=k)
        
        self.time_embeddings = TimeEmbedding(self.time_embed_dim)
        
        self.ups = nn.Sequential(
            *[UBlock(ch, up=True, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        self.bottleneck = UBlock(ch, up=False, bottleneck=True, k=k)
        self.downs = nn.Sequential(
            *[UBlock(ch, up=False, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        self.output_proj = Conv(ch_in=ch, ch_out=input_channels, upsample_ratio=1.0, kernel_sz=k)
        
        self.time_to_ss = nn.ModuleList([nn.Linear(time_embed_dim, 2 * ch) for _ in range(nblocks)])


    def forward(self, x, t): # [b, ch, h, w] -> [b, ch, h, w]
        b = x.shape[0]
        h = self.input_proj(x)
        # h = F.pad(x, (2, 2, 2, 2), mode='constant', value=0) # [b, 1, 32, 32] -> [b, 1, 32, 32]
        t_embeds = self.time_embeddings(t) # add to every UBlock 
        layer_counter = 0 

        for down_block in self.downs: 
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            scale, shift = ss_output.chunk(2, dim=1)  # split into two [b, ch] tensors
            h = down_block(h, scale, shift)
            layer_counter += 1
        
        ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
        scale, shift = ss_output.chunk(2, dim=1)  # split into two [b, ch] tensors
        h = self.bottleneck(h, scale, shift)
        layer_counter += 1

        for up_block in self.ups: 
            ss_output = self.time_to_ss[layer_counter](t_embeds)  # [b, 2*ch]
            scale, shift = ss_output.chunk(2, dim=1)  # split into two [b, ch] tensors
            h = up_block(h, scale, shift)
            layer_counter += 1

        return self.output_proj(h) # [b, ch, h, w], include long-range skip connection 

    pass # Ublock(down) x N -> Ublock(up) x N with middle layers having attn 

# betas is a length T 1-tensor defining noise schedule 
def train(model, dataloader, betas, b=256, ch=1, h=32, w=32, epochs=10, lr=3e-3, print_every_steps=10, verbose=False): # inputs are both [b, ch, h, w], latter is unet(real_batch + true_noise)
    torch.autograd.set_detect_anomaly(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Count and print the number of parameters in the model if verbose
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    betas = betas.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()

    # betas is noise schedule, so increasing noise 
    # alphas is 1- betas, so decreasing
    # cumprod makes it decrease even faster
    # x_t = torch.sqrt(alphas_cumprod[t]) * x_0 + torch.sqrt(1 - alphas_cumprod[t]) * noise
    T = betas.shape[0]
    alphas = 1 - betas # variance of noise added x_t -> x_{t+1}
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
            b = real_batch.shape[0] # actual bs 
            real_batch = F.pad(real_batch, (2, 2, 2, 2), mode='constant', value=0)  # [b, 1, 28, 28] -> [b, 1, 32, 32]
            
            
            t = torch.randint(0, T, (b,), device=device) # fresh batch of [t]
            true_noise = torch.randn(b, ch, h, w, device=device)
            noised_batch = sqrt_cp[t].reshape(b, 1, 1, 1) * real_batch + om_sqrt_cp[t].reshape(b, 1, 1, 1) * true_noise

            pred_noise = model(noised_batch, t) # needs t for time step embeddings 
            # breakpoint()
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


# TODO: read this carefully and rewrite from scratch, for now just use as a test for correctness
def sample(model, betas, n_samples=10, ch=1, h=32, w=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    T = betas.shape[0]
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start with pure noise
    x = torch.randn(n_samples, ch, h, w, device=device)
    
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
                variance = beta_t * (1. - alpha_cumprod_t / (alpha_cumprod_t * alpha_t))
                variance = torch.clamp(variance, min=1e-20)
                noise = noise * torch.sqrt(variance)
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
    parser.add_argument('--ch', type=int, default=1, help='Number of channels')
    parser.add_argument('--h', type=int, default=32, help='Image height')
    parser.add_argument('--w', type=int, default=32, help='Image width')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--beta-start', type=float, default=1e-4, help='Starting beta value')
    parser.add_argument('--beta-end', type=float, default=2e-2, help='Ending beta value')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Noise schedule type')
    parser.add_argument('--print-every', type=int, default=10, help='Print loss every N steps')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--sample', action='store_true', help='Generate samples after training')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = args.timesteps
    
    # Default to cosine noise schedule
    if args.schedule == 'linear':
        betas = torch.linspace(args.beta_start, args.beta_end, T)
    else:  # cosine schedule
        # Implementation of cosine schedule from Improved DDPM paper
        s = 0.008  # small offset to prevent singularity
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=args.beta_start, max=args.beta_end)
    
    if args.wandb:
        wandb.init(project="ddpm-unet", config=vars(args))
    
    model = UNet(
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
        ch=args.ch, 
        h=args.h, 
        w=args.w, 
        epochs=args.epochs, 
        lr=args.lr, 
        print_every_steps=args.print_every,
        verbose=args.verbose
    )
    
    if args.sample:
        print("Generating samples...")
        samples = sample(model, betas, n_samples=10, ch=args.ch, h=args.h, w=args.w)
        
        # Create a grid of images
        grid = torchvision.utils.make_grid(samples, nrow=5)
        
        # Convert to PIL image and save
        grid_img = torchvision.transforms.ToPILImage()(grid)
        grid_img.save('unet_samples.png')
        print("Samples saved to unet_samples.png")
