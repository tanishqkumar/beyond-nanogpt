'''
(https://arxiv.org/pdf/2006.11239) Denoising Diffusion Probabilistic Models (DDPM)

let's implemet ddpm with a u-net with a score-matching (or equivalently, noise prediction) objective 
same setup as in DiT but now with a convolutional backbone. Recall, we take noised_img [b, ch, h, w] -> noise_pred [b, ch, h, w]
with (eps_true - eps_pred).mean() as loss 
new additions here: groupnorm, new type of timeEmbeddings, Ublock and Unet with bottleneck structure compared to DiT 
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
        super().__init__()
        self.upsample_ratio = upsample_ratio
        self.kernel_sz = kernel_sz
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_weights = torch.nn.Parameter(torch.randn(ch_out, ch_in, kernel_sz, kernel_sz))

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
            # padding = int(1/self.upsample_ratio)
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
    def __init__(self, ch=1, channels_per_group=3, eps = 1e-8): 
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
        out = (out - mean)/(torch.sqrt(var) + self.eps)
        # broadcast from [ng] to [1, ng, 1] to can mult with [b, ng, cpg * h * w]
        shift_reshaped = self.shift.view(1, self.num_groups, 1)
        scale_reshaped = self.scale.view(1, self.num_groups, 1)
        out = (out + shift_reshaped) * scale_reshaped  # apply to each group in each batch separately
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
    
    def forward(self, t): # [b] -> [b, dim] where then projection to [b, 2*ch] handled in UNet class
        # sinusoidal embeddings, freqs shape: [dim//2]
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * torch.log(torch.tensor(self.max_period)) / half_dim
        )
        
        args = t[:,None] * freqs[None, :] # [b, dim//2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [b, dim]
        
        # handle odd dim
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
    def __init__(self, nblocks=11, time_embed_dim=64, ch=1, h=32, w=32, k=3): # 5 down, 1 bottleneck, 5 up 
        super().__init__()
        self.nblocks = nblocks
        self.time_embed_dim = time_embed_dim
        self.ch = ch
        self.h = h
        self.w = w
        
        self.time_embeddings = TimeEmbedding(self.time_embed_dim)
        
        self.ups = nn.Sequential(
            *[UBlock(ch, up=True, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        self.bottleneck = UBlock(ch, up=False, bottleneck=True, k=k)
        self.downs = nn.Sequential(
            *[UBlock(ch, up=False, bottleneck=False, k=k) for _ in range(int(nblocks//2))]
        )
        
        self.time_to_ss = nn.ModuleList([nn.Linear(time_embed_dim, 2 * ch) for _ in range(nblocks)])


    def forward(self, x, t): # [b, ch, h, w] -> [b, ch, h, w]
        b = x.shape[0]
        og_x = x.clone()
        h = x
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

        return h + og_x # [b, ch, h, w], include long-range skip connection 

    pass # Ublock(down) x N -> Ublock(up) x N with middle layers having attn 

# betas is a length T 1-tensor defining noise schedule 
def train(model, dataloader, betas, b=16, ch=1, h=32, w=32, epochs=1, lr=3e-4, print_every_steps=10, verbose=False): # inputs are both [b, ch, h, w], latter is unet(real_batch + true_noise)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
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
            loss = F.mse_loss(true_noise, pred_noise)
            loss.backward()
            opt.step()
            opt.zero_grad()

            step +=1
            if verbose:
                if step % print_every_steps == 0: 
                    print(f'Step {step}, epoch {epoch_idx+1}/{epochs}: Loss {loss.item():.6f}')
            elif step % print_every_steps == 0:
                print(f'Step {step}, epoch {epoch_idx+1}: Loss {loss.item()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM UNet on MNIST')
    parser.add_argument('--latent-dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--ch', type=int, default=1, help='Number of channels')
    parser.add_argument('--h', type=int, default=32, help='Image height')
    parser.add_argument('--w', type=int, default=32, help='Image width')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion timesteps')
    parser.add_argument('--beta-start', type=float, default=1e-4, help='Starting beta value')
    parser.add_argument('--beta-end', type=float, default=2e-2, help='Ending beta value')
    parser.add_argument('--print-every', type=int, default=10, help='Print loss every N steps')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = args.timesteps
    betas = torch.linspace(args.beta_start, args.beta_end, T)  # linear noise schedule from small to large
    
    if args.wandb:
        wandb.init(project="ddpm-unet", config=vars(args))
    
    model = UNet(
        latent_dim=args.latent_dim,
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
