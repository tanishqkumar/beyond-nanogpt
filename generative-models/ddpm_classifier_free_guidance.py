'''
(https://arxiv.org/pdf/2207.12598) Classifier-Free Diffusion Guidance 

This is a fork of the train_ddpm.py script that supports classifier-free guidance for conditional sampling. 
Before working through this, understand that as well as classifier-based guidance in ddpm_classifier_guidance.py 
since that is the technique this method seeks to obviate. 

The key changes this makes to train_ddpm.py are (along with motivation/intuition for each ):
    - Introduce a notion of class-conditioning that parallels time-conditioning during training 
        - ie. for a given batch during training, we pass in the labels as part of the conditioning
            - Note this isn't "giving the answer away" because the DDPM is not trained to predict the class, but to reconstruct
            the image exactly. 
        - we make the model learn the unconditional image distribution as well as conditional distributions by only passing in class 
        embeddings some fraction (eg. 80-90%) of the time, called "dropout on class labels." 
            - this forces learning both because guessing a conditional distribution when you don't see a class label will do 
            very poorly in reconstruction on average. 
        - At first glance, since we *do not* modify the noise reconstruction loss function, one might ask why introducing class 
        embeddings does anything at all (ie. forces learning of conditional class sub-distributions), since in principle the model 
        could just learn to ignore them. 
            - But actually the point is when you want to reconstruct an image, being told its an image of a dog (ie. given the class label), 
            is a lot of information, since outputting a "generic dog" will do much better than not being told its a dog at all. So the model 
            will learn to use the class label to output conditional distributions, hence allowing us to do conditional sampling, ie. guidance. 

One might also ask why you even care about being classifier-free in the first place, since classifiers are much smaller than large 
diffusion models, doing a fwd+bwd through those at sampling time isn't much overhead. I think the paper above actually gives 
a very good explanation of this: 
        > We are interested in whether classifier guidance can be performed without a classifier. Classifier
    guidance complicates the diffusion model training pipeline because it requires training an extra
    classifier, and this classifier must be trained on noisy data so it is generally not possible to plug
    in a pre-trained classifier. Furthermore, because classifier guidance mixes a score estimate with
    a classifier gradient during sampling, classifier-guided diffusion sampling can be interpreted as
    attempting to confuse an image classifier with a gradient-based adversarial attack. This raises the
    question of whether classifier guidance is successful at boosting classifier-based metrics such as FID
    and Inception score (IS) simply because it is adversarial against such classifiers. Stepping in direction
    of classifier gradients also bears some resemblance to GAN training, particularly with nonparameteric
    generators; this also raises the question of whether classifier-guided diffusion models perform well
    on classifier-based metrics because they are beginning to resemble GANs, which are already known
    to perform well on such metrics.

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

    def forward(self, x, scale, shift, residual=None): # conv, norm, and skip connection, with optional self-attn
        b, ch, h, w = x.shape 
        og_x = x.clone() # we can't naively add a residual since out main branch changes size
        if self.up: # so this makes sure our residual changes size accordingly to match the size of the main branch 
            og_x = F.interpolate(og_x, scale_factor=2)
        elif not self.up and not self.bottleneck: # self.down 
            og_x = self.skip(og_x)

        scale = scale.view(b, ch, 1, 1)
        shift = shift.view(b, ch, 1, 1)
        
        # apply conditioning
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
        b = c.shape[0]
        valid_indices = c > 0 # -1 means we want no class conditioning, so zero out those embeddings
        embeddings = torch.zeros(b, self.dim, device=c.device, dtype=torch.float32)
        
        if valid_indices.any(): # embed those as usual 
            valid_entries = c[valid_indices]
            valid_embeddings = self.embedding(valid_entries.long())
            embeddings[valid_indices] = valid_embeddings
        
        return self.mlp(embeddings)

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

def train(model, dataloader, betas, b=256, ch_data=1, h=32, w=32, epochs=10, 
          lr=3e-4, print_every_steps=10, verbose=False, cfg_dropout_prob=0.1): 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    betas = betas.to(device)

    if verbose:
        print(f'Model is: ')
        print(model)
        print(f'--' * 10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    
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
            
        for real_batch, c in dataloader_iter:  # real_batch: [b, ch, h, w], real_labels: [b]
            real_batch = real_batch.to(device)
            c = c.to(device)
            b = real_batch.shape[0] # actual batch size in case batch is incomplete/not full 
            real_batch = F.pad(real_batch, (2, 2, 2, 2), mode='constant', value=0)  # [b, 1, 28, 28] -> [b, 1, 32, 32]
            
            t = torch.randint(0, T, (b,), device=device) # fresh batch of [t]
            # generate a [b] tensor of class labels when/where we want to condition and -1 otherwise 
                # we'll handle making sure -1 entries has zero embeddings and thus zero effect 
                # from a conditioning perspective in the embedding class 
            mask = torch.rand(b, device=device) < cfg_dropout_prob
            c = torch.where(mask, -1, c)  

            true_noise = torch.randn(b, ch_data, h, w, device=device)
            noised_batch = sqrt_cp[t].reshape(b, 1, 1, 1) * real_batch + om_sqrt_cp[t].reshape(b, 1, 1, 1) * true_noise

            pred_noise = model(noised_batch, t, c) # now condition on both (t, c) instead 
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


def sample_guided(model, betas, target_class=6, n_samples=10, ch_data=1, h=32, w=32, guidance_scale=3.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    betas = betas.to(device)
    T = betas.shape[0]
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # start with noise 
    x = torch.randn(n_samples, ch_data, h, w, device=device)
    
    # reverse process 
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            c_batch = torch.full((n_samples,), target_class, device=device, dtype=torch.long)
            predicted_noise_uncond = model(x, t_batch, torch.ones_like(t_batch) * -1)
            predicted_noise_cond = model(x, t_batch, c_batch)
            cond_diff = predicted_noise_cond - predicted_noise_uncond

            predicted_noise = predicted_noise_uncond + guidance_scale * cond_diff
            
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            
            if t > 0:
                beta_t = betas[t]
                alpha_cumprod_prev = alphas_cumprod[t-1]
                variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                noise = torch.randn_like(x) * torch.sqrt(variance.clamp(min=1e-20))
            else:
                noise = 0
            
            # Update x using the reverse process formula
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + noise
    
    # denormalize to visualize 
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    
    # remove padding if we added since mnist is 28 x 28 
    if h == 32 and w == 32:
        x = x[:, :, 2:-2, 2:-2]  
    
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM UNet on MNIST')
    parser.add_argument('--ch_data', type=int, default=1, help='Number of channels in data') 
    parser.add_argument('--ch', type=int, default=64, help='Number of channels in our latent space') 
    parser.add_argument('--h', type=int, default=32, help='Image height')
    parser.add_argument('--w', type=int, default=32, help='Image width')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=2e-2, help='Ending beta value')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Amount of guidance conditioning...')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='Noise schedule type')
    parser.add_argument('--print_every', type=int, default=100, help='Print loss every N steps')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--sample_guided', action='store_true', help='Generate samples after training')
    parser.add_argument('--save', action='store_true', help='Save model after training')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = args.timesteps
    
    # Default to cosine noise schedule
    if args.schedule == 'linear':
        betas = torch.linspace(args.beta_start, args.beta_end, T)
    else:  # cosine schedule like in improved ddpm paper (Nichol & Dharival)
        s = 0.008  
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
        verbose=args.verbose,
    )
    
    if args.save:
        save_path = f'ddpm_cfg_trained_e{args.epochs}_ch{args.ch}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    if args.sample_guided:
        print("Generating samples with classifier-free guidance...")
        samples = sample_guided(model, betas, n_samples=10, ch_data=args.ch_data, h=args.h, w=args.w, guidance_scale=args.guidance_scale)
        
        # Create a grid of images
        grid = torchvision.utils.make_grid(samples, nrow=5)
        
        # Convert to PIL image and save
        grid_img = torchvision.transforms.ToPILImage()(grid)
        filename = 'cfg_unet_samples.png'
        grid_img.save(filename)
        print(f"Samples saved to {filename}")


