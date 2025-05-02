'''
(https://arxiv.org/pdf/1312.6114) Auto-Encoding Variational Bayes. 
Here's a quick, informal intro to our Variational Autoencoder (VAE) that builds on the 
vanilla autoencoder you saw in train_autoencoder.py. I will give the first introduction to VAEs
to my knowledge that DOES NOT talk about VARIATIONAL INFERENCE AND ELBO AT ALL. In my opinion, 
it can be explained clearly without those concepts even though obviously you need to understand those 
to do research around generative models. Here's how to get from an auto-encoder to VAE in three steps. 

1) Notice if we pass in z~N(0,1) to an autoencoder, we'll get garbage since there's no reason the decoder 
expects to see random normal vectors, the encoder may have learned something totally different. 
Thus including a KL term to keep latents near N(0,1) encourages latents fed in from encoder to be closer to 
N(0,1) which helps us use the decoder for generation. 

2) Why does the encoder output the [mean, std] of a distribution now where we directly outputted latents before? 
Answer: because outputting latents only (as in vanilla AE) means decoder can simply learn to map those specific 
vectors to their original images (ie. "memorize") so when you give it new z~N(0,1) to generate, you get garbage 
(esp since the specific z drawn will be diff each time you sample). Instead, make sure during training it gets used to 
taking *distributions* to outputted images, so we can give it many z, z' ~N(0,1) at inference and we'll get similar 
outputted/generated images. Using sampling during the forward isn't problematic for autodiff for the below reason. 

3) What is this "reparameterization trick" math jargon/garbage? Ignore the jargon IMO. It's 
merely observing we want to take encoder gradients wrt [mean, std] and can't do so if we write 
x = dist.sample(mean, std), so instead we just write x = N(0,1).sample() * std + mean 
where the key idea is that we don't need to diff through  N(0,1).sample() when we compute x that way 
since it's treated as a constant during autograd and std/mean are "on the outside." This is what allows 
sample() to not break the autograd graph and do backward() as usual. 

These three tweaks let us go from reconstructing existing data to being able to *learn a distribution* and thus 
generate samples. 
'''
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math 
import ssl
    
# hack
ssl._create_default_https_context = ssl._create_unverified_context

class Encoder(nn.Module): # [b, ch, h, w] -> [b, d]
    def __init__(self, ch=1, h=28, w=28, d=32, mult=4, nlayers=0): 
        super().__init__()
        
        layers = [nn.Linear(ch * h * w, d * mult), nn.GELU()]
        
        # can add additional hidden layers to inrease capacity
        for _ in range(nlayers):
            layers.append(nn.Linear(d * mult, d * mult))
            layers.append(nn.GELU())
            
        self.encoder_mlp = nn.Sequential(*layers)
        
        # two heads for computing mean/std of our distb 
        self.mean_layer = nn.Linear(d * mult, d)
        self.std_layer = nn.Linear(d * mult, d)

    def forward(self, x): # [b, ch, h, w] -> 2x [d] mean + std vectors 
        latents = x.reshape(x.shape[0], -1)
        latents = self.encoder_mlp(latents)
        return self.mean_layer(latents), self.std_layer(latents)


class Decoder(nn.Module): # [b, d] -> [b, ch * h * w] -> [b, ch, h, w]
    def __init__(self, ch=1, h=28, w=28, d=32, mult=4, nlayers=0): 
        super().__init__()
        self.ch = ch
        self.h = h
        self.w = w
        self.d = d
        
        # Build decoder MLP with configurable hidden layers
        layers = [nn.Linear(d, d * mult), nn.GELU()]
        
        # Add additional hidden layers if specified
        for _ in range(nlayers):
            layers.append(nn.Linear(d * mult, d * mult))
            layers.append(nn.GELU())
            
        # Add final output layer
        layers.append(nn.Linear(d * mult, ch * h * w))
        
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, x): # [b, d] -> [b, ch, h, w]
        return self.decoder_mlp(x).reshape(x.shape[0], self.ch, self.h, self.w)

class VAE(nn.Module): 
    def __init__(self, ch=1, h=28, w=28, d=32, mult=4, nlayers=0): 
        super().__init__()
        self.enc = Encoder(ch, h, w, d, mult, nlayers)
        self.dec = Decoder(ch, h, w, d, mult, nlayers)

    @staticmethod
    def sample(latent_mean, logvar): # [b, latent] -> sample from this distb [b, latent]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # logvar and thus std are [b, d]
        return latent_mean + eps * std # "reparam trick" as no need to diff through eps

    def forward(self, x): 
        # enc -> sample -> dec 
        mean, logvar = self.enc(x)
        latents = self.sample(mean, logvar)
        return self.dec(latents), mean, logvar

def loss_fn(preds, targets, mean, logvar, beta=0.1): 
    # two error terms, the kl one is key for reconstruction -> generative modeling
    # since it enforces a smooth latent space that is close to gaussian 
    # so that we can give gaussians at inference time to the decoder, which is ultimately 
    # what we want to use for generation (just like generator in GAN is really what we wanted)
    # for generation and discriminator was just a tool to train it 
    # much like the encoder here 
    reconstruction_loss = F.mse_loss(preds, targets)
    # kl for N(mean, var) against N(0,1) our prior can be written in closed form as a fn of mean/var 
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) 

    return reconstruction_loss + beta * kl_loss

def train(ch=1, h=28, w=28, d=32, mult=4, nlayers=0, batch_sz=32, nepochs=100, lr=1e-3, verbose=False, sample=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1)
    
    vae = VAE(ch=ch, h=h, w=w, d=d, mult=mult, nlayers=nlayers).to(device)
    if verbose: 
        print(f'Model has {sum(p.numel() for p in vae.parameters()):,} parameters')
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for epoch in tqdm(range(nepochs)):
        vae.train()
        total_train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            
            reconstructed, mean, log_var = vae(batch)
            loss = loss_fn(reconstructed, batch, mean, log_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        if verbose:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}')

    # allow for sampling from the trained model at the end
    if sample: 
        vae.eval()
        sample_batch_sz = 10 # these will be unconditional samples from training dist, so random numbers from mnist 
        z = torch.randn(sample_batch_sz, d).to(device)

        with torch.no_grad():
            sample_images = vae.dec(z) # we just need decoder at inference time 
            sample_images = sample_images.reshape(sample_batch_sz, ch, h, w).cpu()
            sample_images = (sample_images + 1) / 2
            
            # create and save grid 
            grid = torchvision.utils.make_grid(sample_images, nrow=5, padding=2)
            torchvision.utils.save_image(grid, 'vae_samples.png')
            
            if verbose:
                print(f"Generated samples saved to vae_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train vae on MNIST')
    parser.add_argument('--ch', type=int, default=1, help='Number of channels in input image')
    parser.add_argument('--h', type=int, default=28, help='Height of input image')
    parser.add_argument('--w', type=int, default=28, help='Width of input image')
    parser.add_argument('--d', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--mult', type=int, default=4, help='Multiplier for hidden layer dimensions')
    parser.add_argument('--nlayers', type=int, default=0, help='Number of additional hidden layers in encoder/decoder')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--sample', action='store_true', help='Sample and save')
    
    args = parser.parse_args()
    
    train(
        ch=args.ch,
        h=args.h,
        w=args.w,
        d=args.d,
        mult=args.mult,
        nlayers=args.nlayers,
        batch_sz=args.batch_size,
        nepochs=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
        sample=args.sample,
    )
