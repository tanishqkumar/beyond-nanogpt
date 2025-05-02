'''
Before writing a VAE let's write a simple autoencoder. This is a simple model that tries to compress
high-dimensional data (in MNIST, dim is ch * h * w = 1 * 28 * 28 = 784) into a small latent space. 
Encoder is therefore just an MLP taking [b, ch, h, w] -> [b, ch * h * w] -> [b, d] latent -> [b, ch * h * w] output 
Decoder is [b, d] -> [b, ch, h, w], and we train with reconstruction error. 

This might seem trivial, but the motivation to go from AE -> VAE is that with a simple AE like this, 
if we take a trained decoder and pass in z~N(0,1) we get garbage, because there's no reason the encoder
learns to output latents "like z" so passing in z to the decoder would give it something it hasn't really seen 
during training. 

We want to be able to get an MNIST-like image to have a generative model. 
So the jump from AE -> VAE really requires forcing the latents to resemble z, 
which is really the motivation for the KL term in the VAE that isn't present in the AE. 
The VAE also requires one or two more conceptual tweaks, see `train_vae.py`

'''
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import ssl
    
# hack
ssl._create_default_https_context = ssl._create_unverified_context

class Encoder(nn.Module): # [b, ch, h, w] -> [b, d]
    def __init__(self, ch=1, h=28, w=28, d=32): 
        super().__init__()
        self.encoder_mlp = nn.Sequential(
            nn.Linear(ch * h * w, d), 
            nn.GELU(),
            nn.Linear(d, d)
        )

    def forward(self, x): # x is [b, ch, h, w]
        b, ch, h, w = x.shape 
        latents = x.reshape(b, -1)
        return self.encoder_mlp(latents)


class Decoder(nn.Module): # [b, ch, h, w] -> [b, d]
    def __init__(self, ch=1, h=28, w=28, d=32): 
        super().__init__()
        self.ch = ch
        self.h = h
        self.w = w
        self.d = d
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, ch * h * w)
        )

    def forward(self, x): # [b, d] -> [b, ch, h, w]
        return self.decoder_mlp(x).reshape(x.shape[0], self.ch, self.h, self.w)

def train(ch=1, h=28, w=28, d=32, batch_sz=32, nepochs=100, lr=1e-3, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # setup dataset/loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=1)
    
    # create model
    encoder = Encoder(ch=ch, h=h, w=w, d=d).to(device)
    decoder = Decoder(ch=ch, h=h, w=w, d=d).to(device)
    autoencoder = nn.Sequential(encoder, decoder)
    
    # setup optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    # training loop
    for epoch in tqdm(range(nepochs)):
        # training phase
        autoencoder.train()
        total_train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            
            reconstructed = autoencoder(batch)
            loss = F.mse_loss(reconstructed, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # evaluation phase
        autoencoder.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device)
                reconstructed = autoencoder(batch)
                loss = F.mse_loss(reconstructed, batch)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        
        if verbose:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Autoencoder on MNIST')
    parser.add_argument('--ch', type=int, default=1, help='Number of channels in input image')
    parser.add_argument('--h', type=int, default=28, help='Height of input image')
    parser.add_argument('--w', type=int, default=28, help='Width of input image')
    parser.add_argument('--d', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    train(
        ch=args.ch,
        h=args.h,
        w=args.w,
        d=args.d,
        batch_sz=args.batch_size,
        nepochs=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
    )
