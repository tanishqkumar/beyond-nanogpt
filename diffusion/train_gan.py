# G: latent vector of latent_dim -> image [ch, h, w]
# D: image -> logits for each class (real vs fake)
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import ssl
import argparse

# hack, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context


class Generator(nn.Module): 
    def __init__(self, latent_dim=128, out_ch=1, out_h=28, out_w=28, mult=4, act=nn.GELU()): 
        super().__init__()
        self.latent_dim = latent_dim 
        self.out_ch = out_ch 
        self.out_h = out_h
        self.out_w = out_w 
        self.mult = mult 
        self.w1 = nn.Linear(latent_dim, latent_dim * mult)
        self.w2 = nn.Linear(latent_dim * mult, out_ch * out_h * out_w)
        self.act = act

    def forward(self, x): # [b, latent_dim] -> [b, ch*h*w] -> [b, ch, h, w]
        out = self.act(self.w1(x))
        out = self.w2(out)
        return out.reshape(x.shape[0], self.out_ch, self.out_h, self.out_w)

class Discriminator(nn.Module): # [out_ch, out_h, out_w] -> 2 logits 
    def __init__(self, out_ch=1, out_h=28, out_w=28, mult=4, act=nn.GELU()): 
        super().__init__()
        self.out_ch = out_ch
        self.out_h = out_h 
        self.out_w = out_w
        self.mult = mult
        self.flattened_img_sz = out_ch * out_h * out_w
        self.w1 = nn.Linear(self.flattened_img_sz, self.flattened_img_sz * mult)
        self.w2 = nn.Linear(self.flattened_img_sz * mult, 1) # output P(real), will take sign 
        self.act = act
    
    def forward(self, x): 
        b, _, _, _ = x.shape
        x = x.reshape(b, -1) # [ch, h, w] -> ch * h * w
        logit = self.w2(self.act(self.w1(x))) 
        return torch.sigmoid(logit) # output in [0, 1]


def train(latent_dim=100, out_ch=1, out_h=28, out_w=28, mult=4, 
          batch_sz=128, nepochs=100, lr=2e-4, beta1=0.5, verbose=False):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    D = Discriminator(out_ch=out_ch, out_h=out_h, out_w=out_w, mult=mult) 
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    G = Generator(latent_dim=latent_dim, out_ch=out_ch, out_h=out_h, out_w=out_w, mult=mult)
    G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # setup dataset/loader 
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=batch_sz, shuffle=True, num_workers=1)

    for epoch in tqdm(range(nepochs)): 
        for real_batch, _ in dataloader:
            # update D to maximize [D(real_batch) - D(fake_batch)]
            z = torch.randn(batch_sz, latent_dim).to(real_batch.device)
            fake_batch = G(z).detach() 
            Dfake = D(fake_batch)
            Dreal = D(real_batch) 
            D_loss = -torch.log(Dreal + 1e-8).mean() - torch.log(1 - Dfake + 1e-8).mean()
            D_opt.zero_grad()
            D_loss.backward()
            D_opt.step()

            # update G to maximize prob D makes a mistake 
            z = torch.randn(batch_sz, latent_dim).to(real_batch.device)
            fake_batch = G(z)
            disc_guess = D(fake_batch) # in [0, 1] and want it to be close to 1
            G_loss = -torch.log(disc_guess + 1e-8).mean() # if disc = 1, then loss vanishes 
            G_opt.zero_grad()
            G_loss.backward()
            G_opt.step()

        if verbose:
            print(f'Epoch {epoch}: D_loss = {D_loss.item():.4f}, G_loss = {G_loss.item():.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN on MNIST')
    parser.add_argument('--latent-dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--out-ch', type=int, default=1, help='Number of output channels')
    parser.add_argument('--out-h', type=int, default=28, help='Output height')
    parser.add_argument('--out-w', type=int, default=28, help='Output width')
    parser.add_argument('--mult', type=int, default=4, help='Multiplier for hidden dimensions')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    train(
        latent_dim=args.latent_dim,
        out_ch=args.out_ch,
        out_h=args.out_h, 
        out_w=args.out_w,
        mult=args.mult,
        batch_sz=args.batch_size,
        nepochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        verbose=args.verbose
    )

# D_loss≈1.386 at convergence when its right half the time 