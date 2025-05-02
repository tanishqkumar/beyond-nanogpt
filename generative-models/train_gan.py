'''
(https://arxiv.org/pdf/1406.2661) Generative Adversarial Nets. 

Note. Obviously GANs are not diffusion models. They are included in this section because they were the key 
motivation and precursor to diffusion and modern generative models, and I personally think they are an important 
pre-requisite before one looks at diffusion to understand the setting in which it was invented/motivated. 

GANs pit two neural networks against each other: a generator that creates fake data and a discriminator that tries to tell 
real from fake. Through this adversarial process, the generator gets better at creating realistic data 
while the discriminator improves at spotting fakes. You can also think of it as co-operative where 
the discriminator tells the generator its "weak spots" (ie. portions of its generation space that are not well trained)
and then the generator focusses on "patching" those. 

It's not at all obvious this should work, but in the paper there's a proof (Prop 1) that such a setup
forces a unique Nash eqm where discriminator cannot do better than random and generator
has learned the training distribution. 

G: latent vector of latent_dim -> image [ch, h, w] ("generation")
D: image -> logits for each class (real vs fake) ("discrimination")


Note, D_loss≈1.386 at convergence when its right half the time, ie. no better than chance
at discriminating generator outputs from the true training distribution, so this is the number that would let us know 
our paired nets have convered and G(x) = p_{data}(x) as desired. 

Here we use a minimal example: MNIST as our training distribution, and MLPs as our networks. Larger scale ones 
use harder distributions and convolutional archs. Also, batchNorms are empirically important for stable training. 

'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image  
from torch.utils.data import DataLoader
from tqdm import tqdm 
import ssl
import argparse

# hack, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context

class Generator(nn.Module): 
    def __init__(self, latent_dim=128, out_ch=1, out_h=28, out_w=28, mult=4, nlayers=1, act=nn.GELU()):
        super().__init__()
        self.latent_dim = latent_dim 
        self.out_ch = out_ch 
        self.out_h = out_h
        self.out_w = out_w 
        self.mult = mult 
        hidden_dim = latent_dim * mult
        layers = []

        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)

        for _ in range(nlayers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act)
        # final layer to produce output image pixels
        layers.append(nn.Linear(hidden_dim, out_ch * out_h * out_w))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [b, latent_dim] -> ... -> [b, ch, h, w]
        out = self.net(x)
        out = out.reshape(x.shape[0], self.out_ch, self.out_h, self.out_w)
        return out.tanh()  # tanh squashes into [-1, 1] like mnist 


class Discriminator(nn.Module):  # [out_ch, out_h, out_w] -> 2 logits 
    def __init__(self, out_ch=1, out_h=28, out_w=28, mult=4, nlayers=1, act=nn.GELU()):
        super().__init__()
        self.out_ch = out_ch
        self.out_h = out_h 
        self.out_w = out_w
        self.mult = mult
        self.flattened_img_sz = out_ch * out_h * out_w
        input_dim = self.flattened_img_sz
        hidden_dim = int(input_dim * mult)  # ensure hidden_dim is an integer
        layers = []
        # first hidden layer: from flattened image to hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        
        # nlayers = n_hidden_layers, maybe bad naming by me 
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act)
        
        # final output layer to produce a single logit
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x): 
        b, _, _, _ = x.shape
        x = x.reshape(b, -1)  # flatten image
        return self.net(x)  # output is a logit, bcewithlogits loss takes care of mapping to [0,1] inside
    

def train(latent_dim=100, out_ch=1, out_h=28, out_w=28, mult=4, nlayers=1,
          batch_sz=128, nepochs=100, lr=2e-4, beta1=0.5, verbose=False, sample=False, cripple_factor=4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([ # standard mnist transform from stack overflow 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

    D = Discriminator(out_ch=out_ch, out_h=out_h, out_w=out_w, mult=int(mult/cripple_factor), nlayers=nlayers).to(device) 
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    G = Generator(latent_dim=latent_dim, out_ch=out_ch, out_h=out_h, out_w=out_w, mult=mult, nlayers=nlayers).to(device)
    G_opt = torch.optim.Adam(G.parameters(), lr=lr*cripple_factor, betas=(beta1, 0.999))

    D.train(); G.train()

    # setup dataset/loader 
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=batch_sz, shuffle=True, num_workers=1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(nepochs)):
        for real_batch, _ in dataloader:
            real_batch = real_batch.to(device)
            # update D to maximize [D(real_batch) - D(fake_batch)]
            bsz = real_batch.shape[0]
            z = torch.randn(bsz, latent_dim, device=device)
            fake_batch = G(z).detach()
            Dfake = D(fake_batch)
            Dreal = D(real_batch)
            D_loss = criterion(Dreal, torch.ones(bsz, 1, device=device)) + criterion(Dfake, torch.zeros(bsz, 1, device=device))
            D_opt.zero_grad()
            D_loss.backward()
            D_opt.step()

            # update G to maximize probability that D makes a mistake 
            z = torch.randn(bsz, latent_dim, device=device)
            fake_batch = G(z)
            disc_guess = D(fake_batch)  # in [0, 1] and want it to be close to 1
            G_loss = criterion(disc_guess, torch.ones(bsz, 1, device=device))  # if disc = 1, then loss vanishes 
            G_opt.zero_grad()
            G_loss.backward()
            G_opt.step()

        if verbose:
            print(f'Epoch {epoch}: D_loss = {D_loss.item():.4f}, G_loss = {G_loss.item():.4f}')

    if sample: 
        G.eval()
        nsamples = 10 
        with torch.no_grad(): 
            z = torch.randn(nsamples, latent_dim, device=device)
            generated_minibatch = G(z)

        # denormalize images from [-1, 1] to [0, 1] for proper visualization
        generated_minibatch = (generated_minibatch + 1) / 2
        # save the generated images as a grid; here, arranging 5 images per row (adjust nrow as needed)
        save_image(generated_minibatch, 'generated_minibatch.png', nrow=5)
        print("Saved generated images to generated_minibatch.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN on MNIST')
    parser.add_argument('--latent-dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--out-ch', type=int, default=1, help='Number of output channels')
    parser.add_argument('--out-h', type=int, default=28, help='Output height')
    parser.add_argument('--out-w', type=int, default=28, help='Output width')
    parser.add_argument('--mult', type=int, default=4, help='Multiplier for hidden dimensions')
    parser.add_argument('--nlayers', type=int, default=1, help='Number of hidden layers in both generator and discriminator')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--cripple', type=float, default=4.0, help='Cripple factor for G>D')
    # cripple "cripples" the discriminator to make training easier for the generator
    # this trick is often used in GAN training to allow the generator to reach low loss
    # since "discrimination is easier than generation"
    # and in general getting GANs to train stably is a nightmare
    # which is why so many follow up papers (DCGAN, Wasserstein GANs) are focused on new ideas
    # for stable training as well as partly why Diffusion Models "won the day"
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--sample', action='store_true', help='Sample from GAN at end...')
    
    args = parser.parse_args()
    
    train(
        latent_dim=args.latent_dim,
        out_ch=args.out_ch,
        out_h=args.out_h, 
        out_w=args.out_w,
        mult=args.mult,
        nlayers=args.nlayers,
        batch_sz=args.batch_size,
        nepochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        verbose=args.verbose,
        sample=args.sample,
        cripple_factor=args.cripple,
    )
