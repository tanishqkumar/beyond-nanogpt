'''
(https://arxiv.org/pdf/1611.07004) Image-to-Image Translation with Conditional Adversarial Nets 

This is an important variation of a GAN. It is a conditional GAN that takes pixels to pixels under some 
mapping, like "make this sketch photorealistic" or "fill in the colors." Besides using a conditional GAN, 
it includes a modified loss function with an L1 penalty. 

Differences of Pix2Pix from vanilla GAN. 
    - One data point is now [in_img, out_img], eg. pencil sketch and photorealistic counterpart 
    - Both gen/discriminator take in in_img as conditioning information 
    - Noise z is usually dropped in fwd of generator 
    - Include L1 penalty on generator loss to ensure its close to out_img in pixel space 
    - Often people use U-net for generator and PatchGAN arch for discriminator
        - We've already implemented UNets from scratch (train_ddpm.py) and 
            GANs (train_gan.py) so here we'll write the two archs using native torch 
            fns (eg. Conv2d) rather than from scratch again. 

'''

import os
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image  
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm 
import ssl
import argparse

# hack, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context


## image preprocessing infra, not important ##
def download_and_extract_facades(url = "https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"): 
    try: 
        response = requests.get(url, stream=True)        
        response.raise_for_status() # raise an error if bad http error code given 
        
        z = zipfile.ZipFile(BytesIO(response.content)) 
        z.extractall('./data/facades') # extract all bytes from this zip file into local folder 
        return True 
    except requests.exceptions.RequestException as e: 
        print(f'Failed to get data, {e}')
        return False 
    except zipfile.BadZipFile as e: 
        print(f'Bad file, {e}')
        return False 

class FacadesDataset(Dataset): 
    def __init__(self, root_dir, transform=None): 
        self.root_dir = root_dir 
        self.transform = transform 

        # keep only images 
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self): 
        return len(self.image_files) 

    def __getitem__(self, idx): 
        img_name = os.path.join(self.root_dir, self.image_files[idx])

        try: 
            image = Image.open(img_name).convert('RGB')

            # in facades, mask has same base name but in png 
            base_name = os.path.splitext(self.image_files[idx])[0] # gets base name 
            label_name = os.path.join(self.root_dir, base_name + '.png')

            if os.path.exists(label_name): 
                label = Image.open(label_name).convert('RGB')
            else: 
                print(f'Warning, mask not found for {img_name}')
                label = Image.new('RGB', image.size, (0, 0, 0))

            if self.transform: 
                image = self.transform(image)
                label = self.transform(label)

            return {'image': image, 'label': label}

        except (IOError, OSError) as e: 
            print(f'Error in Dataset.getitem(), {e}')
            placeholder = torch.zeros(3, 256, 256) if self.transform else Image.new('RGB', (256, 256), (0, 0, 0))
            return {'image': placeholder, 'label': placeholder}

## end image preprocessing infra ##

# TODO: change archs to (unet, patchgan) once it's working with MLPs
class Generator(nn.Module): 
    def __init__(self, in_ch=3, in_h=256, in_w=256, out_ch=3, out_h=256, out_w=256, mult=2, nlayers=1, act=nn.GELU()):
        super().__init__()
        self.out_ch = out_ch 
        self.out_h = out_h
        self.out_w = out_w
        self.in_ch = in_ch
        self.in_h = in_h
        self.in_w = in_w
        self.mult = mult 
        self.in_dim = in_ch * in_h * in_w
        hidden_dim = self.in_dim * mult
        layers = []

        layers.append(nn.Linear(self.in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)

        for _ in range(nlayers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act)
        # final layer to produce output image pixels
        layers.append(nn.Linear(hidden_dim, out_ch * out_h * out_w))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [b, ch, h, w] -> ... -> [b, ch, h, w]
        # if x has > 2 dims, reshape to [b, -1]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1) # [b, ch * h * w]
        
        out = self.net(x)
        out = out.reshape(x.shape[0], self.out_ch, self.out_h, self.out_w)
        return out.tanh()  # tanh squashes into [-1, 1] like mnist 


class Discriminator(nn.Module):  # [out_ch, out_h, out_w] -> 2 logits 
    def __init__(self, out_ch=3, out_h=256, out_w=256, mult=2, nlayers=1, act=nn.GELU()):
        super().__init__()
        self.out_ch = out_ch
        self.out_h = out_h 
        self.out_w = out_w
        self.mult = mult
        self.flattened_img_sz = 2 * out_ch * out_h * out_w # 2 is bc we take in in_img and out_img_real_or_fake
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
    
    def forward(self, x): # disc takes in [in_img, out_image_real_or_fake] ie. [b, 2, ch, h, w], outputs scalar P(real)
        x = x.reshape(x.shape[0], -1)  # flatten image
        return self.net(x)  # output is a logit, bcewithlogits loss takes care of mapping to [0,1] inside
    

def train(
    in_ch=3, in_h=256, in_w=256, out_ch=3, out_h=256, out_w=256, 
    mult=2, nlayers=1, bsz=64, nepochs=100, lr=1e-4, beta1=0.5, 
    verbose=False, sample=False, cripple_factor=4, l1_coeff=0.1
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("./data/facades", exist_ok=True)
    
    # Download and extract dataset
    if not os.path.exists("./data/facades/base"):
        if not download_and_extract_facades():
            print("Failed to set up the dataset. Please check your internet connection or download manually.")
            return
    
    # TODO: rewrite from scratch 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])
    
    # Create dataset and dataloader
    dataset_path = "./data/facades/base"

    D = Discriminator(out_ch=out_ch, out_h=out_h, out_w=out_w, mult=int(mult/cripple_factor), nlayers=nlayers).to(device) 
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    G = Generator(in_ch=in_ch, in_h=in_h, in_w=in_w, out_ch=out_ch, out_h=out_h, out_w=out_w, mult=mult, nlayers=nlayers).to(device)
    G_opt = torch.optim.Adam(G.parameters(), lr=lr*cripple_factor, betas=(beta1, 0.999))

    D.train(); G.train()

    dataset = FacadesDataset(root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)
        
    criterion = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"Starting training on {device}")
        print(f"Dataset size: {len(dataset)} images")
        print(f"Generator parameters: {sum(p.numel() for p in G.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters())}")
        print(f"Training for {nepochs} epochs with batch size {bsz}")

    # add L1 to loss, add/cat conditioning to inputs 
    for epoch in tqdm(range(nepochs)):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        
        for batch in dataloader: # [in_img, out_img]
            batch_count += 1
            in_img_batch = batch['image'].to(device)
            out_img_batch = batch['label'].to(device)
            
            # Zero gradients
            D_opt.zero_grad()
            
            # update D to maximize [D(in_img_batch) - D(fake_batch_out_imgs)]
            fake_batch_out_imgs = G(in_img_batch).detach() # no need for noise, just pass in conditioning info in_img
            
            # D expects [b, 2 * ch, h, w]
            D_input_fake = torch.cat([in_img_batch, fake_batch_out_imgs], dim=1)
            D_input_real = torch.cat([in_img_batch, out_img_batch], dim=1)

            Dfake = D(D_input_fake) # discriminator outputs prob its real_out_img
            Dreal = D(D_input_real)
            
            # Make sure batch size matches actual batch
            actual_batch_size = in_img_batch.size(0)
            D_loss = criterion(Dreal, torch.ones(actual_batch_size, 1, device=device)) + criterion(Dfake, torch.zeros(actual_batch_size, 1, device=device))
            D_loss.backward()
            D_opt.step()

            # update G to maximize probability that D makes a mistake, include L1 
            G_opt.zero_grad()
            fake_batch_out_imgs = G(in_img_batch)  # Generate new fake images for G update
            D_input_fake = torch.cat([in_img_batch, fake_batch_out_imgs], dim=1)
            Dfake = D(D_input_fake)
            
            l1_loss = torch.nn.functional.l1_loss(fake_batch_out_imgs, out_img_batch)
            G_loss = criterion(Dfake, torch.ones(actual_batch_size, 1, device=device)) + l1_coeff * l1_loss  # if disc = 1, then loss vanishes 
            G_loss.backward()
            G_opt.step()
            
            epoch_d_loss += D_loss.item()
            epoch_g_loss += G_loss.item()
            
            if verbose and batch_count % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_count}/{len(dataloader)}: D_loss = {D_loss.item():.4f}, G_loss = {G_loss.item():.4f}, L1 = {l1_loss.item():.4f}')

        if verbose:
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
            print(f'Epoch {epoch+1} completed: Avg D_loss = {avg_d_loss:.4f}, Avg G_loss = {avg_g_loss:.4f}')

    if verbose:
        print("Training completed!")

    # TODO: rewrite this from scratch 
    if sample: 
        G.eval()
        # For conditional GAN, we need input images to generate samples
        sample_batch = next(iter(dataloader))
        sample_inputs = sample_batch['image'][:10].to(device)  # Take first 10 images
        
        if verbose:
            print("Generating sample images...")
            
        with torch.no_grad(): 
            generated_minibatch = G(sample_inputs)
            
            # Create comparison grid: input, generated, ground truth
            comparison = []
            for i in range(min(5, len(sample_inputs))):
                comparison.extend([
                    sample_inputs[i],
                    generated_minibatch[i],
                    sample_batch['label'][i].to(device)
                ])

            # denormalize images from [-1, 1] to [0, 1] for proper visualization
            # Images were normalized with transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # which scales to [-1, 1], so we need to convert back to [0, 1] for saving
            comparison_tensor = torch.stack(comparison)
            comparison_tensor = comparison_tensor * 0.5 + 0.5  # Denormalize: x * std + mean
            
            # save the comparison grid
            save_image(comparison_tensor, 'pix2pix_results.png', nrow=3)
            print("Saved comparison images to pix2pix_results.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pix2Pix on Facades')
    parser.add_argument('--in-ch', type=int, default=3, help='Number of input channels')
    parser.add_argument('--in-h', type=int, default=256, help='Input height')
    parser.add_argument('--in-w', type=int, default=256, help='Input width')
    parser.add_argument('--out-ch', type=int, default=3, help='Number of output channels')
    parser.add_argument('--out-h', type=int, default=256, help='Output height')
    parser.add_argument('--out-w', type=int, default=256, help='Output width')
    parser.add_argument('--mult', type=int, default=4, help='Multiplier for hidden dimensions')
    parser.add_argument('--nlayers', type=int, default=1, help='Number of hidden layers in both generator and discriminator')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--cripple', type=float, default=4.0, help='Cripple factor for G>D')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--sample', action='store_true', help='Sample from conditional GAN generator at end...')
    parser.add_argument('--l1-coeff', type=float, default=0.1, help='L1 loss coefficient')
    
    args = parser.parse_args()
    
    if args.verbose: 
        print(f'Parsed args, commencing train()') 

    train(
        in_ch=args.in_ch,
        in_h=args.in_h,
        in_w=args.in_w,
        out_ch=args.out_ch,
        out_h=args.out_h, 
        out_w=args.out_w,
        mult=args.mult,
        nlayers=args.nlayers,
        bsz=args.batch_size,
        nepochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        verbose=args.verbose,
        sample=args.sample,
        cripple_factor=args.cripple,
        l1_coeff=args.l1_coeff,
    )
