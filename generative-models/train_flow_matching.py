'''
[https://arxiv.org/pdf/2210.02747] Flow-Matching Algorithms for Generative Modeling 

Flow-matching is a set of techniques for generative modeling that is conceptually similar to diffusion, but 
mathematically rather distinct. The goal of generative modeling is to learn a distribution Q, and diffusion uses 
a denoising objective that is equivalent to learning the score of Q (grad[-logQ]). Flow matching instead 
learns the optimal path from noise, Z, to Q, in distributional space. The "flow" is from Z --> Q. 

Concretely, to do this, we basically start with (noise, training images) and interpolate between the two, 
and teach a neural network to learn "the line connecting the two." Formally, what is being learned is 
a "velocity field" but all that means is at any point in data space, which for images is [b, ch, h, w], 
our neural network learns the optimal direction to move to get closer to Q. So the input to the net is 
(t, data_dim) and the output is data_dim which represents the optimal direction to move. 

The supervised directional signal comes from the fact that flow matching uses "interpolation paths" 
between noise and images as the ground truth. That is, x[t] = x_start * t + (1-t) * x_goal where 
x_start = noise and x_end = images. 

FAQ: 
    - What's the difference from diffusion? 
        --> There are two critical differences. First, diffusion's reverse sampling process is stochastic in that
        you sample from p(x_t|x_{t+1}) at each denoising step using the newly computed mean. Flow-matching is 
        a deterministic mapping x_0 -> x_T whereas DDPM is not. Second, the flow-matching algorithm follows the shortest 
        path in distributinoal space to go Z --> Q, whereas the means outputted by 
        DDPM follows a very specific path that is not always the shortest. 

    - Is there a formal connection b/w diffusion and flow-matching? 
        --> Yes, they are connected by something called the "probability flow ODE." Loosely, 
        this is the optimal way to move from Z --> Q in the sense of requiring the "least work." Such 
        notions are made precise in the field of optimal transport. At this point I'm reaching the limits 
        of my own understanding, but I'll just point out there is a wealth of theoretical work on this topic 
        eg. see Michael Albergo's Stochastic Interpolants paper and I think this is an active area of research. 

The key implementation details: 
    class Flow: [t, flattened_img_dim] -> flattened_img_dim 
    loss_fn = F.mse_loss(flow(t, x), x1-x0)
    Sample by repeatedly following the directions given to us in data space by the flowNet
        By sophists, this is called "integrating the velocity field with an ODE solver"
            ie. just repeat[ x_{t+1} = x_t + eta * flowNet(x_t, t) ] using x_0 = noise 
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os 
import argparse
import wandb 

class Flow(nn.Module): # [t, 1 * 28 * 28] -> [1 * 28 * 28] trained to predict v(x, t)
    def __init__(self, mult=4, nhidden=0): 
        super().__init__()
        self.mnist_dim = 28 * 28
        self.in_dim = self.mnist_dim + 1
        self.hidden_dim = self.in_dim * mult

        layers = [nn.Linear(self.in_dim, self.hidden_dim), nn.GELU()]
        for _ in range(nhidden):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU()
            ])
        layers.append(nn.Linear(self.hidden_dim, self.mnist_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t): # assumed flatten in/out 
        xt = torch.cat([x, t], dim=-1)
        return self.mlp(xt)

def loss_fn(flow, x_t, t, x0, x1): 
    # x_t is noised data tensor at time t
    v_preds = flow(x_t, t) # out is [b, mnist_dim]
    return F.mse_loss(v_preds, x1-x0)

def get_batch(dataloader, device='cuda'):
    # fetch one batch and move to device, handle dataloader wraparound for multi-epoch 
    try:
        x1, _ = next(dataloader)
    except StopIteration:
        dataloader = get_mnist_dataloader()
        x1, _ = next(dataloader)
        
    b, ch, h, w = x1.shape
    x1 = x1.reshape(b, ch * h * w).to(device)
    mnist_dim = ch * h * w
    # sample noise and times on the same device
    x0 = torch.randn(b, mnist_dim, device=device)
    t = torch.rand(b, 1, device=device)
    # interpolate
    x_t = t * x1 + (1 - t) * x0
    return x_t, t, x0, x1, dataloader

def get_mnist_dataloader(b=32, device='cuda'):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_data = datasets.MNIST(
        root='./data',
        train=True,
        download=not os.path.exists('./data/MNIST'),
        transform=transform
    )
    pin_mem = device.startswith('cuda')
    return iter(DataLoader(mnist_data, batch_size=b, shuffle=True, pin_memory=pin_mem))

# define sample as a simple euler solver given flow and using x0=noise 
@torch.no_grad()
def sample(model, b=10): # want [b, ch, h, w] sampled outputs 
    # sample [b, mnist_dim]
    model.eval()
    device = next(model.parameters()).device
    ch, h, w = 1, 28, 28
    mnist_dim = ch * h * w
    # initialize on the same device as the model
    x = torch.randn(b, mnist_dim, device=device)
    niters = 10
    times = torch.linspace(0, 1.0, niters + 1, device=device)

    # this might superficially resemble the denoising process in DDPMs
    # but really this is "integration against a velocity field"
    # ie. in plain english, "following the direction in data space given to use by our neural net"
    # to get from Z --> Q, where the FlowNet model is the "map" for where to go at each point 
    for i in range(1, niters + 1):
        t = times[i] * torch.ones(b, 1, device=device)
        dt = times[i] - times[i - 1]
        x = x + dt * model(x, t)

    return x.reshape(b, ch, h, w)

def train(lr=3e-4, nsteps=1000, verbose=True, do_sample=False, save=False, b=32, nhidden=0, mult=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Training on device: {device}")
        print(f"Learning rate: {lr}")
        print(f"Batch size: {b}")
        print(f"Number of steps: {nsteps}")

    dataloader = get_mnist_dataloader(b, device=device)
    model = Flow(mult=mult, nhidden=nhidden).to(device)
    if verbose: 
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Model has {n_params:,} parameters')
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(nsteps): 
        x_t, t, x0, x1, dataloader = get_batch(dataloader)
        opt.zero_grad()
        loss = loss_fn(model, x_t, t, x0, x1)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            if verbose:
                print(f'[{step}/{nsteps}]: Loss {loss.item():.6f}')
            if wandb.run is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": step
                })

    if do_sample:
        n_samples = 10
        sampled_batch = sample(model, n_samples)
        sample_path = f'flow_matching_samples_{nsteps}_B{b}_nhidden{nhidden}_mult{mult}.png'
        # move to CPU for saving
        torchvision.utils.save_image(sampled_batch.cpu(), sample_path, nrow=5, normalize=True)
        if wandb.run is not None:
            wandb.log({"samples": wandb.Image(sample_path)})
        print(f'Saved generated sample to {sample_path}!')

    if save:
        save_path = f'flow_model_steps{nsteps}_B{b}_nhidden{nhidden}_mult{mult}.pt'
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Model saved to {save_path}")
        if wandb.run is not None:
            wandb.save(save_path)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Flow Matching model on MNIST')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--steps', type=int, default=5000, help='Number of training steps')
    parser.add_argument('--nhidden', type=int, default=0, help='Number of hidden layers')
    parser.add_argument('--mult', type=int, default=4, help='Hidden multiplier in MLP')
    parser.add_argument('--B', type=int, default=256, help='Batch size')
    parser.add_argument('--save', action='store_true', help='Save model checkpoint')
    parser.add_argument('--sample', action='store_true', help='Sample at end')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--verbose', action='store_true', help='Print detailed training information')
    args = parser.parse_args()

    if args.wandb:
        run_name = f"flow_matching_lr{args.lr}_B{args.B}_steps{args.steps}"
        wandb.init(
            project="train_flow_matching",
            name=run_name,
            config={
                "learning_rate": args.lr,
                "batch_size": args.B,
                "num_steps": args.steps,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        )

    model = train(
        lr=args.lr,
        nsteps=args.steps,
        verbose=args.verbose,
        do_sample=args.sample,
        save=args.save,
        b=args.B, 
        nhidden=args.nhidden,
        mult=args.mult,
    )

    if args.wandb:
        wandb.finish()