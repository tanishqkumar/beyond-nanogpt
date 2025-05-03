'''
(https://arxiv.org/pdf/2105.05233) [Diffusion Models Beat GANs on Image Synthesis]

Classifier guidance for diffusion models that lets us "steer" the sampling process
towards images of a particular class. In an LLM, you can do this by just prompting the model since it was 
trained to change its output distribution based on the prompt. Since diffusion models are not autoregressive, 
this doesn't work there, so if we want to sample from the "dog" distribution of a diffusion model trained on ImageNet, 
we need this trick. The intuition is:

1. Assume a pretrained DDPM, eg. from train_ddpm.py 
2. We train a small classifier f(x) that can predict class labels even from noisy images x_t, eg. an MLP 
3. During sampling, at each step we:
   - Use the DDPM to predict how to denoise x_t -> x_{t-1} as usual
   - BUT we also run the classifier f(x_t) and backprop to get grad[log f(x_t)]
   - This gradient tells us how to tweak x_t to make it "more like" our target class
   - We add this gradient as an extra "steering" term when denoising

So in addition to following the learned denoising process, we're also nudging the samples
towards regions that the classifier thinks look more like our target class. The classifier
acts like a guide helping the diffusion model generate specific types of images.

The cool part is this works even though the classifier and diffusion model were trained
totally separately! 

We implement this by:
1. Training/loading a DDPM from train_ddpm.py
2. Training a simple MLP classifier on MNIST that can handle noisy inputs
    2.1. Note, we must train the classifier on the same data distribution (images + noise + time embeddings)
            as the diffusion model so that we can use its learned data gradients "out of the box
3. During sampling, doing both the regular denoising step AND a classifier gradient step
    3.1. This "nudges" the latent we're denoising toward the desired class. 

'''

import os
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn   
from torchvision import datasets, transforms
import argparse
import math
import wandb
from train_ddpm import TimeEmbedding, UNet

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnist_dim = 28 * 28
        self.hidden_dim = self.mnist_dim * 4
        self.num_classes = 10
        self.mlp = nn.Sequential(
            nn.Linear(self.mnist_dim, self.hidden_dim),
            nn.GELU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.time_embeddings = TimeEmbedding(self.mnist_dim)

    def forward(self, x, t): # [b, ch, h, w] -> [b, num_classes]
        # t is a [b] len 1-tensor of random times 
        if x.shape[1] != self.mnist_dim: 
            x = x.reshape(x.shape[0], -1)
        
        # image_batch, x, is [b, mnist_dim] at this point 

        t_emb = self.time_embeddings(t) # [b, t] -> [b, mnist_dim]
        return self.mlp(x + t_emb) 
    
def train_classifier(betas, look_for_classifier_path = 'mnist_classifier.pt', device='cuda'): 
    T = betas.shape[0]
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # quick and dirty mnist training loop 
    classifier = Classifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    classifier.train()

    # mnist converges fast so hardcode some hypers like lr/epochs
    for _ in range(10): 
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # get [b] t_batch that will be projected to [b, mnist_dim] in classifier fwd
            optimizer.zero_grad()

            # need to noise the images and add time vector [b]
            t = torch.randint(0, T, (images.shape[0],), device=device)
            
            beta_t = betas[t]
            alpha_cumprod_prev = alphas_cumprod[t-1]
            variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alphas_cumprod[t])
            
            noise = torch.randn_like(images) * torch.sqrt(variance.clamp(min=1e-20)).unsqueeze(1) # [b, 1]
            noised_images = noise + images
            
            outputs = classifier(noised_images, t)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        if accuracy > 0.9:  
            break

    # save so that get_classifier can read this 
    torch.save(classifier.state_dict(), look_for_classifier_path)
    return 

# need our classifier to be trained on noised images and given t at each fwd (real_batch, )
def get_classifier(betas, device='cuda', look_for_classifier_path = 'mnist_classifier.pt'): 
    if not os.path.exists(look_for_classifier_path): 
        train_classifier(betas, look_for_classifier_path=look_for_classifier_path)
    
    # load model from path and return it
    classifier = Classifier()
    classifier.load_state_dict(torch.load(look_for_classifier_path))
    classifier.to(device)
    classifier.eval()
    return classifier

# key classifier-guidance logic here 
def get_steer(classifier, x_batch, t_batch, target_class):
    # freeze classifier parameters so only x_batch gets gradients
    for p in classifier.parameters():
        p.requires_grad_(False)

    # prepare input for grad:
    # detach from any existing graph, enable grad on data only
    data = x_batch.detach().requires_grad_(True)

    # if padded to 32×32, crop back to 28×28 for the MNIST classifier
    if data.dim() == 4 and data.size(2) > 28 and data.size(3) > 28:
        data = data[:, :, 2:-2, 2:-2]
        data.retain_grad()

    # forward pass through classifier
    logits = classifier(data, t_batch) # [b, num_classes]

    # define loss = mean(1 – p_target) bc we want to maximize p(target class)
    loss = -F.log_softmax(logits, dim=-1)[:, target_class].mean()
    loss.backward() # backprop to data, not params

    steer_crop = data.grad.detach() # [b, ch, h, w] steering vectors
    steer = F.pad(steer_crop, (2, 2, 2, 2)) # upcast back to ddpm dims

    return steer


# does reverse sampling loop from ddpm 
def denoise(x, model, betas, n_samples=10, device='cuda',
            guidance=True, target_class=6, steer_coeff=0.1):
    betas = betas.to(device)
    T = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # if guidance, get and store classifier 
    if guidance: 
        classifier = get_classifier(betas, device=device)
        classifier.eval() # turns grads off for classifier, since we want to autodif the *data*

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # predict noise (ε) without accumulating gradients in the model
        with torch.no_grad():
            pred_noise = model(x, t_batch)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]

        # compute the posterior variance and sample noise if not the final step
        if t > 0:
            beta_t = betas[t]
            alpha_cumprod_prev = alphas_cumprod[t - 1]
            var = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(x) * torch.sqrt(var.clamp(min=1e-20))
        else:
            noise = torch.zeros_like(x)

        # classifier guidance term
        if guidance:
            # allow gradients w.r.t. x for the steering computation
            with torch.enable_grad():
                steer = get_steer(classifier, x, t_batch, target_class)
        else:
            steer = torch.zeros_like(x)

        # reverse diffusion update + steering
        x = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise
            )
        x = x + noise + steer_coeff * steer

    return x


def sample(model, betas, n_samples=10, ch_data=1, h=32, w=32,
           guidance=True, target_class=6, steer_coeff=0.1):
    """
    Generate samples from a pretrained DDPM with optional classifier guidance.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # start from pure noise
    x = torch.randn(n_samples, ch_data, h, w, device=device)

    # run the reverse diffusion process
    x = denoise(
        x, model, betas,
        n_samples=n_samples,
        device=device,
        guidance=guidance,
        target_class=target_class,
        steer_coeff=steer_coeff
    )

    # map from [-1,1] to [0,1]
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0

    # undo the 28->32 padding if applied
    if h == 32 and w == 32:
        x = x[:, :, 2:-2, 2:-2]

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from a pretrained DDPM UNet with optional classifier guidance."
    )
    parser.add_argument(
        "--model_path", type=str,
        default="ddpm_trained_e5_ch256.pt",
        help="Path to pretrained DDPM model checkpoint"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500,
        help="Number of diffusion timesteps T"
    )
    parser.add_argument(
        "--beta_start", type=float, default=1e-4,
        help="Starting beta for noise schedule"
    )
    parser.add_argument(
        "--beta_end", type=float, default=2e-2,
        help="Ending beta for noise schedule"
    )
    parser.add_argument(
        "--schedule", type=str, default="cosine", choices=["linear", "cosine"],
        help="Noise schedule type"
    )
    parser.add_argument(
        "--ch_data", type=int, default=1,
        help="Number of data channels"
    )
    parser.add_argument(
        "--ch", type=int, default=256,
        help="Base channel width of UNet"
    )
    parser.add_argument(
        "--h", type=int, default=32, help="Image height"
    )
    parser.add_argument(
        "--w", type=int, default=32, help="Image width"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--target_class", type=int, default=6,
        help="Class label to steer towards"
    )
    parser.add_argument(
        "--steer_coeff", type=float, default=0.1,
        help="Guidance strength coefficient"
    )
    parser.add_argument(
        "--no-guidance", dest="guidance", action="store_false",
        help="Disable classifier-based steering"
    )
    parser.add_argument(
        "--output_path", type=str, default="guided_samples.png",
        help="Where to save the generated image grid"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress messages"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log results to Weights & Biases"
    )
    args = parser.parse_args()

    # initialize wandb if requested
    if args.wandb and wandb is not None:
        wandb.init(project="ddpm-classifier-guidance", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build betas
    if args.schedule == "linear":
        betas = torch.linspace(args.beta_start, args.beta_end, args.timesteps)
    else:
        s = 0.008
        steps = torch.arange(args.timesteps + 1, dtype=torch.float32) / args.timesteps
        f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=args.beta_start, max=args.beta_end)
    betas = betas.to(device)

    # load the pretrained UNet
    model = UNet(
        ch_data=args.ch_data,
        ch=args.ch,
        h=args.h,
        w=args.w
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    if args.verbose:
        print(f"Sampling {args.n_samples} images "
              f"(guidance={args.guidance}, class={args.target_class}, coeff={args.steer_coeff})")

    # generate
    samples = sample(
        model, betas,
        n_samples=args.n_samples,
        ch_data=args.ch_data,
        h=args.h,
        w=args.w,
        guidance=args.guidance,
        target_class=args.target_class,
        steer_coeff=args.steer_coeff
    )

    # make and save grid
    grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(args.n_samples)))
    torchvision.transforms.ToPILImage()(grid).save(args.output_path)

    if args.verbose:
        print(f"Saved samples to {args.output_path}")
    if args.wandb and wandb is not None:
        wandb.log({"samples": wandb.Image(grid)})