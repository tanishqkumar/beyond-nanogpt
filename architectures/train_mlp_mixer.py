
'''
MLP Mixer from "MLP-Mixer: An all-MLP Architecture for Vision"
(https://arxiv.org/pdf/2105.01601)

TLDR: 
    We start with a batch of CIFAR-10 images (32x32 pixels with 3 color channels).

    The cool thing about MLP Mixer is that it's super simple - no convolutions or attention! 
    Instead, we just:
    1. Chop each image into small 4x4 patches (giving us a bunch of patches per image)
    2. Flatten and embed each patch into a higher-dimensional space (D)
    3. Then we alternate between two types of MLPs:
    - One that mixes information ACROSS patches (communication between different parts of the image)
    - Another that mixes information WITHIN each patch's features (processing local information)

    Each mixer layer has skip connections (like ResNets) and layer normalization to help training.

    After several mixer layers, we just average all patch representations and classify with a simple linear head.

    What's neat is that this architecture shows you don't need specialized inductive biases like convolutions
    or attention to get decent performance on vision tasks - just MLPs operating on different dimensions! Maybe vanilla 
    neural nets are all you need...
'''

import torch 
import argparse
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import wandb 
import torch.nn as nn 
import ssl 

# hack to get data, don't do in prod 
ssl._create_default_https_context = ssl._create_unverified_context

class MLP(nn.Module): 
    def __init__(self, in_dim, out_dim, mult=4, act=nn.GELU()): 
        super().__init__() 
        self.w_up = nn.Linear(in_dim, in_dim * mult)
        self.w_down = nn.Linear(in_dim * mult, out_dim)
        self.act = act 

    def forward(self, x): # operates on last dim 
        return self.w_down(self.act(self.w_up(x)))
        # this is basically a bottleneck MLP - we project up to a higher dim, apply non-linearity, then project back down
        # the mult=4 is a common choice in transformers too (4x expansion in FFN layers)

class LN(nn.Module): 
    def __init__(self, dim, eps=1e-8): 
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = eps 

    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + self.eps).sqrt() + self.shift
        # layer norm is crucial for training deep networks - it stabilizes activations
        # unlike batch norm, it normalizes across features (last dim) not batch samples
        # this makes it more suitable for sequence models and variable batch sizes

class GlobalAvgPool(nn.Module):
    # For MLP Mixer on CIFAR-10:
    #   Input:  x of shape [batch_size, num_patches, D]
    #   Output: tensor of shape [batch_size, D]
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean(dim=1)
        # simple but effective - we just average across all patches
        # this is a parameter-free way to aggregate spatial information
        # similar to global average pooling in CNNs, but operating on patches instead of pixels


class Embeddings(nn.Module): 
    # [b, 3, 32, 32] -> [b, np, 3, 4, 4] 
    # -> [b, np, 3 * 4 * 4] -> [b, np, D]

    def __init__(self, in_ch=3, in_sz=32, D=256, patch_sz=4):
        super().__init__()
        self.in_ch = in_ch
        self.in_sz = in_sz 
        self.patch_sz = patch_sz
        self.np = (in_sz//patch_sz)**2 # assumes in_sz is divisible by patch_sz
        self.D = D
        self.embeddings = nn.Linear(in_ch * patch_sz * patch_sz, D)
    
    def forward(self, x): # [b, in_ch, in_sz, in_sz] -> [b, np, D]
        b = x.shape[0]
        h = x.reshape(b, self.in_ch, -1) # [b, in_ch, in_sz, in_sz]
        h = h.reshape(b, self.in_ch, self.np, self.patch_sz * self.patch_sz) # [b, in_ch, np, ps * ps]
        h = h.transpose(1, 2) # [b, np, in_ch, ps, ps]
        h = h.reshape(b, self.np, -1) # [b, np, in_ch * ps * ps]
        return self.embeddings(h) # [b, np, D]
        # this reshaping dance is the key to patching - we're rearranging the 2D image into a sequence of patches
        # each patch gets flattened and linearly projected to dimension D
        # this is conceptually similar to the patch embedding in ViT, but implemented with pure reshaping ops

class MixerLayer(nn.Module): # [b, np, D] -> [b, np, D]
    def __init__(self, np, D): 
        super().__init__()
        self.mlp1 = MLP(np, np) # np dim input "token mixing"
        self.ln1 = LN(np)

        self.ln2 = LN(D)
        self.mlp2 = nn.Linear(D, D) # D dim input "channel mixing"

    def forward(self, x): 
        h = self.mlp1(self.ln1(x.transpose(1,2))).transpose(1,2) + x # [b, np, D] after token mixing 
        return self.mlp2(self.ln2(h)) + h
        # the transpose trick is the secret sauce of MLP-Mixer
        # by transposing, we can apply MLPs across different dimensions
        # first MLP mixes across patches (spatial mixing)
        # second MLP mixes across feature dimensions (channel mixing)
        # the skip connections (+ x and + h) are crucial for gradient flow in deep networks

class Mixer(nn.Module): # [b, in_ch, in_sz, in_sz] -> [b, num_classes]
    def __init__(self, in_ch=3, in_sz=32, D=256, num_classes=10, nlayers=8, patch_sz=4): 
        super().__init__()
        self.in_ch = in_ch
        self.in_sz = in_sz
        self.num_classes = num_classes
        self.patch_sz = patch_sz
        self.np = (in_sz//patch_sz)**2 # assumes in_sz is divisible by patch_sz
        self.D = D
        self.nlayers = nlayers 
        
        self.e = Embeddings(in_ch, in_sz, D, patch_sz)
        self.layers = nn.ModuleList([MixerLayer(self.np, D) for _ in range(nlayers)]) 
        self.pool = GlobalAvgPool()
        self.output_head = nn.Linear(D, num_classes)
    
    def forward(self, x): 
        h = self.e(x)
        for layer in self.layers: 
            h = layer(h)
        h = self.pool(h)
        logits = self.output_head(h)
        return logits
        # the overall architecture is remarkably simple - patch embedding, mixer layers, pooling, classification
        # no complex attention mechanisms or convolutions, just MLPs operating on different dimensions
        # this simplicity is what makes MLP-Mixer so interesting as an architectural design

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP Mixer on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--log_steps", type=int, default=100, help="Log metrics every log_steps steps")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--save_model", action="store_true", help="Save the final model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"using device: {device}")

    if args.wandb:
        if wandb is not None:
            # initialize wandb run, may need to set api key/account here 
            wandb.init(project="train_mlp_mixer", config=vars(args))
        else:
            print("wandb not installed, continuing without wandb logging")

    # setup data transforms with cifar10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    # these normalization values are the channel-wise mean and std of CIFAR-10
    # normalizing inputs to zero mean and unit variance helps training converge faster

    # load cifar10 datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # pin_memory=True gives a slight speed boost on GPU by making sure data is loaded into pinned memory
    # shuffling the training data helps prevent the model from learning spurious patterns from data order

    model = Mixer().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Calculate total steps for warmup/decay schedule
    total_steps = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10  # 10% of total steps for warmup
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9),
        weight_decay=args.lr  # Set weight decay equal to learning rate
    )
    # using the same value for weight decay and learning rate is an interesting choice from the paper
    # this creates a balance between optimization and regularization that scales with learning rate

    # details like hypers and lr schedule are taken straight from the paper, not crucial to understand
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps  
        return args.lr * (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))  
        # this implements a linear warmup followed by linear decay
        # warmup helps stabilize early training when gradients might be noisy
        # decay helps fine-tune the model as training progresses

    # training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        step_counter = 0
        start_time = time.time()
        for step, (images, labels) in enumerate(train_loader, 1):
            # Update learning rate
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                param_group['weight_decay'] = lr  # Update weight decay to match lr
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_counter += 1
            global_step += 1
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

            if step_counter % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                avg_acc = 100.0 * train_correct / train_total
                if args.verbose:
                    print(f"epoch [{epoch+1}/{args.epochs}], step [{step}/{len(train_loader)}], lr: {lr:.6f}, loss: {avg_loss:.4f}, acc: {avg_acc:.2f}%")
                if args.wandb and wandb is not None:
                    wandb.log({
                        "train_loss": avg_loss,
                        "train_acc": avg_acc,
                        "learning_rate": lr,
                        "epoch": epoch+1,
                        "step": global_step
                    })
                running_loss = 0.0
                train_total = 0
                train_correct = 0

        # evaluate on test set after each epoch
        model.eval()
        test_loss = 0.0
        total_test = 0
        correct_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_test += labels.size(0)
                correct_test += preds.eq(labels).sum().item()
        avg_test_loss = test_loss / total_test
        test_acc = 100.0 * correct_test / total_test
        print(f"epoch [{epoch+1}/{args.epochs}] test loss: {avg_test_loss:.4f}, test acc: {test_acc:.2f}%")
        if args.wandb and wandb is not None:
            wandb.log({"test_loss": avg_test_loss, "test_acc": test_acc, "epoch": epoch+1})
        
        if args.verbose:
            epoch_duration = time.time() - start_time
            print(f"epoch [{epoch+1}] completed in {epoch_duration:.2f} seconds")

    if args.save_model:
        torch.save(model.state_dict(), "mixer_final.pth")
        if args.verbose:
            print("saved final model as mixer_final.pth")

    if args.wandb and wandb is not None:
        wandb.finish()