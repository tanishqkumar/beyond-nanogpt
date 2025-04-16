'''
Deep Residual Learning for Image Recognition
(https://arxiv.org/pdf/1512.03385)

Legendary paper, IMO, for an overlooked reason. I think most people don't get that ResNets are not always a better
arch than contemporary vision SOTA eg VGG, the key different is that they SCALE better, eg. a ResNet18 is not really much better
than a similarly sized VGG, but a ResNet50 IS much better than a VGG of that size. This was one of the first 
"do simple things that scale" type innovations, not to mention the ResNet is the workhorse for image-recognition
genuinely in production today at most companies who need this (which is a lot more than you'd think). Figure 4
epitomizes the scaling point of view. 

Also instructive because I think implementing an efficient Conv layer using the unfold trick to turn 
serial convolution into a batch matmul is extremely helpful to learn how to "think in tensors and matmul" which 
is ultimately key to going from a DL beginner to researcher. 

'''


import torch 
import torch.nn as nn 
from datasets import load_dataset
import wandb
from torchvision import transforms
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm

class Conv(nn.Module): # we write our own Conv class to understand the key unfold trick underlying nn.Conv2D
    def __init__(self, ch_in=3, ch_out=3, size=3, stride=1):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out 
        self.size = size
        self.stride = stride
        self.padding = (size-1)//2
        self.weights = nn.Parameter(torch.randn(ch_in * size * size, ch_out)) 

    def forward(self, x): 
        """
        The F.unfold trick that turns a convolution into a batched matmul is CRITICAL to understand here. 
        The idea is that 1) a single conv can be written as a dot product/matmul, and that 2) 
        each conv operation is independent, so that (1 + 2) a conv layer can be computed via a batched matmul 
        which of course is exactly what GPUs are built for. 

        Input shape after unfold: [batch, ch_in * kernel_size * kernel_size, h_out * w_out]
        Conv weights shape: [ch_out, ch_in, kernel_size, kernel_size]
        Reshape weights to: [ch_out, ch_in * kernel_size * kernel_size]
        Matrix multiply: [batch, h_out * w_out, ch_in * k * k] @ [ch_in * k * k, ch_out]
        Final shape: [batch, h_out * w_out, ch_out] -> transpose to [batch, ch_out, h_out * w_out]
        """
        b, _, h, w = x.shape 
        
        h_out = (h - self.size + 2*self.padding)//self.stride + 1
        w_out = (w - self.size + 2*self.padding)//self.stride + 1

        ## important to understand what this is doing
        unfolded = F.unfold(x, 
                           kernel_size=self.size, 
                           stride=self.stride, 
                           padding=self.padding) # [b, ch_in * self.size * self.size, h_out * w_out]
        
        out = unfolded.transpose(1,2) @ self.weights # [b, h_out*w_out, ch_out]
        ## end important to understand

        return out.transpose(1,2).reshape(b, self.ch_out, h_out, w_out) # [b, ch_out, h_out, w_out]
    

'''
these days everyone uses LayerNorm in eg. decoder-based transformers
so implementing BatchNorm is quite instructive to see how things are different in vision/"the old days"
plus BN is weird in that it uses batch statistics so is data-dependent in that way 
which requires keeping track of non-trainable statistics in buffers, a useful exercise 
since buffers are used often to track optimizer state and in other ways 

it might seem tempting to gloss over normalization as just a "implementation detail" but actually 
details like these can COMPLETELY change whether an architecture trains stably vs not, and thus whether 
it is widely adopted or not. do NOT give into the temptation. re-implement this from scratch. 
'''
class BatchNorm(nn.Module): # [b, ch, h, w]
    def __init__(self, ch=3, eps=1e-5, mom=0.1):
        super().__init__()
        self.shifts = nn.Parameter(torch.zeros(ch))
        self.scales = nn.Parameter(torch.ones(ch))
        self.eps = eps
        self.momentum = mom 
        self.register_buffer('running_mean', torch.zeros_like(self.shifts))
        self.register_buffer('running_var', torch.ones_like(self.scales))

    def forward(self, x):
        # compute stats 
        curr_means = torch.mean(x, dim=(0, 2, 3)) # [ch]
        curr_vars = torch.var(x, dim=(0, 2, 3)) # [ch]

        # update buffers 
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * curr_means 
        self.running_var = self.momentum * self.running_var + (1-self.momentum) * curr_vars 
        
        # normalize with our stats
        if not self.training: # inference-time, use our stats averaged over training
            reshaped_mean = self.running_mean.reshape(1, -1, 1, 1)
            reshaped_var = self.running_var.reshape(1, -1, 1, 1)
            x_norm = (x - reshaped_mean) / torch.sqrt(reshaped_var + self.eps)
        else: # train, use live stats
            reshaped_mean = curr_means.reshape(1, -1, 1, 1)
            reshaped_var = curr_vars.reshape(1, -1, 1, 1)
            x_norm = (x - reshaped_mean) / torch.sqrt(reshaped_var + self.eps)
        
        shifts = self.shifts.reshape(1, -1, 1, 1)
        scales = self.scales.reshape(1, -1, 1, 1)
        return shifts + scales * x_norm # [b, ch, h, w]
    
class MaxPool2D(nn.Module):
    """
    Standard MaxPool2D layer used in ResNet18:
    - kernel_size: 3x3
    - stride: 2 
    - padding: 1
    Used after initial conv layer to reduce spatial dimensions
    
    Input shape: [batch, channels, height, width]
    Output shape: [batch, channels, height//2, width//2]
    """    
    def __init__(self, stride=2, padding=1, size=3):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.size = size
        
    def forward(self, x):        
        b, ch_in, h, w = x.shape 
        
        h_out = (h - self.size + 2*self.padding)//self.stride + 1 # think visually and this makes sense 
        w_out = (w - self.size + 2*self.padding)//self.stride + 1
        unfolded = F.unfold(x, # unfold already gives us all the weights that would be dotted with K in conv 
                           kernel_size=self.size, # so we can just max over each 
                           stride=self.stride, 
                           padding=self.padding) # [b, ch_in * self.size * self.size, h_out * w_out]
        unfolded = unfolded.reshape(b, ch_in, self.size * self.size, h_out * w_out)
        return torch.max(unfolded, dim=2)[0].reshape(b, ch_in, h_out, w_out) # [b, ch_in, h_out, w_out], the [0] is because max returns (vals, indices)
        

# from here on out, the implementation details (eg. constants) come from the ResNet paper 
class ResBlock(nn.Module):
    """
    Standard ResNet18 residual block:
    - Two 3x3 conv layers with same number of channels
    - Each conv followed by batch norm and ReLU
    - Skip connection adds input to final output
    - If downsample=True:
        - First conv uses stride=2 to halve spatial dims
        - Skip connection uses 1x1 conv with stride=2 to match dims
    - If downsample=False:
        - Both convs use stride=1
        - Skip connection is identity
        
    Input shape: [batch, channels_in, height, width]
    Output shape: 
        If downsample=False: [batch, channels_in, height, width]
        If downsample=True: [batch, channels_out, height//2, width//2]
    """
    def __init__(self, ch_in, ch_out, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.act = nn.ReLU()
        
        stride = 2 if downsample else 1
        
        self.layer1 = Conv(ch_in=ch_in, ch_out=ch_out, stride=stride)
        self.bn1 = BatchNorm(ch=ch_out)
        
        self.layer2 = Conv(ch_in=ch_out, ch_out=ch_out, stride=1) 
        self.bn2 = BatchNorm(ch=ch_out)
        
        if downsample:
            self.skip_conv = Conv(ch_in=ch_in, ch_out=ch_out, stride=2, size=1)
        else:
            self.skip_conv = lambda x: x
            
    def forward(self, x):
        skip = self.skip_conv(x)
        h = self.bn1(self.act(self.layer1(x)))
        h = self.bn2(self.act(self.layer2(h)))
        return self.act(skip + h)
    
class ResNet18(nn.Module):
    def __init__(self, ch=3, h=224, w=224, num_classes=10):
        # resnet18 architecture from the paper. takes in images of shape [b, ch, h, w] and outputs class logits [b, num_classes]
        super().__init__()
        self.ch_in = ch
        self.size = h

        # first conv block reduces spatial dims by 4x:
        # (b, 3, 224, 224) -> (b, 64, 56, 56) through:
        # - 7x7 conv with stride 2: halves spatial dims
        # - 3x3 maxpool with stride 2: halves spatial dims again
        self.conv1 = Conv(ch_in=ch, ch_out=64, size=7, stride=2)
        self.bn1 = BatchNorm(ch=64)
        self.act = nn.ReLU()
        self.maxpool1 = MaxPool2D(size=3)

        # layer1: two residual blocks without downsampling
        # maintains (b, 64, 56, 56) dimensions
        self.layer1 = nn.Sequential(
            ResBlock(ch_in=64, ch_out=64, downsample=False),
            ResBlock(ch_in=64, ch_out=64, downsample=False),
        )

        # layer2: first block downsamples spatially and doubles channels
        # (b, 64, 56, 56) -> (b, 128, 28, 28)
        self.layer2 = nn.Sequential(
            ResBlock(ch_in=64, ch_out=128, downsample=True),
            ResBlock(ch_in=128, ch_out=128, downsample=False),
        )

        # layer3: first block downsamples spatially and doubles channels
        # (b, 128, 28, 28) -> (b, 256, 14, 14)
        self.layer3 = nn.Sequential(
            ResBlock(ch_in=128, ch_out=256, downsample=True),
            ResBlock(ch_in=256, ch_out=256, downsample=False),
        )

        # layer4: first block downsamples spatially and doubles channels
        # (b, 256, 14, 14) -> (b, 512, 7, 7)
        self.layer4 = nn.Sequential(
            ResBlock(ch_in=256, ch_out=512, downsample=True),
            ResBlock(ch_in=512, ch_out=512, downsample=False),
        )

        # final classifier:
        # - global average pool to remove spatial dims: (b, 512, 7, 7) -> (b, 512)
        # - linear layer to get class logits: (b, 512) -> (b, num_classes)
        self.avgpool = lambda x: torch.mean(x, dim=(2, 3))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.maxpool1(self.act(self.bn1(self.conv1(x))))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        out = self.classifier(self.avgpool(h))  # out is [b, num_classes]
        return out

# imagine being so famous you got an initialization scheme named after you lol
# >feelsKaimingMan
def initialize_weights(model): 
    """Initialize model weights using Kaiming (He) initialization"""
    for m in model.modules():
        if isinstance(m, Conv):
            nn.init.kaiming_normal_(m.weights, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, BatchNorm):
            nn.init.constant_(m.shifts, 0)
            nn.init.constant_(m.scales, 1)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"Using device: {device}")
    
    if args.wandb:
        # TODO. you may need to configure your api_key/account here
        wandb.init(project="train_resnet") 
        wandb.config.update(args)
    
    if args.verbose:
        print("Loading dataset...")
    
    transform = transforms.Compose([
        transforms.Resize(224),  # resize to match ResNet input size
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 normalization
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = load_dataset("cifar10", split="train", streaming=True)
    val_dataset = load_dataset("cifar10", split="test", streaming=True)
    
    # standard vision dataloader stuff, understanding composed transforms is
    # not too important these days (2025) but make sure to understand dimensions  
    def collate_fn(batch):
        images = torch.stack([transform(sample['img'].convert('RGB')) for sample in batch])
        labels = torch.tensor([sample['label'] for sample in batch])
        return images, labels
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset.take(args.train_size),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset.take(args.val_size),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = ResNet18(num_classes=10)
    initialize_weights(model) # initialization scheme comes from ResNet paper 
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma
    )
    
    # training and val loop, pretty standard, more important to understand the arch carefully 
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        num_train_batches = 0
        
        if args.verbose:
            print(f"Epoch {epoch+1}/{args.epochs}")
            train_iter = tqdm(train_loader)
        else:
            train_iter = train_loader
            
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            
            # fwd
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # bwd + step 
            optimizer.zero_grad()
            loss.backward()
            
            # clipping is an important detail for stable training 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # log statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            num_train_batches += 1
            
            if args.verbose:
                train_iter.set_description(f"Train Loss: {train_loss/num_train_batches:.4f} | Acc: {100.*train_correct/train_total:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss and accuracy
        epoch_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        epoch_train_acc = 100.*train_correct/train_total if train_total > 0 else 0
        
        if args.wandb:
            wandb.log({"epoch": epoch+1, "train_loss": epoch_train_loss, "train_acc": epoch_train_acc})
        
        # test every so often 
        if (epoch + 1) % max(1, int(args.epochs * args.test_every)) == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            num_val_batches = 0
            
            with torch.no_grad():
                if args.verbose:
                    val_iter = tqdm(val_loader, desc="Validation")
                else:
                    val_iter = val_loader
                    
                for images, labels in val_iter:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    num_val_batches += 1
                    
                    if args.verbose:
                        val_iter.set_description(f"Val Loss: {val_loss/num_val_batches:.4f} | Acc: {100.*val_correct/val_total:.2f}%")
            
            # Calculate validation metrics
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
            
            # Print statistics
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_train_loss:.4f} | "
                  f"Train Acc: {epoch_train_acc:.2f}% | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
            
            # Log validation metrics
            if args.wandb:
                wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_acc": val_acc})
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                if args.save_model:
                    torch.save(model.state_dict(), "resnet18_best.pth")
                    if args.verbose:
                        print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
    
    # save final model
    if args.save_model:
        torch.save(model.state_dict(), "resnet18_final.pth")
        if args.verbose:
            print("Saved final model")
    
    # Finish wandb run
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 from scratch")
    
    # training params 
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    # yes, I too (an LLM person), was surprised at how high ^this^ LR is, but indeed it's common in vision 
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    # also notice how SGD is preferred in vision compared to Adam! 
    # there are interesting papers studying where this asymmetry comes from, eg. 
    # see eg. "Why are Adaptive Methods Good for Attention Models?" (https://arxiv.org/abs/1912.03194)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD optimizer")
    parser.add_argument("--lr_step", type=int, default=30, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")
    
    # dataset params
    parser.add_argument("--train_size", type=int, default=50000, help="Number of training samples to use")
    parser.add_argument("--val_size", type=int, default=10000, help="Number of validation samples to use")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    
    # misc params
    parser.add_argument("--test_every", type=float, default=0.1, help="Fraction of epochs to test validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # train 
    start_time = time.time()
    train(args)
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time/60:.2f} minutes")