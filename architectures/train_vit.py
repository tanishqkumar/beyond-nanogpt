'''
This implements "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(https://arxiv.org/pdf/2010.11929)

The core idea here is that TRANSFORMERS SCALE, so that simply embedding patches of images and 
adding a classification head on a transformer allows it to be competitive and even beat convNets
at large scale (though not at small scale, since inductive bias in ConvNets acts as effective compute). 
'''


import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import ssl
from tqdm import tqdm
import argparse
import wandb
import time


#### SAME AS VANILLA TRANSFORMER ####
class Attention(torch.nn.Module): 
    def __init__(self, D: int = 512, head_dim: int = 64):
        super().__init__()
        self.D = D
        self.wq = torch.nn.Linear(D, D)
        self.wk = torch.nn.Linear(D, D)
        self.wv = torch.nn.Linear(D, D)
        self.wo = torch.nn.Linear(D, D)
        self.head_dim = head_dim 
        assert self.D % self.head_dim == 0 
        self.nheads = self.D // self.head_dim 


    def forward(self, x): # [B, S, D] -> [B, S, D]
        B, S, D = x.shape 
        q, k, v = self.wq(x), self.wk(x), self.wv(x) # [B, S, D]
        q = q.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]
        k = k.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]
        v = v.reshape(B, S, self.nheads, self.head_dim).transpose(1, 2) # [B, nh, S, hd]

        scale = torch.sqrt(torch.tensor(self.head_dim))
        logits = torch.einsum('bnik,bnjk->bnij', q, k) / scale # [B, nh, S, S]
                
        A = F.softmax(logits, dim=-1)
        preout = torch.einsum('bnij,bnjd->bnid', A, v) # [B, nh, S, hd]
        preout = preout.transpose(1, 2).reshape(B, S, D) # [B, S, D]

        return self.wo(preout)


class MLP(torch.nn.Module):
    def __init__(self, D=512, multiplier=4): 
        super().__init__()
        self.multiplier = multiplier
        self.wup = torch.nn.Linear(D, multiplier * D)
        self.wdown = torch.nn.Linear(multiplier * D, D)
        self.act = torch.nn.GELU()
    
    def forward(self, x): 
        return self.wdown(self.act(self.wup(x)))

class LN(torch.nn.Module): # BSD -> BSD, just subtracts mean and rescales by std of tensor
    def __init__(self, D=512, eps=1e-9):
        super().__init__()
        self.D = D
        self.mean_scale = torch.nn.Parameter(torch.ones(D))
        self.std_scale = torch.nn.Parameter(torch.ones(D))
        self.eps = eps 

    def forward(self, x): # BSD -> BSD 
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.std_scale * ((x - mean)/(std + self.eps)) + self.mean_scale

#### DIFFERENT FROM VANILLA TRANSFORMER, now vision-specific ####

class PatchEmbeddings(torch.nn.Module): # (b, ch, h, w) -> (b, np+1, d)
    def __init__(self, D=512, t=16, h=32, w=32, ch=3): # (b, ch, h, w) -> (b, np, t, t, ch) -> (b, np, t * t * ch) -> (b, np, d) -> (b, np+1, d)
        super().__init__()
        self.t = t
        self.D = D 
        self.num_patches = (h//t) * (w//t)
        self.ch = ch 
        self.h = h 
        self.w = w 
        self.patch_proj = torch.nn.Linear(self.t*self.t*ch, D)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, D))
        self.pos_embeds = torch.nn.Parameter(torch.randn(1, self.num_patches+1, D)) # learnable D-vector at each [b, np+1]

    def forward(self, x): # (b, ch, h, w) -> (b, np+1, d) 
        B, _, _, _ = x.shape # batch of images 
        x = x.permute(0, 2, 3, 1).reshape(B, self.num_patches, self.t*self.t*self.ch)  # (b, ch, h, w) -> (b, h, w, ch) -> (b, np, t*t*ch), ie. b*np patches
        x = self.patch_proj(x) # t*t*ch -> D on last dim of x, so we have [b, np, d]
        
        # now we want to 1) add positional embeddings and 2) prepend a CLS token
        cls_tokens = self.cls_token.expand(B, 1, self.D)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embeds
        return x

class ClassificationHead(torch.nn.Module): # remember, the ultimate goal is not generation, but classificaiton here 
    def __init__(self, D=512, multiplier=4, num_classes=100):
        super().__init__()
        self.D = D
        self.w1 = torch.nn.Linear(D, multiplier * D)
        self.w2 = torch.nn.Linear(multiplier * D, num_classes)
        self.act = torch.nn.GELU()
    
    def forward(self, x): # [b, np+1, D] -> [b, D] -> [b, num_classes] of logits-vector for each 
        x = x[:, 0, :]
        return self.w2(self.act(self.w1(x)))

class ViTLayer(torch.nn.Module):
    def __init__(self, D=512):
        super().__init__()
        self.attn = Attention(D)
        self.mlp = MLP(D)
        self.ln1 = LN(D)
        self.ln2 = LN(D)
    
    def forward(self, x):
        h = self.attn(self.ln1(x)) + x
        h = self.mlp(self.ln2(h)) + h
        return h
    
class ViT(torch.nn.Module):
    def __init__(self, D=512, nlayers=6, num_classes=100): 
        super().__init__()
        self.D = D 
        self.nlayers = nlayers 
        self.embed = PatchEmbeddings(D)
        self.classify = ClassificationHead(D, num_classes)
        self.layers = torch.nn.ModuleList([ViTLayer(D) for _ in range(nlayers)])

    def forward(self, x): # (b, ch, h, w) -> (b, num_classes) of logits for each batch element
        h = self.embed(x)
        for layer in self.layers: 
            h = layer(h) 
        logits = self.classify(h)
        return logits

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")

    if args.wandb:
        wandb.init(project="train_vit")
        wandb.config.update(args)

    # get dataset
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # init model/opt, standard stuff
    vit = ViT(args.D, args.nlayers, num_classes=100).to(device)
    optimizer = torch.optim.Adam(vit.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        vit.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        num_train_batches = 0

        if args.verbose:
            train_iter = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            train_iter = trainloader

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = vit(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            num_train_batches += 1

            if args.verbose:
                train_iter.set_description(f"Train Loss: {total_loss/num_train_batches:.4f} | Acc: {100.*train_correct/train_total:.2f}%")

        epoch_train_loss = total_loss / num_train_batches
        epoch_train_acc = 100. * train_correct / train_total

        if args.wandb:
            wandb.log({"epoch": epoch+1, "train_loss": epoch_train_loss, "train_acc": epoch_train_acc})

        if (epoch + 1) % max(1, int(args.epochs * args.test_every)) == 0 or epoch == args.epochs - 1:
            vit.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            num_test_batches = 0

            with torch.no_grad():
                if args.verbose:
                    test_iter = tqdm(testloader, desc="Validation")
                else:
                    test_iter = testloader

                for x, y in test_iter:
                    x, y = x.to(device), y.to(device)
                    logits = vit(x)
                    loss = F.cross_entropy(logits, y)
                    
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    test_total += y.size(0)
                    test_correct += predicted.eq(y).sum().item()
                    num_test_batches += 1

                    if args.verbose:
                        test_iter.set_description(f"Val Loss: {test_loss/num_test_batches:.4f} | Acc: {100.*test_correct/test_total:.2f}%")

            test_acc = 100. * test_correct / test_total
            test_loss = test_loss / num_test_batches

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_train_loss:.4f} | "
                  f"Train Acc: {epoch_train_acc:.2f}% | Val Loss: {test_loss:.4f} | "
                  f"Val Acc: {test_acc:.2f}%")

            if args.wandb:
                wandb.log({"epoch": epoch+1, "val_loss": test_loss, "val_acc": test_acc})

            if test_acc > best_acc:
                best_acc = test_acc
                if args.save_model:
                    torch.save(vit.state_dict(), "vit_best.pth")
                    if args.verbose:
                        print(f"Saved best model with accuracy: {best_acc:.2f}%")

    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")

    if args.save_model:
        torch.save(vit.state_dict(), "vit_final.pth")
        if args.verbose:
            print("Saved final model")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT from scratch")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--D", type=int, default=512, help="Model dimension")
    parser.add_argument("--nlayers", type=int, default=6, help="Number of transformer layers")
    
    # Dataset parameters
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    
    # Misc parameters
    parser.add_argument("--test_every", type=float, default=0.1, help="Fraction of epochs to test validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    start_time = time.time()
    train(args)
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time/60:.2f} minutes")