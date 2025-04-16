import torch 
import torch.nn.functional as F
from torchtyping import TensorType as TT 
import torchvision
import torchvision.transforms as transforms
import ssl
from tqdm import tqdm

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


    def forward(self, x: TT["b", "s", "d"]) -> TT["b", "s", "d"]: # [B, S, D] -> [B, S, D]
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

class ClassificationHead(torch.nn.Module): 
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

if __name__ == "__main__":
    # Disable SSL verification (not recommended for production)
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load train and test datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

    # Create data loaders
    B = 16  # batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)

    # Initialize model and training parameters
    D = 512
    num_classes = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = ViT(D).to(device)
    optimizer = torch.optim.Adam(vit.parameters(), lr=1e-4)
    num_epochs = 10

    # Training loop
    train_losses = []
    test_accuracies = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        vit.train()
        total_loss = 0
        for batch in tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = vit(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        vit.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in testloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = vit(x)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%')