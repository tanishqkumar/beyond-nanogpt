import torch, torch.nn as nn, torch.nn.functional as F 
import chess
from typing import Tuple, List 
from utils import board2input

# force reimport utils to iterate faster when I make changes
import importlib
import sys
if "utils" in sys.modules:
    importlib.reload(sys.modules["utils"])
from utils import board2input

# first up projects channels, last down projects 
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False, last: bool = False, act = nn.GELU):
        super().__init__()
        self.in_ch = in_ch 
        self.out_ch = out_ch
        self.first = first 
        self.last = last 

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) # each [b, 8, 8] channel activations are projected onto sphere
        
        self.act = act()
        
        # Skip connection projection if input and output channels differ
        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_proj = None

    def forward(self, x): # [b, 8, 8, MT + L] -> [b, 8, 8, MT + L] for middle blocks
        og_x = x.clone()

        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        
        if self.skip_proj is not None:
            og_x = self.skip_proj(og_x)
            
        return self.act(h + og_x)


class ChessNet(nn.Module): 
    def __init__(self, nblocks: int = 6, in_ch:int = 119, action_dim: int = 4672): 
        super().__init__()
        self.nblocks = nblocks
        self.action_dim = action_dim
        self.hidden_ch = 128 # hidden layers constant feature dim 
        self.in_ch = in_ch
        self.out_ch = in_ch
        self.hidden_dim_flat = 8 * 8 * in_ch
    
        self.first = ResBlock(in_ch=self.in_ch, out_ch=self.hidden_ch, first=True)
        self.hidden_blocks = nn.ModuleList([
            ResBlock(in_ch=self.hidden_ch, out_ch=self.hidden_ch) for _ in range(nblocks-2) 
        ])
        self.last = ResBlock(in_ch=self.hidden_ch, out_ch=self.out_ch, last=True)


        self.value_head = nn.Linear(self.hidden_dim_flat, 1) # 8 * 8 * (MT + L) -> 1, applied across batch dim 
        self.policy_head = nn.Linear(self.hidden_dim_flat, action_dim)
    
    def mask_legal(self, logits: torch.Tensor, board_history: List[chess.Board]) -> torch.Tensor: 
        # TODO: this sets logits at illegal indices to float('-inf')
        return logits

    def board2input(self, board_history: List[chess.Board]) -> torch.Tensor: 
        return board2input(board_history) 
    
    def forward(self, board_history: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor]: # [b, mtl, 8, 8] -> [[b, 1], [b, nactions]]
        x = self.board2input(board_history).unsqueeze(0) # add batchdim

        h = self.first(x)
        for block in self.hidden_blocks: 
            h = block(h) # residual is handled inside block 
        h = self.last(h)
        h_flat = h.view(h.size(0), -1)  # flatten for linear layers

        values, logits = self.value_head(h_flat), self.policy_head(h_flat)
        # TODO: mask illegal moves using chess infra (will have to understand representation)

        return values, self.mask_legal(logits, board_history)
        
# simple test to tune defaults and sanity check, want it to have 10-20m params by default
if __name__ == "__main__":
    print(f'Initializing ChessNet...')

    mtl = 119 # in channels, M*T + L, see paper on how they represent a cheess board 
        # to give it as input into their alphazero conv neural net 
    nactions = 4672

    net = ChessNet(in_ch = mtl)
    num = sum(p.numel() for p in net.parameters())
    print(f'Net has {num/1e6:.1f}M parameters!')

    print(f'Testing forward pass with real chess board...')
    board = chess.Board()
    history = [board]
    v, l = net(history)
    assert v.shape == (1, 1) and l.shape == (1, nactions)

    v, l = net(board)
    assert v.shape == (1, 1) and l.shape == (1, nactions)
    print(f'Forward test passed with real input!')
    print(f'--' * 20)
    print(f'ALL TESTS PASSED.')