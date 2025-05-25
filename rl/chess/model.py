import torch, torch.nn as nn, torch.nn.functional as F 
import chess
from typing import Tuple, List 
from utils import board2input, legal_mask, move2index

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
    
    def forward(self, board_history: List[chess.Board], nactions: int = 4672) -> Tuple[torch.Tensor, torch.Tensor]: # [b, mtl, 8, 8] -> [[b, 1], [b, nactions]]
        if not isinstance(board_history, list): 
            board_history = [board_history]

        x = board2input(board_history).unsqueeze(0) # add batchdim

        h = self.first(x)
        for block in self.hidden_blocks: 
            h = block(h) # residual is handled inside block 
        h = self.last(h)
        h_flat = h.view(h.size(0), -1)  # flatten for linear layers

        values, logits = self.value_head(h_flat), self.policy_head(h_flat)

        return values, legal_mask(logits, board_history[-1], nactions)
        
