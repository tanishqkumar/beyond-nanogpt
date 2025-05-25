import chess 
import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
from typing import Tuple, List, Optional 
from utils import board2input

class Buffer: 
    def __init__(self, buff_sz: int = 1_000_000): 
        # buff stores [state, value, action]
        # value is scalar, action is [nactions]
        self.state_buff = deque(maxlen=buff_sz)
        self.val_buff = deque(maxlen=buff_sz)
        self.action_buff = deque(maxlen=buff_sz)

    def push(self, state: chess.Board, val: float, action: torch.Tensor): 
        self.state_buff.append(state) 
        self.val_buff.append(val)
        self.action_buff.append(action) # [nactions]

    def get_batch(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        L = len(self.state_buff)
        if bsz > L: 
            bsz = L
        
        indices = torch.randint(0, L, (bsz,))
        s = torch.stack([board2input(self.state_buff[i]) for i in indices]) # [b, 119, 8, 8]
        v = torch.tensor([self.val_buff[i] for i in indices], dtype=torch.float32).view(-1, 1)
        a = torch.stack([self.action_buff[i] for i in indices])

        return s, v, a
    
    def __len__(self): 
        return len(self.state_buff)
        

        