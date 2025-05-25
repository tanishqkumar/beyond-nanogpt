import torch, torch.nn as nn, torch.nn.functional as F
import chess 
from dataclasses import dataclass 
from __future__ import annoations 
from typing import List, Tuple, Optional
from model import ChessNet
from utils import board2input

# these should be moved to config.py eventually 
@dataclass
class TrainConfig: 
    num_rollouts: int = 100
    num_train_steps: int = 1_000 

    lr: float = 1e-4 

    value_width: int = 256
    value_depth: int = 4
    
    policy_width: int = 256
    policy_depth: int = 4

    bsz: int = 512
    verbose: bool = False 


@dataclass 
class MCTSConfig: 
    num_sims: int = 100 
    max_tree_depth: int = 20 
    max_rollout_depth: int = 20 

'''
def train(
        train_cfg: TrainConfig = TrainConfig(), 
        mcts_cfg: MCTSConfig = MCTSConfig(), 
    ): 

    # setup nets and optimizers
    model = ChessNet()
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    buffr = Buffer()

    for step in train_cfg.num_train_steps: 

        for rollout_idx in train_cfg.num_rollouts: 
            self-play a game (to either completion or max rollout depth) using MCTS and our env
            push 

        


need 
MCTS: s -> a
chess env wrapping board (stateful): a -> (s_next, r, done)
buffer storing [board, val] and [board, action]

'''