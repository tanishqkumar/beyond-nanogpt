from __future__ import annotations
from dataclasses import dataclass, field
import chess
from collections import deque
from typing import Optional, TYPE_CHECKING

# trick to avoid circular import errors but still use Node type 
if TYPE_CHECKING: 
    from MCTS import Node 


@dataclass
class ModelConfig: 
    num_blocks: int = 4
    hidden_ch: int = 256 
    mtl: int = 119 
    nactions: int = 4672

@dataclass
class TrainConfig: 
    num_games_per_step: int = 100
    num_train_steps: int = 1_000 
    lr: float = 1e-4 
    lr_schedule: str = "cosine"

    bsz: int = 512
    verbose: bool = False 

    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    buff_sz: int = 100_000
    selection_temp: float = 0.7
    wd: float = 0.01

@dataclass
class MCTSNodeConfig: 
    state: chess.Board = field(default_factory=chess.Board)
    parent: Optional['Node'] = None
    action_taken: Optional[int] = None
    nactions: int = 4672
    history: deque[chess.Board] = field(default_factory=lambda: deque(maxlen=8))
    
@dataclass 
class MCTSConfig: 
    num_sims: int = 512
    noise_scale: float = 0.1 
    cpuct: float = 1.0 
    max_game_duration: int = 200
    batch_sz: int = 512