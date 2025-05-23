import torch, torch.nn as nn, torch.nn.funcational as F
from typing import List, Tuple, Optional 
import chess

# converts a python-chess Board object to something our conv arch can take as a fwd 
def state2input(board_state: chess.Board):
    pass 