from __future__ import annotations 
import torch, torch.nn as nn, torch.nn.functional 
import chess 
from typing import List, Tuple, Optional, Dict, Any 
from collections import deque 
from utils import index2move, board2input, eval_pos, move2index
from copy import deepcopy 

class ChessEnv: 

    ## CORE METHODS ## 
    def __init__(self, board: Optional[chess.Board] = None, max_moves: int = 200, history_len: int = 8): 
        self.max_moves = max_moves 
        self.move_count = 0 
        self.history_len = history_len
        self.board_history = deque(maxlen=history_len)
        self.init_board = board.copy() if board else chess.Board()
        self.board = self.init_board.copy()
        self._update_history()

    def step(self, a_idx: int) -> Tuple[chess.Board, float, bool, Dict]: 
        a_move = index2move(self.board, a_idx) 

        if a_move not in self.board.legal_moves: 
            print("🛑 IllegalStep!", 
              " action index:", a_idx,
              " decoded move:", a_move.uci(),
              " board FEN:", self.board.fen(),
              " legal moves:", [m.uci() for m in self.board.legal_moves])
            raise Exception("Error: ChessEnv.step given an illegal move!")

        self.board.push(a_move)
        self.move_count += 1
        self._update_history()

        reward, done, info = eval_pos(self.board)
        
        # problem: we're returning a tensor when we want state to be a board in buffer 
        return (self.board.copy(), reward, done, info)

    # faster than deepcopying the whole thing, which adds overhead 
        # we call this many times in mcts, so it needs to be lightweight 
    def clone(self) -> ChessEnv: 
        new_env = ChessEnv(self.board, self.max_moves, self.history_len)
        new_env.board_history = deepcopy(self.board_history)
        new_env.move_count = self.move_count

        return new_env

    def reset(self, board: Optional[chess.Board] = None) -> chess.Board: 
        self.move_count = 0
        self.board = board if board else chess.Board()
        self.init_board = self.board.copy()
        self.board_history = deque(maxlen=self.history_len)
        self._update_history()
        return self.board.copy()

    ## SECONDARY/HELPER METHODS ## 
    def _update_history(self): 
        self.board_history.append(self.board.copy())

    def __repr__(self): 
        return f"ChessEnv(move_count={self.move_count}, turn={self.board.turn})"

    def __str__(self): 
        return str(self.board)


