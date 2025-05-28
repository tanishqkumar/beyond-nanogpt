from __future__ import annotations 
'''
self-play profiling notes 

max moves per game = 20, num sims = 512

0) baseline 
    Time spent in MCTS: 111.316s (99.9%)
    Time spent outside MCTS: 0.91s (0.1%)
    Total time: 110.164s

1) add torch.no_grad and amp.bf16 around mcts
    Time spent in MCTS: 110.062s (99.9%)
    Time spent outside MCTS: 0.102s (0.1%)
    Total time: 110.164s
    
2) batch expand fwd passes in mcts
    Game completed in 20 moves
    Time spent in MCTS: 5.206s (98.9%)
    Time spent outside MCTS: 0.057s (1.1%)
    Total time: 5.263s

    wow!! collating all individal model() calls into one big batched gemm made things much faster!

at this point: 
    MCTS_batched timing: Model Forward (GPU): 0.0013s (0.7%), Other (CPU): 0.1745s (99.3%), Total: 0.1758s
    MCTS_batched timing: Model Forward (GPU): 0.0107s (5.4%), Other (CPU): 0.1872s (94.6%), Total: 0.1979s

    which means we're CPU bound now by algorithmic operations in utils.py -- 
        so now, we should a) make those more efficient, vectorized/clever where possible and 
        b) add multithreading to make CPU ops 20x faster where possible 

    
3) vectorize python loops in utils

    - make empty return a constant 
        did nothing 
    - precache move2index by storing legal indices when a node created (so we don't have to in legal_mask)
        did nothing
    


'''
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Optional
import time
from model import ChessNet
from utils import board2input, legal_mask
from config import TrainConfig, MCTSConfig, MCTSNodeConfig
from MCTS import MCTS, MCTS_batched, Node, MCTS_batched_optimized
from env import ChessEnv
from buffer import Buffer


def self_play_game(
    env: ChessEnv, 
    model: ChessNet, 
    buffer: Buffer, 
    mcts_cfg: MCTSConfig, 
    selection_temp: float, 
    verbose: bool = False, 
):
    s = env.reset()
    data = []
    duration = 0
    done = False
    mcts_time = 0.0
    other_time = 0.0
    
    
    if verbose: print(f'Beginning a self-play game...')
    while not done and duration < mcts_cfg.max_game_duration: 
        normalize = bool(selection_temp == 0.)
        
        # Time MCTS
        start_time = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16): 
            # already normalized and legal inside 
            # mcts_logits = MCTS_batched(mcts_cfg, s, model, env, normalize=normalize, record_time=True) # mcts consists of [selection, expansion, simulation, backprop]
            mcts_logits = MCTS_batched_optimized(mcts_cfg, s, model, env, normalize=normalize, record_time=True) # mcts consists of [selection, expansion, simulation, backprop]
        mcts_time += time.time() - start_time
        
        # Time everything else
        start_time = time.time()

        if selection_temp == 0.:
            # Greedy selection - pick the action with highest probability
            mcts_policy = torch.zeros_like(mcts_logits)
            best_action = torch.argmax(mcts_logits)
            mcts_policy[best_action] = 1.0
        else:
            # Temperature-based selection
            mcts_policy = F.softmax(mcts_logits / selection_temp, dim=0)
        
        a_next = torch.multinomial(mcts_policy, num_samples=1, replacement=True).item()
        data.append([s, mcts_policy, None])
        s, r, done, _ = env.step(a_next)
        other_time += time.time() - start_time
        
        if duration % 10 == 0 and verbose: print(f'{duration} moves into the self play game...')
        if done: # about to exit, 
            if verbose: print(f'Game reached termination, winner is {env.board.turn}, moving on...')
            for _, (state, policy, _) in enumerate(data): 
                z = r if state.turn == env.board.turn else -r
                buffer.push(state, z, policy) # each mcts call contributes a single example 
        elif duration == mcts_cfg.max_game_duration - 1: 
            if verbose: print(f'Reached max game duration in self-play, calling it a draw and moving on!')
            for _, (state, policy, _) in enumerate(data): 
                z = 0  # draw
                buffer.push(state, z, policy) # each mcts call contributes a single example 
        duration += 1
    
    print(f"Game completed in {duration} moves")
    print(f"Time spent in MCTS: {mcts_time:.3f}s ({mcts_time/(mcts_time+other_time)*100:.1f}%)")
    print(f"Time spent outside MCTS: {other_time:.3f}s ({other_time/(mcts_time+other_time)*100:.1f}%)")
    print(f"Total time: {mcts_time+other_time:.3f}s")


if __name__ == "__main__":
    # Minimal setup for profiling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configs
    train_cfg = TrainConfig()
    mcts_cfg = MCTSConfig()
    
    # Setup model
    model = ChessNet(train_cfg.model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count / 1e6:.2f}M parameters")
    
    # Setup environment and buffer
    env = ChessEnv()
    buffer = Buffer(buff_sz=train_cfg.buff_sz)
    
    print("Starting self-play game for profiling...")
    self_play_game(env, model, buffer, mcts_cfg, train_cfg.selection_temp, verbose=True)
