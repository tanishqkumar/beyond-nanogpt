from __future__ import annoations 
import torch, torch.nn as nn, torch.nn.functional as F
import chess 
from dataclasses import dataclass 
from typing import List, Tuple, Optional
from model import ChessNet
from utils import board2input
from config import TrainConfig, MCTSConfig

'''

# need to add ucb computation/logic and node/mcts defn 
# include dirichlet noise + temp in mcts 

# this fn should self play a game either to end or max depth and 
# should write examples to buffer, one per step 
def self_play(env, model, buffer, num_mcts_sims): 
    s = env.reset()
    done = False
    data = []

    while not done: 
        mcts_policy = mcts(s, env.clone()) # mcts consists of [selection, expansion, simulation, backprop]
        a_next = torch.argmax(mcts_policy)
        s_next, r, done = env.step(a_next)
        data.append([s_next, mcts_policy, None])

        if done: # about to exit, 
            for i, (s, p, _) in enumerate(data): 
                z = r if s.board.turn == env.board.turn else -r
                buffer.push([s, z, p]) # each mcts call contributes a single example 



def train(
        train_cfg: TrainConfig = TrainConfig(), 
        mcts_cfg: MCTSConfig = MCTSConfig(), 
    ): 

    # setup nets and optimizers
    model = ChessNet()
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    buffer = Buffer()

    for step in train_cfg.num_train_steps: 

        for rollout_idx in train_cfg.num_rollouts: 
            # this does mcts for num_sims, pushes a single example to Buffer
            self_play(env.clone(), model, buffer, train_cfg.num_mcts_sims) 

        # train networks using buffer data
        state_batch, action_batch, val_batch = buffer.get_batch(train_cfg.bsz)
        state_batch_t = torch.stack(list(map(board2input, state_batch))) # state_batch is a list of boards 
        vals, logits = model(state_batch)
        loss = F.cross_entropy_loss(logits, action_batch) + F.mse_loss(vals, val_batch)
        loss.backward()
        opt.step()
        opt.zero_grad()



'''