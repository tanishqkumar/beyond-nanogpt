from __future__ import annotations 
import torch, torch.nn as nn, torch.nn.functional as F
import chess 
from dataclasses import dataclass 
from typing import List, Tuple, Optional
import argparse
from model import ChessNet
from utils import board2input, legal_mask
from config import TrainConfig, MCTSConfig
from MCTS import MCTS
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
    if verbose: print(f'Beginning a self-play game...')
    while not done and duration < mcts_cfg.max_game_duration: 
        normalize = bool(selection_temp == 0.)
        mcts_logits = MCTS(mcts_cfg, s, model, env, normalize=normalize) # mcts consists of [selection, expansion, simulation, backprop]
        mcts_logits = legal_mask(mcts_logits.unsqueeze(0), [env.board]).squeeze(0)
        
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
        
        if duration % 10 == 0 and verbose: print(f'{duration} moves into the self play game...')
        if done: # about to exit, 
            if verbose: print(f'Game reached termination, winner is {state.turn}, moving on...')
            for _, (state, policy, _) in enumerate(data): 
                z = r if state.turn == env.board.turn else -r
                buffer.push(state, z, policy) # each mcts call contributes a single example 
        elif duration == mcts_cfg.max_game_duration - 1: 
            if verbose: print(f'Reached max game duration in self-play, calling it a draw and moving on!')
            for _, (state, policy, _) in enumerate(data): 
                z = 0  # draw
                buffer.push(state, z, policy) # each mcts call contributes a single example 
        duration += 1

def train(
        train_cfg: TrainConfig = TrainConfig(), 
        mcts_cfg: MCTSConfig = MCTSConfig(),
        verbose: bool = False,
        wandb: bool = False,
    ): 

    torch.manual_seed(69)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    if wandb:
        import wandb
        wandb.init(project="chess-alphazero", config={
            "train_cfg": train_cfg.__dict__,
            "mcts_cfg": mcts_cfg.__dict__,
            "device": str(device),
            "lr_schedule": train_cfg.lr_schedule
        })

    # setup nets and optimizers
    model = ChessNet(train_cfg.model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.wd)
    
    # Setup learning rate scheduler
    if train_cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.num_train_steps)
    elif train_cfg.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=train_cfg.num_train_steps)
    else:
        raise ValueError(f"Unsupported lr_schedule: {train_cfg.lr_schedule}. Use 'cosine' or 'linear'.")
    
    buffer = Buffer(buff_sz=train_cfg.buff_sz)
    env = ChessEnv()

    if verbose:
        print(f"Starting training with {train_cfg.num_train_steps} steps")
        print(f"Model config: {train_cfg.model_cfg}")
        print(f"MCTS config: {mcts_cfg}")
        print(f"LR schedule: {train_cfg.lr_schedule}")

    for step in range(train_cfg.num_train_steps): 

        # if verbose and step % 100 == 0:
        print(f"Step {step}/{train_cfg.num_train_steps}")

        for game_idx in range(train_cfg.num_games_per_step): 
            # this does mcts for num_sims at each step of the game, pushes single example to Buffer
            self_play_game(env.clone(), model, buffer, mcts_cfg, train_cfg.selection_temp, verbose)
            
            if verbose: #  and step % 100 == 0 and game_idx % 10 == 0
                print(f"  Completed rollout {game_idx+1}/{train_cfg.num_games_per_step}, buffer size: {buffer.size}")

        # train networks using buffer data
        state_batch, val_batch, action_batch, board_list = buffer.get_batch(train_cfg.bsz, device=device)
        # action batch is [b, nactions]
        action_batch_indices = torch.argmax(action_batch, dim=1) # [b]

        vals, logits = model(state_batch) 
        
        masked_logits = legal_mask(logits, board_list) # returns [b, nactions]
        
        loss = F.cross_entropy(masked_logits, action_batch_indices) + F.mse_loss(vals.squeeze(), val_batch.squeeze())
        loss.backward()
        opt.step()
        opt.zero_grad() 
        
        scheduler.step()

        if verbose and step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Training loss: {loss.item():.6f}, LR: {current_lr:.6f}")

        if wandb:
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "buffer_size": buffer.size,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        # TODO: once train working, compute win rate against random policy 
        # if step % train_cfg.eval_every == 0: 
        #     elo = eval(model)
        #     if verbose:
        #         print(f'Model has elo {elo} at step {step}')
        #     if wandb:
        #         wandb.log({"elo": elo})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train AlphaZero on Chess')
    # fix these 
    parser.add_argument('--num-games-per-step', type=int, default=3, help='Number of rollouts per training step')
    parser.add_argument('--num-train-steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--bsz', type=int, default=256, help='Batch size')
    parser.add_argument('--num-mcts-sims', type=int, default=50, help='Number of MCTS simulations')
    parser.add_argument('--buff-sz', type=int, default=10_000, help='Buffer size')
    parser.add_argument('--max-game-duration', type=int, default=100, help='Buffer size')
    parser.add_argument('--selection-temp', type=float, default=0.7, help='Temperature for selection sampling')
    parser.add_argument('--cpuct', type=float, default=1.0, help='CPUCT parameter for MCTS')
    parser.add_argument('--noise-scale', type=float, default=0.1, help='Noise scale for MCTS root')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight deacy for training')
    parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['cosine', 'linear'], help='Learning rate schedule')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    args = parser.parse_args()
    
    train_cfg = TrainConfig(
        num_games_per_step=args.num_games_per_step,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        bsz=args.bsz,
        buff_sz=args.buff_sz,
        verbose=args.verbose,
        selection_temp=args.selection_temp, 
        wd=args.wd,
        lr_schedule=args.lr_schedule,
    )
    
    mcts_cfg = MCTSConfig(
        num_sims=args.num_mcts_sims,
        cpuct=args.cpuct,
        noise_scale=args.noise_scale,
        max_game_duration=args.max_game_duration, 
    )
    
    train(
        train_cfg=train_cfg,
        mcts_cfg=mcts_cfg,
        verbose=args.verbose,
        wandb=args.wandb,
    )

