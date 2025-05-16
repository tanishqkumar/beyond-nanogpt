'''
(https://arxiv.org/pdf/1602.01783) Asynchronous Methods for Deep Reinforcement Learning

A3C stands for "Asynchronous Advantage Actor-Critic." 
Starting point is train_a1c.py that we modify to arrive at this file, which is essentially 
an asynchronous version of that on a multi-CPU setting (no GPUs involved)! 

Changes to take A1C -> A3C
    - Multiple CPU workers run in parallel, each with their own environment instance
    - Workers share a global policy and value network stored in shared memory
    - Each worker computes gradients locally, then updates the global networks asynchronously
    - Periodic synchronization between local and global networks to ensure workers have recent parameters

The TLDR is that each worker [does a batch of rollouts, computes a small batch grad, updates GLOBAL params]
in a *yolo* sort of way (ie. the global params get lots of small updates from workers). 
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import warnings
import torch.multiprocessing as mp
from dataclasses import dataclass

# we can port these primitives from vanilla actor-critic, this file is mainly about getting async logic right
from train_a1c import ValueNet, PolicyNet, loss_fn, get_batch, eval 

# clean up logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")

device = torch.device('cpu') # a3c is cpu-only, shows parallel rollouts on cpu can compete with gpu implementations 

# abstract away some RL hypers into a dataclass to avoid code bloat in kwargs of _worker
@dataclass
class WorkerConfig: 
    b: int = 32
    max_rollout_len: int = 200
    lr: float = 3e-4
    gamma: float = 0.99
    verbose: bool = False
    nstates: int = 3
    policy_hidden_dim: int = 32
    value_hidden_dim: int = 32
    sync_freq: int = 10
    max_grad_norm: float = 1.0
    max_steps: int = 10_000
    eval_every: int = 20

def _worker(
    i: int, 
    global_steps: mp.Value, 
    global_policy: nn.Module, 
    global_value_net: nn.Module, 
    global_opt_pol: torch.optim.Optimizer, 
    global_opt_val: torch.optim.Optimizer, 
    global_lock: mp.Lock,
    config: WorkerConfig = WorkerConfig(),
) -> None:
    
    # make a separate env instance for each worker
    env = gym.make('Pendulum-v1')
    
    local_policy = PolicyNet(hidden_dim=config.policy_hidden_dim, nstates=config.nstates).to(device)
    local_value_net = ValueNet(hidden_dim=config.value_hidden_dim, nstates=config.nstates).to(device)

    if config.verbose:
        print(f'On worker thread: {i}')
        if i == 0: 
            print(f"Using device: {device}")
            print(f"Starting training with {config.max_steps} max steps, batch size {config.b}, max rollout length {config.max_rollout_len}")
            print(f"Learning rate: {config.lr}, discount factor (gamma): {config.gamma}")
            print(f"Policy network: {local_policy}")
            print(f"Value network: {local_value_net}")

    local_step = 0
    while global_steps.value < config.max_steps:
        if global_steps.value % 20 == 0: 
            print(f'Thread {i}: we are on global step {global_steps.value}')
            
        if global_steps.value % config.eval_every == 0: 
            if config.verbose: print(f'Evaluating global policy...')
            avg_r = eval(global_policy, env)
            print(f'[Worker {i}, global step {global_steps.value}/{config.max_steps}] EVAL: Average reward {avg_r.item():.3f}')
        
        with global_steps.get_lock():
            global_steps.value += 1


        if local_step % config.sync_freq == 0: 
            if config.verbose: print(f'Worker {i} is syncing local with global...')
            # sync local model to global for rollouts to avoid being too stale 
            local_policy.load_state_dict(global_policy.state_dict())
            local_value_net.load_state_dict(global_value_net.state_dict())
            # print(f'Loading state dict complete!')

        # we have to put things on cpu by hand 
        local_policy = local_policy.to(device)
        local_value_net = local_value_net.to(device)
        
        batch_rewards, batch_logprobs, batch_states = get_batch(
            local_policy, 
            env, 
            config.nstates, 
            config.b, 
            config.max_rollout_len, 
            False, # verbose 
            global_steps.value
        )
        
        batch_rewards = batch_rewards.to(device)
        batch_logprobs = batch_logprobs.to(device)
        batch_states = batch_states.to(device)
        

        loss = loss_fn(
            local_policy, 
            local_value_net, 
            batch_rewards, 
            batch_logprobs, 
            batch_states, 
            gamma=config.gamma, 
            max_rollout_len=config.max_rollout_len, 
            val_coeff=0.5, 
            ent_coeff=0.01, 
        )

        local_policy.zero_grad()
        local_value_net.zero_grad()  
        loss.backward()
        
        # clip grads, rl optimization is often finnicky and this helps stabilize 
        torch.nn.utils.clip_grad_norm_(local_policy.parameters(), config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(local_value_net.parameters(), config.max_grad_norm)
        
        # copy grads to global and global_opt.step(), layerwise 
        with global_lock:
            for (local_param, global_param) in zip(local_policy.parameters(), global_policy.parameters()): 
                global_param.grad = local_param.grad.clone()
            global_opt_pol.step()
            
            for (local_param, global_param) in zip(local_value_net.parameters(), global_value_net.parameters()): 
                global_param.grad = local_param.grad.clone()
            global_opt_val.step()

        if global_steps.value % 25 == 0 and config.verbose and i == 0: # only master should print loss to avoid too much printing 
            print(f"Worker {i}, [{global_steps.value:4d}/{config.max_steps:4d}]  ||   Loss = {loss.item():.4f} ")
        
        local_step += 1

    # Close environment when done
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Actor-Critic on Pendulum')
    parser.add_argument('--max_steps', type=int, default=10_000, help='Maximum number of global steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max-rollout-len', type=int, default=20, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy-hidden-dim', type=int, default=64, help='Hidden dimension size for policy network')
    parser.add_argument('--value-hidden-dim', type=int, default=64, help='Hidden dimension size for value network')
    parser.add_argument('--nprocs', type=int, default=16, help='Number of parallel worker threads')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--sync-freq', type=int, default=50, help='Frequency to sync local models with global model')
    parser.add_argument('--eval_every', type=int, default=50, help='Frequency to eval')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm for clipping')

    args = parser.parse_args()

    if args.verbose:
        print("Starting Actor-Critic training on Pendulum-v1 environment")
        print(f"Arguments: {args}")
        
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    ## here we'll set up processes/shared objects and set off all the workers 
        # each worker will [do rollout, compute local grad, update global params] updating its local model every so often 

    nstates = 3 # hardcode for pendulum 
    global_policy = PolicyNet(hidden_dim=args.policy_hidden_dim, nstates=nstates).to(device)
    global_value_net = ValueNet(hidden_dim=args.value_hidden_dim, nstates=nstates).to(device)
    global_policy.share_memory()
    global_value_net.share_memory()
    
    global_opt_pol = torch.optim.AdamW(global_policy.parameters(), lr=args.lr)
    global_opt_val = torch.optim.AdamW(global_value_net.parameters(), lr=args.lr)

    # paper says keeping shared stats across processes does best
        # perhaps unsurprising, the moments/statistics get more accurate as we see more examples 
    for state in global_opt_pol.state.values(): 
        for k, v in state.items(): 
            if torch.is_tensor(v): 
                state[k] = v.share_memory()

    for state in global_opt_val.state.values(): 
        for k, v in state.items(): 
            if torch.is_tensor(v): 
                state[k] = v.share_memory()

    # Create worker config with CLI args
    worker_config = WorkerConfig(
        max_steps=args.max_steps,
        b=args.batch_size,
        max_rollout_len=args.max_rollout_len,
        lr=args.lr,
        gamma=args.gamma,
        verbose=args.verbose,
        nstates=nstates,
        policy_hidden_dim=args.policy_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        sync_freq=args.sync_freq,
        max_grad_norm=args.max_grad_norm, 
        eval_every=args.eval_every, 
    )

    # set up processes and their args 
    all_procs = []
    global_lock = mp.Lock()  
    
    SHARED_ARGS = (
        global_policy,
        global_value_net,
        global_opt_pol,
        global_opt_val,
        global_lock,  
        worker_config,  
    )
    
    global_steps = mp.Value('i', 0)  

    for idx in range(args.nprocs): 
        p = mp.Process(target=_worker, args=(idx, global_steps, *SHARED_ARGS))
        p.daemon = True 
        p.start()
        all_procs.append(p)
    
    # cleanup 
    for p in all_procs:
        p.join()
    
    if args.verbose:
        print("Training completed. Model saved.")
