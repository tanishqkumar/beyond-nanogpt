'''
(https://arxiv.org/pdf/1802.01561) IMPALA: Scalable Distributed Deep-RL with Importance Weighted
Actor-Learner Architectures

Differences from A3C: 
    - instead of writing local grads into global params, we decouple acting and learning
        - acting = rollouts, learning = grads + step
        - in a3c, local models (actors) are also the objects on which grads are computed (learners)
        - in IMPALA, local models (actors) pass *only rollout batches, 
            torch.stack([s, a, r])* to the learner (*global* model) which takes grads

    # actors (fwd only, small bsz=1) on cpu, learners (fwd+bwd, large batch) on gpu 
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

# TODO: make sure actors on cpu and main thread on cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# avoid code bloat due to hypers/kwargs in workers by defining a dataclass
@dataclass
class ActorWorkerConfig: 
    max_rollout_len: int = 200
    gamma: float = 0.99
    verbose: bool = False
    nstates: int = 3
    policy_hidden_dim: int = 32
    sync_freq: int = 10
    max_steps: int = 10_000
    eval_every: int = 20

def _actor_worker(
    worker_id: int, # TODO: use this to print if verbose 
    global_step: mp.Value, 
    global_lock: mp.Lock, 
    global_policy_net: nn.Module, 
    global_q_rewards: torch.Tensor, 
    global_q_logprobs: torch.Tensor, 
    global_q_states: torch.Tensor, 
    global_q_actions: torch.Tensor, 
    global_free_slots: mp.Queue, 
    config: ActorWorkerConfig = ActorWorkerConfig(),
) -> None:
    
    # make a separate env instance for each worker
    env = gym.make('Pendulum-v1')
    
    local_policy = PolicyNet(hidden_dim=config.policy_hidden_dim, nstates=config.nstates).to(device)

    local_step = 0
    while global_step.value < config.max_steps:

        if local_step % 10 == 0: 
            print(f'Getting rollouts in worker {worker_id}, local 
                  step {local_step} and global step {global_step.value}.')
    
        with global_step.get_lock():
            global_step.value += 1

        if local_step % config.sync_freq == 0: 
            # sync local model to global for rollouts to avoid being too stale 
            local_policy.load_state_dict(global_policy_net.state_dict())
            local_value_net.load_state_dict(global_value_net.state_dict())

        # we have to put things on cpu by hand 
        local_policy = local_policy.to(device)
        local_value_net = local_value_net.to(device)
        
        # gets a single rollout as IMPALA desires 
        batch_rewards, batch_logprobs, batch_states, batch_actions = get_batch(
            local_policy, 
            env, 
            config.nstates, 
            1, 
            config.max_rollout_len, 
            False, # set verbose = False as an arg (not kwarg)
            global_step.value, 
            return_actions=True, 
        )
    
        # update shared memory with rollout, create mp.Queue passing around free slots 
        with global_lock: 
            free_slot_idx = global_free_slots.get()
            global_q_rewards[free_slot_idx:free_slot_idx+1, :].copy_(batch_rewards)
            global_q_logprobs[free_slot_idx:free_slot_idx+1, :].copy_(batch_logprobs)
            global_q_states[free_slot_idx:free_slot_idx+1, :, :].copy_(batch_states)
            global_q_actions[free_slot_idx:free_slot_idx+1, :].copy_(batch_actions)

        local_step += 1

    # Close environment when done
    env.close()


# this [consumes a batch from global buffers, does fwd to get global_model.logprobs, does bwd using (global_lps, batch)]
def learner_step(
    policy_net: nn.Module,
    value_net: nn.Module,
    opt_pol: torch.optim.Optimizer,
    opt_val: torch.optim.Optimizer,
    global_lock: mp.Lock,
    global_step: mp.Value,
    global_q_rewards: torch.Tensor,
    global_q_logprobs: torch.Tensor,
    global_q_states: torch.Tensor,
    global_q_actions: torch.Tensor,
    global_free_slots: mp.Queue,
    config: ActorWorkerConfig
) -> None:
    pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Actor-Critic on Pendulum')
    parser.add_argument('--max_steps', type=int, default=10_000, help='Maximum number of global steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max-rollout-len', type=int, default=20, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy-hidden-dim', type=int, default=64, help='Hidden dimension size for policy network')
    parser.add_argument('--value-hidden-dim', type=int, default=64, help='Hidden dimension size for value network')
    parser.add_argument('--nactors', type=int, default=8, help='Number of parallel actor threads')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--sync-freq', type=int, default=50, help='Frequency to sync local models with global model')
    parser.add_argument('--eval_every', type=int, default=50, help='Frequency to eval')
    parser.add_argument('--buffer_sz', type=int, default=1_000, help='Size of global shared buffers for rollouts')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm for clipping')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    nstates = 3 # hardcode for pendulum 
    global_policy_net = PolicyNet(hidden_dim=args.policy_hidden_dim, nstates=nstates).to(device)
    global_value_net = ValueNet(hidden_dim=args.value_hidden_dim, nstates=nstates).to(device)
    global_policy_net.share_memory()
    global_value_net.share_memory()
    
    global_opt_pol = torch.optim.AdamW(global_policy_net.parameters(), lr=args.lr)
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
    actor_worker_config = ActorWorkerConfig(
        max_steps=args.max_steps,
        max_rollout_len=args.max_rollout_len,
        gamma=args.gamma,
        verbose=args.verbose,
        nstates=nstates,
        policy_hidden_dim=args.policy_hidden_dim,
        sync_freq=args.sync_freq,
        eval_every=args.eval_every,
    )

    # set up processes and their args 
    all_procs = []
    global_lock = mp.Lock()  
    global_step = mp.Value('i', 0)  

    global_q_rewards = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory()
    global_q_logprobs = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory()
    global_q_states = torch.zeros(args.buffer_sz, args.max_rollout_len, nstates, device=device).share_memory()
    global_q_actions = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory()
    
    global_free_slots = mp.Queue()
    for i in range(args.buffer_sz): 
        global_free_slots.put(i)
    
    ACTOR_ARGS = (
        global_policy_net,
        global_value_net,
        global_opt_pol,
        global_opt_val,
        global_lock,
        global_step,
        global_q_rewards,
        global_q_logprobs,
        global_q_states,
        global_q_actions,
        global_free_slots,
        actor_worker_config,
    )
    

    for worker_id in range(args.nactors): 
        p = mp.Process(target=_actor_worker, args=(worker_id, *ACTOR_ARGS))
        p.daemon = True 
        p.start()
        all_procs.append(p)

    
    ### learner/gpu/main thread logic here ###
    # one learner process, the main thread running on our gpu 

    # learner() or just put main logic here 

    # eval logic in main thread with learner 
    # if global_step.value % config.eval_every == 0: 
            # avg_r = eval(global_policy_net, env)

    ### learner/gpu/main thread logic here ###            
    
    # cleanup 
    for p in all_procs:
        p.join()
