'''
(https://arxiv.org/pdf/1802.01561) IMPALA: Scalable Distributed Deep-RL with Importance Weighted
Actor-Learner Architectures

Differences from A3C: 
    - instead of writing local grads into global params, we decouple acting and learning
        - acting = rollouts, learning = grads + step
        - in a3c, local models (actors) are also the objects on which grads are computed (learners)
        - in IMPALA, local models (actors) pass *only rollout batches, 
            torch.stack([s, a, r])* to the learner (*global* model) which takes grads

    - intuition for vtrace -- what needs correction, and why? 
        - review of actor-critic math, A_t = G_t - V_t and why we want E_rollouts[G_t(s_t) - V_t(s_t)] = 0

    - relation to LLM RL 
        (decoupling acting and learning, doing policy-grad style stuff with half off policy rollouts)

    # actors (fwd only, small bsz=1) on cpu, learners (fwd+bwd, large batch) on gpu 
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import torch
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import argparse
import warnings
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple, List, Optional 
import math 

# we can port these primitives from vanilla actor-critic, this file is mainly about getting async logic right
from train_a1c import ValueNet, PolicyNet, get_batch, eval 

# clean up logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")

# TODO: make sure actors on cpu and main thread on cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## ACTOR LOGIC ##

# avoid code bloat due to hypers/kwargs in workers by defining a dataclass
@dataclass
class ActorWorkerConfig: 
    max_rollout_len: int = 200
    gamma: float = 0.99
    verbose: bool = False
    nstates: int = 3
    policy_hidden_dim: int = 32
    sync_freq: int = 10
    max_actor_steps: int = 10_000
    eval_every: int = 20

def _actor_worker(
    worker_id: int, # TODO: use this to print if verbose 
    global_actor_step: mp.Value, 
    global_buffer_writelock: mp.Lock, 
    global_policy_net: nn.Module, 
    global_q_rewards: torch.Tensor, 
    global_q_logprobs: torch.Tensor, 
    global_q_states: torch.Tensor, 
    global_q_actions: torch.Tensor, 
    global_free_slots: mp.Queue, 
    global_filled_slots: mp.Queue, 
    global_actor_to_sync: mp.Value, 
    config: ActorWorkerConfig = ActorWorkerConfig(),
) -> None:
    
    # make a separate env instance for each worker
    env = gym.make('Pendulum-v1')
    
    local_policy = PolicyNet(hidden_dim=config.policy_hidden_dim, nstates=config.nstates).to(device)

    local_step = 0
    while global_actor_step.value < config.max_actor_steps:

        if local_step % 10 == 0: 
            print(f'Getting rollouts in worker {worker_id}, local 
                  step {local_step} and global step {global_actor_step.value}.')
    
        with global_actor_step.get_lock():
            global_actor_step.value += 1

        if (local_step % config.sync_freq == 0) or global_actor_to_sync.value: 
            # sync local model to global for rollouts to avoid being too stale 
            local_policy.load_state_dict(global_policy_net.state_dict())
        
        # gets a single rollout as IMPALA desires 
        batch_rewards, batch_logprobs, batch_states, batch_actions = get_batch(
            local_policy, 
            env, 
            config.nstates, 
            1, 
            config.max_rollout_len, 
            False, # set verbose = False as an arg (not kwarg)
            global_actor_step.value, 
            return_actions=True, 
            include_init_state=True, 
        )
    
        # update shared memory with rollout, create mp.Queue passing around free slots 
        with global_buffer_writelock: 
            free_slot_idx = global_free_slots.get()
            global_q_rewards[free_slot_idx:free_slot_idx+1, :].copy_(batch_rewards)
            global_q_logprobs[free_slot_idx:free_slot_idx+1, :].copy_(batch_logprobs)
            global_q_states[free_slot_idx:free_slot_idx+1, :, :].copy_(batch_states)
            global_q_actions[free_slot_idx:free_slot_idx+1, :].copy_(batch_actions)
            global_filled_slots.put(free_slot_idx)

        local_step += 1

    # Close environment when done
    env.close()



## END ACTOR LOGIC ## 


## LEARNER LOGIC ## 
def get_batch_from_buffers(
    buffers: Tuple[torch.Tensor], 
    global_free_slots: mp.Queue, 
    global_filled_slots: mp.Queue, 
    global_slot_lock: mp.Lock,
    big_bsz: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    global_q_rewards, global_q_logprobs, global_q_states, global_q_actions = buffers
    
    with global_slot_lock: 
        to_read_slots = []
        for i in range(big_bsz): 
            if global_filled_slots.empty():
                if i == 0: 
                    raise RuntimeError("No data in global rollout buffer when learner tried to access it!")
                break 
            slot = global_filled_slots.get()
            to_read_slots.append(slot)

    batch_rewards = global_q_rewards[to_read_slots]
    batch_local_lps = global_q_logprobs[to_read_slots]
    batch_states = global_q_states[to_read_slots]
    batch_actions = global_q_actions[to_read_slots]

    with global_slot_lock: 
        for used_slot in to_read_slots: 
            global_free_slots.put(used_slot)

    return (batch_rewards, batch_local_lps, batch_states, batch_actions)


# global fwd pass, done on cuda since large batch 
def get_global_lps(batch_states: torch.Tensor, batch_actions: torch.Tensor, global_policy: nn.Module): 
    # states and actions are [B,T,3] and [B, T] resp
    # want the probs of the empirical actions 
    # policy(states) will be [B, T, nactions] logits 
    global_policy = global_policy.cuda()
    batch_states = batch_states.cuda()
    batch_actions = batch_actions.cuda()
    logits = global_policy(batch_states) # this is [B, T, nactions] and we want [B, T]
    
    B, T = batch_actions.shape
    rb = torch.arange(B, device='cuda')[:, None]
    rt = torch.arange(T, device='cuda')
    
    # [B, T] now, ready for log_softmax: logits -> logprobs 
    selected_logits = logits[rb, rt, batch_actions] 
    
    return F.log_softmax(selected_logits, dim=-1) # [B, T] logprob under global policy for [state_ij -> action_ij]


def loss_fn(global_policy, global_lps, vals, target_vals, adv, val_coeff=0.01, ent_coeff=1e-4, verbose=False): # first two are [b, ms]

    # return -Reward = -E[R_t * logprob] as loss like in REINFORCE
    policy_loss_t = (-adv * global_lps).mean()

    # get loss to update value network so it's better at predicting returns
    value_loss_t = F.mse_loss(vals, target_vals)

    # compute entropy_loss_t using policy.log_std of action distribution
    # Proper entropy calculation for Normal distribution
    # Entropy of Normal(μ,σ) = 0.5 * log(2πeσ²)
    action_std = torch.exp(global_policy.log_std) # this implies converence 
    pi_tensor = torch.tensor(math.pi, device=action_std.device)
    entropy_loss_t = 0.5 * (torch.log(2 * pi_tensor * math.e * action_std.pow(2))).mean()


    # Print the magnitude of each loss component for monitoring/debugging
    if verbose: 
        print(f"Policy Loss: {policy_loss_t.item():.4f}, Value Loss (x{val_coeff}): {value_loss_t.item():.4f}, Entropy Loss (x{ent_coeff}): {entropy_loss_t.item():.4f}")

    return policy_loss_t + val_coeff * value_loss_t - ent_coeff * entropy_loss_t


# outputs G_t, A_t for loss compuation, both are [B, T], more intuition on the math behind vtrace at the top of this file 
# TODO: understand the math more carefully here, maybe even rewrite this for intuition 
    # try vectorize the loop 
def vtrace(
    global_value_net: nn.Module, 
    global_lps: torch.Tensor, 
    local_lps: torch.Tensor, 
    values: torch.Tensor, 
    batch_rewards: torch.Tensor, 
    clip_rho: float = 1.0, 
    clip_c: float = 1.0, 
    gamma: float = 0.99, 
):
    B, T = batch_rewards.shape 

    # get importance weights global/local 
    rho = torch.exp(global_lps - local_lps) # [B, T]
    rho_bar = torch.clamp(rho, max=clip_rho)
    c_bar = torch.clamp(rho, max=clip_c)

    # build vs, the vtraced(values) 
    vs = torch.zeros(B, T, device=batch_rewards.device) # [B, T]
    acc = values[:, -1] # V_T from our value net, our base case for backward accumulation 
    # see the V^trace_t = V_t + sum gamma^{i-t} (prod_j c_bar_j) delta_i 

    for t in reversed(range(T)): 
        delta = rho_bar[:, t] * (batch_rewards[:, t] + gamma * values[:, t+1] - values[:, t])
        acc = values[:, t] + delta + gamma * c_bar[:, t] * (acc - values[:, t])
        vs[:, t] = acc 

    # use vs, IS-normalized-values to compute advantage instead
    last_el = acc.unsqueeze(1) # [B, 1]
    next_vs = torch.cat([vs[:, 1:], last_el], dim=1) # [B, T] shifted by 1 since A_t is in terms of vs[:, t+1]
    adv = rho_bar * (batch_rewards + gamma * next_vs - values[:, :-1])

    return vs, adv

# this [consumes a batch from global buffers, does fwd to get global_model.logprobs, does bwd using (global_lps, batch)]
def learner_step(
    big_bsz: int, 
    global_policy_net: nn.Module,
    global_value_net: nn.Module,
    global_opt_pol: torch.optim.Optimizer,
    global_opt_val: torch.optim.Optimizer,
    global_slot_lock: mp.Lock,
    global_actor_step: mp.Value,
    buffers: Tuple[torch.Tensor], 
    global_free_slots: mp.Queue,
    global_filled_slots: mp.Queue,
    clip_grad_norm: float = 1.0, 
) -> None:
    
    batch_rewards, batch_local_lps, batch_states, batch_actions = get_batch_from_buffers(buffers, global_free_slots, global_filled_slots, global_slot_lock, big_bsz)
    global_lps = get_global_lps(batch_states, batch_actions, global_policy_net) # fwd with global model getting logprobs of empirical actions 
    
    # vtrace basically adjusts for data being off-policy in an algorithm (actor critic, policy grad) that 
        # otherwise assumes data is generated on-policy (policy grad thm requires this)
        # other methods that use off-policy data (eg. DQN) don't assume/require policy gradient theorem
    # target_val is used as target in value network MSE computation, adv as advantage in policy loss
    values = global_value_net(batch_states) # # now [B, T+1] because batch_states is [B, T+1]
    target_vals, adv = vtrace(
        global_value_net, 
        global_lps, 
        batch_local_lps, 
        values, 
        batch_rewards, 
        clip_rho=1.0, 
        clip_c=1.0,
        gamma=0.99
    )
    
    global_opt_pol.zero_grad()
    global_opt_val.zero_grad()

    loss = loss_fn(global_policy_net, global_lps, values, target_vals, adv)
    loss.backward()
    
    if global_actor_step.value % 10 == 0: 
        print(f'Main process computed loss {loss.item()} on global step {global_actor_step.value}.')    

    torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), clip_grad_norm)
    torch.nn.utils.clip_grad_norm_(global_value_net.parameters(), clip_grad_norm)

    global_opt_pol.step()
    global_opt_val.step()


    return # broadcast global (policy, value) nets to actors after this 

## END LEARNER LOGIC ## 

## MAIN LOGIC ## 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Actor-Critic on Pendulum')
    parser.add_argument('--max_actor_steps', type=int, default=10_000, help='Maximum number of global steps')
    parser.add_argument('--max_learner_steps', type=int, default=1_000, help='Maximum number of global steps')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--max-rollout-len', type=int, default=20, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy-hidden-dim', type=int, default=64, help='Hidden dimension size for policy network')
    parser.add_argument('--value-hidden-dim', type=int, default=64, help='Hidden dimension size for value network')
    parser.add_argument('--nactors', type=int, default=16, help='Number of parallel actor threads')
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
    global_policy_net.share_memory_()
    global_value_net.share_memory_()
    
    global_opt_pol = torch.optim.AdamW(global_policy_net.parameters(), lr=args.lr)
    global_opt_val = torch.optim.AdamW(global_value_net.parameters(), lr=args.lr)

    # paper says keeping shared stats across processes does best
        # perhaps unsurprising, the moments/statistics get more accurate as we see more examples 
    for state in global_opt_pol.state.values(): 
        for k, v in state.items(): 
            if torch.is_tensor(v): 
                state[k] = v.share_memory_()

    for state in global_opt_val.state.values(): 
        for k, v in state.items(): 
            if torch.is_tensor(v): 
                state[k] = v.share_memory_()

    # Create worker config with CLI args
    actor_worker_config = ActorWorkerConfig(
        max_actor_steps=args.max_actor_steps,
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
    global_buffer_writelock = mp.Lock()  
    global_slot_lock = mp.Lock()  
    global_actor_step = mp.Value('i', 0)  
    global_actor_to_sync = mp.Value('i', 0)  

    global_q_rewards = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory_()
    global_q_logprobs = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory_()
    global_q_states = torch.zeros(args.buffer_sz, args.max_rollout_len, nstates, device=device).share_memory_()
    global_q_actions = torch.zeros(args.buffer_sz, args.max_rollout_len, device=device).share_memory_()
    
    global_free_slots = mp.Queue()
    global_filled_slots = mp.Queue()
    
    for i in range(args.buffer_sz): 
        global_free_slots.put(i)
    
    buffers = (global_q_rewards,
        global_q_logprobs,
        global_q_states,
        global_q_actions,
    )
    
    ACTOR_ARGS = (
        global_policy_net,
        global_buffer_writelock,
        global_actor_step,
        buffers, 
        global_free_slots,
        global_filled_slots,
        global_actor_to_sync,
        actor_worker_config,
    )
    
    for worker_id in range(args.nactors): 
        p = mp.Process(target=_actor_worker, args=(worker_id, *ACTOR_ARGS))
        p.daemon = True 
        p.start()
        all_procs.append(p)

    
    ### learner/gpu/main thread logic here ###
    # one learner process, the main thread running on our gpu 

    LEARNER_ARGS = (
        args.big_bsz,
        global_policy_net,
        global_value_net,
        global_opt_pol,
        global_opt_val,
        global_slot_lock,
        global_actor_step,
        buffers,
        global_free_slots,
        global_filled_slots,
        args.clip_grad_norm,
    )

    for step in range(args.max_learner_steps): 
        if step % args.eval_every: 
            env = gym.make('Pendulum-v1')
            avg_r = eval(global_policy_net, env)
            if args.verbose: 
                print(f'On step {step}, average reward in main process is {avg_r}.')
        learner_step(*LEARNER_ARGS) 
        # TODO: wait until workers synced using a barrier 


    ### learner/gpu/main thread logic here ###            
    
    # cleanup 
    for p in all_procs:
        p.join()
