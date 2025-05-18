'''
Changes to take REINFORCE -> vanilla advantage actor-critic (A2C)
    - change to Pendulum environment from Cartpole, where now action is scalar
        - new nstates=3, nactions=1 scalar
        - this now means we treat the scalar output as the mean of an action distribution 
        - no need for mask/dones since pendulum doesn't truncate early 
    - value net has scalar output given a vector of states, ie. nstates -> 1
    - general n-step loss as opposed to n=T
    - value net also trained in loss_fn wrt returns, and we also have an entropy component
    - put everything on a single cpu & make networks smaller

        --> The introduction of a value net, n-step loss, and modified loss function are what 
        separate REINFORCE from actor-critic. 

'''


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
from typing import Tuple, Optional 
import warnings
import math  

# clean up logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")

env = gym.make('Pendulum-v1')
# device = 'cpu' # a3c uses only cpu, but many in parallel doing async rollouts and writing to shared mem
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNet(nn.Module): # states -> action probs, modified now for pendulum instead of Pendulum, so 3 -> 1
    def __init__(self, nstates=3, nactions=1, hidden_dim=128, act=nn.GELU()):
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, nactions)
        self.act = act
        self.log_std = nn.Parameter(torch.zeros(nactions))

    def _forward(self, x): # x is [nstates], want [nactions] as logits
        return self.w2(self.act(self.w1(x)))

    def forward(self, state, env_left=-2.0, env_right=2.0): # [nstates] -> (action scalar, lp scalar)
        action_mean = self._forward(state) # internal forward returning mean, forward is a distributional wrapper
        action_std = torch.exp(self.log_std)

        distb = torch.distributions.Normal(action_mean, action_std)
        action = torch.clamp(distb.sample(), env_left, env_right) # scalar
        log_prob = distb.log_prob(action)

        if torch.abs(log_prob) > 1e6:
            raise ValueError("Log probability is too large, causing numerical instability")
        
        return action, log_prob

class ValueNet(nn.Module): # states -> values for states
    def __init__(self, nstates=3, hidden_dim=16, act=nn.GELU()):
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, x): # [b, nstates] -> [b]
        return self.w2(self.act(self.w1(x)))


# rewards -> final returns incl. n-step discounting and critic bootstrapping + V_{s+n}
def rewards2returns(value_net, rewards, states, gamma=0.99, max_rollout_len=50, n=10):
    idx = torch.arange(max_rollout_len, device=rewards.device)
    power_mat = idx[:,None] - idx[None,:]     # (k,t)=k−t
    n_step_mask = ((power_mat >= 0) & (power_mat < n)).float()
    G = torch.tril(gamma ** power_mat)  # lower‐triangular: gives gamma^(i-j) for i>=j
    n_step_returns = rewards @ (G * n_step_mask)  # [B, T] @ [T, T] -> [B, T]

    # add (gamma ** n) * V(s_{t+n}) for valid timesteps only
    B, T, S = states.shape
    # Create shifted states tensor where states_shifted[b,t] = states[b,t+n] if t+n < T else zeros
    states_shifted = torch.zeros_like(states) # Will be on the same device as states
    if T > n: # Avoid negative indexing if T <= n
        states_shifted[:, :-n] = states[:, n:]  # Shift states by n steps
    flat_states_shifted = states_shifted.reshape(B*T, S)
    with torch.no_grad(): # Ensure value net forward pass doesn't track gradients here
        V_shifted = value_net(flat_states_shifted).reshape(B, T, -1).squeeze(-1) # [B, T, 1] -> [B, T]

    # don't want to update val_net based on policy loss, so detach (already done via torch.no_grad)
    return n_step_returns + (gamma ** n) * V_shifted

def loss_fn(policy, value_net, rewards, logprobs, states, gamma=0.99, max_rollout_len=50, val_coeff=0.01, ent_coeff=1e-4, n=10, verbose=False): # first two are [b, ms]
    batch_size, seq_len = states.shape[0], states.shape[1]
    returns = rewards2returns(
        value_net,
        rewards,
        states,
        gamma=gamma,
        max_rollout_len=max_rollout_len,
        n=n
    )

    with torch.no_grad(): 
        advantages = returns - value_net(states).reshape(batch_size, seq_len)

    # Detach advantages used for policy loss calculation, but not for value loss
    advantages_detached = advantages.detach()
    norm_adv = (advantages_detached - advantages_detached.mean())/(advantages_detached.std() + 1e-8)
    # return -Reward = -E[R_t * logprob] as loss like in REINFORCE
    policy_loss_t = (-norm_adv * logprobs).mean()

    # get loss to update value network so it's better at predicting returns
        # since that's how we're using it,
    reshaped_states = states.view(batch_size * seq_len, -1)
    values = value_net(reshaped_states).view(batch_size, seq_len)
    # Use the original advantages (with gradients) for value loss
    value_loss_t = F.mse_loss(values, returns)

    # compute entropy_loss_t using policy.log_std of action distribution
    # Proper entropy calculation for Normal distribution
    # Entropy of Normal(μ,σ) = 0.5 * log(2πeσ²)
    action_std = torch.exp(policy.log_std)
    # Ensure pi is a tensor on the correct device
    pi_tensor = torch.tensor(math.pi, device=action_std.device)
    entropy_loss_t = 0.5 * (torch.log(2 * pi_tensor * math.e * action_std.pow(2))).mean()


    # Print the magnitude of each loss component for monitoring/debugging
    if verbose: 
        print(f"Policy Loss: {policy_loss_t.item():.4f}, Value Loss (x{val_coeff}): {value_loss_t.item():.4f}, Entropy Loss (x{ent_coeff}): {entropy_loss_t.item():.4f}")

    return policy_loss_t + val_coeff * value_loss_t - ent_coeff * entropy_loss_t


def get_batch(
    policy,
    env,
    nstates,
    b: int = 64,
    max_rollout_len: int = 200,
    verbose: bool = False,
    step: int = 0,
    return_actions: bool = False,
    include_init_state: bool = False, 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    batch_rewards, batch_logprobs, batch_actions = [], [], []  # both will be [b, sm]
    batch_states = []

    if verbose and step % 5 == 0:
        print(f"\nGenerating batch {step} of rollouts...")

    for batch_idx in range(b):  # in contrast to dqn which stores a buffer where we may be learning from
        i = 0

        states_len = max_rollout_len + 1 if include_init_state else max_rollout_len

        # init tensors
        rollout_rewards = torch.zeros(max_rollout_len, device=device)
        rollout_logprobs = torch.zeros(max_rollout_len, device=device)
        rollout_states = torch.zeros(states_len, nstates, device=device)
        if return_actions: 
            rollout_actions = torch.zeros(max_rollout_len, device=device)
        
        # rollout_actions
        state, _ = env.reset()

        state = torch.from_numpy(state).float().to(device)

        if include_init_state: 
            rollout_states[0].copy_(state)

        # generate a single rollout
        while i < max_rollout_len:
            next_action, logprob_next_action = policy(state)
            # Ensure action is detached before converting to numpy/float
            next_action_float = float(next_action.detach().cpu().numpy())
            next_state, r, _, _, _ = env.step([next_action_float])

            # Store data on the correct device
            rollout_rewards[i] = torch.tensor(float(r), device=device)
            rollout_logprobs[i] = logprob_next_action # Already on device from policy output
            
            # Store the current state, not the next state
            state_idx = i if not include_init_state else i + 1
            rollout_states[state_idx].copy_(state)
            
            if return_actions: 
                rollout_actions[i] = next_action

            state = torch.from_numpy(next_state).float().to(device)
            i += 1

            if i == max_rollout_len:
                if verbose and step and step % 100 == 0:
                    print(f"  Rollout {batch_idx}: length {i}, final reward {r:.2f}")

        # add tensor to list to stack later
        batch_rewards.append(rollout_rewards[:i])  # Only include actual steps taken
        batch_logprobs.append(rollout_logprobs[:i])
        
        # For states, include one more state if include_init_state is True
        states_to_include = i + 1 if include_init_state else i
        batch_states.append(rollout_states[:states_to_include])
        
        if return_actions: 
            batch_actions.append(rollout_actions)

    # stack logprobs and r into two tensors, they are list of lists overall [b, mrl]
    # Tensors in the lists are already on the correct device
    batch_rewards = torch.stack(batch_rewards)
    batch_logprobs = torch.stack(batch_logprobs)
    batch_states = torch.stack(batch_states)
    if return_actions: 
        batch_actions = torch.stack(batch_actions)
    
    if return_actions: 
        return batch_rewards, batch_logprobs, batch_states, batch_actions 
    else: 
        # [B, T], [B, T], [B, T+1, states] = [B, T+1 if include_init_state else T, 3]
        return batch_rewards, batch_logprobs, batch_states
        

def eval(policy, env, max_rollout_len: int = 200, b: int = 64): 
    batch_rewards = []

    for batch_idx in range(b):
        i = 0

        # init tensors
        rollout_rewards = torch.zeros(max_rollout_len, device=device)
        state, _ = env.reset()

        state = torch.from_numpy(state).float().to(device)
        # generate a single rollout
        while i < max_rollout_len:
            with torch.no_grad():  # Disable gradient computation during evaluation
                next_action, _ = policy(state)
            # Ensure action is detached before converting to numpy/float
            next_action_float = float(next_action.detach().cpu().numpy())
            next_state, r, terminated, truncated, _ = env.step([next_action_float])

            # Store data on the correct device
            rollout_rewards[i] = torch.tensor(float(r), device=device)

            state = torch.from_numpy(next_state).float().to(device)
            i += 1
            
            # Break if episode is done
            if terminated or truncated:
                break

        # add tensor to list to stack later
        batch_rewards.append(rollout_rewards[:i].mean())  # Only average the actual steps taken

    return torch.tensor(batch_rewards, device=device).mean()

def train(nsteps=100, batch_size=16, max_rollout_len=200, lr=1e-3, gamma=0.99, verbose=False, nstates=3, policy_hidden_dim=128, value_hidden_dim=128):
    policy = PolicyNet(hidden_dim=policy_hidden_dim, nstates=nstates).to(device)
    value_net = ValueNet(hidden_dim=value_hidden_dim, nstates=nstates).to(device)
    opt_pol = torch.optim.AdamW(policy.parameters(), lr=lr * 3)
    opt_pol.zero_grad()
    opt_val = torch.optim.AdamW(value_net.parameters(), lr=lr)
    opt_val.zero_grad()
    b = batch_size

    if verbose:
        print(f"Using device: {device}")
        print(f"Starting training with {nsteps} steps, batch size {batch_size}, max rollout length {max_rollout_len}")
        print(f"Learning rate: {lr}, discount factor (gamma): {gamma}")
        print(f"Policy network: {policy}")
        print(f"Value network: {value_net}")

    for step in tqdm(range(nsteps)):
        # on-policy, ie. the batch we'll step on a few times in training step has JUST been generated by our current policy
        if verbose: # and step % 1 == 0
            print(f"Computing loss for batch {step}...")

        batch_rewards, batch_logprobs, batch_states = get_batch(policy, env, nstates, b, max_rollout_len, verbose, step)
        loss = loss_fn(policy, value_net, batch_rewards, batch_logprobs, batch_states, gamma=gamma, max_rollout_len=max_rollout_len)
        loss.backward()
        opt_pol.step()
        opt_pol.zero_grad()
        opt_val.step()
        opt_val.zero_grad()

        if step % 5 == 0 and verbose:
            print(f"[{step:4d}/{nsteps:4d}]  ||   Loss = {loss.item():.4f} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Actor-Critic on Pendulum')
    parser.add_argument('--nsteps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy-hidden-dim', type=int, default=128, help='Hidden dimension size for policy network')
    parser.add_argument('--value-hidden-dim', type=int, default=128, help='Hidden dimension size for value network') # Added arg for value net
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')

    args = parser.parse_args()

    if args.verbose:
        print("Starting Actor-Critic training on Pendulum-v1 environment")
        print(f"Arguments: {args}")
        

    train(
        nsteps=args.nsteps,
        batch_size=args.batch_size,
        max_rollout_len=args.max_rollout_len,
        lr=args.lr,
        gamma=args.gamma,
        verbose=args.verbose,
        nstates=env.observation_space.shape[0], 
        policy_hidden_dim=args.policy_hidden_dim,
        value_hidden_dim=args.value_hidden_dim
    )