'''
changes to get a3c: 
    (a1c)
    - change to pendulum environment where action is scalar 
        - new nstates=3, nactions=1 scalar
        - also means we output the mean of a distribution now from which we sample actions
        - no need for mask since pendulum always runs to a max step budget, no "dones" need to be stored 
    - value net has scalar output given a vector of states, ie. nstates -> 1
    - general n-step loss as opposed to n=T 
    - value net trained in loss_fn wrt returns
        - include entropy loss 
    - put everything on a single cpu & make networks smaller
    
    (a3c)
    - add a worker_fn that 
        - does a batch of rollouts
        - compute loss 
        - updates torch.shared_memory global parameters

        make helper fn for each of these and then define _worker to be all three in succession 
        wrapped in a while loop reading a global step counter (nupdates)

'''


import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import argparse
import numpy as np
import warnings

# clean up logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")


env = gym.make('Pendulum-v1')
device = 'cpu' # a3c uses only cpu, but many in parallel doing async rollouts and writing to shared mem 

class PolicyNet(nn.Module): # states -> action probs, modified now for pendulum instead of cartpole, so 3 -> 1
    def __init__(self, nstates=3, nactions=1, hidden_dim=32, act=nn.GELU()):
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, nactions)
        self.act = act
        self.log_std = nn.Parameter(torch.zeros(nactions))

    def _forward(self, x): # x is [nstates], want [nactions] as logits
        return self.w2(self.act(self.w1(x)))

    def forward(self, state, env_left=-2.0, env_right=2.0): # [nstates] -> (action scalar, lp scalar)
        action_mean = self._forward(state) # internal forward returning mean, this is a distributional wrapper 
        action_std = torch.exp(self.log_std)

        distb = torch.distributions.Normal(action_mean, action_std)
        action = torch.clamp(distb.sample(), env_left, env_right) # scalar
        log_prob = distb.log_prob(action)

        return action, log_prob

# TODO: in practice, we share params as two heads on the same backbone 
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
    n_step_mask = (power_mat>=0 & power_mat<n).float()
    G = torch.tril(gamma**power_mat)  # already lower‐triangular
    n_step_returns = rewards @ (G * n_step_mask) # [b, msl] @ [msl, msl] -> [b, msl]

    # add (gamma ** n) * (V(s_{t+n}) * mask)
    B, T, S = states.shape 
    flat_states = states.reshape(B*T, S)
    V = value_net(flat_states).reshape(B, T, -1).unsqueeze(-1) # [B, T, 1] -> [B, T]
    
    
    # for each token t we want V_{t+n} 
    # V[i,j] = value of states[i,j] which is a [nstates] vector 
    # in returns we have for each token \sum_{i=0}^n g^i rewards

    value_term = (gamma ** n) * V[:, n]


    
    
    return n_step_returns + value_term 

def loss_fn(policy, value_net, rewards, logprobs, states, gamma=0.99, max_rollout_len=50, val_coeff=0.5, ent_coeff=0.01, n=10): # first two are [b, ms]
    returns = rewards2returns(
        value_net,
        rewards,
        states,
        gamma=gamma,
        max_rollout_len=max_rollout_len,
        n=n
    )

    norm_returns = (returns - returns.mean())/(returns.std() + 1e-8)
    # return -Reward = -E[R_t * logprob] as loss like in REINFORCE
    policy_loss_t = (-norm_returns * logprobs).mean()

    # get loss to update value network so it's better at predicting returns 
        # since that's how we're using it, 
    batch_size, seq_len = states.shape[0], states.shape[1]
    reshaped_states = states.view(batch_size * seq_len, -1)
    values = value_net(reshaped_states).view(batch_size, seq_len)    
    value_loss_t = ((returns - values) ** 2).mean()
    
    # compute entropy_loss_t using policy.log_std of action distribution
    entropy_loss_t = 0.5 * policy.log_std.mean()  

    return policy_loss_t + val_coeff * value_loss_t - ent_coeff * entropy_loss_t


def train(nsteps=100, batch_size=16, max_rollout_len=200, lr=1e-3, gamma=0.99, verbose=False, nstates=3):
    policy = PolicyNet().to(device)
    value_net = ValueNet().to(device)
    opt_pol = torch.optim.AdamW(policy.parameters(), lr=lr)
    opt_pol.zero_grad()
    opt_val = torch.optim.AdamW(value_net.parameters(), lr=lr)
    opt_val.zero_grad()
    b = batch_size
    
    if verbose:
        print(f"Starting training with {nsteps} steps, batch size {batch_size}, max rollout length {max_rollout_len}")
        print(f"Learning rate: {lr}, discount factor (gamma): {gamma}")
        print(f"Policy network: {policy}")
    
    for step in tqdm(range(nsteps)):
        # on-policy, ie. the batch we'll step on a few times in training step has JUST been generated by our current policy 
        batch_rewards, batch_logprobs = [], []  # both will be [b, sm]
        batch_states = []

        if verbose and step % 5 == 0:
            print(f"\nGenerating batch {step} of rollouts...")
            
        for batch_idx in range(b):  # in contrast to dqn which stores a buffer where we may be learning from 
            i = 0 
            rollout_rewards, rollout_logprobs = torch.zeros(max_rollout_len), torch.zeros(max_rollout_len)
            rollout_states = torch.zeros(max_rollout_len, nstates)
            state, _ = env.reset()

            state = torch.from_numpy(state).float()
            # generate a single rollout
            while i < max_rollout_len:
                next_action, logprob_next_action = policy(state)
                next_action_float = float(next_action.detach().numpy())
                next_state, r, _, _, _ = env.step([next_action_float])
                rollout_rewards[i] = float(r)
                rollout_logprobs[i] = logprob_next_action
                rollout_states[i].copy_(state)

                state = torch.from_numpy(next_state).float()
                i += 1
                
                if i == max_rollout_len:
                    true_lens[batch_idx] = i 
                    if verbose and batch_idx % 10 == 0 and step % 5 == 0:
                        print(f"  Rollout {batch_idx}: length {i}, final reward {r}")

            # add tensor to list to stack later 
            batch_rewards.append(rollout_rewards)
            batch_logprobs.append(rollout_logprobs)
            batch_states.append(rollout_states)

        # stack logprobs and r into two tensors, they are list of lists overall [b, mrl]
        batch_rewards = torch.stack(batch_rewards).to(device)
        batch_logprobs = torch.stack(batch_logprobs).to(device)
        batch_states = torch.stack(batch_states).to(device)

        if verbose and step % 5 == 0:
            print(f"Computing loss for batch {step}...")

        # TODO: learn to take multiple steps without backward repeat issue, ie. need new forward/probs under new policy?     
        loss = loss_fn(policy, value_net, batch_rewards, batch_logprobs, batch_states, gamma=gamma, max_rollout_len=max_rollout_len)
        loss.backward()    
        opt_pol.step()
        opt_pol.zero_grad()
        opt_val.step()
        opt_val.zero_grad()

        if step % 5 == 0 and verbose:
            print(f"[{step:4d}/{nsteps:4d}]  ||   Loss = {loss.item():.4f} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train REINFORCE on CartPole')
    parser.add_argument('--nsteps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension size for policy network')
    parser.add_argument('--early-stop', action='store_true', help='Stop training once environment is solved (avg reward > 195)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting REINFORCE training on CartPole-v1 environment")
        print(f"Arguments: {args}")
    
    train(
        nsteps=args.nsteps,
        batch_size=args.batch_size,
        max_rollout_len=args.max_rollout_len,
        lr=args.lr,
        gamma=args.gamma,
        verbose=args.verbose
    )