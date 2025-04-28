'''
Proximal Policy Optimization. 
(https://arxiv.org/pdf/1707.06347) 

A key modern policy gradient optimization algorithm that is basically PPO++ but is 
important enough that it's worth knowing deeply. Basically all the changes are 
in the loss computation `loss_fn` and things otherwise stay the same. 

Key differences from PPO:  
    1) (Critical) Stabilize reward estimation by subtracting the "average value of a state" so that 
        now our actions are not weighted by return/reward, but *advantage* which is an estimate of how much 
        BETTER an action a in state s is than the average action, eg. R(s,a) - V(s) where V is now a *new* neural network 
        called a *value network*. We use this to estimate the advantage using an algorithm called Generalized Advantage Estimation. 
        This is the biggest difference from PPO, where either you normalize or subtract a constant baseline. 
    2) We use one batch for multiple steps and a buffer-like mechanism to store recent rollouts. 
        To make this close to on-policy, clip our step size each time so we don't move too far from data-generating policy. 
    3) We add an entropy regularization term in loss to encourage exploration. 

The key implementational difference in training is just collecting more data on each experience. Instead of 
(reward, logprob) being an "experience" in a rollout, 
now we want (curr_state, action/logprobs, next_state, rewards, done, value[curr_state], value[next_state])
since we'll need all of these to compute the new loss. So we modify our training code to collect/store these new pieces of information 
in addition to using them in the new loss_fn. Otherwise things are conceptually the same. This new loss was motivated 
by consideration to do with the loss/optimization landscape and a desire for sample efficiency, see the paper for 
full details -- it's worth reading!
'''


import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import argparse
import wandb

env = gym.make('CartPole-v1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyNet(nn.Module): # states -> action probs 
    def __init__(self, NSTATES=4, nactions=2, hidden_dim=128, act=nn.GELU()):
        super().__init__()
        self.w1 = nn.Linear(NSTATES, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, nactions)
        self.act = act

    def forward(self, x): # x is [NSTATES], want [nactions] as logits
        return self.w2(self.act(self.w1(x)))

class ValueNet(nn.Module): # states -> state values 
    def __init__(self, NSTATES=4, hidden_dim=128, act=nn.GELU()):
        super().__init__()
        self.w1 = nn.Linear(NSTATES, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1) # output a scalar given a state vector 
        self.act = act

    def forward(self, x): # x is [NSTATES], want [nactions] as logits
        return self.w2(self.act(self.w1(x)))

# PPO loss computation and what it takes in is key difference from REINFORCE
def loss_fn(policy_net, value_net, 
            batch_rewards, batch_logprobs, batch_actions, batch_dones, batch_states,
            batch_values, batch_next_values, mask, gamma=0.99, lamb=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_rollout_len=50): # all are are [b, ms]

    batch_values = batch_values.detach()
    batch_next_values = batch_next_values.detach()

    # compute td error 
    deltas = batch_rewards + gamma * batch_next_values * (1.0 - batch_dones.float()) - batch_values # deltas is [b, T]
    
    # compute advantage using td error
    B = deltas.shape[0]
    A = torch.zeros_like(deltas)
    A_prev = torch.zeros(B, device=deltas.device) # accum for recurrence below 
    # A[t] = d[t] + gam * lam * A_prev recurrence by defn of advantage in paper 
    for t in range(max_rollout_len-1, -1, -1): 
        mask_t = 1.0 - batch_dones[:, t].to(torch.float)
        A[:, t] = deltas[:, t] + gamma * lamb * mask_t * A_prev
        A_prev = A[:, t] # can this be done by cumprod? not clear...
    
    # compute ratio term using old/new logprobs
    new_logits = policy_net(batch_states) # [b, T, nstates] -> [b, T, nactions]
    new_logprobs_all = F.log_softmax(new_logits, dim=-1)   # [B,T,nactions]
    new_logprobs = new_logprobs_all.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)

    r = torch.exp(new_logprobs - batch_logprobs) # batch logprobs is [b, T] for chosen action

    # finally, compute clipped obj ie. policy loss 
    clipped_r = torch.max(torch.min(r, 1. + clip), 1. - clip)
    min_term = torch.min(r * A, clipped_r * A)
    policy_loss = -torch.sum(min_term * mask)/mask.sum()
    
    # compute value_net loss 
    value_preds = value_net(batch_states).flatten() # [b, T, nstates]  -> [b, T]
    returns = (A + batch_values).squeeze(-1)
    mse = F.mse_loss(value_preds, returns, reduction='none')
    value_loss = (mse * mask).sum() / mask.sum()

    # compute entrop loss term 
    new_probs = F.softmax(new_logits, dim=-1) # [b, T, nactions]
    entropy = -(new_probs * new_logprobs).sum(dim=-1) # [b, T]
    entropy_loss = (entropy * mask).sum()/mask.sum()

    # return full obj, a linear combo of the three loss terms
    return policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

def train(nsteps=100, batch_size=64, max_rollout_len=50, lr=1e-3, gamma=0.99, hidden_dim=128, early_stop=False, wandb_log=False, verbose=False):
    policy = PolicyNet(hidden_dim=hidden_dim).to(device)
    value_net = ValueNet(hidden_dim=hidden_dim).to(device)
    opt_pol = torch.optim.AdamW(policy.parameters(), lr=lr)
    opt_val = torch.optim.AdamW(value_net.parameters(), lr=lr)
    opt_pol.zero_grad(); opt_val.zero_grad()
    done = False 
    b = batch_size
    
    if verbose:
        print(f"Starting training with {nsteps} steps, batch size {batch_size}, max rollout length {max_rollout_len}")
        print(f"Learning rate: {lr}, discount factor (gamma): {gamma}")
        print(f"Policy network: {policy}")
    
    for step in tqdm(range(nsteps)):
        # on-policy, ie. the batch we'll step on a few times in training step has JUST been generated by our current policy 
        batch_rewards, batch_logprobs, batch_actions, batch_states, batch_next_states, batch_dones = [], [], [], [], [], []  # both will be [b, sm]
        true_lens = torch.zeros(b)  # entry i is true len of batch element i, use this with arange to mask mask of size [b, s]
        
        if verbose and step % 5 == 0:
            print(f"\nGenerating batch {step} of rollouts...")
            
        for batch_idx in range(b):  # in contrast to dqn which stores a buffer where we may be learning from 
            i = 0 
            NSTATES = 4
            rollout_rewards = torch.zeros(max_rollout_len).to(device)
            rollout_logprobs = torch.zeros(max_rollout_len).to(device)
            rollout_actions = torch.zeros(max_rollout_len).to(device)
            # need curr and next states, dones, and values for both 
            rollout_states = torch.zeros(max_rollout_len, NSTATES).to(device)
            rollout_next_states = torch.zeros(max_rollout_len, NSTATES).to(device)
            rollout_dones = torch.zeros(max_rollout_len, dtype=torch.bool).to(device)

            done = False 
            state, _ = env.reset()
            state = torch.tensor(state).to(device)
            
            # generate a single rollout
            while not done and i < max_rollout_len:
                dist = F.softmax(policy(state), dim=-1)
                logprobs = torch.log(dist)
                next_action = torch.multinomial(dist, 1).item()

                next_state, r, terminated, truncated, _ = env.step(next_action)
                done = terminated or truncated
                next_state = torch.tensor(next_state).to(device)
                
                # update info for this experience 
                rollout_rewards[i] = r
                rollout_logprobs[i] = logprobs[next_action]
                rollout_actions[i] = next_action
                rollout_dones[i] = done 
                rollout_states[i] = state 
                rollout_next_states[i] = next_state 

                state = next_state
                i += 1
                
                if done or i == max_rollout_len:
                    true_lens[batch_idx] = i 
                    if verbose and batch_idx % 10 == 0 and step % 5 == 0:
                        print(f"  Rollout {batch_idx}: length {i}, final reward {r}, terminated: {terminated}")

            # add tensor to list to stack later 
            batch_rewards.append(rollout_rewards)
            batch_logprobs.append(rollout_logprobs)
            batch_actions.append(rollout_actions)
            batch_dones.append(rollout_dones)
            batch_states.append(rollout_states)
            batch_next_states.append(rollout_next_states)

        # list of experiences to feed into loss computation for this 
        batch_rewards = torch.stack(batch_rewards).to(device)
        batch_logprobs = torch.stack(batch_logprobs).to(device)
        batch_actions = torch.stack(batch_actions).to(device)
        batch_dones = torch.stack(batch_dones).to(device)
        batch_states = torch.stack(batch_states).to(device)
        batch_next_states = torch.stack(batch_next_states).to(device)
        batch_values = value_net(batch_states)
        batch_next_values = value_net(batch_next_states)

        mask = (torch.arange(max_rollout_len, device=device)[None]
                < true_lens[:, None].to(device)).float()  # [b, s], this is some cute tensor golf you should make sure to understand

        if verbose and step % 5 == 0:
            print(f"Computing loss for batch {step}...")

        data = (batch_rewards, batch_logprobs, batch_actions, batch_dones, batch_states, batch_values, batch_next_values, mask)
        loss = loss_fn(policy, value_net, *data, gamma=gamma, max_rollout_len=max_rollout_len)
        loss.backward()    
        opt_pol.step()
        opt_val.step()
        opt_val.zero_grad()
        opt_pol.zero_grad()

        avg_length = true_lens.mean().item()
        if step % 5 == 0:
            min_length = true_lens.min().item()
            max_length = true_lens.max().item()
            if verbose:
                print(f"[{step:4d}/{nsteps:4d}]  ||   Loss = {loss.item():.4f}  ||  Reward(Avg Ep Len) = {avg_length:.1f}  ||  Min/Max Len = {min_length:.1f}/{max_length:.1f}")
            if wandb_log:
                wandb.log({
                    "loss": loss.item(),
                    "avg_episode_length": avg_length,
                    "min_episode_length": min_length,
                    "max_episode_length": max_length
                })

        if early_stop and avg_length > 195:
            print(f"Environment solved at step {step} with average reward {avg_length}!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO on CartPole')
    parser.add_argument('--nsteps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size for policy network')
    parser.add_argument('--early-stop', action='store_true', help='Stop training once environment is solved (avg reward > 195)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting PPO training on CartPole-v1 environment")
        print(f"Arguments: {args}")

    if args.wandb:
        wandb.init(project="cartpole-ppo")
        wandb.config.update(args)
    
    train(
        nsteps=args.nsteps,
        batch_size=args.batch_size,
        max_rollout_len=args.max_rollout_len,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        early_stop=args.early_stop,
        wandb_log=args.wandb,
        verbose=args.verbose
    )