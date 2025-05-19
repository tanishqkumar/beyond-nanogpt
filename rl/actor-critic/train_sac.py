'''
(https://arxiv.org/pdf/1801.01290) Soft Actor-Critic: Off-Policy 
                    Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

DDPG -> SAC: 
    - Adding noise at rollouts is crude, want "learned diversity" 
        -> use entropy term in loss and stochastic policy (diff from DDPG)
        Motivation: SAC actually redefines its notion of value. Rather than Q(s,a) being the 
        action-value of (s,a), we now hypothesize that being in a state where pi(_|s) has high entropy is 
        to some extent *inherently valuable* because it increases future exploration. This means 
        where before we had just Q(s,a) (or max_a Q(s,a) or Q(s, mu(s)), etc) in both the actor/critic 
        losses, we'll now have [Q(s,a) + k * H(log pi(_|s))] where k is a hyperparam constant. 
        -> 

    
    - Two critics instead of one, and we use min(Q1, Q2) to compute targets 
        - Both are trained with E[Q(s, mu(s))] as usual 
        - Motivation: usual actor critic methods, incl A2C and DDPG, are known to suffer from 
            the max_a Q(s,a) being too large when Q is being learned, ie. an overestimate. This was 
            pointed out in several other papers as well, leading to the "double DQN" paper that identifies 
            this problem and introduces the notion of using two Q-networks and using the min of the two Q-values. 
            SAC simply ports this double DQN innovation into continuous world (DDPG). 


TODO 
'''

import gym 
import torch
import torch.nn as nn
from tqdm import tqdm 
import argparse
import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)
env = gym.make('Pendulum-v1', render_mode=None)

# 3 states -> action is a scalar we'll sample from since DDPG requires continuous action space 
class PolicyNet(nn.Module): 
    def __init__(
        self,
        nstates=3,
        action_dim=1,
        hidden_dim=128,
        act=nn.GELU(),
        ls_lower=-20.0, # clamp params from paper
        ls_upper=2.0, 
    ):
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim) # torch.cat([s, a]) --> values 
        self.w2_mean = nn.Linear(hidden_dim, action_dim) # two heads, one for mean one for std 
        self.w2_logstd = nn.Linear(hidden_dim, action_dim) 
        self.act = act  
        self.ls_lower = ls_lower 
        self.ls_upper = ls_upper 

    # deterministic, ie. outputs (mean, logstd)
    def forward(self, x): # x is [nstates], want [nactions] as logits
        h = self.act(self.w1(x))
        mean = self.w2_mean(h)
        log_std = self.w2_logstd(h) # [b, 1]
        log_std = log_std.clamp(self.ls_lower, self.ls_upper) 
        
        return mean, log_std 

    def sample(self, x, eps=1e-5):
        h = self.act(self.w1(x))
        mean = self.w2_mean(h).squeeze(-1) # [b] mean 
        log_std = self.w2_logstd(h).squeeze(-1)
        std = log_std.exp()
        distb = torch.distributions.Normal(mean, std)
        z = distb.sample()
        action = torch.tanh(z) # squash to be reasonable, ie. soft clamp 

        # want to also return logprobs of these actions
            # but have to correct closed-form gaussian logprob for tanh 
            # its known how to do this correction online, i just copied it 
        log_p = distb.log_prob(z)
        log_p = log_p.sum(dim=-1, keepdim=True)
        # subtract fat term to correct logprob for tanh squash 
        log_p = log_p - (2*(z - action + eps).abs().exp().log()).sum(dim=-1, keepdim=True)

        return action, log_p, mean, log_std 


class QNet(nn.Module): 
    def __init__(self, nstates=3, action_dim=1, hidden_dim=128, act=nn.GELU()): 
        super().__init__()
        self.w1 = nn.Linear(nstates + action_dim, hidden_dim) # torch.cat([s, a]) --> values 
        self.w2 = nn.Linear(hidden_dim, 1) # 
        self.act = act 

    def forward(self, states, actions): # [nstates + action_dim]
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)  # Add dimension to match expected shape
        x = torch.cat([states, actions], dim=-1) 
        return self.w2(self.act(self.w1(x))).squeeze(-1)


class ReplayBuffer: 
    def __init__(self, max_buffer_sz=1_001, state_sz=3):
        self.max_buffer_sz = max_buffer_sz
        
        self.states = torch.zeros(max_buffer_sz, state_sz) # states are each 3-vectors tho rewards/dones/actions are scalars
        self.actions = torch.zeros(max_buffer_sz)
        self.rewards = torch.zeros(max_buffer_sz)
        self.next_states = torch.zeros(max_buffer_sz, state_sz)
        self.done = torch.zeros(max_buffer_sz)
        
        self.curr_ptr = 0
        self.size = 0    

    def push(self, sartd): # sartd is a 5-tuple of tensors, s and t are states and next states
        bs = sartd[0].shape[0] # and a = actions, d = dones and r = rewards 
        buffers = [self.states, self.actions, self.rewards, self.next_states, self.done]
        for idx, data in enumerate(sartd):     
            buff = buffers[idx]
            space_left = self.max_buffer_sz - self.curr_ptr
            if bs > space_left:
                buff[self.curr_ptr:].copy_(data[:space_left])
                buff[:bs-space_left].copy_(data[space_left:])
            else:
                buff[self.curr_ptr:self.curr_ptr + bs].copy_(data)

        # use these to make sure we wrap around buffer and return batch correctly 
        self.curr_ptr = (self.curr_ptr + bs) % self.max_buffer_sz
        self.size = min(self.size + bs, self.max_buffer_sz)
        
        return

    def get_batch(self, bs=64): 
        if self.size < bs: 
            bs = self.size
        
        indices = torch.randint(0, self.size, (bs,)) # randomly sample from our buffer, called "experience replay"
        states = self.states[indices]                   
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.done[indices]
        
        return (states, actions, rewards, next_states, dones)

# Bellman MSE to learn the right Q-value function using q_net1
    # policy_net, the actor, gets its grads in a separate loss fn doing 
    # gradient ascent on E[Q(s, policy(s))] since it's deterministic 
def critic_loss_fn(sartd, q_net1, q_net2, policy_net_target, q_net_target1, q_net_target2, gamma=0.995): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    states, actions, rewards, next_states, dones = sartd # unpack 
    
    states = states.to(device) # [batch, nstates]
    next_states = next_states.to(device)
    actions = actions.to(device) # [batch]
    rewards = rewards.to(device)
    dones = dones.to(device)
    
    # twin critics to avoid overestimation 
    with torch.no_grad(): 
        mu = policy_net_target(next_states) # [b] actions
        qtargets1 = q_net_target1(next_states, mu) 
        qtargets2 = q_net_target2(next_states, mu)
        mintargets = torch.min(qtargets1, qtargets2)

    targets = rewards + gamma * (1 - dones) * mintargets 
    
    if qtargets1 > qtargets2: # 2 is min 
        return ((q_net2(states, actions) - targets)**2).mean()
    else: 
        return ((q_net1(states, actions) - targets)**2).mean()

# gradient descent on -Q(s, policy(s)).mean()
def actor_loss_fn(sartd, q_net1, q_net2, policy_net): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    states, _, _, _, _ = sartd # unpack only states
    
    states = states.to(device)
    mu = policy_net(states)

    # twin critics in SAC 
    mean1 = q_net1(states, mu).mean()
    mean2 = q_net2(states, mu).mean()
    return -torch.min(mean1, mean2) # [b] -> scalar

# function to evaluate the current policy, infra, not important conceptually
def eval_policy(policy_net, n_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_env = gym.make('Pendulum-v1')
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        state = torch.tensor(state)
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action = policy_net(state.to(device)).cpu()
            
            next_state, reward, terminated, truncated, _ = eval_env.step([action] if isinstance(action, float) else action)
            done = terminated or truncated
            state = torch.tensor(next_state)
            episode_reward += reward
        
        total_reward += episode_reward
    
    eval_env.close()
    return total_reward / n_episodes

def train(
    nsteps=1_000,
    max_buffer_sz=10_000,
    actor_lr=0.001,
    critic_lr=0.001,
    max_rollout_len=200,
    num_updates_per_step=5,
    train_batch_sz=128,
    rollout_batch_sz=64,
    verbose=False, 
    target_ema=0.995,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = PolicyNet().to(device)
    q_net1 = QNet().to(device)
    q_net2 = QNet().to(device)
    buffer = ReplayBuffer(max_buffer_sz)
 
    opt_critic = torch.optim.Adam(q_net1.parameters(), lr=critic_lr)
    opt_critic.zero_grad()
    opt_actor = torch.optim.Adam(policy_net.parameters(), lr=actor_lr)
    opt_actor.zero_grad()

    # moving average copies for use in policy loss computation for training stability   
        # no need for opts for these since they are just EMA shadows of the policy_net and q_net1 above 
    policy_net_target = PolicyNet().to(device)
    q_net_target1 = QNet().to(device)
    q_net_target2 = QNet().to(device)
    
    # Initialize target networks with the same weights as the online networks
    policy_net_target.load_state_dict(policy_net.state_dict())
    q_net_target1.load_state_dict(q_net1.state_dict())
    q_net_target2.load_state_dict(q_net2.state_dict())

    nstates = 3  # pendulum has 3 states and one (scalar) action 

    ep_len_moving_avg = 0 
    for step in tqdm(range(nsteps), disable=not verbose): 
        rollout_batch_states = torch.zeros(rollout_batch_sz, nstates)
        rollout_batch_actions = torch.zeros(rollout_batch_sz)
        rollout_batch_rewards = torch.zeros(rollout_batch_sz)
        rollout_batch_next_states = torch.zeros(rollout_batch_sz, nstates)
        rollout_batch_done = torch.zeros(rollout_batch_sz)
        
        ttl = 0 # ttl is total experiences (summed over all rollouts in this step)
                    # so far in this step
        for _ in range(rollout_batch_sz): # number of rollouts per step 
            done = False
            state, _ = env.reset()
            state = torch.tensor(state)
            
            i = 0 
            while i < max_rollout_len and not done and ttl < rollout_batch_sz: # one rollout
                # removed noise since now stochastic policy 
                next_action, log_p, mean, log_std  = policy_net.sample(state.to(device)).cpu() 
                
                next_state, r, terminated, truncated, _ = env.step([next_action] if isinstance(next_action, float) else next_action)
                done = terminated or truncated
                next_state = torch.tensor(next_state)

                rollout_batch_states[ttl].copy_(state)
                rollout_batch_actions[ttl] = next_action
                rollout_batch_rewards[ttl] = r
                rollout_batch_next_states[ttl].copy_(next_state)
                rollout_batch_done[ttl] = done
                
                state = next_state if not done else torch.tensor(env.reset()[0])
                i += 1
                ttl += 1
                
                if done:
                    ep_len_moving_avg = 0.9 * ep_len_moving_avg + 0.1 * i
                    i = 0
        
        # push all newly collected experiences into our buffer
        buffer.push((rollout_batch_states, rollout_batch_actions, rollout_batch_rewards, 
                rollout_batch_next_states, rollout_batch_done))


        for i in range(num_updates_per_step): 
            batch = buffer.get_batch(train_batch_sz)

            critic_loss = critic_loss_fn(batch, q_net1, q_net2, policy_net_target, q_net_target1, q_net_target2)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(q_net2.parameters(), max_norm=1.0)
            opt_critic.step()

            actor_loss = actor_loss_fn(batch, q_net1, q_net2, policy_net)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            opt_actor.step()
            
            opt_actor.zero_grad()
            opt_critic.zero_grad() 


        # Run evaluation every 100 nsteps
        if step % 500 == 0:
            avg_reward = eval_policy(policy_net)
            if verbose:
                print(f'[{step}/{nsteps}]: Critic Loss {critic_loss.item()}, Actor Loss {actor_loss.item()},  Avg Reward {avg_reward:.2f}')

        # Soft update of target network parameters, difference from DQN, done every step 
        with torch.no_grad():
            for tp, pp in zip(q_net_target1.parameters(), q_net1.parameters()): 
                tp.data.copy_(target_ema * tp.data + (1-target_ema) * pp.data)
            
            for tp, pp in zip(q_net_target2.parameters(), q_net2.parameters()): 
                tp.data.copy_(target_ema * tp.data + (1-target_ema) * pp.data)

            for tp, pp in zip(policy_net_target.parameters(), policy_net.parameters()): 
                tp.data.copy_(target_ema * tp.data + (1-target_ema) * pp.data)


# same structure hypers/args as most files in this project 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPG on Pendulum')
    parser.add_argument('--nsteps', type=int, default=10_000, help='Number of training nsteps')
    parser.add_argument('--buffer-size', type=int, default=100_000, help='Size of replay buffer')
    parser.add_argument('--actor-lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--epsilon-final', type=float, default=0.01, help='Final exploration rate')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--updates-per-step', type=int, default=3, help='Number of updates per step')
    parser.add_argument('--train-batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--rollout-batch-size', type=int, default=64, help='Rollout batch size')
    parser.add_argument('--target-ema', type=float, default=0.995, help='Smoothing rate for target reset')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    train(
        nsteps=args.nsteps,
        max_buffer_sz=args.buffer_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        epsilon_final=args.epsilon_final,
        max_rollout_len=args.max_rollout_len,
        num_updates_per_step=args.updates_per_step,
        train_batch_sz=args.train_batch_size,
        rollout_batch_sz=args.rollout_batch_size,
        verbose=args.verbose, 
        target_ema=args.target_ema,
    )
