'''
(https://arxiv.org/pdf/1509.02971) Continuous control with deep reinforcement learning

Problem: because of the max_a Q(s,a) in DQN update rule, it doesn't work with large or continuous action spaces. 
Policy gradient / actor critic methods do, but they are online methods that don't work where rollouts are 
very expensive (DQN does, since it uses an offline pool of data). We want the best of both (continuous action space, 
offline data use). 

DDPG does exactly this. You can think of it as (DQN + actor critic) that gets the best of both worlds. 

Changes DQN -> DDPG: 
    - We use Pendulum environment instead of cartpole, since it has continuous action space 
        - nstates is 3, not 4, and nactions is 1 (a scalar in [-2, 2]) instead of 2 (left or right, or ±1)
    - Instead of outputting a mean over action distb like in actor critic on continuous domains,
    we use output as action itself deterministically. This allows Q(s, policy(s)) to be differentiable, 
    whereas it isn't if you sample from policy(s). 
    - We approximate the max_a Q(s,a) in the DQN Bellman-error with Q(s, mu_theta(s)), where theta trained with gradient ascent 
    - Change from updating target net (critic) every n-steps to an EMA

    Key takeaway: 
        In DQN there is no separate actor; your Q-network doubles as policy.
            In DDPG you explicitly split "actor" (μ) and "critic" (Q), and maintain two targets 
            (μ', Q') which are EMA copies of their online counterparts.
            EMA replaces the "copy-every-N steps" trick.


Output looks something like below with default hypers (scroll right for numbers)! 
    You can see reward increasing steadily over time (getting less negative), which means it's working!
            
  5%|█████████▌                                                                                                                                                                                       | 496/10000 [00:11<03:42, 42.79it/s][500/10000]: Critic Loss 0.1101377010345459, Actor Loss 20.757001876831055, Noise Std 0.0955, Avg Reward -1484.45
 10%|███████████████████▏                                                                                                                                                                             | 996/10000 [00:23<03:31, 42.55it/s][1000/10000]: Critic Loss 0.07868068665266037, Actor Loss 35.48066329956055, Noise Std 0.0910, Avg Reward -1431.16
 15%|████████████████████████████▋                                                                                                                                                                   | 1496/10000 [00:35<03:19, 42.63it/s][1500/10000]: Critic Loss 0.09234171360731125, Actor Loss 49.64954376220703, Noise Std 0.0865, Avg Reward -1076.92
 20%|██████████████████████████████████████▎                                                                                                                                                         | 1996/10000 [00:47<03:08, 42.49it/s][2000/10000]: Critic Loss 0.185881108045578, Actor Loss 55.5531005859375, Noise Std 0.0820, Avg Reward -1127.65
 25%|███████████████████████████████████████████████▉                                                                                                                                                | 2496/10000 [00:59<02:56, 42.46it/s][2500/10000]: Critic Loss 0.6029446125030518, Actor Loss 56.4275016784668, Noise Std 0.0775, Avg Reward -742.59
 30%|█████████████████████████████████████████████████████████▌                                                                                                                                      | 2996/10000 [01:11<02:44, 42.64it/s][3000/10000]: Critic Loss 1.1585023403167725, Actor Loss 63.518165588378906, Noise Std 0.0730, Avg Reward -1509.08
 35%|███████████████████████████████████████████████████████████████████                                                                                                                             | 3496/10000 [01:23<02:32, 42.66it/s][3500/10000]: Critic Loss 1.5256766080856323, Actor Loss 68.82205200195312, Noise Std 0.0685, Avg Reward -190.22
 40%|████████████████████████████████████████████████████████████████████████████▋                                                                                                                   | 3996/10000 [01:35<02:20, 42.63it/s][4000/10000]: Critic Loss 2.092850685119629, Actor Loss 68.82980346679688, Noise Std 0.0640, Avg Reward -193.57
 45%|██████████████████████████████████████████████████████████████████████████████████████▎                                                                                                         
 48%|███████████████████████████████████████████████████████████████████████████████████████████▊                       
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
    def __init__(self, nstates=3, action_dim=1, hidden_dim=128, act=nn.GELU(), a_lower=-2.0, a_upper=2.0): 
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim) # torch.cat([s, a]) --> values 
        self.w2 = nn.Linear(hidden_dim, action_dim) # one action, a scalar output on pendulum in [-2., 2.]
        self.act = act  
        self.a_lower = a_lower 
        self.a_upper = a_upper 

    def forward(self, x): # x is [nstates], want [nactions] as logits
        out = self.w2(self.act(self.w1(x))).squeeze(-1)
        return torch.clamp(out, self.a_lower, self.a_upper) # [b] outputs 

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

# Bellman MSE to learn the right Q-value function using q_net
    # policy_net, the actor, gets its grads in a separate loss fn doing 
    # gradient ascent on E[Q(s, policy(s))] since it's deterministic 
def critic_loss_fn(sartd, q_net, policy_net_target, q_net_target, gamma=0.995): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    states, actions, rewards, next_states, dones = sartd # unpack 
    
    # move to device only for forward pass
    states = states.to(device) # [batch, nstates]
    next_states = next_states.to(device)
    actions = actions.to(device) # [batch]
    rewards = rewards.to(device)
    dones = dones.to(device)
    
    # this is critic, ie. QNet, loss, actor (policy_net) trained separately
        # and _target nets are EMA shadows, so not trained at all 
    with torch.no_grad(): 
        mu = policy_net_target(next_states) # [b] actions
        qtargets = q_net_target(next_states, mu) # 

    # a key change from DQN: approx torch.max(next_values) with our new DPG network (targets)
        # which we train with gradient ascent 
    targets = rewards + gamma * (1 - dones) * qtargets 
    
    # empirical bellman mse optimized, just as in DQN, but using our policy mu to approx max_a Q(s, a)
    return ((q_net(states, actions) - targets)**2).mean()

# gradient descent on -Q(s, policy(s)).mean()
def actor_loss_fn(sartd, q_net, policy_net): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    states, _, _, _, _ = sartd # unpack only states
    
    # move to device only for forward pass
    states = states.to(device)

    # Don't use no_grad here - we need gradients for the policy network
    mu = policy_net(states)

    # Pass mu (actions from policy) to q_net to evaluate policy
    return -q_net(states, mu).mean() # [b] -> scalar

# function to evaluate the current policy, infra, not important conceptually
def eval_policy(policy_net, n_episodes=5, a_lower=-2.0, a_upper=2.0):
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
                # No exploration noise during evaluation
                action = torch.clamp(action, a_lower, a_upper).item()
            
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
    epsilon_final=0.01,
    max_rollout_len=200,
    num_updates_per_step=5,
    train_batch_sz=128,
    rollout_batch_sz=64,
    verbose=False, 
    target_ema=0.995,
    fwd_noise_std=0.2, 
    fwd_noise_std_final=0.01,
    a_upper=2.0, 
    a_lower=-2.0, 
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = PolicyNet().to(device)
    q_net = QNet().to(device)
    buffer = ReplayBuffer(max_buffer_sz)
 
    opt_critic = torch.optim.Adam(q_net.parameters(), lr=critic_lr)
    opt_critic.zero_grad()
    opt_actor = torch.optim.Adam(policy_net.parameters(), lr=actor_lr)
    opt_actor.zero_grad()

    # moving average copies for use in policy loss computation for training stability   
        # no need for opts for these since they are just EMA shadows of the policy_net and q_net above 
    policy_net_target = PolicyNet().to(device)
    q_net_target = QNet().to(device)
    
    # Initialize target networks with the same weights as the online networks
    policy_net_target.load_state_dict(policy_net.state_dict())
    q_net_target.load_state_dict(q_net.state_dict())

    epsilon = 1.0  
    epsilon_decay = (1.0 - epsilon_final) / (nsteps * 0.8)  # we decay so we explore more at beginning and less at end
    
    # Setup noise decay schedule
    noise_decay = (fwd_noise_std - fwd_noise_std_final) / nsteps
    current_noise_std = fwd_noise_std
    
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
                next_action = policy_net(state.to(device)).cpu() 
            
                # add noise as in paper to help exploration since now policy is deterministic 
                distb = torch.distributions.Normal(0, current_noise_std)
                noise = distb.sample(next_action.shape)
                next_action = next_action + noise 
                next_action = torch.clamp(next_action, a_lower, a_upper).item() 
                
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

            critic_loss = critic_loss_fn(batch, q_net, policy_net_target, q_net_target)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            opt_critic.step()

            actor_loss = actor_loss_fn(batch, q_net, policy_net)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            opt_actor.step()
            
            opt_actor.zero_grad()
            opt_critic.zero_grad() 
            # subtle: it's important opt_critic is zero_grad here and not before actor bwd
                # because critic is used on actor fwd -E[critic(s,actor(s))] and so is given .grad
                # in opt_actor.bwd() even if those grads aren't applied, so we need to clear those 


        epsilon = max(epsilon_final, epsilon - epsilon_decay)
        current_noise_std = max(fwd_noise_std_final, current_noise_std - noise_decay)

        # Run evaluation every 100 nsteps
        if step % 500 == 0:
            avg_reward = eval_policy(policy_net, a_lower=a_lower, a_upper=a_upper)
            if verbose:
                print(f'[{step}/{nsteps}]: Critic Loss {critic_loss.item()}, Actor Loss {actor_loss.item()}, '
                      f'Noise Std {current_noise_std:.4f}, Avg Reward {avg_reward:.2f}')

        # Soft update of target network parameters, difference from DQN, done every step 
        with torch.no_grad():
            for tp, pp in zip(q_net_target.parameters(), q_net.parameters()): 
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
    parser.add_argument('--fwd-noise-std', type=float, default=0.1, help='Initial forward noise standard deviation')
    parser.add_argument('--fwd-noise-std-final', type=float, default=0.01, help='Final forward noise standard deviation')
    parser.add_argument('--a-upper', type=float, default=2.0, help='Upper bound for actions')
    parser.add_argument('--a-lower', type=float, default=-2.0, help='Lower bound for actions')
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
        fwd_noise_std=args.fwd_noise_std,
        fwd_noise_std_final=args.fwd_noise_std_final,
        a_upper=args.a_upper,
        a_lower=args.a_lower
    )
