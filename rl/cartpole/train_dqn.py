'''
State: 4 values representing the cart-pole system
- Cart Position: Position of the cart on the track
- Cart Velocity: Velocity of the cart  
- Pole Angle: Angle of the pole with respect to vertical
- Pole Angular Velocity: Rate of change of the pole angle

Environment: CartPole-v1
- Goal: Keep the pole balanced by moving the cart left or right
- Reward: +1 for each timestep the pole remains upright
- Episode ends when:
  1. Pole angle is more than 15 degrees from vertical
  2. Cart position is more than 2.4 units from center
  3. Episode length reaches 500 timesteps
- Actions: 0 (push cart left), 1 (push cart right)


Q(s) returns q-values for all actions, ie. 2 outputs for us 
Loss = (  Q(s,a) - [R(s,a) + gamma * max_a(Q(s', a))]  )^2
ie. given (sart) and gamma where t is next state
where policy_net is what we train and target_net defines the target
    note, these should of course technically be the same, using two different nets  
    where target is reset to policy every few steps is a trick done for training stability
    so that target is not moving too much as the policy improves itself in a "bootstrapping" manner 
(policy_net(s)[a] - (r + gamma * max_a(target_net(t))))**2
Bellman: Q(s, a) = R(s,a) + gamma * max_a(Q(s', a))
'''

import gym 
import torch
import torch.nn as nn
from tqdm import tqdm 
import argparse

env = gym.make('CartPole-v1')

class PolicyNet(nn.Module): 
    def __init__(self, nstates=4, nactions=2, hidden_dim=128, act=nn.GELU()): 
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, nactions)
        self.act = act 

    def forward(self, x): # x is [nstates], want [nactions] as logits
        return self.w2(self.act(self.w1(x))) 

# Environment API provided by the gym library: 
# env.reset() -> initial_state: Returns initial observation
# env.step(action) -> (next_state, reward, done, info): Takes action, returns next state, reward, terminal flag, and debug info
# env.render() -> None: Visualizes current environment state
# env.close() -> None: Closes the environment

class ReplayBuffer: 
    def __init__(self, max_buffer_sz=1001, state_sz=4):
        self.max_buffer_sz = max_buffer_sz
        
        self.states = torch.zeros(max_buffer_sz, state_sz) # states are each 4-vectors tho rewards/dones/actions are scalars
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
        
        indices = torch.randint(0, self.size, (bs,)) # randomly sample from our buffer 
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.done[indices]
        
        return (states, actions, rewards, next_states, dones)


def loss_fn(sartd, policy_net, target_net, gamma=0.9): # [b, 5] tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # states = sartd[:, 0] is [b, state_dim]
    # then logits = policy(states) is [b, nactions]
    states, actions, rewards, next_states, dones = sartd # unpack 
    
    # Move to device only for forward pass
    states = states.to(device)
    next_states = next_states.to(device)
    actions = actions.to(device) # [batch]
    rewards = rewards.to(device)
    dones = dones.to(device)
    
    logits = policy_net(states)
    # logits is [b, nactions]
    # want pred_qvals to be [b] where we get the logit for the right action 
    
    # never backprop target net, its moving avg of policy net by hand 
    with torch.no_grad(): 
        next_values = target_net(next_states) # [b, nactions]

    # this is the key math: the update rule for deep q networks 
    # we just want the Bellman equations satisfied, so we penalize the deviation from them 
    targets = rewards + gamma * (1 - dones) * torch.max(next_values, dim=1)[0] 

    # this line is subtle, one may be tempted to do (I was)
    # the more naive logits[:, actions] or logits[:, actions.long()]
    # but BOTH will FAIL. 
    pred_qvals = logits.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    
    
    return ((pred_qvals - targets)**2).mean()

def train(epochs=4000, max_buffer_sz=10000, lr=0.0001, epsilon_final=0.01,
          max_rollout_len=200, num_updates_per_step=5,
          train_batch_sz=512, rollout_batch_sz=128, reset_target=100, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNet().to(device)
    target_net = PolicyNet().to(device)

    buffer = ReplayBuffer(max_buffer_sz)
    opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    opt.zero_grad()

    epsilon = 1.0  
    epsilon_decay = (1.0 - epsilon_final) / (epochs * 0.8)  # we decay so we explore more at beginning and less at end
    nstates = 4  # cartpole has 4 state dimensions for (pos, velocity) of both (cart, pole)
    nactions = 2  # cartpole has 2 actions (left/right)

    ep_len_moving_avg = 0 
    for step in tqdm(range(epochs)): 
        rollout_batch_states = torch.zeros(rollout_batch_sz, nstates)
        rollout_batch_actions = torch.zeros(rollout_batch_sz)
        rollout_batch_rewards = torch.zeros(rollout_batch_sz)
        rollout_batch_next_states = torch.zeros(rollout_batch_sz, nstates)
        rollout_batch_done = torch.zeros(rollout_batch_sz)
        
        ttl = 0
        for _ in range(rollout_batch_sz):
            done = False
            state, _ = env.reset()
            state = torch.tensor(state)
            
            i = 0 
            while i < max_rollout_len and not done and ttl < rollout_batch_sz:
                logits = policy_net(state.to(device)) 
                if torch.rand(1).item() > epsilon: 
                    next_action = torch.argmax(logits).item()
                else: 
                    next_action = torch.randint(0, nactions, (1,)).item()
                next_state, r, terminated, truncated, _ = env.step(next_action)
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

        buffer.push((rollout_batch_states, rollout_batch_actions, rollout_batch_rewards, 
                rollout_batch_next_states, rollout_batch_done))

        for i in range(num_updates_per_step): 
            batch = buffer.get_batch(train_batch_sz)
            loss = loss_fn(batch, policy_net, target_net)
            loss.backward()
            opt.step()
            opt.zero_grad()

        epsilon = max(epsilon_final, epsilon - epsilon_decay)

        if step % 200 == 0: 
            print(f'[{step}/{epochs}]: Loss {loss.item()}, Reward (moving avg ep len): {round(ep_len_moving_avg, 3)}')

        if step % reset_target == 0: 
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on CartPole')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--buffer-size', type=int, default=2000, help='Size of replay buffer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon-final', type=float, default=0.01, help='Final exploration rate')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--updates-per-step', type=int, default=5, help='Number of updates per step')
    parser.add_argument('--train-batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--rollout-batch-size', type=int, default=128, help='Rollout batch size')
    parser.add_argument('--reset-target', type=int, default=100, help='Steps between target network updates')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        max_buffer_sz=args.buffer_size,
        lr=args.lr,
        epsilon_final=args.epsilon_final,
        max_rollout_len=args.max_rollout_len,
        num_updates_per_step=args.updates_per_step,
        train_batch_sz=args.train_batch_size,
        rollout_batch_sz=args.rollout_batch_size,
        reset_target=args.reset_target,
        verbose=args.verbose
    )
