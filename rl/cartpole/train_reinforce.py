## IN PROGRESS... ## 

import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartPole-v1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyNet(nn.Module): # states -> action probs 
    def __init__(self, nstates=4, nactions=2, hidden_dim=128, act=nn.GELU()): 
        super().__init__()
        self.w1 = nn.Linear(nstates, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, nactions)
        self.act = act 

    def forward(self, x): # x is [nstates], want [nactions] as logits
        return self.w2(self.act(self.w1(x))) 

def loss_fn(rollout, mask, gamma=0.99, max_rollout_len=50): 
    rewards, logprobs = rollout[0], rollout[1] # [b, msl]
    # transform r into return R_t
    gammas = torch.ones(max_rollout_len, device=device).fill_(gamma).cumprod(dim=1)
    rewards = gammas * rewards # discount before cumsum 
    flipped = rewards.flip(dims=[1])
    returns = torch.cumsum(flipped, dim=0).flip(dims=[1]) # [b, msl]

    # return -Reward = -E[R_t * logprob] as loss 
    loss_t = -mask * returns * logprobs # [true_len]
    return (loss_t).sum() / mask.sum()

policy = PolicyNet().to(device)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
done = False 
max_rollout_len = 50 # max seqlen in llm speak 
b = 64 
rollout = torch.zeros(2, b, max_rollout_len).to(device) # [2, b, s] where 0 = rewards and 1 = logprobs for each token 
true_lens = torch.zeros(b) # entry i is true len of batch element i, use this with arange to mask mask of size [b, s]
nsteps = 100 
n_grads_per_step = 3 

for step in range(nsteps): 
  # on-policy, ie. the batch we'll step on a few times in training step has JUST been generate dby our current policy 
  for batch_idx in range(b): # in contrast to dqn which stores a buffer where we may be learning from 
    i = 0 
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
        rollout[0, batch_idx, i] = r
        rollout[1, batch_idx, i] = logprobs[next_action]

        # this assignment breaks the computation graph 
        # instead of keeping/assigning to a tensor of rollouts, 
        # we should just accumulate the loss object by adding -r * logp for each rollout one at a time 
        # and taking a running mean and calling .backward() at the end of batch creation, so that it 
        # just becomes an "episode"
        
        state = torch.tensor(next_state).to(device)
        i+=1
        
        if done: 
            true_lens[batch_idx] = i 

  # rollout.shape, true_lens.shape # [2, b, s]
  mask = (torch.arange(max_rollout_len, device=device)[None]
        < true_lens[:,None].to(device)).float() # [b, s], this is some cute tensor golf you should make sure to understand
  
  # take a few steps using the constructed batch 
  for _ in range(n_grads_per_step): 
    loss = loss_fn(rollout, mask, max_rollout_len=max_rollout_len)
    loss.backward()    
    opt.step()
    opt.zero_grad()


  if step % 2 == 0: 
    print(f'[{step}/{nsteps}]: {loss.item()}')