'''
(https://arxiv.org/pdf/1705.08439) Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration)

Changes from MPC
    - MPC uses the cross-entropy method taking in (f_theta, s) to output an action, (mean[0]). 
        - This is compute-heavy, even for inference. 
    - Expert iteration just says, hey, let's just distill this "expert algorithm" (MPC, in this case, but 
    the expert is generically any compute-heavy planning algorithm) into a small and lean policy that we can 
    use as a proxy at inference time, which is cheap to evaluate. 
        - Planning is taking (f_theta, s) -> action, by "rollout out against internal world model" 
        - MCTS is another planning algorithm, and the one we'll use here 
            - It's typically used in discrete spaces, compared to eg. cross-entropy method which is for continuous spaces 

This implementation uses LunarLander because it's a nontrivial discrete action space (so MCTS is actually useful). 
This is a "prototype" for the full chess engine we'll be implementing, which will also basically use 
expert iteration with MCTS, albeit in a more scalable way with more bells and whistles. But this is a good 
setting in which to deeply understand the core algorithmic idea. 

See this good TLDR from the paper: 
        Algorithm 1 Expert Iteration
        1: π̂₀ = initial_policy()
        2: π*₀ = build_expert(π̂₀)
        3: for i = 1; i ≤ max_iterations; i++ do
        4:     Sᵢ = sample_self_play(π̂ᵢ₋₁)
        5:     Dᵢ = {(s, imitation_learning_target(π*ᵢ₋₁(s))) | s ∈ Sᵢ}
        6:     π̂ᵢ = train_policy(Dᵢ)
        7:     π*ᵢ = build_expert(π̂ᵢ)

My personal TLDR: it's just repeat[expensive inference-time search, distill resulting (s,a_best) into a small policy]

Note. Some view modern, LLM-based generative reward models as "distilling inference time compute into the base model" using the 
reward signal, and it's interesting that you can think of this idea as coming from expert iteration, or even before. 
The entire premise is to do expensive compute to get optimal (s -> a) during training then distill those 
into a policy (what the paper calls the "apprentice") that is actually what will be used at inference-time. 
The whole process is unsupervised, ie. good (s -> a) pairs are generated in a bootstrapped manner rather than having 
to be fed in. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym 
from dataclasses import dataclass 
from typing import List, Tuple 
from __future__ import annotations 
import argparse
from tqdm import tqdm
from copy import deepcopy 
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# buffer stores/feeds batches for policy, ie. (s, a_opt) pairs so [sdim, adim]
class PolicyBuffer: 
    def __init__(self, buff_sz=10_000, sdim=8, adim=4): 
        self.buff = torch.zeros(buff_sz, sdim + adim, device=device)
        self.ptr = 0 
        self.sdim = sdim 
        self.adim = adim 
        self.size = 0
        self.max_size = buff_sz

    def push(self, batch): # gets [sdim, adim sdim]
        # batch = torch.tensor(batch) #[b, sdim + adim + sdim]
        b, _ = batch.shape 

        if self.ptr + b > self.buff.shape[0]: 
            # loop around to beginning and overwrite 
            overflow = self.ptr + b - self.buff.shape[0]
            self.buff[self.ptr:] = batch[:b-overflow]
            self.buff[:overflow] = batch[b-overflow:]
            self.ptr = overflow 
        else: # usual case
            self.buff[self.ptr:self.ptr+b] = batch 
            self.ptr = (self.ptr + b) % self.max_size

        self.size = min(self.size + b, self.max_size)
        return 

    def get_batch(self, batch_size): 
        if self.size < batch_size:
            batch_size = self.size
            
        indices = torch.randint(0, self.size, (batch_size,), device=device)
        batch = self.buff[indices]
        
        # split the batch into state, action, and delta state
        s_batch = batch[:, :self.sdim]
        a_batch = batch[:, self.sdim:]
        
        return s_batch, a_batch


# stores [s, val] pairs from mcts rollouts to train valueNet on, so [sdim, 1]
class ValueBuffer: 
    def __init__(self, buff_sz=10_000, sdim=8): 
        self.buff = torch.zeros(buff_sz, sdim + 1, device=device)
        self.ptr = 0 
        self.sdim = sdim 
        self.size = 0
        self.max_size = buff_sz

    def push(self, batch): # gets [sdim, 1]
        b, _ = batch.shape 

        if self.ptr + b > self.buff.shape[0]: 
            # loop around to beginning and overwrite 
            overflow = self.ptr + b - self.buff.shape[0]
            self.buff[self.ptr:] = batch[:b-overflow]
            self.buff[:overflow] = batch[b-overflow:]
            self.ptr = overflow 
        else: # usual case
            self.buff[self.ptr:self.ptr+b] = batch 
            self.ptr = (self.ptr + b) % self.max_size

        self.size = min(self.size + b, self.max_size)
        return 

    def get_batch(self, batch_size): 
        if self.size < batch_size:
            batch_size = self.size
            
        indices = torch.randint(0, self.size, (batch_size,), device=device)
        batch = self.buff[indices]
        
        # split the batch into state, val 
        s = batch[:, :self.sdim]
        v = batch[:, -1]
        
        return s, v


# in hindsight, I should probably use one buffer class 
    # with diff .policy_buff and .val_buff and get_value_batch and get_policy_batch 
    # functions, but nbd


class Node: 
    def __init__(self, state, nactions=4, action_taken=None, prior=None, parent=None):
        self.state = state
        self.parent = parent # another Node 
        self.action_taken = action_taken # int in [nactions]
        self.val = 0 
        self.count = 0

        self.children = {} # action -> Node 

        # not aesthetically optimal that prior is tensor and untried_actions a list but nbd
        self.prior = prior if prior is not None else torch.ones(nactions) / nactions # action -> prob from policy(state)
        self.untried_actions = list(range(nactions)) # all actions to start 
    
def is_leaf(node: Node) -> bool: 
    return len(node.untried_actions) > 0

# uses UCB: Node.state -> choice of action maximizing state-action value 
def get_action(node: Node, cpuct: float = 1.0) -> int: 
    # Calculate UCB score for each action
    ucb_scores = {}
    for action in node.children: 
        child = node.children[action] # Node object 
        Q = child.val / child.count if child.count > 0 else 0
        U = cpuct * node.prior[action] * (node.count ** 0.5) / (1 + child.count)
        ucb_scores[action] = Q + U 
    
    # return action with maximum UCB score
    return max(ucb_scores, key=ucb_scores.get)

# update values and visit counts through the path we came 
def backprop(node, final_val): 
    curr = node
    while curr:
        curr.count += 1
        curr.val += final_val 
        curr = curr.parent 
    return 
        
# made a helper because of softmax, ie. when we need distb over actions 
def action_distb(s, policy): 
    return F.softmax(policy(s), dim=-1)


# each MCTS sim in num_sums can be broken down into 
    # [Selection, Expansion, Simulation, Backprop] and I've annotated in the code
def MCTS(
    env: gym.Env, 
    state: torch.Tensor, # starting state s0 from env to use as root.state = s0
    policy_net: nn.Module, # gives priors for exploration for use in action_distb
    value_net: nn.Module, # used to compute values of nodes at terminal state for +V(s_term)
    num_sims: int = 100, 
    max_depth: int = 20, # depth of exploration starting from a leaf 
    cpuct: float = 1.0, # exploration constant
    discount: float = 0.99, 
    ) -> int: 
    
    nactions = env.action_space.n

    root = Node(
        state, 
        nactions=nactions, 
        action_taken=None, 
        prior=action_distb(state, policy_net), 
        parent=None, 
    )
    
    for _ in range(num_sims): 
        env_copy = deepcopy(env) # clone this original env so we don't pollute 
        node = root 
        depth = 0
        done = False 
        ttl_r = 0

        # Selection 
        while not is_leaf(node) and not done: 
            next_a = get_action(node, cpuct=cpuct)  # selection 

            if next_a in node.children: # seen it before, go to it 
                node = node.children[next_a]
                new_state, r, done, _, _ = env_copy.step(next_a)
                ttl_r += r * (discount ** depth)

        # Expansion on a non-terminal leaf node 
        if is_leaf(node) and not done and depth < max_depth:
            depth += 1
            next_a = node.untried_actions.pop(0)
            new_state, r, done, _, _ = env_copy.step(next_a)
            ttl_r += r * (discount ** (depth + 1))
            new_child = Node(
                new_state,
                nactions=nactions,
                action_taken=next_a,
                prior=action_distb(torch.tensor(new_state), policy_net),
                parent=node,
            )
            node.children[next_a] = new_child
            node = new_child

        # Simulation, do random rollout from node 
            # don't use UCB -- that's for selection. This is supposed to be "noisy exploration"
        while depth < max_depth and not done: 
            next_a = env.action_space.sample() # random rollout 
            _, r, done, _, _ = env.step(next_a)
            ttl_r += (discount ** depth) * r
            depth += 1

        # rollout complete, get final value by casework on how we ended simulation 
        if done:
            # terminal state reached
            final_val = ttl_r  
        else: # add V(s') where s' is our current node.state, here we terminated
                    # because we reached max_depth 
            final_val = ttl_r + value_net(torch.tensor(node.state)).item()

        # Backpropagation
        backprop(node, final_val)            
    
    # action which led to most visited state is our output as best action 
    return max(
        root.children.keys(), 
        key= lambda a: root.children[a].count 
    )

# [b, sdim] -> [b, adim]
class PolicyNet: 
    def __init__(self, sdim=8, adim=4, hidden_dim=128, act=nn.GELU()): 
        self.sdim = sdim 
        self.adim = adim 
        self.hidden_dim = hidden_dim 
        self.act = act

        self.net = nn.Sequential(
            nn.Linear(sdim, hidden_dim), 
            self.act, 
            nn.Linear(hidden_dim, adim)
        )

    def forward(self, x):  
        return self.net(x) 
    
# [b, sdim] -> [b, 1]
class ValueNet: 
    def __init__(self, sdim=8, hidden_dim=128, act=nn.GELU()): 
        self.sdim = sdim 
        self.hidden_dim = hidden_dim 
        self.act = act

        self.net = nn.Sequential(
            nn.Linear(sdim, hidden_dim), 
            self.act, 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  
        return self.net(x) 


@dataclass 
class TrainConfig: 
    lr: float = 1e-4 
    batch_size: int = 128 
    nsteps: int = 1_000
    num_rollouts: int = 100
    buff_sz: int = 10_000
    num_sims: int = 500
    num_policy_updates_per_step: int = 10 
    eval_policy_every: int = 100 
    sdim: int = 8 
    adim: int = 4
    env_name: str = 'LunarLander-v2'
    cpuct: float = 1.0 
    discount: float = 0.99 
    

def train(
    cfg: TrainConfig = TrainConfig()
): 
    env = gym.make(cfg.env_name)
    policy_net = PolicyNet(); value_net = ValueNet()
    policy_opt = torch.optim.AdamW(policy_net.parameters(), lr=cfg.lr)
    value_opt = torch.optim.AdamW(value_net.parameters(), lr=cfg.lr)
    policy_buffer = PolicyBuffer()
    value_buffer = ValueBuffer()


    MCTS_ARGS = (
        env.clone(), 
        s, 
        policy_net, 
        value_net, 
        cfg.num_sums, 
        cfg.max_depth, 
        cfg.cpuct, 
        cfg.discount
    )

    for step in range(cfg.nsteps): 
        for _ in range(cfg.num_rollouts): 
            done = False 
            s, _ = env.reset() 
            while not done: 
                a_next = MCTS(*MCTS_ARGS)
                singleton_batch = torch.tensor([[s, a_next]]) # [1, sdim adim]
                policy_buffer.push(singleton_batch) 
                _, _, done, _, _ = env.step(a_next)

        for _ in range(cfg.num_policy_updates_per_step): 
            state_batch, opt_action_batch = policy_buffer.get_batch(cfg.batch_size)
            loss = F.cross_entropy(policy_net(state_batch), opt_action_batch)
            loss.backward()
            policy_opt.step()
            policy_opt.zero_grad()

        # update value net 



        if step % cfg.eval_policy_every == 0: 
            print(f'Policy gets reward {eval(policy_net)}')

        
# TODO: add argparse to take in args and launch train, etc 