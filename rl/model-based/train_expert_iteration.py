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

Some key takeaways overall: 
    - Expert iteration just refers to repeat[Plan (inference-time search, populate buffer), distill (train policy from buffer)]
    - The actual "planning" algorithm is agnostic -- it's often MCTS, but it could easily be MPC too. 
    - The heart of "expert iteration" itself is very simple, it's essentially just the train() loop 
    - The bulk of this file goes toward implementing MCTS correctly
        - This is worth doing, because LunarLander is a simple discrete "warm up" setting, and we'll need to 
            do MCTS in a more serious, scalable way for the neural chess engine in `rl/chess` so it's worth 
            understanding carefully in this "toy" setting. 

Note 1. Expert iteration is TECHNICALLY NOT MODEL-BASED RL BECAUSE WE NEVER LEARN A DYNAMICS
MODEL f_theta (see train_mpc.py for an example). It's just typically lumped in with model based methods 
because it involves *planning* (inference-time search with MCTS), a key feature of 
model-based RL (you learn the model to plan using rollouts). Thus the philosophy of 
planning moves using search is the same. 


Note 2. Some view modern, LLM-based generative reward models as "distilling inference time compute into the base model" using the 
reward signal, and it's interesting that you can think of this idea as coming from expert iteration, or even before. 
The entire premise is to do expensive compute to get optimal (s -> a) during training then distill those 
into a policy (what the paper calls the "apprentice") that is actually what will be used at inference-time. 
The whole process is unsupervised, ie. good (s -> a) pairs are generated in a bootstrapped manner rather than having 
to be fed in. 
'''

from __future__ import annotations 
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym 
from dataclasses import dataclass 
from typing import List, Tuple 
import argparse
from tqdm import tqdm
from copy import deepcopy 
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.manual_seed(69_420)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# buffer stores/feeds batches for policy, ie. (s, a_opt) pairs so [sdim, adim]
class Buffer: 
    def __init__(self, policy_buff_sz=10_000, value_buff_sz=10_000, sdim=8, adim=4): 
        self.policy_buff = torch.zeros(policy_buff_sz, sdim + adim, device=device)
        self.value_buff = torch.zeros(value_buff_sz, sdim + 1, device=device)
        
        self.policy_ptr = 0 
        self.value_ptr = 0 
        self.policy_size = 0
        self.value_size = 0

        self.sdim = sdim 
        self.adim = adim 
        self.policy_max_size = policy_buff_sz
        self.value_max_size = value_buff_sz

    def push(self, batch, policy=True): # gets [sdim, adim sdim]
        buff = self.policy_buff if policy else self.value_buff
        ptr = self.policy_ptr if policy else self.value_ptr 
        max_size = self.policy_max_size if policy else self.value_max_size

        batch = batch.to(device)
        b, _ = batch.shape 

        if ptr + b > buff.shape[0]: 
            # loop around to beginning and overwrite 
            overflow = ptr + b - buff.shape[0]
            buff[ptr:] = batch[:b-overflow]
            buff[:overflow] = batch[b-overflow:]
            ptr = overflow 
        else: # usual case
            buff[ptr:ptr+b] = batch 
            ptr = (ptr + b) % max_size

        if policy:
            self.policy_size = min(self.policy_size + b, self.policy_max_size)
            self.policy_ptr = ptr
        else:
            self.value_size = min(self.value_size + b, self.value_max_size)
            self.value_ptr = ptr
        return 

    def get_batch_policy(self, batch_size): 
        if self.policy_size < batch_size:
            batch_size = self.policy_size
            
        indices = torch.randint(0, self.policy_size, (batch_size,), device=device)
        batch = self.policy_buff[indices]
        
        # split the batch into state, action states
        s_batch = batch[:, :self.sdim]
        a_batch = batch[:, self.sdim:]
        
        return s_batch, a_batch

    def get_batch_value(self, batch_size): 
        if self.value_size < batch_size:
            batch_size = self.value_size
            
        indices = torch.randint(0, self.value_size, (batch_size,), device=device)
        batch = self.value_buff[indices]
        
        # split the batch into state, val 
        s = batch[:, :self.sdim]
        v = batch[:, -1]
        
        return s, v

class Node: 
    def __init__(self, state, nactions=4, action_taken=None, prior=None, parent=None):
        self.state = state
        self.parent = parent # another Node 
        self.action_taken = action_taken # int in [nactions]
        self.val = 0 
        self.count = 0

        self.children = {} # action -> Node 

        # not aesthetically optimal that prior is tensor and untried_actions a list but nbd
        self.prior = prior if prior is not None else torch.ones(nactions, device=device) / nactions # action -> prob from policy(state)
        self.untried_actions = list(range(nactions)) # all actions to start 
    
def is_leaf(node: Node) -> bool: 
    return len(node.untried_actions) > 0

# uses UCB: Node.state -> choice of action maximizing state-action value 
def get_action(node: Node, cpuct: float = 1.0) -> int: 
    # calculate UCB score for each action
    ucb_scores = {}
    for action in node.children: 
        child = node.children[action] # Node object 
        Q = child.val / child.count if child.count > 0 else 0
        U = cpuct * node.prior[action] * (node.count ** 0.5) / (1 + child.count)
        ucb_scores[action] = Q + U 
    
    # return action with maximum UCB score
    return max(ucb_scores, key=ucb_scores.get)

# update values and visit counts of nodes visited in [select, expand]
# Notes: 
    # - we *only touch (backprop up to) nodes seen in [select, expand]*, not rollout 
        # - this is bc a node only needs to be backprop'd once it's *in the tree*
        # - being "in the tree" means being  Node with pointers to parents and stats like Node.count, etc
        # - a state only becomes a node when it is *expanded*, at which point it's included in future backprop 
def backprop(node: Node, rewards: List[float], leaf_return: float, discount: float, buffer: Buffer): 
    # assume node = new_child that is outcome of expansion, we'll walk it backwards 
    G = leaf_return # running future reward
    
    # backward induction to propagate reward
    for r in reversed(rewards): 
        # return from this node's POV 
        G_node = r + discount * G

        node.count += 1
        node.val += G_node 

        state_t, g_t = make_2tensor(node.state), make_2tensor(G_node)
        singleton_batch = torch.cat([state_t, g_t], dim=1)
        buffer.push(singleton_batch, policy=False) 

        G = G_node
        node = node.parent 
        if node is None:
            break

    pass 
        
# made a helper because of softmax, ie. when we need distb over actions 
def action_distb(s, policy): 
    return F.softmax(policy(s), dim=-1)

# helper taking an object/scalar t -> [1, t_dim] 
def make_2tensor(t): 
    return torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)

# takes leaf and rollouts it out with random policy, returns total value at leaf for backprop up the chain 
def rollout(env: gym.Env, max_rollout_depth: int, discount: float, value_net: nn.Module, done: bool) -> float: 
    rollout_reward = 0. 
    rollout_depth = 0

    while rollout_depth < max_rollout_depth and not done: 
        next_a = env.action_space.sample() # random rollout 
        s, r, term, trunc, _ = env.step(next_a)
        done = trunc or term 
        rollout_reward +=  r * (discount ** rollout_depth)
        rollout_depth += 1

    if done: 
        val_end_state = 0. 
    elif rollout_depth >= max_rollout_depth: 
        val_end_state = value_net(make_2tensor(s)).item()

    return rollout_reward + val_end_state

def expand(node: Node, env: gym.Env, nactions: int, policy_net: nn.Module, rewards: List[float]): 
    # expand an unexplored action with the highest prior
    next_a = max(
        node.untried_actions, 
        key=lambda a: node.prior[a].item()
    )
    node.untried_actions.remove(next_a)

    new_state, r, term, trunc, _ = env.step(next_a)
    rewards.append(r)
    done = term or trunc 
    new_child = Node(
        new_state,
        nactions=nactions,
        action_taken=next_a,
        prior=action_distb(make_2tensor(new_state), policy_net).squeeze(0),
        parent=node,
    )
    node.children[next_a] = new_child
    
    return new_child, done, rewards 


def select_nonterminal(node: Node, env: gym.Env, cpuct: float, tree_max_depth: int, rewards: List[float]): 
    tree_depth = 0
    done = False

    while not is_leaf(node) and not done and tree_depth < tree_max_depth: 
        # selection is guided by UCB (upper confidence bound) policy implemented in get_action
            # this balances exploration/exploitation and is conceptually the heart of MCTS
                # along with the random rollouts 
        next_a = get_action(node, cpuct=cpuct)  

        if next_a in node.children: # seen it before, go to it 
            node = node.children[next_a]
            _, r, term, trunc, _ = env.step(next_a)
            rewards.append(r)
            done = term or trunc
            tree_depth += 1

    return node, done, rewards

# each MCTS sim in num_sums can be broken down into 
    # [Selection, Expansion, Simulation, Backprop] and I've annotated in the code as [1, 2, 3, 4]
def MCTS(
        env: gym.Env, 
        state: torch.Tensor, # starting state s0 from env to use as root.state = s0
        policy_net: nn.Module, # gives priors for exploration for use in action_distb
        value_net: nn.Module, # used to compute values of nodes at terminal state for +V(s_term)
        buffer: Buffer, 
        num_sims: int = 100, 
        tree_max_depth: int = 20, # depth of exploration starting from a leaf 
        rollout_max_depth: int = 20, # depth of exploration starting from a leaf 
        cpuct: float = 1.0, # exploration constant
        discount: float = 0.99, 
    ) -> int: # output the optimal action, an integer in [nactions/adim]
    
    nactions = env.action_space.n
    
    root = Node(
        state, 
        nactions=nactions, 
        action_taken=None, 
        prior=action_distb(make_2tensor(state), policy_net).squeeze(0), 
        parent=None, 
    )
    
    for _ in range(num_sims): 
        env_copy = deepcopy(env) # clone this original env so we don't pollute 
        node = root 
        tree_depth = 0 # tree depth is steps during selection, rollut depth during rollout 
        done = False 
        rewards = [] # bookkeeping of reward trajecotry for backprop 

        # 1. Selection, find non-terminal node, ie. has unexplored action 
        node, done, rewards = select_nonterminal(node, env_copy, cpuct, tree_max_depth, rewards)

        # 2. Expansion (create a child at non-terminal node, will rollout from child)
        if is_leaf(node) and not done:
            new_child, done, rewards = expand(node, env_copy, nactions, policy_net, rewards)
            node = new_child
            tree_depth += 1

        # 3. Simulation, explore that child and get its value w/ random policy 
        leaf_return = rollout(env_copy, rollout_max_depth, discount, value_net, done) 

        # 4. Backpropagation, update all earlier nodes in the path with the value information 
            # this includes adding to buffer examples to train the value net 
        backprop(node, rewards, leaf_return, discount, buffer) 
    
    # action which led to most visited state is our output as best action 
    return max(
        root.children.keys(), 
        key= lambda a: root.children[a].count 
    )


# [b, sdim] -> [b, adim]
class PolicyNet(nn.Module): 
    def __init__(self, sdim=8, adim=4, hidden_dim=128, nlayers=2, act=nn.GELU()): 
        super().__init__()
        self.sdim = sdim 
        self.adim = adim 
        self.hidden_dim = hidden_dim 
        self.nlayers = nlayers
        self.act = act

        layers = []
        layers.append(nn.Linear(sdim, hidden_dim))
        layers.append(self.act)
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act)
        
        layers.append(nn.Linear(hidden_dim, adim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):  
        return self.net(x) 
    
# [b, sdim] -> [b, 1]
class ValueNet(nn.Module): 
    def __init__(self, sdim=8, hidden_dim=128, nlayers=2, act=nn.GELU()): 
        super().__init__()
        self.sdim = sdim 
        self.hidden_dim = hidden_dim 
        self.nlayers = nlayers
        self.act = act

        layers = []
        layers.append(nn.Linear(sdim, hidden_dim))
        layers.append(self.act)
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act)
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):  
        return self.net(x) 


@dataclass 
class TrainConfig: 
    lr: float = 1e-4 
    batch_size: int = 128 
    nsteps: int = 1_000
    num_rollouts: int = 100
    policy_buff_sz: int = 10_000
    value_buff_sz: int = 10_000
    num_sims: int = 500
    tree_max_depth: int = 20
    rollout_max_depth: int = 20
    num_policy_updates_per_step: int = 10 
    eval_policy_every: int = 100 
    sdim: int = 8 
    adim: int = 4
    hidden_dim: int = 128
    nlayers: int = 2
    env_name: str = 'LunarLander-v2'
    cpuct: float = 1.0 
    discount: float = 0.99 


def eval_policy(policy_net, env_name='LunarLander-v2', num_episodes=10):
    """Evaluate policy by running episodes and returning average reward"""
    env = gym.make(env_name)
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_probs = F.softmax(policy_net(state_tensor), dim=-1)
                action = torch.argmax(action_probs).item()
            
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
        total_reward += episode_reward
    
    env.close()
    return total_reward / num_episodes


def train(cfg: TrainConfig = TrainConfig(), verbose: bool = False, use_wandb: bool = False): 
    if use_wandb:
        import wandb
        wandb.init(project="expert-iteration", config=cfg.__dict__)
    
    env = gym.make(cfg.env_name)
    policy_net = PolicyNet(sdim=cfg.sdim, adim=cfg.adim, hidden_dim=cfg.hidden_dim, nlayers=cfg.nlayers).to(device)
    value_net = ValueNet(sdim=cfg.sdim, hidden_dim=cfg.hidden_dim, nlayers=cfg.nlayers).to(device)
    policy_opt = torch.optim.AdamW(policy_net.parameters(), lr=cfg.lr)
    value_opt = torch.optim.AdamW(value_net.parameters(), lr=cfg.lr)
    buffer = Buffer(policy_buff_sz=cfg.policy_buff_sz, value_buff_sz=cfg.value_buff_sz, sdim=cfg.sdim, adim=cfg.adim)

    for step in range(cfg.nsteps): 
        if verbose and step % 100 == 0:
            print(f"Step {step}/{cfg.nsteps}")
            
        for _ in range(cfg.num_rollouts): 
            done = False 
            s, _ = env.reset() 
            while not done: 
                with torch.no_grad(): 
                    a_next = MCTS(
                        env, 
                        s, 
                        policy_net, 
                        value_net, 
                        buffer,
                        cfg.num_sims, 
                        cfg.tree_max_depth, 
                        cfg.rollout_max_depth, 
                        cfg.cpuct, 
                        cfg.discount
                    )

                # convert action to one-hot encoding for discrete actions
                a_next_onehot = torch.zeros(cfg.adim, dtype=torch.float32, device=device)
                a_next_onehot[a_next] = 1.0
                a_next_onehot = a_next_onehot.unsqueeze(0) # [1, adim]
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) # [1, sdim]

                singleton_batch = torch.cat([s_t, a_next_onehot], dim=1) # [1, sdim + adim]
                buffer.push(singleton_batch, policy=True)  # push policy examples into buffer 
                s, _, done, _, _ = env.step(a_next)

        policy_losses = []
        for _ in range(cfg.num_policy_updates_per_step): 
            state_batch, opt_action_batch = buffer.get_batch_policy(cfg.batch_size)
            # convert one-hot back to class indices for cross entropy
            opt_action_indices = torch.argmax(opt_action_batch, dim=1)
            loss = F.cross_entropy(policy_net(state_batch), opt_action_indices)
            loss.backward()
            policy_opt.step()
            policy_opt.zero_grad()
            policy_losses.append(loss.item())

        # update value net 
        state_batch, val_batch = buffer.get_batch_value(cfg.batch_size)
        val_batch = val_batch.unsqueeze(1)  # [b, 1]
        value_loss = F.mse_loss(value_net(state_batch), val_batch)
        value_loss.backward()
        value_opt.step()
        value_opt.zero_grad() 

        # logging
        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        
        if verbose and step % 100 == 0:
            print(f"Step {step}: Policy Loss {avg_policy_loss:.6f}, Value Loss {value_loss.item():.6f}")
            print(f"Buffer sizes - Policy: {buffer.policy_size}, Value: {buffer.value_size}")

        if use_wandb:
            wandb.log({
                "step": step,
                "policy_loss": avg_policy_loss,
                "value_loss": value_loss.item(),
                "policy_buffer_size": buffer.policy_size,
                "value_buffer_size": buffer.value_size
            })

        if step % cfg.eval_policy_every == 0: 
            avg_reward = eval_policy(policy_net, cfg.env_name)
            if verbose:
                print(f'Step {step}: Policy average reward {avg_reward:.2f}')
            
            if use_wandb:
                wandb.log({
                    "step": step,
                    "avg_reward": avg_reward
                })

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train Expert Iteration with MCTS')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--nsteps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--num_rollouts', type=int, default=50, help='Number of rollouts per step')
    parser.add_argument('--policy_buff_sz', type=int, default=5000, help='Policy buffer size')
    parser.add_argument('--value_buff_sz', type=int, default=5000, help='Value buffer size')
    parser.add_argument('--num_sims', type=int, default=100, help='Number of MCTS simulations')
    parser.add_argument('--tree_max_depth', type=int, default=15, help='Maximum depth for MCTS tree exploration')
    parser.add_argument('--rollout_max_depth', type=int, default=15, help='Maximum depth for MCTS rollouts')
    parser.add_argument('--num_policy_updates_per_step', type=int, default=5, help='Policy updates per step')
    parser.add_argument('--eval_policy_every', type=int, default=50, help='Evaluate policy every N steps')
    parser.add_argument('--sdim', type=int, default=8, help='State dimension')
    parser.add_argument('--adim', type=int, default=4, help='Action dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for networks')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers for networks')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2', help='Environment name')
    parser.add_argument('--cpuct', type=float, default=1.4, help='CPUCT exploration constant')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        nsteps=args.nsteps,
        num_rollouts=args.num_rollouts,
        policy_buff_sz=args.policy_buff_sz,
        value_buff_sz=args.value_buff_sz,
        num_sims=args.num_sims,
        tree_max_depth=args.tree_max_depth,
        rollout_max_depth=args.rollout_max_depth,
        num_policy_updates_per_step=args.num_policy_updates_per_step,
        eval_policy_every=args.eval_policy_every,
        sdim=args.sdim,
        adim=args.adim,
        hidden_dim=args.hidden_dim,
        nlayers=args.nlayers,
        env_name=args.env_name,
        cpuct=args.cpuct,
        discount=args.discount
    )
    
    train(cfg, verbose=args.verbose, use_wandb=args.wandb)


if __name__ == "__main__":
    main()