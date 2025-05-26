from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import chess
from typing import Optional, List, Tuple
from config import MCTSNodeConfig, MCTSConfig
from env import ChessEnv
from utils import move2index, eval_pos, board2input, legal_mask
from collections import deque 

class Node: 
    def __init__(self, cfg: MCTSNodeConfig): 
        # state: chess.Board, parent: Optional[Node] = None, nactions: int = 4672
        self.state = cfg.state # a chess board 
        self.val = 0 
        self.count = 0 
        self.parent = cfg.parent 
        self.priors = {} # action -> float
        self.action_taken = cfg.action_taken
        self.untried_children = list(map(lambda move: move2index(self.state, move), self.state.legal_moves))
        self.children = {}
        
        # will need history to compute val/policy of state in model fwd pass
        if cfg.history and len(cfg.history) > 0:
            self.history = cfg.history
        elif self.parent:
            self.history = self.parent.history.copy()
            self.history.append(self.state.copy())
        else:
            self.history = deque([self.state.copy()], maxlen=8)
    
    def is_fully_expanded(self) -> bool: 
        return len(self.untried_children) == 0
    
    def is_terminal(self) -> bool: 
        return self.state.is_game_over()


def get_action_ucb(node: Node, cpuct: float = 1.0) -> int: 
    if not node.children: 
        raise ValueError("Error: trying to get_action_ucb from node, but it has no expanded actions.")

    scores = {}
    for action_idx, child in node.children.items(): # child is a Node, action_idx is an int
        Q = child.val/max(child.count, 1)
        U = cpuct * node.priors[action_idx] * (node.count ** 0.5) / (1 + child.count)
        scores[action_idx] = Q + U

    return max(
        scores.keys(), 
        key=lambda k: scores[k]
    )

def Select(node: Node, env: ChessEnv, cpuct: float = 1.0, done: bool = False) -> Tuple[Node, List[float]]: 
    rewards = []
    while node.is_fully_expanded() and not done and not node.is_terminal(): 
        a_idx = get_action_ucb(node, cpuct)
        _, r, done, _ = env.step(a_idx)
        rewards.append(r)
        node = node.children[a_idx]
        
    return node, rewards, done

def Expand(
    node: Node,
    model: nn.Module,
    env: ChessEnv, # clone
    rewards: List[float], 
) -> Tuple[Node, List[float]]: 
    # make child
    most_likely_action_idx = max(
        node.untried_children, # need max prior over just untried actions
        key=lambda a: node.priors[a]
    )

    # legals = [move2index(node.state, m) for m in node.state.legal_moves]
    # print(f'in Expand, selected {most_likely_action_idx} and legal moves is {legals}')
    _, r, done, _ = env.step(most_likely_action_idx) 
    rewards.append(r)
    
    child_cfg = MCTSNodeConfig(
        state=env.board, # use the updated board state from env
        parent=node, 
        action_taken=most_likely_action_idx, 
    )
    new_child = Node(child_cfg) 

    # update our node with new child for future 
    node.untried_children.remove(most_likely_action_idx)
    node.children[most_likely_action_idx] = new_child
    
    # populate its priors with our policy, need to apply legal masking
    # Convert history to proper input format
    device = next(model.parameters()).device
    model_input = board2input(new_child.history, device=device).unsqueeze(0)  # Add batch dimension
    child_value_t, new_prior_logits = model(model_input)
    
    # Apply legal masking to the policy logits
    masked_logits = legal_mask(new_prior_logits, [new_child.state])
    new_prior_vals = F.softmax(masked_logits, dim=-1)
    new_child.priors = dict(enumerate(new_prior_vals.squeeze()))
    
    if done: 
        leaf_value, _, _ = eval_pos(new_child.state)
    else: 
        leaf_value = child_value_t.item()

    return new_child, rewards + [leaf_value]


# discount defaults to 1. because AlphaZero doesn't discount, but included for completeness 
def Backprop(rewards: List[float], new_child: Node, discount: float = 1.) -> None: 
    # traverse backwards from new_child to root, updating val/counts
    node = new_child
    G = 0.
    sign = 1.

    for reward in reversed(rewards): 
        if node is None:
            break
        node.count += 1
        node.val += (reward + discount * G) * sign
        G = reward + discount * G

        node = node.parent 
        sign *= -1.


def get_empirical_policy(root: Node, nactions: int = 4672, normalize: bool = True, device: torch.device = torch.device("cpu")): # returns [nactions] tensor
    policy = torch.zeros(nactions, device=device)
    # look at all kids and their visit count
    indices = list(root.children.keys())
    
    if not indices:  # handle case where no children exist
        return torch.zeros(nactions, device=device)
    
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)  # use long for indexing
    
    f = lambda a_idx: root.children[a_idx].count
    
    counts = list(map(f, indices))
    counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
    
    # apply legal mask here 
    # print(f'counts tensor has shape {counts_tensor.shape}') # [1, 5]
    policy[indices_tensor] = counts_tensor
    # policy = legal_mask(policy.unsqueeze(0), [root.state]).squeeze(0) # only kids are zero, but we softmax so even 0 mass is not low enough for illegal moves

    # Convert counts to probabilities
    if policy.sum() > 0:
        if normalize: 
            return policy / policy.sum()
        else: 
            return policy
    else:
        return torch.zeros(nactions, device=device)


def get_root(state: chess.Board, model: nn.Module, nactions: int = 4672, noise_scale: float = 0.1, device: torch.device = torch.device("cpu")): 
    root_cfg = MCTSNodeConfig(
        state=state, 
        parent=None, 
    )
    root = Node(root_cfg)
    
    # Convert history to proper input format and apply legal masking
    model_input = board2input(root.history, device=device).unsqueeze(0)  # Add batch dimension
    _, root_prior_logits = model(model_input)
    # Apply legal masking to the policy logits
    masked_logits = legal_mask(root_prior_logits, [root.state])
    root_prior_vals = F.softmax(masked_logits, dim=-1)
    
    # add dirichlet noise to prior of root only, as done in alphazero paper 
    noise = torch.distributions.dirichlet.Dirichlet(torch.ones(nactions, device=device) * 0.03).sample()
    noisy_priors = (1 - noise_scale) * root_prior_vals.squeeze() + noise_scale * noise
    
    root.priors = dict(enumerate(noisy_priors))
    
    return root


def MCTS(cfg: MCTSConfig, state: chess.Board, model: nn.Module, env: ChessEnv, nactions: int = 4672, normalize: bool = False): 
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Initialize the root node
    root = get_root(state, model, nactions, noise_scale=cfg.noise_scale, device=device)
    
    
    # Run the Monte Carlo Tree Search simulations
    for _ in range(cfg.num_sims):
        done = False 

        clone = env.clone()
        # select, get to a terminal node, use ucb policy to traverse with pointers
        terminal, rewards, done = Select(root, clone, cfg.cpuct, done)
        # expand, create a child we'll run rollout from 
        if not done: 
            new_child, rewards = Expand(terminal, model, clone, rewards)
        # no need to simulate, no random rollouts in alphazero, and val of end_state is in Expand()
            Backprop(rewards, new_child)
        else: 
            Backprop(rewards, terminal)
    

    # should return [nactions] empirical (val/count) over all explored actions
        # and use our prior or zero for unvisite actions? 
    return get_empirical_policy(root, normalize=normalize, device=device) 