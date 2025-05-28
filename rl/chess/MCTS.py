from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import chess
import time
from typing import Optional, List, Tuple
from config import MCTSNodeConfig, MCTSConfig
from env import ChessEnv
from utils_optimized import move2index, eval_pos, board2input, legal_mask
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
        self.legal_indices = [move2index(self.state, m) for m in cfg.state.legal_moves]
        
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

def Select_batch(node: Node, env: ChessEnv, cpuct: float = 1.0, done: bool = False) -> Tuple[Node, List[float]]: 
    rewards = []
    # while node.is_fully_expanded() and not done and not node.is_terminal(): 
    while node.is_fully_expanded() and node.children and not done and not node.is_terminal(): 
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
    
    ### need to batch above by doing all above and writing to model_input_batch
    child_value_t, new_prior_logits = model(model_input)
    
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
    
    policy[indices_tensor] = counts_tensor
    
    # Apply legal mask
    policy = legal_mask(policy.unsqueeze(0), [root]).squeeze(0)
    
    # Convert counts to probabilities
    if normalize: 
        # Avoid dividing by sum when there are -inf values
        policy_sum = policy[policy > float("-inf")].sum()
        return policy / policy_sum if policy_sum > 0 else policy
    else: 
        return policy


def get_root(state: chess.Board, model: nn.Module, nactions: int = 4672, noise_scale: float = 0.1, device: torch.device = torch.device("cpu")): 
    root_cfg = MCTSNodeConfig(
        state=state.copy(), 
        parent=None, 
    )
    root = Node(root_cfg)
    
    # Convert history to proper input format and apply legal masking
    model_input = board2input(root.history, device=device).unsqueeze(0)  # Add batch dimension
    _, root_prior_logits = model(model_input)

    # Apply legal masking to the policy logits
    masked_logits = legal_mask(root_prior_logits, [root])
    root_prior_vals = F.softmax(masked_logits, dim=-1)
    
    # Create a mask for legal moves
    legal_moves_mask = masked_logits.squeeze() > float("-inf")
    legal_moves_indices = torch.nonzero(legal_moves_mask).squeeze(-1)
    num_legal_moves = legal_moves_indices.size(0)
    
    # Generate Dirichlet noise only for legal moves
    noise = torch.zeros(nactions, device=device)
    if num_legal_moves > 0:
        legal_moves_noise = torch.distributions.dirichlet.Dirichlet(
            torch.ones(num_legal_moves, device=device) * 0.03
        ).sample()
        noise[legal_moves_indices] = legal_moves_noise
    
    # Apply noise only to legal moves
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

# batched version 
def MCTS_batched(cfg: MCTSConfig, state: chess.Board, model: nn.Module, env: ChessEnv, nactions: int = 4672, normalize: bool = False, record_time: bool = False):
    # Get device from model parameters
    device = next(model.parameters()).device
    root = get_root(state, model, nactions, noise_scale=cfg.noise_scale, device=device)

    # Timing variables
    model_time = 0.0
    other_time = 0.0
    total_time_start = time.time() if time else None

    # Run the Monte Carlo Tree Search simulations in batches
    for _ in range(0, cfg.num_sims, cfg.batch_sz):
        batch_data = []

        # Start timing non-model operations
        if record_time:
            other_time_start = time.time()

        # Collect up to cfg.batch_sz expansions
        for _ in range(cfg.batch_sz):
            done = False
            clone = env.clone()

            # Selection: traverse down to a leaf
            terminal, rewards, done = Select_batch(root, clone, cfg.cpuct, done)

            # If not terminal and has untried children, prepare an expansion
            if done or terminal.is_fully_expanded(): 
                Backprop(rewards, terminal)
                continue 

            # Pick the child with highest prior
            action = max(terminal.untried_children, key=lambda a: terminal.priors[a])
            _, r, done, _ = clone.step(action)
            rewards.append(r)

            # Create the new child node
            child_cfg = MCTSNodeConfig(
                state=clone.board,
                parent=terminal,
                action_taken=action,
            )
            new_child = Node(child_cfg)

            # trick: attach after priors 
            terminal.untried_children.remove(action)
            # terminal.children[action] = new_child

            # Prepare network input for the new child
            model_input = board2input(new_child.history, device=device)
            batch_data.append({
                'terminal': terminal, 
                'action': action, 
                'new_child': new_child,
                'rewards': rewards,
                'done': done,
                'model_input': model_input
            })
        
        # Update other time
        if record_time:
            other_time += time.time() - other_time_start

        # If we have expansions, do a single batched forward pass
        if batch_data:
            # Start timing model operations
            if record_time:
                model_time_start = time.time()

            model_inputs = torch.stack([d['model_input'] for d in batch_data], dim=0)
            child_values_batch, prior_logits_batch = model(model_inputs) # bmm 

            # Update model time
            if record_time:
                model_time += time.time() - model_time_start
                other_time_start = time.time()

            # Process each expansion in the batch
            for i, item in enumerate(batch_data):
                new_child = item['new_child']
                rewards = item['rewards']
                done = item['done']
                terminal = item['terminal']
                action = item['action']


                # Mask illegal moves and convert logits to priors
                masked = legal_mask(prior_logits_batch[i:i+1], [new_child])
                priors = F.softmax(masked, dim=-1).squeeze(0)
                new_child.priors = dict(enumerate(priors))

                # terminal.untried_children.remove(action)
                terminal.children[action] = new_child

                # Determine the leaf value
                if done:
                    leaf_value, _, _ = eval_pos(new_child.state)
                else:
                    leaf_value = child_values_batch[i].item()

                # Backpropagate the result up the tree
                Backprop(rewards + [leaf_value], new_child)
            
            # Update other time
            if record_time:
                other_time += time.time() - other_time_start

    # Print timing information if requested
    if record_time:
        total_time = time.time() - total_time_start
        model_percent = (model_time / total_time) * 100
        other_percent = (other_time / total_time) * 100
        print(f"MCTS_batched timing: Model Forward (GPU): {model_time:.4f}s ({model_percent:.1f}%), Other (CPU): {other_time:.4f}s ({other_percent:.1f}%), Total: {total_time:.4f}s")

    # Return the empirical policy from the root
    return get_empirical_policy(root, normalize=normalize, device=device)

######################
class Node: 
    def __init__(self, cfg: MCTSNodeConfig): 
        self.state = cfg.state
        self.val = 0 
        self.count = 0 
        self.parent = cfg.parent 
        self.action_taken = cfg.action_taken
        
        # OPTIMIZATION 1: Use numpy arrays instead of lists for legal moves
        self.legal_indices = [move2index(self.state, m) for m in cfg.state.legal_moves]
        self.untried_children = list(self.legal_indices)  # Just copy instead of recomputing
        self.children = {}
        
        # OPTIMIZATION 2: Don't store priors as dict initially
        self.priors = None  # Will be a tensor when set
        self.prior_indices = None  # Indices where priors are valid
        
        # History management
        if cfg.history and len(cfg.history) > 0:
            self.history = cfg.history
        elif self.parent:
            self.history = self.parent.history.copy()
            self.history.append(self.state.copy())
        else:
            self.history = deque([self.state.copy()], maxlen=8)
    
    def get_prior(self, action_idx: int) -> float:
        """Get prior for a specific action"""
        if self.priors is None:
            return 0.0
        return self.priors[action_idx].item()
    
    def is_fully_expanded(self) -> bool: 
        return len(self.untried_children) == 0 
    
    def is_terminal(self) -> bool: 
        return self.state.is_game_over()


def get_action_ucb_optimized(node: Node, cpuct: float = 1.0) -> int:
    """Optimized UCB calculation using vectorized operations"""
    if not node.children:
        raise ValueError("Error: trying to get_action_ucb from node, but it has no expanded actions.")
    
    # Vectorized UCB calculation
    actions = list(node.children.keys())
    n_actions = len(actions)
    
    # Pre-allocate tensors
    Q_values = torch.zeros(n_actions)
    counts = torch.zeros(n_actions)
    priors = torch.zeros(n_actions)
    
    for i, action_idx in enumerate(actions):
        child = node.children[action_idx]
        Q_values[i] = child.val / max(child.count, 1)
        counts[i] = child.count
        priors[i] = node.get_prior(action_idx)
    
    # Vectorized UCB formula
    U_values = cpuct * priors * (node.count ** 0.5) / (1 + counts)
    ucb_scores = Q_values + U_values
    
    best_idx = torch.argmax(ucb_scores).item()
    return actions[best_idx]


def get_root_optimized(state: chess.Board, model: nn.Module, nactions: int = 4672, noise_scale: float = 0.1, device: torch.device = torch.device("cpu")): 
    """Optimized root initialization"""
    root_cfg = MCTSNodeConfig(
        state=state.copy(), 
        parent=None, 
    )
    root = Node(root_cfg)
    
    # Convert history to proper input format
    model_input = board2input(root.history, device=device).unsqueeze(0)
    
    # OPTIMIZATION: Do everything on GPU and avoid dict conversion
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _, root_prior_logits = model(model_input)
    
    # Create legal mask efficiently
    legal_mask_tensor = torch.zeros(nactions, dtype=torch.bool, device=device)
    legal_indices_tensor = torch.tensor(root.legal_indices, dtype=torch.long, device=device)
    legal_mask_tensor[legal_indices_tensor] = True
    
    # Apply mask and softmax in one go
    root_prior_logits = root_prior_logits.squeeze(0)
    root_prior_logits[~legal_mask_tensor] = float('-inf')
    root_prior_vals = F.softmax(root_prior_logits, dim=-1)
    
    # Apply Dirichlet noise efficiently
    if noise_scale > 0 and len(root.legal_indices) > 0:
        # Generate noise only for legal moves
        num_legal = len(root.legal_indices)
        dirichlet_noise = torch.distributions.dirichlet.Dirichlet(
            torch.ones(num_legal, device=device) * 0.03
        ).sample()
        
        # Apply noise using advanced indexing
        root_prior_vals = root_prior_vals * (1 - noise_scale)
        root_prior_vals[legal_indices_tensor] += noise_scale * dirichlet_noise
    
    # OPTIMIZATION: Store as tensor, not dict
    root.priors = root_prior_vals
    root.prior_indices = legal_indices_tensor
    
    return root


def MCTS_batched_optimized(cfg: MCTSConfig, state: chess.Board, model: nn.Module, env: ChessEnv, nactions: int = 4672, normalize: bool = False, record_time: bool = False):
    """Optimized batched MCTS"""
    device = next(model.parameters()).device
    
    # Use optimized root initialization
    root = get_root_optimized(state, model, nactions, noise_scale=cfg.noise_scale, device=device)
    
    # Timing variables
    model_time = 0.0
    other_time = 0.0
    total_time_start = time.time() if record_time else None

    # Run simulations in batches
    for _ in range(0, cfg.num_sims, cfg.batch_sz):
        batch_data = []

        if record_time:
            other_time_start = time.time()

        # Collect expansions
        for _ in range(cfg.batch_sz):
            done = False
            clone = env.clone()

            # Selection
            terminal, rewards, done = Select_batch(root, clone, cfg.cpuct, done)

            if done or terminal.is_fully_expanded(): 
                Backprop(rewards, terminal)
                continue 

            # OPTIMIZATION: Use tensor operations for finding best prior
            untried_tensor = torch.tensor(terminal.untried_children, dtype=torch.long, device=device)
            untried_priors = terminal.priors[untried_tensor]
            best_idx = torch.argmax(untried_priors).item()
            action = terminal.untried_children[best_idx]
            
            _, r, done, _ = clone.step(action)
            rewards.append(r)

            # Create new child
            child_cfg = MCTSNodeConfig(
                state=clone.board,
                parent=terminal,
                action_taken=action,
            )
            new_child = Node(child_cfg)

            terminal.untried_children.remove(action)

            # Prepare for batch processing
            model_input = board2input(new_child.history, device=device)
            batch_data.append({
                'terminal': terminal, 
                'action': action, 
                'new_child': new_child,
                'rewards': rewards,
                'done': done,
                'model_input': model_input
            })
        
        if record_time:
            other_time += time.time() - other_time_start

        # Batch forward pass
        if batch_data:
            if record_time:
                model_time_start = time.time()

            model_inputs = torch.stack([d['model_input'] for d in batch_data], dim=0)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                child_values_batch, prior_logits_batch = model(model_inputs)

            if record_time:
                model_time += time.time() - model_time_start
                other_time_start = time.time()

            # OPTIMIZATION: Batch process legal masking and softmax
            batch_size = len(batch_data)
            
            # Pre-create legal masks for all nodes in batch
            legal_masks = torch.zeros((batch_size, nactions), dtype=torch.bool, device=device)
            for i, item in enumerate(batch_data):
                legal_indices = torch.tensor(item['new_child'].legal_indices, dtype=torch.long, device=device)
                legal_masks[i, legal_indices] = True
            
            # Apply masks and softmax in one operation
            prior_logits_batch[~legal_masks] = float('-inf')
            priors_batch = F.softmax(prior_logits_batch, dim=-1)
            
            # Process results
            for i, item in enumerate(batch_data):
                new_child = item['new_child']
                rewards = item['rewards']
                done = item['done']
                terminal = item['terminal']
                action = item['action']

                # Store priors as tensor
                new_child.priors = priors_batch[i]
                
                # Attach child
                terminal.children[action] = new_child

                # Get leaf value
                if done:
                    leaf_value, _, _ = eval_pos(new_child.state)
                else:
                    leaf_value = child_values_batch[i].item()

                # Backpropagate
                Backprop(rewards + [leaf_value], new_child)
            
            if record_time:
                other_time += time.time() - other_time_start

    if record_time:
        total_time = time.time() - total_time_start
        model_percent = (model_time / total_time) * 100
        other_percent = (other_time / total_time) * 100
        print(f"MCTS_batched timing: Model Forward (GPU): {model_time:.4f}s ({model_percent:.1f}%), Other (CPU): {other_time:.4f}s ({other_percent:.1f}%), Total: {total_time:.4f}s")

    # Return the empirical policy
    return get_empirical_policy_optimized(root, nactions, normalize=normalize, device=device)


def get_empirical_policy_optimized(root: Node, nactions: int = 4672, normalize: bool = True, device: torch.device = torch.device("cpu")):
    """Optimized policy extraction"""
    policy = torch.zeros(nactions, device=device)
    
    if not root.children:
        return policy
    
    # Vectorized count extraction
    actions = list(root.children.keys())
    counts = torch.tensor([root.children[a].count for a in actions], dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    
    policy[actions_tensor] = counts
    
    # Apply legal mask efficiently
    legal_mask_tensor = torch.zeros(nactions, dtype=torch.bool, device=device)
    legal_indices = torch.tensor(root.legal_indices, dtype=torch.long, device=device)
    legal_mask_tensor[legal_indices] = True
    policy[~legal_mask_tensor] = float('-inf')
    
    if normalize:
        policy_sum = policy[legal_mask_tensor].sum()
        if policy_sum > 0:
            policy = policy / policy_sum
    
    return policy