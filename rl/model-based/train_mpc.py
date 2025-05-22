'''
Model Predictive Control 

MPC is an oldschool method for general control that has existed for decades, way back into the 20th century. 
It's being included here because it's an essential prereq for understanding model-based RL. 
This author was not familiar with model-based RL, and found it laid a solid and fundamental understanding. 

Fundamental idea of model-based RL compared to Q-learning/policy gradient: 
    - Normally we never explicitly learn P(s'|a, s) the environment dynamics 
        - In model-based RL, we do, with a neural network f_theta that maps (s, a) -> predicted_s'
        - This allows us to select actions by "planning" 
        - Where planning is just repeat[sample from action distb, simulate rollout with f_theta and r(s,a), update action distb to favor 
            actions that led to high rewards, take action step on real env with best_action_rollout[0]]
                - every so often, we update our f_theta will the actual env.step data we collected 

Intuitively, 
    f_theta > Q(s,a) > pi

    in terms of information / inductive bias contained, so that 

    model-based < Q-learning < policy gradient 
        in terms of sample complexity. 

In MPC in particular, we "plan" by doing a specific kind of optimization in the inner loop [rollout, update distb] -- 
we use the "cross-entropy method," an oldschool derivative-free optimizer that essentially keeps updating an action distribution, 
pushing it in the direction of the best actions sampled in the previous distribution, a kind of iterative bootstrapping. This results, 
over num_cem_fit_iterations, in a "good action distb" that we then use to sample the next action. We do this iterative fitting 
over num_rollouts (batched) using our f_theta neural network and computable rewards (which are assumed) as the guide to get 
the rollout return for each action sequence. Important typecheck: action distb (mean, std) are both H-vectors, where H is the horizon 
of the rollouts (cov is assumed diagonal). The H-vector of actions sampled from this induces a rollout using [f_theta, r(\cdot)]. 

Understanding MPC reduces to understanding
    - the cross entropy method of optimization 
    - the notion of an action distribution we iteratively improve 
    - the fact our neural network is learning f_theta

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym 
from dataclasses import dataclass 
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# s is [N, 3] and a is [N]
# assume r(s,a) known analytically, else we have to learn it too
# for pendulum and most mujoco tasks, reward is a pure analytical fn of the following form 
    # loss defined in pendulum to be: th^2 + 0.1 th_dot^2 + 0.001 sigma^2
def r(s, a): 
    cos, sin, dot = s.unbind(-1) # [N, 3] -> [N], [N], [N]
    theta = torch.atan2(sin, cos) # construct theta from cos_theta, sin_theta
    # ensure a is properly squeezed to match theta's dimensions
    a_squeezed = a.squeeze(-1)  # Remove the last dimension to make it [N]
    loss = theta**2 + 0.1 * dot**2 + 0.001 * a_squeezed**2 
    return -loss # [N] rewards 

def batch_rollout(
    actions,
    s0, 
    f_theta, 
    sdim=3, 
    gamma=0.99, 
): 
    # s is from env.reset(), want s to start as [N, sdim] = [s0, s0, s0, ...]
    N, H, adim = actions.shape
    s = s0.unsqueeze(0).expand(N, sdim).clone() # [N, sdim]

    rollouts = torch.zeros(N, H, 2 * sdim + adim + 1, device=device) # [s, a, r, s']
    discounts = gamma ** torch.arange(H, device=s.device)  # [H]
    # might need rollout_states, rollout_actions, rollout_rewards, rollout_next_states as a dict 
    for t in range(H): 
        a_t = actions[:, t, :] # [N, adim]
        r_t = r(s, a_t)
        ds = f_theta(s, a_t)  # ([N, sdim], [N, adim]) -> [N, sdim]
        rollouts[:, t, :sdim] = s 
        rollouts[:, t, sdim:sdim+adim] = a_t
        rollouts[:, t, sdim+adim] = r_t
        s = s + ds
        rollouts[:, t, sdim+adim+1:] = s # s = s + ds at this point 

    return (rollouts[..., sdim+adim] * discounts).sum(dim=1) # [N, H] -> [N]

def update_distb(elites, old_mu, old_std, momentum=0.95, eps=1e-5): # elites is [k, h]
    new_mu = elites.mean(dim=0) # new_mu is h 
    new_std = torch.sqrt(((elites - new_mu.unsqueeze(0))**2).mean(dim=0) + eps)

    new_mu = momentum * old_mu + (1 - momentum) * new_mu 
    new_std = momentum * old_std + (1 - momentum) * new_std 
    
    return new_mu, new_std 

class EnvNet(nn.Module): # f_theta simulating f_env, ie. learning "env model"
    def __init__(self, n_hidden_layers=1, sdim=3, adim=1, hidden_width=128, act=nn.GELU()): 
        super().__init__()
        self.sdim = sdim 
        self.adim = adim 
        self.in_dim = sdim + adim
        self.act = act
        self.out_dim = sdim # ds
        self.hidden_width = hidden_width

        layers = []
        layers.append(nn.Linear(self.in_dim, hidden_width))
        layers.append(self.act)
        
        # Add hidden layers based on n_hidden_layers parameter
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(self.act)
            
        layers.append(nn.Linear(hidden_width, self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a): # [s, a] -> ds
        input = torch.cat([s, a], dim=-1) # [b, sdim] and [b, adim]
        return self.net(input) # [b, sdim + adim] -> [b, sdim] for ds

# store [s, a, ds] to teach the envnet 
# buffer is stack([s, a, ds]) ie. 
    # torch.zeros(buff_sz, sdim + adim + sdim)
class Buffer: 
    def __init__(self, buff_sz=10_000, sdim=3, adim=1): 
        self.buff = torch.zeros(buff_sz, sdim + adim + sdim, device=device)
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

    def get_batch(self, bsz): 
        if self.size < bsz:
            bsz = self.size
            
        indices = torch.randint(0, self.size, (bsz,), device=device)
        batch = self.buff[indices]
        
        # split the batch into state, action, and delta state
        s = batch[:, :self.sdim]
        a = batch[:, self.sdim:self.sdim+self.adim]
        ds = batch[:, self.sdim+self.adim:]
        
        return s, a, ds
    
# every so often, use f_theta to induce a policy and see how well it performs 
def eval(
    f_theta: torch.nn.Module,
    num_runs: int = 5, 
    env_name: str = 'Pendulum-v1',
    horizon: int = 20,
    num_cem_iters: int = 5,
    num_rollouts: int = 50,
    top_pct: int = 10,
    a_min: float = -2.0,
    a_max: float = 2.0,
    gamma: float = 0.9, 
    sdim: int = 3,
    adim: int = 1,
): 
    ttl_reward = 0
    k = int(num_rollouts * (top_pct / 100))
    
    for _ in range(num_runs): 
        env = gym.make(env_name)
        state, _ = env.reset()
        s = torch.tensor(state, dtype=torch.float32, device=device)
        episode_reward = 0
        done = False
        
        while not done:
            # use cem to find the best action sequence
            mean = torch.zeros(horizon, adim, device=device)
            std = torch.ones(horizon, adim, device=device)
            
            for _ in range(num_cem_iters):
                # vectorized rollouts with our f_theta given a batch_dim = num_rollouts
                eps = torch.randn(num_rollouts, horizon, adim, device=device)
                actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(a_min, a_max)
                rewards = batch_rollout(actions, s, f_theta, sdim, gamma=gamma)
                _, elite_idxs = torch.topk(rewards, k=k)
                elite_actions = actions[elite_idxs]
                mean, std = update_distb(elite_actions, mean, std)

            # take first action from optimized sequence
            action = mean[0].cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            s = torch.tensor(next_state, dtype=torch.float32, device=device)
            episode_reward += reward
            
        ttl_reward += episode_reward

    return ttl_reward / num_runs
    
@dataclass
class TrainConfig: 
        lr: float = 1e-4
        bsz: int = 64
        sdim: int = 3
        adim: int = 1

def train_mpc(
        f_theta: nn.Module, 
        num_train_steps: int = 1_000,
        num_cem_fit_steps: int = 20,
        num_rollouts: int = 50,
        H: int = 20,
        env_name: str ='Pendulum-v1', 
        top_pct: int = 10, # top % that become elite rollouts 
        update_dynamics_model: int = 10, 
        eval_every: int = 10, 
        a_min: float = -2.0, 
        a_max: float = 2.0, 
        gamma=0.9, 
        cfg: TrainConfig = TrainConfig(),
        verbose: bool = False,
        wandb_log: bool = False,
        ):

        if wandb_log:
            import wandb
            wandb.init(project="model-based-rl", name="mpc")

        opt = torch.optim.AdamW(f_theta.parameters(), lr=cfg.lr)
        env = gym.make(env_name)
        buffer = Buffer(sdim=cfg.sdim, adim=cfg.adim) 
        state, _ = env.reset()
        s = torch.tensor(state, dtype=torch.float32, device=device)

        k = int(num_rollouts*(top_pct/100))
        
        mean = torch.zeros(H, cfg.adim, device=device)
        std = torch.ones(H, cfg.adim, device=device)
        for step in tqdm(range(num_train_steps), disable=not verbose):
                for _ in range(num_cem_fit_steps): 
                    # vectorized rollouts with our f_theta given a batch_dim = num_rollouts
                    eps = torch.randn(num_rollouts, H, cfg.adim, device=device)
                    actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(a_min, a_max)
                    s0 = s.clone()  # Use current state instead of resetting
                    rewards = batch_rollout(actions, s0, f_theta, cfg.sdim, gamma=gamma)
                    _, elite_idxs = torch.topk(rewards, k=k)
                    elite_actions = actions[elite_idxs]
                    mean, std = update_distb(elite_actions, mean, std)

                next_action = mean[0].unsqueeze(0)  # First action from sequence
                next_action_np = next_action.cpu().numpy().flatten()  # Convert to numpy array for env.step
                next_state, _, _, _, _ = env.step(next_action_np)
                next_s = torch.tensor(next_state, dtype=torch.float32, device=device)
                
                buffer.push(torch.cat([s.unsqueeze(0), next_action, (next_s-s).unsqueeze(0)], dim=1))
                s = next_s 

                if step % eval_every == 0: 
                    r = eval(
                        f_theta, 
                        env_name=env_name, 
                        horizon=H, 
                        num_rollouts=num_rollouts, 
                        top_pct=top_pct, 
                        a_min=a_min, 
                        a_max=a_max, 
                        gamma=gamma,
                        sdim=cfg.sdim,
                        adim=cfg.adim,
                    )
                    if verbose:
                        print(f'Average reward on step {step} is {r:.3f}')
                    if wandb_log:
                        wandb.log({"eval_reward": r, "step": step})

                if step % update_dynamics_model == 0 and buffer.size > cfg.bsz:
                    f_theta.train()
                    states, actions, ds = buffer.get_batch(cfg.bsz) # each is [b, sdim] or [b, adim]
                    ds_pred = f_theta(states, actions)
                    loss = F.mse_loss(ds_pred, ds)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    f_theta.eval()
                    
                    if verbose and step % 100 == 0:
                        print(f"Step {step}, f_theta Loss: {loss.item():.6f}")
                    
                    if wandb_log:
                        wandb.log({"loss": loss.item(), "step": step})
                            
        if wandb_log:
            wandb.finish()
            
        return f_theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MPC on Pendulum')
    parser.add_argument('--num-train-steps', type=int, default=1_000, help='Number of training steps')
    parser.add_argument('--num-cem-fit-steps', type=int, default=10, help='Number of CEM fitting steps')
    parser.add_argument('--num-rollouts', type=int, default=64, help='Number of rollouts per CEM iteration')
    parser.add_argument('--n-hidden-layers', type=int, default=3, help='Number of f_theta hidden layers')
    parser.add_argument('--hidden_width', type=int, default=256, help='Hidden width multiplier for f_theta')
    parser.add_argument('--horizon', type=int, default=20, help='Planning horizon')
    parser.add_argument('--env-name', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--top-pct', type=int, default=20, help='Top percentage for elite samples')
    parser.add_argument('--update-dynamics-model', type=int, default=5, help='Update dynamics model every N steps')
    parser.add_argument('--eval-every', type=int, default=100, help='Update dynamics model every N steps')
    parser.add_argument('--a-min', type=float, default=-2.0, help='Minimum action value')
    parser.add_argument('--a-max', type=float, default=2.0, help='Maximum action value')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # setup
    torch.manual_seed(args.seed)
    f_theta = EnvNet(
        n_hidden_layers=args.n_hidden_layers,
        hidden_width=args.hidden_width,
        sdim=3,
        adim=1
    ).to(device)

    if args.verbose:
        param_count = sum(p.numel() for p in f_theta.parameters())
        print(f'Our dynamics model, f_theta, has {param_count/1e6:.3f}M params.')

    cfg = TrainConfig(
        lr=args.lr,
        bsz=args.batch_size,
        sdim=3, 
        adim=1, # hardcode for pendulum 
    )
    
    train_mpc(
        f_theta=f_theta,
        num_train_steps=args.num_train_steps,
        num_cem_fit_steps=args.num_cem_fit_steps,
        num_rollouts=args.num_rollouts,
        H=args.horizon,
        env_name=args.env_name,
        top_pct=args.top_pct,
        update_dynamics_model=args.update_dynamics_model,
        eval_every=args.eval_every,
        a_min=args.a_min,
        a_max=args.a_max,
        cfg=cfg,
        verbose=args.verbose,
        wandb_log=args.wandb,
    )
