import torch
import torch.nn as nn
import torch.nn.functional as F
import gym 
from dataclasses import dataclass 
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# assume r(s,a) known analytically, else we have to learn it too
# for pendulum and most mujoco tasks, reward is a pure analytical fn of the following form 
    # loss defined in Pendulum to be: th^2 + 0.1 th_dot^2 + 0.001 sigma^2
def r(s, a): 
    cos, sin, dot = s
    theta = torch.atan2(sin, cos) # construct theta from cos_theta, sin_theta
    loss = theta**2 + 0.1 * dot ** 2 + 0.001 * a**2 
    return -loss

def get_rollout(actions, s, f_theta): # a is [rollout_len] sampled from mu -> get [s, a, r, s'] ie. [2 * sdim + adim + 1]
    # s is the initial state
    H = len(actions)

    sdim = 3
    adim = 1 # hardcode for pendulum 
    rollout = torch.zeros(H, 2 * sdim + adim + 1) # [s, a, r, s']

    # might need rollout_states, rollout_actions, rollout_rewards, rollout_next_states as a dict 
    for i in range(H): 
        a = actions[i]
        reward = r(s, a)
        # print(f'in get rollout loopH, passing in s shape {s.unsqueeze(0)} and a shape {a.unsqueeze(0)}')
        ds = f_theta(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)  # [b, ]
        rollout[i, :sdim] = s 
        rollout[i, sdim:sdim+adim] = a
        rollout[i, sdim+adim] = reward
        s = s + ds
        rollout[i, sdim+adim+1:] = s # s = s + ds at this point 

    return rollout 

def rollout2reward(rollout, sdim=3, adim=1, gamma=0.99): # [H, 2*sdim + adim +1] -> scalar, 5 is [s, a, r, s']
    rs = rollout[:, sdim+adim]
    discount = gamma ** torch.arange(len(rs), device=rs.device)  # [1, gamma, gamma^2, ...]
    return (rs * discount).sum()  # Sum the discounted rewards

def update_distb(elites, old_mu, old_std, momentum=0.95, eps=1e-5): # elites is [K, H]
    new_mu = elites.mean(dim=0) # new_mu is H 
    new_std = torch.sqrt(((elites - new_mu.unsqueeze(0))**2).mean(dim=0) + eps)

    new_mu = momentum * old_mu + (1 - momentum) * new_mu 
    new_std = momentum * old_std + (1 - momentum) * new_std 
    
    return new_mu, new_std 

class EnvNet(nn.Module): # f_theta simulating f_env, ie. learning "env model"
    def __init__(self, sdim=3, adim=1, mult=4, act=nn.GELU()): 
        super().__init__()
        self.sdim = sdim 
        self.adim = adim 
        self.in_dim = sdim + adim
        self.act = act
        self.out_dim = sdim # ds

        self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.in_dim * mult), 
                self.act, 
                nn.Linear(self.in_dim * mult, self.out_dim)
        )

    def forward(self, s, a): # [s, a] -> ds
        input = torch.cat([s, a], dim=-1) # [b, sdim] and [b, adim]
        return self.net(input) # [b, sdim + adim] -> [b, sdim] for ds

# store [s, a, ds] to teach the EnvNet 
# Buffer is stack([s, a, ds]) ie. 
    # torch.zeros(buff_sz, sdim + adim + sdim)
class Buffer: 
    def __init__(self, buff_sz=10_000, sdim=3, adim=1): 
        self.buff = torch.zeros(buff_sz, sdim + adim + sdim)
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
            
        indices = torch.randint(0, self.size, (bsz,))
        batch = self.buff[indices]
        
        # Split the batch into state, action, and delta state
        s = batch[:, :self.sdim]
        a = batch[:, self.sdim:self.sdim+self.adim]
        ds = batch[:, self.sdim+self.adim:]
        
        return s, a, ds
    
# every so often, use f_theta to induce a policy and see how well it performs 
def eval(
    f_theta: nn.Module, 
    num_runs: int = 5, 
): 
    
    pass 
    
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
        eval_every_steps: int = 10, 
        a_min: float = -2.0, 
        a_max: float = 2.0, 
        cfg: TrainConfig = TrainConfig(),
        verbose: bool = False,
        wandb_log: bool = False,
        ):

        if wandb_log:
            import wandb
            wandb.init(project="model-based-rl", name="mpc")

        opt = torch.optim.AdamW(f_theta.parameters(), lr=cfg.lr)
        env = gym.make(env_name)
        state, _ = env.reset()
        s = torch.tensor(state, dtype=torch.float32)
        buffer = Buffer() 

        k = int(num_rollouts*(top_pct/100))
        
        mean, std = torch.zeros(H), torch.ones(H) 
        for step in tqdm(range(num_train_steps), disable=not verbose):
                for _ in range(num_cem_fit_steps): 
                        all_rollouts = []
                        for _ in range(num_rollouts): 
                                distb = torch.distributions.Normal(mean, std)
                                actions = distb.sample().clamp(a_min, a_max).unsqueeze(-1) # [H, adim]
                                with torch.no_grad(): # no grads on f_theta here 
                                    rollout = get_rollout(actions, s, f_theta)
                                all_rollouts.append(rollout)
                        
                        # Convert to tensor and get top k rollouts
                        all_rollouts_tensor = torch.stack(all_rollouts) # [num_rollouts, H, 2 * sdim + adim + 1]
                        rewards = torch.tensor([rollout2reward(roll) for roll in all_rollouts]) # [num_rollouts, 1]
                        _, elite_idxs = torch.topk(rewards, k=k)
                        elites = all_rollouts_tensor[elite_idxs] # [num_elites, 2 * sdim + adim + 1]

                        # get actions from elites to update distb 
                        elite_actions = elites[..., cfg.sdim: cfg.sdim + cfg.adim].squeeze(-1) # [num_elites, adim] = [num_elites, 1] -> [num_elites]
                        mean, std = update_distb(elite_actions, mean, std)

                distb = torch.distributions.Normal(mean, std)
                next_action = mean[0].unsqueeze(0) # preserve shape so this is now a 1-tensor for cat later
                next_state, _, _, _, _ = env.step([next_action.item()])
                next_s = torch.tensor(next_state, dtype=torch.float32)
                
                buffer.push(torch.cat([s, next_action, next_s-s]).unsqueeze(0)) # expects 2d tensor input
                s = next_s 

                if verbose and step % eval_every_steps == 0: 
                    r = eval(f_theta)
                    print(f'Average reward on step {step} is {r:.3f}')

                if step % update_dynamics_model == 0:
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
    parser.add_argument('--num-cem-fit-steps', type=int, default=20, help='Number of CEM fitting steps')
    parser.add_argument('--num-rollouts', type=int, default=50, help='Number of rollouts per CEM iteration')
    parser.add_argument('--horizon', type=int, default=20, help='Planning horizon')
    parser.add_argument('--env-name', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--top-pct', type=int, default=10, help='Top percentage for elite samples')
    parser.add_argument('--update-dynamics-model', type=int, default=10, help='Update dynamics model every N steps')
    parser.add_argument('--a-min', type=float, default=-2.0, help='Minimum action value')
    parser.add_argument('--a-max', type=float, default=2.0, help='Maximum action value')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create model
    f_theta = EnvNet()
    
    # Create config
    cfg = TrainConfig(
        lr=args.lr,
        bsz=args.batch_size,
        sdim=3, 
        adim=1, # hardcode for pendulum 
    )
    
    # Train
    train_mpc(
        f_theta=f_theta,
        num_train_steps=args.num_train_steps,
        num_cem_fit_steps=args.num_cem_fit_steps,
        num_rollouts=args.num_rollouts,
        H=args.horizon,
        env_name=args.env_name,
        top_pct=args.top_pct,
        update_dynamics_model=args.update_dynamics_model,
        a_min=args.a_min,
        a_max=args.a_max,
        cfg=cfg,
        verbose=args.verbose,
        wandb_log=args.wandb,
    )
