'''
This is a variant of train_grpo_gsm.py aiming to train a model to tell "knock knock" jokes in particular. 
I was curious about performance on nonverifiable tasks. Of course, this task is verifiable since our result just verifies the presence of 
a word in the solution, but I'm confident if a model can learn how to do this, it can learn close variants that are non verifiable (eg. 
"chicken crossing the road" jokes, which can appear in many different ways that do not have any particular lexical patterns -- you could 
just use an LLM judge to verify a certain joke is from a certain genre). 

To much great pleasure, most models trained with this script do (after a few hundred steps) learn to output 
knock knock jokes 100% of the time! In practice they often converge to producing a single knock knock joke 
in every sample, a sort of "RL collapse" that people have observed before. Nonetheless, seeing strong hill climbing on a 
sort of arbitrary like this is pretty fun to watch. 

''' 
import torch 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Union 
import math 
from tqdm import tqdm 
from copy import deepcopy 
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR 
import argparse
import wandb

PROMPT = """
You are a helpful language model assistant that tells funny jokes. You should be creative and humorous in your responses. You will be shown some examples of 
how to tell good jokes, and then you will be asked to tell a joke. You can think step by step to come up with the joke, but include <joke> and </joke> tags around the joke.
--- 
Examples: 

<joke>
    Knock knock.
    Who's there?
    Interrupting cow.
    Interrupting cow w--
    MOO!
</joke>

<joke>
    Why did the chicken cross the road?
    To get to the other side!
</joke>

<joke>
What do you call a fake noodle?
An impasta!
</joke>

<joke>
Knock knock.
Who's there?
Lettuce.
Lettuce who?
Lettuce in, it's cold out here!
</joke>

<joke>
I told my wife she was drawing her eyebrows too high.
She looked surprised.
</joke>

<joke>
What's the best thing about Switzerland?
I don't know, but the flag is a big plus.
</joke>

--- 
Now come up with your own joke in a similar way, and include <joke> and </joke> tags around the joke.
"""

def get_model_tokenizer(name: str = "meta-llama/Llama-3.2-1B-Instruct") -> Tuple[AutoModelForCausalLM, AutoTokenizer]: 
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="flash_attention_2",
        )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    return model, tokenizer

def extract(text: str) -> str: 
    if "<joke>" not in text:
        return ""
    try:
        answer_part = text.split("<joke>")[1].split("</joke>")[0].strip()
        return answer_part
    except:
        return ""

def get_batch_preds(
        model: AutoModelForCausalLM,
        prompts: List[str],
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256
    ) -> List[str]:

    device = next(model.parameters()).device
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad(): 
        outputs_t = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        )

    # Get only the new tokens by slicing off the input tokens
    new_tokens = outputs_t[:, inputs['input_ids'].shape[1]:]
    
    outputs_s = tokenizer.batch_decode(
        new_tokens, 
        skip_special_tokens=True, 
    )
    
    return outputs_s


# output is [bsz] tensor of sequence-wise rewards, uses extract() on completions
def output_format_reward(pred: str) -> float: 
    r = 0 
    if "<joke>" in pred: 
        r += 0.5 
    if "</joke>" in pred: 
        r += 0.5
    return r

def correctness_reward(pred: str) -> float: 
    return 5.0 if "knock" in pred.lower() else 0.0

def reward_fn(completions: List[str]) -> torch.Tensor: 
    preds = list(map(lambda c: extract(c), completions))
    rewards = []
    
    for pred in preds:
        r = correctness_reward(pred)
        r += output_format_reward(pred)
        rewards.append(r)
    
    return torch.tensor(
        rewards, 
        dtype=torch.bfloat16, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

# implementes GRPO: (rewards, ref_lps, policy_lps) -> final objective (-reward)
def loss_fn(
    rewards: torch.Tensor, 
    policy_lps: torch.Tensor, 
    ref_lps: torch.Tensor, 
    mask: torch.Tensor, # should be [B, max_completion_len], don't want padding to influence reward
    beta: float = 0, 
    G: int = 16,
    num_groups_per_batch: int = 1, 
    eps_denom: float = 1e-5, 
    use_dr_len: bool = False, 
    use_dr_std: bool = False, 
) -> torch.Tensor:

    # using no-clip version without importance sampling ratios 
    # rewards is [B], broadcast to [B, max_completion_len]
    _, L = policy_lps.shape
    A = torch.empty_like(rewards) # [B]

    rewards_reshape = rewards.reshape(num_groups_per_batch, G) 
    means = rewards_reshape.mean(dim=-1, keepdim=True)
    stds = rewards_reshape.std(dim=-1, keepdim=True)
    eps_t = torch.full_like(stds, eps_denom, dtype=torch.bfloat16)
    denom = (stds + eps_t) if use_dr_std else torch.ones_like(stds)
        
    A = ((rewards_reshape - means)/denom).flatten() # [B]

    A = A.unsqueeze(1).expand(-1, L).detach()  # broadcast from seqs to tokens [B, L], dont want diff thru rewards 
    sum_over_rollouts = (A * policy_lps * mask).sum(dim=-1) 
    if use_dr_len:
        sum_over_rollouts = sum_over_rollouts / mask.sum(dim=-1)
    adv = sum_over_rollouts.mean()
    
    kl_loss = beta * get_kl(policy_lps, ref_lps, mask)
    reward = adv - kl_loss
    return -reward


# seqs is [bsz] where each is prompt_i + completion_{i, j} where i in [num_prompts_per_step] and j in [num_completions_per_prompt]
def get_model_lps(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    seqs: List[str], 
    grad: bool = False, 
) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = tokenizer(
        seqs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    if grad: 
        outputs = model(**inputs)
    else: 
        with torch.no_grad(): 
            outputs = model(**inputs)

    logits = outputs.logits[:, :-1, :] # BLV
    
    lps = F.log_softmax(logits, dim=-1)  # [B, L, V] ~ 16 * 768 * 150_000 * 2 ~ 3Gb
    
    # Gather log probs for actual tokens
    input_ids = inputs['input_ids']  # [B, L]
    lps = lps.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L]

    return lps # B, L

# compute closed form for loss_fn 
def get_kl(policy_lps: torch.Tensor, ref_lps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
    diff_m = (ref_lps.float() - policy_lps.float()) * mask
    kl_t = torch.exp(diff_m) - (diff_m) - 1
    return (kl_t.sum() / mask.sum()).float()

# returns completions for each prompt, may need to return mask from .generate or input_ids? 
def get_completions_and_mask(
    prompts: List[str], 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int = 256
) -> Tuple[List[str], torch.Tensor]:
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()} # we pinned dataloader memory so this h2d is async 

    with torch.no_grad(): 
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True, 
            temperature=1.0, # nontrivial exploration
        )

    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs.sequences[:, input_len:]  # [B, max_completion_len]
    completions = tokenizer.batch_decode(
        generated_tokens, 
        skip_special_tokens=True, 
    )

    # Create mask for non-padded tokens
    mask = (generated_tokens != tokenizer.pad_token_id).to(torch.bfloat16)

    return completions, mask  # all are [B, L] where L = max_completion_len

def eval_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    b: int = 16,
    max_new_tokens: int = 256,
) -> Tuple[List[str], float]:
    

    jokes = get_batch_preds(
        model,
        [PROMPT] * b,
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    jokes_lst = list(map(lambda s: extract(s), jokes))

    return jokes_lst, reward_fn(jokes).mean().item()

def log_metrics(
    step: int,
    loss: torch.Tensor,
    rewards_b: torch.Tensor,
    policy_lps_b: torch.Tensor,
    ref_lps_b: torch.Tensor,
    mask_b: torch.Tensor,
    args,
    cumulative_rewards: torch.Tensor = None,
    cumulative_policy_lps: torch.Tensor = None,
    cumulative_ref_lps: torch.Tensor = None,
    cumulative_mask: torch.Tensor = None,
):
    # Use cumulative metrics if provided, otherwise use current batch
    if cumulative_rewards is not None:
        rewards_for_metrics = cumulative_rewards
        policy_lps_for_metrics = cumulative_policy_lps
        ref_lps_for_metrics = cumulative_ref_lps
        mask_for_metrics = cumulative_mask
    else:
        rewards_for_metrics = rewards_b
        policy_lps_for_metrics = policy_lps_b
        ref_lps_for_metrics = ref_lps_b
        mask_for_metrics = mask_b

    mean_reward = rewards_for_metrics.mean().item()
    loss_value = loss.item()
    reward_std = rewards_for_metrics.std().item()
    correct_predictions = (rewards_for_metrics > 1.0).sum().item()
    accuracy = correct_predictions / len(rewards_for_metrics)

    generation_lengths = mask_for_metrics.sum(dim=1)
    mean_generation_len = generation_lengths.float().mean().item()

    if args.verbose:
        print(
            f"Step {step}: Loss = {loss_value:.4f}, Mean Reward = {mean_reward:.4f}")
        print(
            f"  Accuracy: {accuracy:.2%} ({correct_predictions}/{len(rewards_for_metrics)})")
        print(
            f"  Mean generation length: {mean_generation_len:.1f}")
        print(
            f"  Policy log probs mean: {policy_lps_for_metrics.mean().item():.4f}")
        print(
            f"  Ref log probs mean: {ref_lps_for_metrics.mean().item():.4f}")
        print(
            f"  KL divergence: {get_kl(policy_lps_for_metrics, ref_lps_for_metrics, mask_for_metrics).item():.4f}")

    if args.wandb:
        wandb.log({
            "step": step,
            "train_loss": loss_value,
            "train_mean_reward": mean_reward,
            "train_reward_std": reward_std,
            "train_accuracy": accuracy,
            "train_mean_generation_len": mean_generation_len,
            "train_kl_divergence": get_kl(policy_lps_for_metrics, ref_lps_for_metrics, mask_for_metrics).item(),
            "train_policy_lps_mean": policy_lps_for_metrics.mean().item(),
            "train_ref_lps_mean": ref_lps_for_metrics.mean().item(),
        })

def main():
    parser = argparse.ArgumentParser(description="Train a language model using GRPO on humor dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the model to load from HuggingFace")
    parser.add_argument("--G", type=int, default=16, help="Number of completions per prompt (group size)")
    parser.add_argument("--num_prompts_per_step", type=int, default=4, help="Total batch size for training")
    parser.add_argument("--b", type=int, default=16, help="Total batch size for training") # batch size since we grad accum 
    parser.add_argument("--beta", type=float, default=1e-2, help="KL divergence weight coefficient")
    parser.add_argument("--num_rl_steps", type=int, default=5_000, help="Number of RL training steps")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate for the optimizer")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate model every N steps") # every inner_updates * eval_every grad upates
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches to use for evaluation")
    parser.add_argument("--use_dr_len", action="store_true", help="Enable length-based reward shaping")
    parser.add_argument("--use_dr_std", action="store_true", help="Enable std-based reward shaping")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name for the wandb run")
    parser.add_argument("--group", type=str, default=None, help="Group for the wandb run")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="b-ngpt-rl",
            config=vars(args),
            name=args.wandb_run_name,
            group=args.group
        )

    G = args.G
    num_prompts_per_step = args.num_prompts_per_step
    # num_prompts_per_step 
    # G = groups_per_step 

    b = args.b
    assert b>=G, "b must be greater than or equal to G"
    num_groups_per_batch = int(b/G)
    num_rl_steps = args.num_rl_steps
    beta = args.beta
    lr = args.lr
    wd = args.wd
    accum_steps = num_prompts_per_step * G // b

    if args.verbose:
        print(f"Starting GRPO training with the following configuration:")
        print(f"  Model: {args.model_name}")
        print(f"  Group size (G): {G}")
        print(f"  Batch size: {b}")
        print(f"  Groups per batch: {num_groups_per_batch}")
        print(f"  Number of RL steps: {num_rl_steps}")
        print(f"  Beta (KL weight): {beta}")
        print(f"  Learning rate: {lr}")
        print(f"  Max new tokens: {args.max_new_tokens}")
        print(f"  Evaluation every: {args.eval_every} steps")
        print(f"  Number of eval batches: {args.num_eval_batches}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"  Wandb logging: {args.wandb}")
        print()

    if args.verbose:
        print("Loading model and tokenizer...")
    policy, tokenizer = get_model_tokenizer(args.model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # sanity check
    if args.verbose: 
        test_prompt = PROMPT
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_output = policy.generate(**test_inputs, max_new_tokens=20)
        print(f'shape: {test_output[0].shape}')
        print("Sanity check on joke:", tokenizer.decode(test_output[0]), '\n')

    if args.verbose:
        print(f"Model loaded with {sum(p.numel() for p in policy.parameters())/1e9:.2f}B parameters")
        
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))
    

    def get_cosine_schedule(step: int, warmup: int, num_rl_steps: int) -> float: 
        if step < warmup: 
            return step/warmup
        else: 
            progress = (step - warmup)/(num_rl_steps - warmup)  
            return math.cos(math.pi/2 * progress)

    warmup = int(0.1 * num_rl_steps)

    scheduler = LambdaLR(
        opt, 
        lr_lambda= lambda step: get_cosine_schedule(step, warmup, num_rl_steps),
    )

    if args.verbose:
        print("Creating reference model...")
    ref = deepcopy(policy)
    for p in ref.parameters(): 
        p.requires_grad = False 

    # Create a simple dataset of prompts - we just use the same prompt repeatedly
    # since we're training on joke generation with the same prompt format
    num_dummy_examples = 10000  # Large enough to not run out during training
    dummy_data = [(PROMPT, "")] * num_dummy_examples

    # let dataloader "batch" be just prompts for num_groups_per_batch
        # then expand those to group_sz each and pass to get_rollouts
        # then your batch is (expanded_prompts, all_completions)
    loader = DataLoader(
        dummy_data, 
        shuffle=True, 
        batch_size=num_groups_per_batch, 
        pin_memory=True,
        )

    if args.verbose:
        print(f"Starting training loop for {num_rl_steps} steps...")
        print()

    step = 0 
    print(f'G={G}, num_groups_per_batch={num_groups_per_batch}, b={b}, accum_steps={accum_steps}')
    
    # Initialize cumulative metrics storage
    cumulative_rewards = []
    cumulative_policy_lps = []
    cumulative_ref_lps = []
    cumulative_mask = []
    
    while True:
        for prompt_b, _ in tqdm(loader):
            # expand by a factor group_sz to num_prompts_per_step each (prompts)
            flat_prompts = []
            for p in prompt_b: # num groups, each tiled by group_sz
                flat_prompts += [p] * G

            completions_b, mask_b = get_completions_and_mask(
                flat_prompts, 
                policy, 
                tokenizer,
                max_new_tokens=args.max_new_tokens
            ) 
            full_seqs = [p + c for p, c in zip(flat_prompts, completions_b)]
            ref_lps_b = get_model_lps(
                ref, tokenizer, full_seqs, grad=False, 
            )
            rewards_b = reward_fn(
                completions_b, 
            )
            # extract only the completion portion of ref_lps
            L = mask_b.shape[1] # max completion len 
            ref_lps_b = ref_lps_b[:, -L:]
        
            policy_lps_b = get_model_lps(
                policy, tokenizer, full_seqs, grad=True,
            )
            policy_lps_b = policy_lps_b[:, -L:]

            # Accumulate metrics
            cumulative_rewards.append(rewards_b)
            cumulative_policy_lps.append(policy_lps_b)
            cumulative_ref_lps.append(ref_lps_b)
            cumulative_mask.append(mask_b)

            loss = loss_fn(
                rewards_b,
                policy_lps_b,
                ref_lps_b,
                mask_b,
                beta=beta,
                G=G,
                num_groups_per_batch=num_groups_per_batch,
                use_dr_len=args.use_dr_len,
                use_dr_std=args.use_dr_std,
            ) / accum_steps

            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
                scheduler.step()           

                # Concatenate cumulative metrics (pad variable sequence lengths)
                max_L = max(
                    [t.shape[1] for t in cumulative_mask] +
                    [t.shape[1] for t in cumulative_policy_lps] +
                    [t.shape[1] for t in cumulative_ref_lps]
                )
                pad_to = lambda x: F.pad(x, (0, max_L - x.shape[1]))
                cum_rewards = torch.cat(cumulative_rewards, dim=0)
                cum_policy_lps = torch.cat([pad_to(x) for x in cumulative_policy_lps], dim=0)
                cum_ref_lps = torch.cat([pad_to(x) for x in cumulative_ref_lps], dim=0)
                cum_mask = torch.cat([pad_to(x) for x in cumulative_mask], dim=0)

                log_metrics(
                    step,
                    loss,
                    rewards_b,
                    policy_lps_b,
                    ref_lps_b,
                    mask_b,
                    args,
                    cumulative_rewards=cum_rewards,
                    cumulative_policy_lps=cum_policy_lps,
                    cumulative_ref_lps=cum_ref_lps,
                    cumulative_mask=cum_mask
                )

                # Reset cumulative metrics
                cumulative_rewards = []
                cumulative_policy_lps = []
                cumulative_ref_lps = []
                cumulative_mask = []

            if step % args.eval_every == 0: # we want to eval on step 0 now so took out (if step > 0) condition
                if args.verbose:
                    print(f"\nRunning evaluation at step {step}...")
                
                jokes_lst, eval_reward = eval_model(
                    policy,
                    tokenizer,
                    b=b,
                    max_new_tokens=256,
                )
                
                if args.wandb:
                    wandb.log({
                        "step": step,
                        "eval_reward": eval_reward,
                    })
                
                if args.verbose:
                    print(f"Evaluation reward: {eval_reward:.4f}")
                    print("Sample jokes:")
                    for i, joke in enumerate(jokes_lst[:3]):
                        print(f"  {i+1}: {joke}")
                    print()
        
            if step >= num_rl_steps: 
                print(f"Training complete after {step} steps")
                import sys 
                sys.exit(0)

            step += 1
        

if __name__ == "__main__":
    main()