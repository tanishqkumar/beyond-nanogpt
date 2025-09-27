'''
(https://arxiv.org/pdf/2402.03300) DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
(https://arxiv.org/pdf/2501.12948) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

The first paper introduces GRPO, and the second uses it to improve reasoning for a base model (V3 --> R1). 
This implements something equivalent to using GRPOTrainer in HuggingFace, the algorithmic logic is easier 
than maybe you'd expect. 

There are some important subtleties, though -- RL is particularly sensitive to hypers/implementation details. 
For instance, if you read the HF implementation notes [https://huggingface.co/docs/trl/main/en/grpo_trainer] 
you'll see there are a number of ablations to the core GRPO algorithm that are common in modern LLM RL in production, 
and their efficacy is contested. HF itself does NOT actually implement the formula in the paper -- for instance, 
with their default hypers it's much more like vanilla REINFORCE but with advantage computed from group statistics 
(Monte-carlo). 
    `inner_updates` for instance, determines how much we re-use a given set of rollouts (ie. re-compute logprobs)
    under policy and policy0 = ref and use those new logprobs and the same advantages for new updates. The larger it is, 
    the more you are "off-policy" and hence clipping is important in the GRPO update since typically it's >1 (which is 
    totally on-policy). However, HF uses inner_updates = 1 (which they call `mu`), and so this means there is no "th_old" 
    as in the GRPO formula, so that the two entries in clip become identical and thus clipping is not necessary 
    (hence why you don't see clipping in my code). 

    It's also a bit weird because Dr GRPO (a new memory efficient version) makes two key ablations: first, 
    not including .std() in the computation of advantages (see reward_scale=False), and second, not including 1/|o_i| in the GRPO formula. 
    HF does NOT include the first ablation, but DOES include the second, so it's a kind of GRPO / Dr GRPO mishmash. 

    I also try to use their default GRPROTrainer hypers where possible, but you can fiddle with those 
    (eg. they set beta = 0 by default but I found our code often has KL divergence blowing up so I set it to 
    a nonzero value so that the policy stays close to its initial distribution, and this led to the training 
    being well-behaved). 

    Nonetheless, if you read the GRPO paper and take away the core idea that 1) GRPO's core contribution is 
    the Monte Carlo advantage estimation from within-group statistics (as opposed to training a separate value head 
    on lots of data), and 2) there are some subtle implementation details that can be ablated, and actually these 
    matter a lot, then you'll have gotten most everything you need to out of this implementation. 
    
An important philosophical point here is also that when we want to RL on a new task, we basically only have to 
change reward_fn, since everything else in the algorithm stays the says RLing LLMs across different envs 
(for writing, instruction tuning, math, etc). This is why reward-shaping and "environment design" are so key 
in modern LLM-based RL. 

Ultimately, the only meaningful test of correctness here is whether eval accuracy/reward goes up. And if you 
run this code -- you'll see indeed it does!
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

### FUNCTIONS TAKEN FROM evals/eval_gsm8k.py ###
PROMPT_TEMPLATE = """
You are a helpful language model assistant correctly solving math problems. You think step by step to reach 
the correct answer. You should output your final answer as `#### answer` after you are done thinking. You will be shown some examples of 
how to answer such questions, and then you will be asked the question you should solve. 
--- 
Examples: 

{icl_examples}

--- 
Question: 
{question}

Answer: 
"""

PROMPT_TEMPLATE_NO_ICL = """
You are a helpful language model assistant correctly solving math problems. You think step by step to reach 
the correct answer. You should output your final answer as `#### answer` after you are done thinking.

Question: 
{question}

Answer: 
"""

def get_icl_examples(test: Dataset, num_fewshot: int = 4) -> str: 
    
    # use .select instead of list splicing for Dataset object to avoid screwing it up 
    num_fewshot_lst = test.select(range(len(test)-num_fewshot, len(test)))

    # format the list of examples properly so we can slot this right into a prompt 
    examples = []

    for q, a in zip(num_fewshot_lst['question'], num_fewshot_lst['answer']):
        examples.append(f"Question: {q}\nAnswer: {a}")
    
    return '\n\n'.join(examples)

# TODOs: format reward, check off by 1 and correctness, launch sweeps

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

def extract(text: str) -> Union[int, float]: 
    if "####" not in text:
        return float('inf')
    try:
        answer_part = text.split("####")[1].strip().replace(",", "").replace("$", "")
        return int(answer_part)
    except:
        return float('inf')

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
### END TAKEN FROM evals/eval_gsm8k.py ###


# output is [bsz] tensor of sequence-wise rewards, uses extract() on completions
def output_format_reward(pred: int) -> float: 
    return (0.5 if pred != float('inf') else 0.)

def correctness_reward(pred: int, golden: int) -> float: 
    if pred == float('inf'): return 0. 
    else: 
        return (2.0 if pred == golden else 0.)

def reward_fn(completions: List[str], goldens: List[int]) -> torch.Tensor: 
    preds = list(map(lambda c: extract(c), completions))
    rewards = []
    
    for pred, golden in zip(preds, goldens):
        r = correctness_reward(pred, golden)
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
            temperature=0.9, # nontrivial exploration
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
    test_dataset: Dataset,
    icl_examples: str,
    b: int = 16,
    num_batches: int = None,
    max_new_tokens: int = 256,
    verbose: bool = False,
    num_fewshot: int = 4
) -> float:
    
    # Prepare test dataset - exclude ICL examples
    test = test_dataset.select(range(len(test_dataset) - num_fewshot))
    if num_batches:
        test = test.select(range(num_batches * b))
    
    loader = DataLoader(test, shuffle=False, batch_size=b)
    
    num_correct = 0
    num_total = 0
    
    if num_fewshot > 0:
        q2prompt = lambda q: PROMPT_TEMPLATE.format(icl_examples=icl_examples, question=q)
    else:
        q2prompt = lambda q: PROMPT_TEMPLATE_NO_ICL.format(question=q)
    
    for batch in loader:
        questions = batch['question']
        answers = batch['answer']
        
        prompts = list(map(q2prompt, questions))
        
        preds = get_batch_preds(
            model,
            prompts,
            tokenizer,
            max_new_tokens=max_new_tokens,
        )
        
        preds_int = list(map(lambda s: extract(s), preds))
        preds_int = [(i if i != float('inf') else None) for i in preds_int]
        answers_int = list(map(lambda a: extract(a), answers))
        
        num_correct += sum([p == a for p, a in zip(preds_int, answers_int) if p is not None])
        num_total += len(answers_int)
    
    accuracy = num_correct / num_total
    
    if verbose:
        print(f"Evaluation: Accuracy = {accuracy:.4f} ({num_correct}/{num_total})")
    
    return accuracy

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
    parser = argparse.ArgumentParser(description="Train a language model using GRPO on GSM8K dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the model to load from HuggingFace")
    parser.add_argument("--G", type=int, default=16, help="Number of completions per prompt (group size)")
    parser.add_argument("--num_prompts_per_step", type=int, default=16, help="Total batch size for training")
    parser.add_argument("--b", type=int, default=16, help="Total batch size for training") # batch size since we grad accum 
    parser.add_argument("--beta", type=float, default=1e-2, help="KL divergence weight coefficient")
    parser.add_argument("--num_rl_steps", type=int, default=1_000, help="Number of RL training steps")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate for the optimizer")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Name of the dataset to load")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration to use")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use for training")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate model every N steps") # every inner_updates * eval_every grad upates
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches to use for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=4, help="Number of few-shot examples to use in prompts")
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
    num_fewshot = args.num_fewshot
    accum_steps = num_prompts_per_step * G // b

    if args.verbose:
        print(f"Starting GRPO training with the following configuration:")
        print(f"  Model: {args.model_name}")
        print(f"  Dataset: {args.dataset_name} ({args.dataset_config}, {args.dataset_split})")
        print(f"  Group size (G): {G}")
        print(f"  Batch size: {b}")
        print(f"  Groups per batch: {num_groups_per_batch}")
        print(f"  Number of RL steps: {num_rl_steps}")
        print(f"  Beta (KL weight): {beta}")
        print(f"  Learning rate: {lr}")
        print(f"  Max new tokens: {args.max_new_tokens}")
        print(f"  Number of few-shot examples: {num_fewshot}")
        print(f"  Evaluation every: {args.eval_every} steps")
        print(f"  Number of eval batches: {args.num_eval_batches}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"  Wandb logging: {args.wandb}")
        print()

    if args.verbose:
        print("Loading dataset...")
    train_ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    test_ds = load_dataset(args.dataset_name, args.dataset_config, split="test")

    if args.verbose:
        print(f"Dataset loaded with {len(train_ds)} examples")
        print("Loading model and tokenizer...")
    policy, tokenizer = get_model_tokenizer(args.model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # sanity check
    if args.verbose: 
        test_prompt = "What is 2 + 2?"
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_output = policy.generate(**test_inputs, max_new_tokens=20)
        print("Sanity check on 2+2:", tokenizer.decode(test_output[0]), '\n')

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

    if num_fewshot > 0:
        icl_examples = get_icl_examples(test_ds, num_fewshot)
        q2prompt = lambda q: PROMPT_TEMPLATE.format(question=q, icl_examples=icl_examples)
    else:
        q2prompt = lambda q: PROMPT_TEMPLATE_NO_ICL.format(question=q)
        icl_examples = ""

    prompt_qs = list(map(q2prompt, train_ds["question"]))

    # let dataloader "batch" be just prompts/golden for num_groups_per_batch
        # then expand those to group_sz each and pass to get_rollouts
        # then your batch is (expanded_prompts, all_completions, expanded_goldens)
    loader = DataLoader(
        list(zip(prompt_qs, train_ds["answer"])), 
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
    
    while step < num_rl_steps:
        for prompt_b, golden_b_str in tqdm(loader):
            # extract from golden
            golden_b_int = list(map(lambda c: extract(c), golden_b_str)) # [num_groups]
            # expand by a factor group_sz to num_prompts_per_step each (prompts, goldens)
            flat_prompts, flat_goldens = [], []
            for p, g in zip(prompt_b, golden_b_int): # num groups, each tiled by group_sz
                flat_prompts += [p] * G
                flat_goldens += [g] * G 

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
                completions_b, flat_goldens
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

            if step % (args.eval_every * accum_steps) == 0: # we want to eval on step 0 now so took out (if step > 0) condition
                if args.verbose:
                    print(f"\nRunning evaluation at step {step}...")
                
                eval_accuracy = eval_model(
                    policy,
                    tokenizer,
                    test_ds,
                    icl_examples,
                    b=b,
                    num_batches=args.num_eval_batches,
                    max_new_tokens=256,
                    verbose=args.verbose,
                    num_fewshot=num_fewshot
                )
                
                if args.wandb:
                    wandb.log({
                        "step": step,
                        "eval_accuracy": eval_accuracy,
                    })
                
                if args.verbose:
                    print(f"Evaluation accuracy: {eval_accuracy:.4f}\n")
        
                
            step += 1
            if step >= num_rl_steps: break 
        

if __name__ == "__main__":
    main()