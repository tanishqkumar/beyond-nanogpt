import torch 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple 
import re 
from tqdm import tqdm 
from copy import deepcopy 
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import argparse

### FUNCTIONS TAKEN FROM evals/eval_gsm8k.py ###
PROMPT_TEMPLATE = """
You are a helpful language model assistant correctly solving math problems. You think step by step to reach 
the correct answer. You should output your final answer as `#### answer`. You will be shown some examples of 
how to answer such questions, and then you will be asked the question you should solve. 
--- 
Examples: 

{icl_examples}

--- 
Question: 
{question}

Answer: 
"""

def get_icl_examples(test: Dataset, num_fewshot: int = 4) -> Tuple[str, Dataset]: 
    # use .select instead of list splicing for Dataset object to avoid screwing it up 
    num_fewshot_lst = test.select(range(len(test)-num_fewshot, len(test)))

    # format the list of examples properly so we can slot this right into a prompt 
    examples = []

    for q, a in zip(num_fewshot_lst['question'], num_fewshot_lst['answer']):
        examples.append(f"Question: {q}\nAnswer: {a}")
    
    return '\n\n'.join(examples)

def get_model_tokenizer(name: str = "meta-llama/Llama-3.2-1B-Instruct") -> Tuple[AutoModelForCausalLM, AutoTokenizer]: 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    return model, tokenizer

def extract(text: str, pattern: str = r'####\s*([0-9]+)') -> Optional[int]: 
    match = re.search(pattern, text)
    if match is None: 
        return float('inf')
    else: 
        return int(match.group(1))

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
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
def reward_fn(completions: List[str], goldens: List[int]) -> torch.Tensor: 
    preds = list(map(lambda c: extract(c), completions))
    rewards = []
    
    for pred, golden in zip(preds, goldens):
        r = 1.0 if pred == golden else 0.0
        if pred is not float('inf'): # format reward 
            r += 0.1
        rewards.append(r)
    return torch.tensor(
        rewards, 
        dtype=torch.float32, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

# implementes GRPO: (rewards, ref_lps, policy_lps) -> final objective (-reward)
def loss_fn(
    rewards: torch.Tensor, 
    policy_lps: torch.Tensor, 
    ref_lps: torch.Tensor, 
    mask: torch.Tensor, # should be [B, max_completion_len], don't want padding to influence reward
    beta: float = 0.1, 
    G: int = 8,
    num_groups_per_batch: int = 2, 
) -> torch.Tensor:

    # using no-clip version without importance sampling ratios 
    # rewards is [B], broadcast to [B, max_completion_len]
    _, L = policy_lps.shape
    A = torch.empty_like(rewards) # [B]

    for i in range(num_groups_per_batch): 
        group_rewards = rewards[i*G:(i+1)*G]  # [G]
        mean = group_rewards.mean()
        A[i*G:(i+1)*G] = group_rewards - mean # no /std bc using Dr GRPO

    A = A.unsqueeze(1).expand(-1, L).detach()  # broadcast from seqs to tokens [B, L], dont want diff thru rewards 
    kl_loss_term = beta * get_kl(policy_lps, ref_lps, mask)
    adv_term = ((A * policy_lps * mask).sum(dim=1)).mean() # do .mean() and not .sum()/mask.sum() on purpose for Dr GRPO

    return -(adv_term - kl_loss_term) # loss = -reward, and inside parentheses is reward 
    

# seqs is [bsz] where each is prompt_i + completion_{i, j} where i in [num_prompts] and j in [num_completions_per_prompt]
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
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if grad: 
        outputs = model(**inputs)
    else: 
        with torch.no_grad(): 
            outputs = model(**inputs)

    logits = outputs.logits # BLV
    
    lps = F.log_softmax(logits, dim=-1)  # [B, L, V]
    
    # Gather log probs for actual tokens
    input_ids = inputs['input_ids']  # [B, L]
    lps = lps.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]

    return lps # B, L

# compute closed form for loss_fn 
def get_kl(policy_lps: torch.Tensor, ref_lps: torch.Tensor, mask: torch.Tensor) -> torch.float32: 
    diff_m = (policy_lps - ref_lps) * mask
    kl_t = torch.exp(diff_m) - (diff_m) - 1
    return kl_t.mean()

# returns completions for each prompt, may need to return mask from .generate or input_ids? 
def get_completions_and_mask(
    prompts: List[str], 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int = 256
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): 
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
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
    mask = (generated_tokens != tokenizer.pad_token_id).float()

    return completions, mask  # all are [B, L] where L = max_completion_len

def eval_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_dataset: Dataset,
    icl_examples: str,
    batch_size: int = 16,
    num_batches: int = None,
    max_new_tokens: int = 256,
    verbose: bool = False
) -> float:
    """Evaluate model on GSM8K test set, returns accuracy"""
    
    # Prepare test dataset
    test = test_dataset.select(range(len(test_dataset) - 8))  # exclude ICL examples
    if num_batches:
        test = test.select(range(num_batches * batch_size))
    
    loader = DataLoader(test, shuffle=False, batch_size=batch_size)
    
    num_correct = 0
    num_total = 0
    
    q2prompt = lambda q: PROMPT_TEMPLATE.format(icl_examples=icl_examples, question=q)
    
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


def main():
    parser = argparse.ArgumentParser(description="Train a language model using GRPO on GSM8K dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the model to load from HuggingFace")
    parser.add_argument("--G", type=int, default=16, help="Number of completions per prompt (group size)")
    parser.add_argument("--batch_size", type=int, default=16, help="Total batch size for training")
    parser.add_argument("--beta", type=float, default=1e-2, help="KL divergence weight coefficient")
    parser.add_argument("--num_rl_steps", type=int, default=1_000, help="Number of RL training steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for the optimizer")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_inner_updates", type=int, default=2, help="Num grad steps on single set of completions")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Name of the dataset to load")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration to use")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use for training")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate model every N steps") # every 200 grad updates 
    parser.add_argument("--num_eval_batches", type=int, default=25, help="Number of batches to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # Initialize wandb if requested
    if args.wandb:
        import wandb
        wandb.init(
            project="grpo-gsm8k-mine",
            config=vars(args)
        )

    G = args.G  # completions per prompt, ie. group size 
    batch_size = args.batch_size
    # need num_rl_steps and num_inner_updates 
    assert batch_size % G == 0, "ERROR: Completions per prompt must divide batch size" 

    num_groups_per_batch = int(batch_size/G)
    num_rl_steps = args.num_rl_steps
    beta = args.beta  # KL weight 
    lr = args.lr
    wd = args.wd
    num_inner_updates = args.num_inner_updates

    if args.verbose:
        print(f"Starting GRPO training with the following configuration:")
        print(f"  Model: {args.model_name}")
        print(f"  Dataset: {args.dataset_name} ({args.dataset_config}, {args.dataset_split})")
        print(f"  Group size (G): {G}")
        print(f"  Batch size: {batch_size}")
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
        print("Loading dataset...")
    train_ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    test_ds = load_dataset(args.dataset_name, args.dataset_config, split="test")

    if args.verbose:
        print(f"Dataset loaded with {len(train_ds)} examples")
        
    if args.verbose:
        print("Loading model and tokenizer...")
    policy, tokenizer = get_model_tokenizer(args.model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # After loading the model, do a quick test
    if args.verbose: 
        test_prompt = "What is 2 + 2?"
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_output = policy.generate(**test_inputs, max_new_tokens=20)
        print("Sanity check on 2+2:", tokenizer.decode(test_output[0]), '\n')

    if args.verbose:
        print(f"Model loaded with {sum(p.numel() for p in policy.parameters())/1e9:.2f}B parameters")
        
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))

    if args.verbose:
        print("Creating reference model...")
    ref = deepcopy(policy) # for KL computation, on cpu bc we don't need grads on it, save memory
    for p in ref.parameters(): 
        p.requires_grad = False 

    # let dataloader "batch" be just prompts/golden for num_groups_per_batch
        # then expand those to group_sz each and pass to get_rollouts
        # then your batch is (expanded_prompts, all_completions, expanded_goldens)

    icl_examples = get_icl_examples(test_ds)
    q2prompt = lambda q: PROMPT_TEMPLATE.format(question=q, icl_examples=icl_examples)
    prompt_qs = list(map(q2prompt, train_ds["question"]))
    loader = DataLoader(
        list(zip(prompt_qs, train_ds["answer"])), 
        shuffle=True, 
        batch_size=num_groups_per_batch, 
        )

    if args.verbose:
        print(f"Starting training loop for {num_rl_steps} steps...")
        print()

    step = 0 
    while step < num_rl_steps: # this upper loop resets dataloader when it runs dry 
        for prompt_b, golden_b_str in tqdm(loader):
            # extract from golden
            golden_b_int = list(map(lambda c: extract(c), golden_b_str)) # [num_groups]
            # expand by a factor group_sz to batch_size each (prompts, goldens)
            flat_prompts, flat_goldens = [], []
            for p, g in zip(prompt_b, golden_b_int): # num groups, each tiled by group_sz
                flat_prompts += [p] * G
                flat_goldens += [g] * G 
            # flats are now [batch_sz]

            if args.verbose and step % 5 == 0:
                print(f"Step {step}: Processing batch with {len(flat_prompts)} prompts")
                print(f"  Sample golden answer: {flat_goldens[0]}")

            # get completions and policy/ref lps 
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
            
            for _ in range(num_inner_updates): 
                # create full sequences (prompts + completions) for ref model
                policy_lps_b = get_model_lps(
                    policy, tokenizer, full_seqs, grad=True, 
                )
                policy_lps_b = policy_lps_b[:, -L:]

                opt.zero_grad()
                # compute loss 
                loss = loss_fn(
                    rewards_b, 
                    policy_lps_b, 
                    ref_lps_b, 
                    mask_b, 
                    beta=beta, 
                    G=G, 
                    num_groups_per_batch=num_groups_per_batch, 
                )

                # step 
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                loss.backward()
                opt.step()
            
            # Evaluation
            if step > 0 and step % args.eval_every == 0:
                if args.verbose:
                    print(f"\nRunning evaluation at step {step}...")
                
                eval_accuracy = eval_model(
                    policy,
                    tokenizer,
                    test_ds,
                    icl_examples,
                    batch_size=16,
                    num_batches=args.num_eval_batches,
                    max_new_tokens=256,
                    verbose=args.verbose
                )
                
                if args.wandb:
                    wandb.log({
                        "step": step,
                        "eval_accuracy": eval_accuracy,
                    })
                
                if args.verbose:
                    print(f"Evaluation accuracy: {eval_accuracy:.4f}\n")
            
            # Logging
            if step % 5 == 0:
                mean_reward = rewards_b.mean().item()
                loss_value = loss.item()
                reward_std = rewards_b.std().item()
                correct_predictions = (rewards_b > 1.0).sum().item()
                accuracy = correct_predictions / len(rewards_b)
                
                # Compute mean generation length
                generation_lengths = mask_b.sum(dim=1)  # sum of non-padded tokens per sequence
                mean_generation_len = generation_lengths.float().mean().item()
                
                if args.verbose:
                    print(f"Step {step}: Loss = {loss_value:.4f}, Mean Reward = {mean_reward:.4f}")
                    print(f"  Accuracy: {accuracy:.2%} ({correct_predictions}/{len(rewards_b)})")
                    print(f"  Mean generation length: {mean_generation_len:.1f}")
                    print(f"  Policy log probs mean: {policy_lps_b.mean().item():.4f}")
                    print(f"  Ref log probs mean: {ref_lps_b.mean().item():.4f}")
                    print(f"  KL divergence: {get_kl(policy_lps_b, ref_lps_b, mask_b).item():.4f}")
                
                if args.wandb:
                    wandb.log({
                        "step": step,
                        "train_loss": loss_value,
                        "train_mean_reward": mean_reward,
                        "train_reward_std": reward_std,
                        "train_accuracy": accuracy,
                        "train_mean_generation_len": mean_generation_len,
                        "train_kl_divergence": get_kl(policy_lps_b, ref_lps_b, mask_b).item(),
                        "train_policy_lps_mean": policy_lps_b.mean().item(),
                        "train_ref_lps_mean": ref_lps_b.mean().item(),
                    })
            
            step += 1

if __name__ == "__main__":
    main()