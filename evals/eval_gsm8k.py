'''
(https://arxiv.org/abs/2110.14168) Training Verifiers to Solve Math Word Problems

This module implements an evaluation system for the GSM8K dataset (math word problems), which is 
a canonical generative evaluation (ie. models generate until they output the answer in a given format)
which tests arithmetic ability. 

Key architectural decisions:
- Uses in-context learning with examples provided via PROMPT_TEMPLATE
- Leverages transformers library for model loading (defaults to Llama-3.2-3B)
- Extracts numerical answers using regex pattern matching on "#### answer" format
- Handles tokenizer padding token setup for generation
- Implements batch processing with DataLoader for efficient evaluation
'''
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, List
import re 
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

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

def get_model_tokenizer(name: str = "meta-llama/Llama-3.2-3B") -> Tuple[AutoModelForCausalLM, AutoTokenizer]: 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def extract(text: str, pattern: str = r'####\s*([0-9]+)') -> Optional[int]: 
    match = re.search(pattern, text)
    if match is None: 
        return None 
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
        )

    # Get only the new tokens by slicing off the input tokens
    new_tokens = outputs_t[:, inputs['input_ids'].shape[1]:]
    
    outputs_s = tokenizer.batch_decode(
        new_tokens, 
        skip_special_tokens=True, 
    )
    
    return outputs_s

def get_icl_examples(test: Dataset, num_fewshot: int = 8) -> Tuple[str, Dataset]: 
    # use .select instead of list splicing for Dataset object to avoid screwing it up 
    num_fewshot_lst = test.select(range(len(test)-num_fewshot, len(test)))
    test = test.select(range(len(test)-num_fewshot))

    # format the list of examples properly so we can slot this right into a prompt 
    examples = []

    for q, a in zip(num_fewshot_lst['question'], num_fewshot_lst['answer']):
        examples.append(f"Question: {q}\nAnswer: {a}")
    
    return '\n\n'.join(examples), test 

def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K performance")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Model to eval")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=8, help="Number of few-shot examples")
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_fewshot = args.num_fewshot
    num_batches = args.num_batches
    model_name = args.model 

    if args.wandb:
        import wandb
        wandb.init(project="gsm8k-eval")
        wandb.config.update(vars(args))

    test = load_dataset("openai/gsm8k", "main", split="test")
    icl_examples_s, test = get_icl_examples(test, num_fewshot)

    model, tokenizer = get_model_tokenizer(model_name)
    tokenizer.padding_side = 'left'

    if num_batches: 
        test = test.select(range(num_batches * batch_size))
    
    loader = DataLoader(test, shuffle=True, batch_size=batch_size)

    num_correct = 0
    num_total = 0

    progress_bar = tqdm(loader) if args.verbose else loader

    for batch_idx, batch in enumerate(progress_bar):
        if num_batches is not None and batch_idx >= num_batches:
            break
            
        questions = batch['question']
        answers = batch['answer']

        q2prompt = lambda q: PROMPT_TEMPLATE.format(icl_examples=icl_examples_s, question=q)
        prompts = list(map(q2prompt, questions))

        preds = get_batch_preds(
            model,
            prompts,
            tokenizer,
        )  # output is List[str]

        preds_int = list(map(lambda s: extract(s), preds))  # List[int]    
        preds_int = [(i if i is not None else float('inf')) for i in preds_int] # extraction can fail, should be marked as wrong 
        answers_int = list(map(lambda a: extract(a), answers))  # List[int]
        
        num_correct += sum([p == a for p, a in zip(preds_int, answers_int)])
        num_total += len(answers_int)
        
        if args.verbose and isinstance(progress_bar, tqdm):
            progress_bar.set_description(f"Accuracy: {num_correct/num_total:.4f}")

    accuracy = num_correct/num_total
    print(f'Accuracy = {accuracy:.4f}')
    
    if args.wandb:
        wandb.log({"accuracy": accuracy, "num_correct": num_correct, "num_total": num_total})

if __name__ == "__main__":
    main()