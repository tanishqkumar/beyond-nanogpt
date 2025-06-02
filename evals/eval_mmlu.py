'''
(https://arxiv.org/pdf/2009.03300) Measuring Massive Multitask Language Understanding

This file implements evaluation code for the classic MMLU benchmark,
which tests language models across 57 diverse academic subjects ranging from math to law to history.

The core idea is to evaluate how well a language model can answer multiple-choice questions when given
few-shot examples (in-context learning). We take the max logit out of the choices [A, B, C, D] when 
the model is given the question in context (really, it's given instructions, few shot exampls, then the question). 
    We use here a permutation-based approach to make
    the results more robust: instead of just asking "what's the answer to this question?", it asks the same
    question multiple times with the answer choices shuffled in different orders, then averages the results.

TLDR: 
1. Load a pre-trained language model and the MMLU dataset
2. For each test question, create multiple versions with different orderings of the answer choices (A/B/C/D)
3. For each permutation, show the model some few-shot examples followed by the target question
4. Extract the model's predictions by looking at the logits for tokens 'A', 'B', 'C', 'D'
5. Convert predictions back to the original answer indexing and compute accuracy across all permutations
6. Average the per-permutation accuracies to get a final score for that question

This permutation approach helps reduce bias from position effects - some models might have learned
to prefer certain answer positions (like always picking 'A'), so by shuffling and averaging,
we get a more fair assessment of the model's actual knowledge rather than its positional biases.

Key functions:
- get_sysprompt_tensor(sys_prompt, batch_sz, tokenizer, device) -> [batch_sz, sys_prompt_len]
- get_icl_example_t(mmlu, tokenizer, q_idx, batch_sz, device) -> [batch_sz, example_len]  
- get_question_tensor(mmlu, tokenizer, q_idx, num_options, device) -> (question_tensor, mask_tensor)

These are concatenated to form the full input sequence: [sys_prompt | icl_examples | target_question]
and eval_mmlu() uses this to construct the data, feeds it through the models, extracts logits from the outputs, 
gets true answer from dataset, accounts for permuted options and permutes answers accordingly to ensure it's always 
got the right answer choice, and logs metrics about how often the highest logit amongst answer choices 
was the correct answer. 

'''

import argparse
import torch 
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForCausalLM
import math 
from typing import List, Dict, Optional 
import itertools
from tqdm import tqdm 
import numpy as np

SYS_PROMPT = ''' 
You are a helpful assistant. You will be asked some multiple-choice questions, 
and you should answer them to the best of your ability by outputting the letter of the 
answer choice that is your answer. You are shown some examples of this before being 
asked the real question at the end, for which you should give your answer as a single letter. 
''' 

def get_sysprompt_tensor(
    sys_prompt: str, 
    batch_sz: int, 
    tokenizer,
    device: torch.device = torch.device("cuda"), 
) -> torch.Tensor: 
    ids = tokenizer.encode(sys_prompt, return_tensors="pt")
    batch = ids[0].unsqueeze(0).expand(batch_sz, -1)  # [batch_sz, sys_prompt_len]
    return batch.to(device)

def get_indices(len_dataset: int, num: int = 200) -> List[int]: 
    return torch.randperm(len_dataset)[:num].tolist()

def get_icl_example_t(
        mmlu, 
        tokenizer, 
        q_idx: int, 
        batch_sz: int = 24, 
        device: torch.device = torch.device("cuda"), 
    ) -> torch.Tensor: # for icl examples, returns [batch_sz, q_prefix_len + choices_len + q_len + answer_len]
    
    question_text = mmlu["test"]["question"][q_idx]
    question_ids = tokenizer.encode(question_text, return_tensors="pt")
    question_ids_t = question_ids[0].to(device)

    options_text_list = mmlu["test"]["choices"][q_idx] 
    options_s = ''.join([f'{chr(65+i)}: {perm_s}\n' for i, perm_s in enumerate(options_text_list)])
    
    options_text_ids = tokenizer(options_s, return_tensors="pt")
    options_ids_t = options_text_ids["input_ids"][0].to(device)

    answers = ["A", "B", "C", "D"]
    answer_idx = mmlu["test"]["answer"][q_idx]
    answer_str = answers[answer_idx]
    answer_text = "Answer: " + answer_str + "\n"

    answer_ids = tokenizer.encode(answer_text, return_tensors="pt")
    answer_ids_t = answer_ids[0].to(device)

    icl_example = torch.cat([question_ids_t, options_ids_t, answer_ids_t], dim=0)
    return icl_example.unsqueeze(0).expand(batch_sz, -1)  # [batch_sz, total_example_len]

def get_question_tensor(
        mmlu, 
        tokenizer, 
        q_idx: int, 
        num_options: int = 4, 
        device: torch.device = torch.device("cuda"), 
    ) -> torch.Tensor: # for icl examples, returns [batch_sz, q_prefix_len + choices_len + q_len + answer_len]
    
    num_perms = math.factorial(num_options)  # 24 permutations for 4 choices

    question_text = mmlu["test"]["question"][q_idx]
    question_ids = tokenizer.encode(question_text, return_tensors="pt")
    question_ids_t = question_ids[0].unsqueeze(0).expand(num_perms, -1).to(device)  # [num_perms, q_len]

    options_text_list = mmlu["test"]["choices"][q_idx] 
    perms_list = list(itertools.permutations(options_text_list))
    # list of lists, each sublist is text, need to format into coherent text and batch tokenize
    options_s_list = []
    for perm in perms_list: 
        # perm is ['option 1 text', ..., 'option 4 text'], turn into one string so we have a list of num_perm strings
        options_s = ''.join([f'{chr(65+i)}: {perm_s}\n' for i, perm_s in enumerate(perm)])
        options_s_list.append(options_s)
    options_text_ids, mask = tokenizer(options_s_list, return_tensors="pt", padding=True).values()  # batch tokenize all permutations
    
    options_ids_t = options_text_ids.to(device)  # [num_perms, max_options_len]
    mask = mask.to(device)

    answer_text = "Answer: "
    answer_ids = tokenizer.encode(answer_text, return_tensors="pt")
    answer_ids_t = answer_ids[0].unsqueeze(0).expand(num_perms, -1).to(device)

    mask = torch.cat([torch.ones_like(question_ids_t), mask, torch.ones_like(answer_ids_t)], dim=-1)  # full mask for concatenated sequence

    return torch.cat([question_ids_t, options_ids_t, answer_ids_t], dim=-1), mask
    

def get_batch(
    mmlu,
    tokenizer, 
    q_idx: int,
    sys_prompt: str = SYS_PROMPT, 
    num_options: int = 4,
    device: torch.device = torch.device("cuda"), 
    num_fewshot: int = 8, 
) -> torch.Tensor: 

    if not tokenizer.pad_token: 
        tokenizer.pad_token = tokenizer.eos_token

    len_dataset = len(mmlu["test"])

    num_perms = math.factorial(num_options)
     
    sys_prompts = get_sysprompt_tensor(sys_prompt, num_perms, tokenizer, device)  # [num_perms, sys_len]
    
    icl_examples_list = []
    all_masks = [torch.ones_like(sys_prompts)]

    for i in range(num_fewshot): 
        eg = get_icl_example_t(mmlu, tokenizer, len_dataset-1-i, device=device)  # use last examples for few-shot
        icl_examples_list.append(eg)
        all_masks.append(torch.ones_like(eg))

    icl_examples_t = torch.cat(icl_examples_list, dim=-1)  # [num_perms, total_icl_len]
    real_qs_t, qs_m = get_question_tensor(mmlu, tokenizer, q_idx, num_options, device)
    all_masks.append(qs_m)
    
    batch_qs = torch.cat([sys_prompts, icl_examples_t, real_qs_t], dim=-1)  # [num_perms, full_seq_len]
    batch_masks = torch.cat(all_masks, dim=-1)
    
    return batch_qs, batch_masks 


# main script will argparse into this 
def eval_mmlu(
    model_name: str = "Qwen/Qwen3-1.7B", 
    num_questions: int = 100, 
    num_fewshot: int = 8, 
    device: torch.device = torch.device("cuda"), 
    verbose: bool = False, 
    num_choices: int = 4, 
): 

    mmlu = load_dataset("cais/mmlu", "all")
    question_indices = get_indices(len(mmlu["test"]), num_questions)
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    choice_indices = [
        tokenizer.encode('A')[0],
        tokenizer.encode('B')[0],
        tokenizer.encode('C')[0],
        tokenizer.encode('D')[0]
    ]
    
    question_accs = [] 
    window_size = 10  # For running average if verbose 
    
    for i, q_idx in enumerate(tqdm(question_indices)): 
        batch_input_t, batch_mask_t = get_batch(mmlu, tokenizer, q_idx, num_fewshot=num_fewshot, device=device)
        logits = model(batch_input_t, attention_mask=batch_mask_t).logits  # [batch_sz, seq_len, vocab_sz]
        preds = logits[:, -1, choice_indices].argmax(dim=-1)  # [num_perms] - get choice predictions for each permutation

        perms = list(itertools.permutations(range(num_choices)))
        ans_idx = mmlu["test"]["answer"][q_idx]
        ans = torch.tensor(list(map(lambda p: p.index(ans_idx), perms)), dtype=torch.long, device=device)  # correct answer index for each permutation
        accuracy = (preds == ans).float().mean().item()  # average accuracy across all permutations
        question_accs.append(accuracy)

        if verbose:
            correct_answer = ["A", "B", "C", "D"][mmlu["test"]["answer"][q_idx]]
            predicted_answers = [["A", "B", "C", "D"][p.item()] for p in preds]
            
            print(f"\nQuestion {i+1}/{len(question_indices)} (idx {q_idx}):")
            print(f"  Correct answer: {correct_answer}")
            print(f"  Predicted: {predicted_answers}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Running averages
            overall_avg = sum(question_accs) / len(question_accs)
            print(f"  Overall avg so far: {overall_avg:.4f} ({overall_avg*100:.2f}%)")
            
            if len(question_accs) >= window_size:
                recent_avg = sum(question_accs[-window_size:]) / window_size
                print(f"  Last {window_size} avg: {recent_avg:.4f} ({recent_avg*100:.2f}%)")
    
    mean_acc = sum(question_accs) / len(question_accs)
    std_acc = np.std(question_accs)
    
    print("\n" + "="*50)
    print("MMLU EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Questions evaluated: {len(question_accs)}")
    print(f"Few-shot examples: {num_fewshot}")
    print("-"*50)
    print(f"Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"Standard deviation: {std_acc:.4f}")
    print(f"Min accuracy: {min(question_accs):.4f}")
    print(f"Max accuracy: {max(question_accs):.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLU performance")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", 
                        help="Name of the model to evaluate")
    parser.add_argument("--num_questions", type=int, default=10,
                        help="Number of questions to evaluate")
    parser.add_argument("--num_fewshot", type=int, default=2,
                        help="Number of few-shot examples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda, cpu, mps)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    eval_mmlu(
        model_name=args.model_name,
        num_questions=args.num_questions,
        num_fewshot=args.num_fewshot,
        device=device,
        verbose=args.verbose
    )

