'''
(https://arxiv.org/pdf/2009.03300) Measuring Massive Multitask Language Understanding

This file implements evaluation code for the classic MMLU benchmark,
which tests language models across 57 diverse academic subjects ranging from math to law to history.

The core idea is to evaluate how well a language model can answer multiple-choice questions when given
few-shot examples (in-context learning). We take the max logit out of the choices [A, B, C, D] when 
the model is given the question in context (really, it's given instructions, few shot examples, then the question). 
    We use here a permutation-based approach to make
    the results more robust: instead of just asking "what's the answer to this question?", it asks the same
    question multiple times with the answer choices shuffled in different orders, then averages the results to avoid 
    LLM bias toward choosing the first/last options blindly (which is known to happen in weaker LLMs). 

Overview:
• The main eval flow is: load MMLU dataset → create MMLUDataset that handles permutations → 
  batch process through DataLoader → extract logits for A/B/C/D tokens → accumulate accuracies 
  per question by using max choices logit as prediction and comparing to label in dataset.
    

• Key functions: MMLUDataset generates all permutations of answer choices for each question and 
  formats them with few-shot examples. mmlu_collate_fn handles batching with proper padding. 
  eval_mmlu() orchestrates everything - loads model/tokenizer, creates dataset, runs inference, 
  and reports per-question + aggregate accuracy stats.

'''
from __future__ import annotations 
import argparse
import torch 
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForCausalLM
import math 
from typing import List, Dict, Optional 
import itertools
from tqdm import tqdm 
import random 
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


SYS_PROMPT = ''' 
You are a helpful assistant. You will be asked some multiple-choice questions, 
and you should answer them to the best of your ability by outputting the letter of the 
answer choice that is your answer. You are shown some examples of this before being 
asked the real question at the end, for which you should give your answer as a single letter. 
''' 

def get_indices(len_dataset: int, num: int = 200) -> List[int]: 
    return torch.randperm(len_dataset)[:num].tolist()

def get_icl_example_t(
        mmlu, 
        tokenizer, 
        q_idx: int, 
        device: torch.device = torch.device("mps"), 
    ) -> torch.Tensor: # for icl examples, returns [batch_sz, q_prefix_len + choices_len + q_len + answer_len]
    
    question_text = mmlu["test"]["question"][q_idx]
    question_ids = tokenizer.encode(question_text, return_tensors="pt")
    question_ids_t = question_ids[0].to(device)

    options_text_list = mmlu["test"]["choices"][q_idx] 
    # format choices as "A: option1\nB: option2\n..." 
    options_s = ''.join([f'{chr(65+i)}: {perm_s}\n' for i, perm_s in enumerate(options_text_list)])
    
    options_text_ids = tokenizer(options_s, return_tensors="pt")
    options_ids_t = options_text_ids["input_ids"][0].to(device)

    answers = ["A", "B", "C", "D"]
    answer_idx = mmlu["test"]["answer"][q_idx]
    answer_str = answers[answer_idx]
    answer_text = "Answer: " + answer_str + "\n"

    answer_ids = tokenizer.encode(answer_text, return_tensors="pt")
    answer_ids_t = answer_ids[0].to(device)

    # concat everything to form a complete in-context learning example
    icl_example = torch.cat([question_ids_t, options_ids_t, answer_ids_t], dim=0)
    return icl_example


# let's make a dataset to wrap into a dataloader, dataset only for questions, not sysprompt/icl examples 
    # it should absorb get question and get batch and use get syspromp/get icl examples 
    # each batch should have many questions, and p
    # key thing is dataset vs dataloader w/ collate allows separation of example construction (dataset) with batch construction (dataloader)
class MMLUDataset: 
    def __init__(
        self,
        full_mmlu, 
        filtered_mmlu, 
        tokenizer, 
        len_dataset_full: int, 
        num_choices: int = 4, 
        num_fewshot: int = 8, 
        device: torch.device = torch.device("mps"), 
    ): 
        
        self.full_dataset = full_mmlu 
        self.dataset = filtered_mmlu 
        self.tokenizer = tokenizer 
        # generate all possible permutations of answer choices (4! = 24 for A,B,C,D)
        self.perms = list(itertools.permutations(range(num_choices)))
        self.len_dataset = len(self.dataset["test"]["question"])
        self.len_dataset_full = len_dataset_full
        self.num_choices = num_choices
        self.device = device 

        global SYS_PROMPT
        self.sys_prompt_t = tokenizer.encode(SYS_PROMPT, return_tensors="pt")[0].to(self.device) # 1d tensor
        
        # build few-shot examples from the end of the full dataset to avoid overlap with test questions
        icl_examples = []
        for i in range(num_fewshot): 
            icl_examples.append(get_icl_example_t(self.full_dataset, tokenizer, self.len_dataset_full-1-i, device))

        self.icl_examples_t = torch.cat(icl_examples, dim=0) # 1d tensor 
        self.answer_prompt_t = tokenizer.encode("Answer: ", return_tensors="pt")[0].to(self.device) # 1d tensor

    def __len__(self): 
        # total dataset size is num_questions * num_permutations_per_question
        return self.len_dataset * len(self.perms)

    def __getitem__(self, idx: int): # idx is in [0, ..., num_questions * num_perms_per_q]
        # decode the flat index into question index and permutation index
        q_idx, perm_idx = idx // len(self.perms), idx % len(self.perms)

        q_text = self.dataset["test"]["question"][q_idx]
        q_t = self.tokenizer.encode(q_text, return_tensors="pt")[0].to(self.device)
        this_perm = self.perms[perm_idx]

        options_text_list = self.dataset["test"]["choices"][q_idx] 
        # apply the permutation to shuffle answer choices
        options_s = ''.join([f'{chr(65+i)}: {options_text_list[this_perm[i]]}\n' for i in range(self.num_choices)])
    
        options_text_ids = self.tokenizer(options_s, return_tensors="pt")
        choices_t = options_text_ids["input_ids"][0].to(self.device)

        full_prompt_t = torch.cat([self.sys_prompt_t, self.icl_examples_t, q_t, choices_t, self.answer_prompt_t])

        og_ans_idx = self.dataset["test"]["answer"][q_idx]
        # find where the original correct answer ended up after permutation
        answer_idx_this_perm = this_perm.index(og_ans_idx)
        answer_text = self.dataset["test"]["choices"][q_idx][og_ans_idx]

        return { 
            "full_prompt_t": full_prompt_t, 
            "q_idx": q_idx, 
            "perm_idx": perm_idx, 
            "answer_text": answer_text, 
            "answer_idx_this_perm": answer_idx_this_perm
        }
        
def mmlu_collate_fn(
        batch, # list of items each from dataset 
        tokenizer, 
        device: torch.device = torch.device("mps"), 
    ): 
    

    prompts = [item["full_prompt_t"] for item in batch]
    labels = [item["answer_idx_this_perm"] for item in batch]
    q_indices = [item["q_idx"] for item in batch]

    # pad sequences to same length for batching
    input_t = pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    mask_t = (input_t != tokenizer.pad_token_id).to(torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    q_indices_t = torch.tensor(q_indices, dtype=torch.long, device=device)

    return (input_t, mask_t, labels_t, q_indices_t) 


# main script will argparse into this 
def eval_mmlu(
    model_name: str = "HuggingFaceTB/SmolLM-135M", 
    num_questions: int = 10, 
    num_fewshot: int = 4, 
    device: torch.device = torch.device("mps"), 
    batch_size: int = 4, 
    verbose: bool = False, 
    num_choices: int = 4, 
): 

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mmlu = load_dataset("cais/mmlu", "all")
    
    len_dataset_full = len(mmlu["test"])
    question_indices = get_indices(len_dataset_full, num_questions)
    # only use a random num_questions subset of the full dataset
    filtered_mmlu = {
        "test": mmlu["test"].select(question_indices),
    }
    
    filtered_dataset = MMLUDataset(
        full_mmlu=mmlu, 
        filtered_mmlu=filtered_mmlu, 
        tokenizer=tokenizer, 
        len_dataset_full=len_dataset_full, 
        num_choices=num_choices, 
        num_fewshot=num_fewshot, 
        device=device,
    )

    num_perms = math.factorial(num_choices)
    collate_fn = lambda batch: mmlu_collate_fn(batch, tokenizer, device)
    dataloader = DataLoader(filtered_dataset, collate_fn=collate_fn, batch_size=batch_size)
    
    # get token ids for A, B, C, D choices to extract their logits
    choice_indices = [
        tokenizer.encode('A')[0],
        tokenizer.encode('B')[0],
        tokenizer.encode('C')[0],
        tokenizer.encode('D')[0]
    ]
    
    correct_counts = torch.zeros(num_questions, device=device)
    all_counts = torch.zeros(num_questions, device=device)

    for (batch_input_t, batch_mask_t, batch_labels_t, batch_q_indices_t) in tqdm(dataloader): 
        with torch.no_grad():
            logits = model(batch_input_t, attention_mask=batch_mask_t).logits  # [batch_sz, seq_len, vocab_sz]
    
        # extract logits for A/B/C/D tokens at the last position (where model should predict answer)
        preds = logits[:, -1, choice_indices].argmax(dim=-1)  # [batch_size] - get choice predictions for each permutation

        batch_accuracy_t = (preds == batch_labels_t) # [batch_size] 
        
        # accum results -- bincount exists for exactly these sort of accum ops eg. taking 
            # [0, 1, 1, 1, 4] -> [1, 3, 0, 0, 1] 
            # for us batch_q_indices_t is the question indices (in [len_filtered_dataset] = num_questions)
            # so index 1 is the "first filtered question in question_indices" 
            # which of course may be an arb index in the og dataset 
        correct_counts += torch.bincount(batch_q_indices_t, weights=batch_accuracy_t.float(), minlength=num_questions)
        all_counts += torch.bincount(batch_q_indices_t, minlength=num_questions)

    # Calculate per-question accuracies
    question_accs = (correct_counts / all_counts).cpu().numpy()

    if verbose:
        for i, acc in enumerate(question_accs):
            print(f"Question {i+1} (idx {question_indices[i]}): {acc:.4f} ({acc*100:.2f}%)")
    
    mean_acc = np.mean(question_accs)
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
    print(f"Min accuracy: {np.min(question_accs):.4f}")
    print(f"Max accuracy: {np.max(question_accs):.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLU performance")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M", 
                        help="Name of the model to evaluate")
    parser.add_argument("--num_questions", type=int, default=20,
                        help="Number of questions to evaluate")
    parser.add_argument("--num_fewshot", type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (total size = num_qs * num_perms_per_q)")
    parser.add_argument("--device", type=str, default="mps",
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
        batch_size=args.batch_size, 
        verbose=args.verbose
    )
