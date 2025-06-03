'''
(https://arxiv.org/pdf/2411.04368) SimpleQA: Measuring short-form factuality in large language models

This introduces the notion of using an LLM judge as an "autograder" for an eval that's sometimes
ambiguous to grade. It might seem surprising a factual eval is "hard to grade" or "sometimes ambiguous" 
but consider the question "When was Anne Frank born?" Would an answer of "1929" (the correct year) -- be correct
without day/month? If asked how many people in America in 2024, would an answer of "around 330 million" be correct, 
when the real number is 332.18 million? An LLM grader will produce outputs most humans would agree with on these 
issues, which is why we use it. 

We run inference on the eval prompts locally on our student (a small model we want to eval), 
send responses to an LLM API (eg. Llama-3.3 70B hosted on Together/OpenRouter) and get a grade
for the student's answer, and accumulate results, printing them at the end. Prompts are given in 
./prompts/ folder, taken verbatim from the paper where applicable. 
''' 
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from together import Together
from prompts.simpleqa_templates import GRADER_TEMPLATE, STUDENT_TEMPLATE
import argparse 
from tqdm import tqdm 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # so tqdm + tokenizer don't yell at us 

def grade(
        user_prompt: str, 
        client: Together,
        teacher_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
        system_prompt: str = "You are an AI judge evaluating the correctness of answers to factual questions.", 
    ):
    try:
        response = client.chat.completions.create(
            model=teacher_name,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt, 
                },
                {
                    "role": "user",
                    "content": user_prompt, 
                }
            ]
        )
        
        return response
        
    except Exception as e:
        print(f"API request failed: {e}")
        return None

def generate_and_grade(
        q, 
        student: AutoModelForCausalLM, 
        teacher_name: str, 
        tokenizer: AutoTokenizer, 
        client: Together, 
        device: torch.device = torch.device("mps"), 
        verbose: bool = False, 
    ):
    
    
    student_prompt = STUDENT_TEMPLATE.format(
        question=q["problem"], 
    )

    if verbose:
        print(f'STUDENT PROMPT: {student_prompt}')
        print(f'--'*20)

    # tokenize student prompt
    inputs = tokenizer.encode(
        student_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
    )

    input_ids, mask = inputs.input_ids.to(device), inputs.mask.to(device) # attn mask makes sure we don't attend to pad/eos tokens
    # for us this isn't a problem single we're doign bsz=1 inference so we don't need to pad, but a mature implemenation here 
    # would do high-batch inference of many questions' prompts in parallel, in which case we'd need to pad generations
    # to the length of the longest generation and an attention mask would be necessary (see eval_mmlu.py for logic like this)
    student_answer_ids = student.generate(input_ids, pad_token_id=tokenizer.eos_token_id, attention_mask=mask, max_new_tokens=128) # always returns a batch, so our text is in [0]
    student_answer_text = tokenizer.decode(student_answer_ids[0], skip_special_tokens=True) # skip ensures no <bos> or <unk> or <pad> appears in text 

    if verbose:
        print(f'STUDENT ANSWER: {student_answer_text}')
        print(f'--'*20)

    grader_prompt = GRADER_TEMPLATE.format(
        question=q["problem"],
        target=q["answer"],
        predicted_answer=student_answer_text, 
    )

    teacher_response = grade(grader_prompt, client=client, teacher_name=teacher_name)
    
    if verbose:
        print(f'GRADER RESPONSE: ', teacher_response.choices[0].message.content)
        print(f'--'*20)

    return teacher_response.choices[0].message.content 

if __name__ == "__main__": # hit together API for LLM as a judge grading of a response 
    # a more sophisticated implementation would hit the LLM judge API asynchronously 
        # and batch generation over student while that is happening
    parser = argparse.ArgumentParser(description="Evaluate SimpleQA using LLM as judge")
    parser.add_argument("--api_key", type=str, default="TODO, set in your env", help="Together API key")
    parser.add_argument("--dataset_name", type=str, default="basicv8vc/SimpleQA", help="Dataset name")
    # the student is the model whose factuality we are evaluating, the teacher is the LLM judge grading its responses
        # SimpleQA is challenging for even frontier models, so we shouldn't expect too much from small/tiny models. 
        # Something like Llama3.1 8B would be a good student here, but I can't use that for generation even on a H100 lol
    parser.add_argument("--student_name", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct", help="Student model name") # or eg. unsloth/gemma-3-4b-it-unsloth-bnb-4bit
    parser.add_argument("--teacher_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", help="Teacher model name")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--num_questions", type=int, default=10, help="Number of questions")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    client = Together(api_key=args.api_key) # feel free to replace with eg. OpenAI or OpenRouter, etc
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.student_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.student_name)

    # set pad token and create attn mask 
    if tokenizer.pad_token is None: 
        tokenier.pad_token = tokenizer.eos_token
        tokenier.pad_token_id = tokenizer.eos_token_id
    
    dataset = load_dataset(args.dataset_name, split="test")
    nright, nidk, nwrong = 0, 0, 0 # correspond to [A, C, B] in grading schema

    for i in tqdm(range(args.num_questions)): 
        first_q = dataset[i]
        resp = generate_and_grade(
            first_q,
            model,
            args.teacher_name,
            tokenizer,
            client,
            device,
            args.verbose
        ).strip()
        if resp == "A": # correct
            nright += 1
        elif resp == "B": # incorrect
            nwrong += 1
        elif resp == "C": # not attempted
            nidk += 1
        
        if args.verbose:
            print(f'Question {i+1}/{args.num_questions} - Correct: {nright}, Incorrect: {nwrong}, Not Attempted: {nidk}')
            print(f'--'*20)
    
    # Final results
    print(f'\nFinal Results:')
    print(f'Correct: {nright}')
    print(f'Incorrect: {nwrong}')
    print(f'Not Attempted: {nidk}')
    print(f'Total: {nright + nwrong + nidk}')
    
    # calculate probabilities to summary 
    total = nright + nwrong + nidk
    attempted = nright + nwrong
    
    p_right = nright / total if total > 0 else 0
    p_right_attempted = nright / attempted if attempted > 0 else 0
    
    print(f'P(right) = {p_right:.3f}')
    print(f'P(right|attempted) = {p_right_attempted:.3f}')
