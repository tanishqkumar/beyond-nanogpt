'''
This file implements a minimal RAG app: we ask a model factual questions, it does poorly, then we use 
an embedding+reranker to put the literal answers in context, ask again, and it does well (no wonder). 
    --> This introduces how embedding/reranking models work with local files and fit into a broader eval loop. 
    A more sophisticated version of this is in `agents/train_search_agent.py` which instead uses a 
    small loop/scaffold to have the model search the answer to the question on the internet 
    instead of just being given it in the prompt. 

Worklog: We start with `evals/eval_simpleqa.py` code, then change the following. 
    - Student inference also via API rather than locally (we use Llama 3.1 8B) so we get signal on a hard eval 
    
    - Add asyncio logic in API requests for 10x faster evals (since Together doesn't support batch inference)
        --> doing this took eval time for 100 questions from ~4 mins down to ~4 seconds
        --> which shows we're heavily IO (network/API) bound, since we're not doing any work ourselves

    - Benchmark student before, expect poor performance
        --> Final Results without RAG [Llama-3.1 8B student, DeepSeek V3 teacher]:
                Correct: 14
                Incorrect: 202
                Not Attempted: 34
                Total: 250
                P(right) = 0.056
                P(right|attempted) = 0.065

    - Get a simple off-the-shelf embedding and reranking model 
        --> Embedding = Biencoder, SentenceTransformer('all-MiniLM-L6-v2')
        --> Reranking = Crossencoder, CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    - Implement simple RAG to get relevant answer for the question from (dataset["questions"], dataset["answers"])

    - Pass RAG-selected answer to each question in as context during API request. 

    - Expect to see much better performance -- and we do!
        - This confirms the minimal RAG pipeline is working 
        --> Final Results with RAG [Llama-3.1 8B student, DeepSeek V3 teacher]:
                Correct: 72
                Incorrect: 153
                Not Attempted: 25
                Total: 250
                P(right) = 0.288
                P(right|attempted) = 0.320
        --> Why not perfect performance? 
            --> The selected answer for question N should ideally always be answers[N], 
            but in practice it's not because at that point we're bottlenecked by the embedding/reranking models. 
            For instance, while "What is the capital of France" will always have high similarity to 
            "Paris is the capital of France," our (and many other) QA datasets do not have answers like the latter, 
            but just "Paris" -- and RAG methods are known to struggle to extract semantic information when there aren't 
            many words to work with in the answer. Not to mention that it's hard for a RAG method to 
            correctly match (ans) to (question) in semantic similarity for obscure questions
            (which is the entirety of SimpleQA) requires sort of knowing the answer to the obscure question 
            already, for instance if you're asked about who wrote an obscure manuscript and are given 200 possible plausible-sounding names
            (as the retriever) -- you can't really rank them well if you don't know yourself who wrote the manuscript!

''' 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse 
import asyncio 
import aiohttp 
from typing import List, Dict, Optional, Tuple 

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from together import Together
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from tqdm import tqdm 

from prompts.simpleqa_templates import GRADER_TEMPLATE, STUDENT_TEMPLATE, STUDENT_RAG_TEMPLATE

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def api(
        user_prompt: str, 
        client: Together,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
        system_prompt: str = "You are helpful language model assistant, designed to produce factual and correct responses to queries.", 
    ):
    try:
        response = client.chat.completions.create(
            model=model_name,
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
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        return None


def grade(
        user_prompt: str, 
        client: Together,
        teacher_name: str = "deepseek-ai/DeepSeek-V3", 
        system_prompt: str = "You are an AI judge evaluating the correctness of answers to factual questions.", 
    ):
    return api(user_prompt, client, teacher_name, system_prompt)


async def generate_and_grade(
        dataset, 
        q, 
        q_idx, 
        client: Together, 
        student_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
        teacher_name: str = "deepseek-ai/DeepSeek-V3", 
        verbose: bool = False, 
        student_prompt: str = "",
    ):
    
    if not student_prompt:  # no rag template 
        student_prompt = STUDENT_TEMPLATE.format(question=q["problem"])
    
    # use API for student inference instead of local model
    student_system_prompt = "You are a helpful language model responding to factual queries concisely and correctly."
    student_answer_text = await asyncio.to_thread(
        api,
        student_prompt,
        client,
        student_name,
        student_system_prompt, 
    )
    
    if student_answer_text is None:
        raise Exception("Error: Student API call failed in generate_and_grade...")
    
    grader_prompt = GRADER_TEMPLATE.format(
        question=q["problem"],
        target=q["answer"],
        predicted_answer=student_answer_text, 
    )

    teacher_response_text = await asyncio.to_thread(
        grade, grader_prompt, client, teacher_name,
    )
    
    if teacher_response_text is None:
        raise Exception("Error: Grader API call failed in generate_and_grade...")

    return teacher_response_text


async def eval(
        dataset, 
        q_indices, 
        questions, 
        api_key: str,
        student_name: str,
        teacher_name: str,
        num_questions: int,
        verbose: bool = False, 
        rag_student_prompts: List[str] = [],
    ):

    client = Together(api_key=api_key)  # feel free to replace with eg. OpenAI or OpenRouter, etc
    nright, nwrong, nidk = 0, 0, 0  # correspond to [A, B, C] in grading schema
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, q in enumerate(questions):
            student_prompt = "" if not rag_student_prompts else rag_student_prompts[i]
            tasks.append(generate_and_grade(dataset, q, i, client, student_name, teacher_name, verbose, student_prompt))
        responses = await asyncio.gather(*tasks)

    for i, resp in enumerate(responses): 
        if resp is None:
            raise Exception("Error: response from generate_and_grade was None...")
        resp = resp.strip()
        if resp == "A":  # correct
            nright += 1
            if verbose: 
                print(f"Question {i+1}: Got it right!")
        elif resp == "B":  # incorrect
            nwrong += 1
            if verbose: 
                print(f"Question {i+1}: Got it wrong")
        elif resp == "C":  # not attempted
            nidk += 1
            if verbose: 
                print(f"Question {i+1}: Did not attempt")

    return nright, nidk, nwrong
    

def get_sims(
        embedder: nn.Module, 
        dataset: Dataset, 
        k: int = 20,
        sims_path: str = "sims_cache.pkl", 
        device: torch.device = torch.device("mps"),
    ) -> torch.Tensor: 
    
    print('Precomputing sims matrix, may take a second...')
    questions, answers = dataset["problem"], dataset["answer"]
    q_embs = embedder.encode(questions, convert_to_tensor=True, device=device) 
    q_embs = F.normalize(q_embs, p=2, dim=1)
    a_embs = embedder.encode(answers, convert_to_tensor=True, device=device) 
    a_embs = F.normalize(a_embs, p=2, dim=1)
    sims = torch.einsum('qd,ad->qa', q_embs, a_embs)
    
    return sims  # [num_qs, num_as] similarity matrix that will immediately give topk embeddings 


def get_reranked_topk(
        topk_embeds: torch.Tensor,  # [num_qs, embed_k answer indices]
        dataset: Dataset, 
        reranker: nn.Module, 
        k: int = 5, 
        device: torch.device = torch.device("cuda"), 
    ) -> torch.Tensor:  # output is [num_qs, rerank_k] indices
    # reranker takes in List[(q, a)] and outputs List[score] which we'll need to topk again 
    num_qs, embed_k = topk_embeds.shape
    questions, answers = dataset["problem"], dataset["answer"]
    
    out = torch.empty((num_qs, k), dtype=torch.long, device=device)
    for i in range(num_qs): 
        q = questions[i]
        top_answer_indices = topk_embeds[i]
        top_answers = [answers[idx] for idx in top_answer_indices]

        qas = [(q, top_answers[j]) for j in range(embed_k)]
        scores_i = reranker.predict(qas)
        temp = torch.topk(torch.tensor(scores_i, device=device), k).indices
        out[i] = torch.tensor([topk_embeds[i][t] for t in temp], device=device)  # temp is indices in basis of topk_embeds, we want in basis of num_answers 

    return out


def get_rag_student_prompts(
        embedder: nn.Module, 
        reranker: nn.Module, 
        dataset: Dataset, 
        embed_k: int = 20, 
        rerank_k: int = 5, 
        device: torch.device = torch.device("mps"),
    ) -> List[str]: 

    # get all embedding sims 
    sims = get_sims(embedder, dataset, embed_k, device=device)  # [num_qs, num_as]
    # use to get embedding topk for all questions 
    topk_embeds = torch.topk(sims, embed_k).indices  # [num_qs, embed_k]
    
    topk_reranks = get_reranked_topk(topk_embeds, dataset, reranker, rerank_k, device=device)  # [num_questions, rerank_k]
    # now get the answers for each question and cat them 
    questions = dataset["problem"]
    answers = dataset["answer"]
    out = []
    # construct contexts list of len [num_questions] where is a catted string of the relevant lookups 
    for i, q in enumerate(questions): 
        # make contexts_i 
        top_indices = topk_reranks[i].tolist()
        top_answer_list = [answers[idx] for idx in top_indices]
        context = '\n' + '\n'.join(top_answer_list)
        template = STUDENT_RAG_TEMPLATE.format(
            context=context, 
            question=q,
        )
        out.append(template)
    # plug into student prompt template, return a string 
    return out


def print_results(nright: int, nidk: int, nwrong: int, before=True): 
    if before: 
        print('\nFinal Results without RAG:')
    else: 
        print('\nFinal Results with RAG:')

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


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate SimpleQA using LLM as judge")
    # replace this with your fav LLM provider, this code is openai-compatible so it should will just work if you change the client
    parser.add_argument("--api_key", type=str, default=os.getenv("TOGETHER_API_KEY"), help="Together API key")
    parser.add_argument("--dataset_name", type=str, default="basicv8vc/SimpleQA", help="Dataset name")
    parser.add_argument("--student_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", help="Student model name") 
    parser.add_argument("--teacher_name", type=str, default="deepseek-ai/DeepSeek-V3", help="Teacher model name")
    parser.add_argument("--num_questions", type=int, default=250, help="Number of questions")
    parser.add_argument("--verbose", action="store_true", help="More logging") 
    
    args = parser.parse_args()
    device = torch.device("mps")  # TODO make general before pushing 

    if args.verbose:
        print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="test")
    q_indices = torch.randint(0, len(dataset), (args.num_questions,)).tolist()
    dataset = dataset.select(q_indices)

    if args.verbose:
        print(f"Selected {args.num_questions} questions from dataset")
        print(f"Using device: {device}")
        print(f"Student model: {args.student_name}")
        print(f"Teacher model: {args.teacher_name}")
        print("\nLoading embedding and reranking models...")

    # Load models once
    embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').to(device)

    if args.verbose:
        print("Running evaluation without RAG...")
 
    nright, nidk, nwrong = asyncio.run(eval(
        dataset, 
        q_indices, 
        dataset,
        api_key=args.api_key,
        student_name=args.student_name,
        teacher_name=args.teacher_name,
        num_questions=args.num_questions,
        verbose=args.verbose, 
        rag_student_prompts=[], 
    ))
    
    # Final results
    print_results(nright, nidk, nwrong, before=True)

    ### RAG 

    if args.verbose:
        print("\n" + "="*50)
        print("Starting RAG evaluation...")
        print("Generating RAG student prompts...")

    # precompute all student system prompts, pass into eval 
    rag_student_prompts = get_rag_student_prompts(
        embedder, 
        reranker, 
        dataset, 
    )

    if args.verbose:
        print(f"Generated {len(rag_student_prompts)} RAG prompts")
        print("Running evaluation with RAG...")

    # repeat eval with rag=True 
    nright_rag, nidk_rag, nwrong_rag = asyncio.run(eval(
        dataset,
        q_indices,
        dataset,
        api_key=args.api_key,
        student_name=args.student_name,
        teacher_name=args.teacher_name,
        num_questions=args.num_questions,
        verbose=args.verbose, 
        rag_student_prompts=rag_student_prompts, 
    ))

    # Final results with RAG
    print_results(nright_rag, nidk_rag, nwrong_rag, before=False)