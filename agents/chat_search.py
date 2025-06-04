'''
This file implements an interactive chat agent that can search the web to answer factual questions.
The goal is to show the core of how to go from the (next token prediction) level of abstraction to 
(interactive agent with access to tools) level of abstraction. It's as simple as putting dialogue in a loop, 
storing past conversation in memory (which we occassionally summarize), and letting it output tool call requests in 
a structured format like <tool>args</tool> which we execute and give back to it. That's one simple view of what an agent is. 

How this file works. 
    1. User asks a factual question in the terminal
    2. Model responds, but if it's uncertain, it outputs <shallowSearchQuery>some search query</shallowSearchQuery>
    3. We parse that tag, execute the search using Google Search API (you'll need to get a key), and give the results back to the model
    4. Model can search again if needed, or output its final answer
    5. Loop continues until model stops searching and gives final response

Key abstractions:
    - Search scaffold: Parse <shallowSearchQuery> tags and execute web searches
    - Memory compression: Keep conversation history manageable by summarizing old exchanges
    - Interactive loop: Terminal interface that feels like chatting with an AI that can "Google things"

This is much more powerful than static RAG because the model can iteratively refine its searches
based on what it learns, rather than being stuck with whatever retrieval happened up front.
It's like giving the model the ability to think step-by-step through research, which is how
humans actually approach answering complex factual questions they don't immediately know.

Some learnings taking an LLM -> agent: 
    - You can feel the uplift from a stronger base model using tools better. Llama-3.1 8B would sometimes
    refuse to use search and just give a wrong answer from memory. 70B consistently used search, but wouldn't know 
    when to use shallow vs deep serach. 405B was perfect: it knew what was approppriate, when, and would be a 
    delightful factual assistant to talk to! 
    - You can see how model-agent codesign will be a thing, ie. including lots of 
    agentic data during midtraining and lots of tool-use RL will make things like 8B models aware of how/when to use 
    the tools, which will massively amplify their usefulness and autonomy. 
    - You can see why long-context is important. While no model will just die with long (eg. 20k+ tokens), often smaller 
    or poorly trained ones will reduce in quality (ie. fail to do multi-hop reasoning with information scattered around context), 
    and in agentic settings the important information will be scattered throughout context (one key sentence in an obscure 
    search result, a request early transcript, etc). The best models will be able to surface the key helpful things from context 
    and make new tool calls, etc. 

''' 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse 
from typing import List, Dict, Optional, Tuple 
from datasets import load_dataset, Dataset
from together import Together
from tqdm import tqdm 
import requests 
import re
from bs4 import BeautifulSoup
from prompts import * 

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

def compress_mem(mem: str, client: Together, model_name: str) -> str: 
    compress_user_prompt = COMPRESS_USER_PROMPT.format(
        transcript=mem, 
    )
    return api(
        compress_user_prompt, 
        client, 
        model_name=model_name,
        system_prompt=COMPRESS_SYSTEM_PROMPT, 
    ) 


def get_page(url: str) -> str: 
    res = requests.get(url)
    html = res.text 
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text() + str(f'--'*10) + '\n'

def get_search_results(query: str, top_k_shallow: int = 5, top_k_deep: int = 2, deep: bool = False) -> str: 
    # google_search_key = TODO, get one from https://developers.google.com/custom-search/v1/overview
    # search_engine_id = TODO, same as above 
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': google_search_key, 
        'q': query, 
        'cx': search_engine_id,
    }

    res = requests.get(url, params=params)
    out = ""

    if res.status_code == 200 and not deep:
        data = res.json()
        items = data.get('items', [])
        for i, item in enumerate(items[:top_k_shallow], 1):  
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No snippet')
            out += f"Result {i}: {title}\n{snippet}\n\n"
            
        print(f'Returning search info of len {len(out)} chars.')
        return out
    elif res.status_code == 200 and deep: 
        # scrape top 2 urls - this actually downloads full page content
        data = res.json()
        items = data.get('items', [])
        for i, item in enumerate(items[:top_k_deep], 1):  
            url = item.get('link', None)
            out += get_page(url)
        print(f'Returning search info of len {len(out)} chars.')
            
        return out
    else:
        print(f"Error: {res.status_code}")
        print(res.text)
        return None


def parse_search(s: str) -> str: # return tool_outputs only
    # Handle shallow search queries with correct XML-style tags
    shallow_pattern = r'<shallowSearchQuery>(.*?)</shallowSearchQuery>'
    matches = list(re.finditer(shallow_pattern, s, re.DOTALL))
    
    # process matches in reverse order to maintain string positions
    for match in reversed(matches):
        print(f'[ShallowSearching...]')
        query = match.group(1)
        start_pos = match.start()
        end_pos = match.end()

        res = get_search_results(query, deep=False)
        replacement = f'<shallowSearchOutput>{res}</shallowSearchOutput>'
        s = s[:start_pos] + replacement + s[end_pos:]
    
    # Handle deep search queries with correct XML-style tags
    deep_pattern = r'<deepSearchQuery>(.*?)</deepSearchQuery>'
    matches = list(re.finditer(deep_pattern, s, re.DOTALL))
    
    # process matches in reverse order to maintain string positions
    for match in reversed(matches):
        print(f'[DeepSearching...]')
        query = match.group(1)
        start_pos = match.start()
        end_pos = match.end()

        res = get_search_results(query, deep=True)
        replacement = f'<deepSearchOutput>{res}</deepSearchOutput>'
        s = s[:start_pos] + replacement + s[end_pos:]
    
    return s


def contains_search_call(s: str) -> bool: 
    pattern = r'<shallowSearchQuery>(.*?)</shallowSearchQuery>'
    matches_shallow = re.findall(pattern, s, re.DOTALL)
    
    pattern = r'<deepSearchQuery>(.*?)</deepSearchQuery>'
    matches_deep = re.findall(pattern, s, re.DOTALL)

    return bool(matches_shallow or matches_deep)


def back_and_forth_with_tools(
        api_prompt, 
        api_client, 
        mem,
        model_name: str,
        use_tools: bool = True,
        verbose: bool = True,
    ): 
    if use_tools:
        system_prompt = get_system_prompt(mem)
    else:
        system_prompt = "You are helpful language model assistant."
    
    res = api(api_prompt, 
                api_client, 
                model_name=model_name,
                system_prompt=system_prompt, 
            )
    
    if not use_tools:
        return res
        
    calls = 0
    # keep calling model until it stops requesting searches
    while contains_search_call(res): 
        res = parse_search(res) 

        res = api(res, 
                api_client, 
                model_name=model_name,
                system_prompt=system_prompt, 
            )
        calls +=1 
        if calls > 5: 
            if verbose:
                print(f'Tool call limit exceeded!')
            break 

    return res 

def get_system_prompt(mem: str) -> str: 
    return FULL_SYSTEM_PROMPT.format(
                transcript=mem, 
                tool_info=SEARCH_TOOL_PROMPT, 
            )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Interactive chat with AI model")

    parser.add_argument("--big", action="store_true",
                        help="Use the larger 70B model instead of 8B")
    parser.add_argument("--huge", action="store_true",
                        help="Use the larger 405B model")
    parser.add_argument("--notools", action="store_true", 
                        help="Disable search tools")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    LLAMA_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    LLAMA_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    model_name = LLAMA_405B if args.huge else (LLAMA_70B if args.big else LLAMA_8B)
    
    client = Together()
    mem = ""

    while True: 
        prompt = input("You: ")
        
        if "quit" in prompt or "exit" in prompt: 
            break 
        elif "print_mem()" in prompt: 
            print(f'Memory is length {len(mem)} character.')
            print('MEMORY: ')
            print(f'--'*10)
            print(mem)
            print(f'--'*10)
        else: 
            mem += prompt + "\n"
            res = back_and_forth_with_tools(
                prompt, 
                client, 
                mem,
                model_name=model_name,
                use_tools=not args.notools,
                verbose=args.verbose,
            ) # includes parsing tools until no tool calls found 
            if res is not None:
                mem += res + "\n"

                if args.verbose:
                    print(f'--'*10)
                    print(f'AI: {res}')
                    print(f'--'*10)
                else:
                    print(res)

                # if mem is very long, hit the api behind the scenes to summarize and compress it and use that as new mem. 
                if len(mem) > 5_000: 
                    if args.verbose:
                        print(f'[System Message: Conversation has gone long, so transcript of past messages was summarized. Proceed...]')
                    mem = compress_mem(mem, client, model_name)
            else:
                print("AI: Sorry, I encountered an error. Please try again.")
