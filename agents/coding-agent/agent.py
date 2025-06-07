import re
import argparse
import json  
from typing import List, Dict, Optional, Tuple 
import shutil 
import os 

from together import Together
from anthropic import Anthropic 

from tools.search_tool import SearchToolInput, SearchToolOutput, SearchTool 
from global_constants import AGENT_SCRATCH_DIR
from tools.tool_validation_utils import contains_tool_calls, check_and_fix_tool_calls
from memory import Memory 
from api import api 

from prompts.system_prompts import BASE_SYSTEM_PROMPT, FULL_SYSTEM_PROMPT, FINALIZER_SYSTEM_PROMPT, REFLECT_OR_FINALIZE_SYSTEM_PROMPT
from prompts.memory_prompts import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, UPDATE_GLOBAL_MEM_PROMPT
from tools.registry import ALL_TOOL_PROMPTS, ALL_TOOLS


# get the last occurrence
def get_final_output(res: str) -> str: 
    pattern = r'<output>(.*?)</output>'
    matches = re.findall(pattern, res, re.DOTALL)
    if matches:
        return matches[-1].strip()  # take the last output tag in case there are multiple
    return "[RETURNING EMPTY STRING BECAUSE RES DIDNT CONTAIN OUTPUT]"

def finalize(res: str, memory: Memory, client: Together, model_name: str) -> str: 
    # final cleanup pass to extract clean output from the tool interaction mess
    finalizer_prompt = FINALIZER_SYSTEM_PROMPT.format(
        transcript=memory.local_tool_memory, 
    ) 
    out = api(res, 
            client, 
            model_name=model_name,
            system_prompt=finalizer_prompt, 
        ) 
    
    # update global memory with a summary of what happened during tool use
    memory.update_global_mem(model_name)
    return get_final_output(out)

def back_and_forth_with_tools(
        user_prompt: str, 
        client: Together,
        memory: Memory,
        model_name: str,
        use_tools: bool = True,
        verbose: bool = True,
        MAX_NUM_TOOL_CALLS: int = 100,  # prevent infinite loops
    ): 
    if use_tools:
        system_prompt = get_system_prompt(memory.global_mem)
    else:
        system_prompt = "You are helpful language model assistant."
    
    # initial response from the model
    res = api(user_prompt, 
                client, 
                model_name=model_name,
                system_prompt=system_prompt, 
            ) 
    
    # if no tools needed or tools disabled, return immediately
    if not use_tools or not contains_tool_calls(res):
        return get_final_output(res)
        
    # initialize tool memory with global context
    memory.local_tool_memory = memory.global_mem + "\n \n" + "Tool Interaction Memory: \n "
    res_after_tools = res

    calls = 0
    # keep going until model stops requesting tools or gives final output
    while contains_tool_calls(res_after_tools) and (not '<output>' in res_after_tools): 
        if verbose:
            print(f'--'*20)
            print(f'Res after tools: ')
            print(res_after_tools)
            print(f'--'*20)
        
        # clean up malformed json in tool calls (newlines, quotes, etc)
        res_after_tools = check_and_fix_tool_calls(res_after_tools) 
        
        # execute all tool requests in the response
        for tool in ALL_TOOLS:
            res_after_tools = tool.call(res_after_tools)  # replaces <tool_request> with <tool_output>
        
        # accumulate all tool interactions for context
        memory.local_tool_memory += res_after_tools + "\n\n\n" 
        
        # ask model to reflect on tool results and decide next steps
        reflect_or_finalize_prompt = REFLECT_OR_FINALIZE_SYSTEM_PROMPT.format(
            transcript=memory.local_tool_memory,
            tool_info="\n".join([t.prompt for t in ALL_TOOLS])
        )

        res_after_tools = api(
            res_after_tools, 
            client, 
            model_name=model_name,
            system_prompt=reflect_or_finalize_prompt,
        )
        
        if verbose: 
            print(res_after_tools)

        # model decided it's done, extract final answer
        if not contains_tool_calls(res_after_tools):  
            return finalize(res_after_tools, memory, client, model_name)

        calls += 1
        if calls > MAX_NUM_TOOL_CALLS: 
            print(f'WARNING: Max tool calls exceeded, breaking out to finalizer...')
            return finalize(res_after_tools, memory, client, model_name)

    return finalize(
        res, 
        memory, 
        client, 
        model_name, 
    )

def get_system_prompt(global_mem: str) -> str: 
    # inject tool descriptions into system prompt so model knows what's available
    tool_info_prompt = "\n".join([t.prompt for t in ALL_TOOLS])
    
    return FULL_SYSTEM_PROMPT.format(
                transcript=global_mem, 
                tool_info=tool_info_prompt, 
            )

def wipe_agent_scratch(): 
    # clean slate for each session to avoid file conflicts
    if os.path.exists(AGENT_SCRATCH_DIR):
        shutil.rmtree(AGENT_SCRATCH_DIR)
    
    os.makedirs(AGENT_SCRATCH_DIR, exist_ok=True) 

    print(f"Wiped and recreated {AGENT_SCRATCH_DIR}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Interactive chat with AI model")

    parser.add_argument("--small", action="store_true",
                        help="Use the 8B model (Meta-Llama-3.1-8B-Instruct-Turbo)")
    parser.add_argument("--deepseek", action="store_true",
                        help="Use DeepSeek V3")
    parser.add_argument("--huge", action="store_true",
                        help="Use the 405B model (Meta-Llama-3.1-405B-Instruct-Turbo)")
    parser.add_argument("--anthropic", action="store_true",
                        help="Use Claude")
    parser.add_argument("--notools", action="store_true", 
                        help="Disable search tools")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # default to the balanced 70B model
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo" 
    if args.small:
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    elif args.deepseek:
        model_name = "deepseek-ai/DeepSeek-V3"
    elif args.huge:
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    
    client = Together()

    if args.anthropic:
        client = Anthropic() 

    memory = Memory(client, "", model_name)

    wipe_agent_scratch()  # start fresh each time

    while True: 
        prompt = input("You: ")
        
        if prompt == "quit" or prompt == "exit": 
            break 
        elif "print_mem()" in prompt: 
            memory.show()
        else: 
            # add user message to conversation history
            memory.global_mem += "\nUser: " + prompt + "\n"
            res = back_and_forth_with_tools(
                prompt, 
                client, 
                memory,
                model_name=model_name,
                use_tools=not args.notools,
                verbose=args.verbose,
            ) 
            if res is not None:
                print(f"AI: {res}")
                # add ai response to conversation history
                memory.global_mem += "\n AI:" + res + "\n"

                # compress memory when it gets too long to avoid context limits
                if len(memory.global_mem) > 200_000:  # roughly 50k tokens
                    if args.verbose:
                        print(f'[System Message: Conversation has gone long, so transcript of past messages was summarized. Proceed...]')
                    memory.compress_global_mem(memory.global_mem, client, model_name)
            else:
                print("[ERROR: response from back_and_forth was None]")
