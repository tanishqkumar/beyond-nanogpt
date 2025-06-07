from pydantic import BaseModel 
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union 

from together import Together
from anthropic import Anthropic

from tools.search_tool import SearchToolInput, SearchToolOutput, SearchTool 
from api import api 

from prompts.system_prompts import BASE_SYSTEM_PROMPT, FULL_SYSTEM_PROMPT, FINALIZER_SYSTEM_PROMPT, REFLECT_OR_FINALIZE_SYSTEM_PROMPT
from prompts.memory_prompts import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, UPDATE_GLOBAL_MEM_PROMPT
from prompts.tool_prompts import SEARCH_TOOL_PROMPT
from tools.registry import ALL_TOOL_PROMPTS, ALL_TOOLS


class Memory: 
    def __init__(self, client: Union[Together, Anthropic], init_mem: str = "", api_model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"): 
        self.global_mem = init_mem  # persistent conversation history
        self.local_tool_memory = ""  # temporary memory for current tool interaction session
        self.client = client 
        self.model_name = api_model_name


    def compress_global_mem(self, global_mem: str, client: Union[Together, Anthropic], model_name: str): 
        # use llm to summarize long conversations to stay under context limits
        compress_user_prompt = COMPRESS_USER_PROMPT.format(
            transcript=global_mem, 
        )
        self.global_mem = api(
            compress_user_prompt, 
            client, 
            model_name=model_name,
            system_prompt=COMPRESS_SYSTEM_PROMPT, 
        ) 

    def update_global_mem(self, model_name: str): 
        # extract key insights from tool interactions and add to permanent memory
        update_mem_sysprompt = UPDATE_GLOBAL_MEM_PROMPT.format(
            tool_memory=self.local_tool_memory,     
        )
        local_tool_memory_summary  = api(
            self.global_mem, 
            self.client, 
            model_name, 
            system_prompt=update_mem_sysprompt, 
        )

        # append tool session summary to global memory with special tags
        self.global_mem += "\n\n <internal_tool_use_summary>" + local_tool_memory_summary + "</internal_tool_use_summary> \n\n"
        self.local_tool_memory = self.global_mem  # reset local memory to global state

    def show(self): 
        print(f'Memory is length {len(self.global_mem)} characters.')
        print('MEMORY:')
        print('--' * 20)
        print(self.global_mem)
        print('--' * 20)