from pydantic import BaseModel # for data and types
from abc import ABC, abstractmethod # for behavior/methods
from typing import List, Union, Optional, Dict, Any, Tuple
import re 
import json  


class BaseTool(ABC): 
    def __init__(self, name: str, description: str, prompt: str, input_type: type): 
        self.name = name 
        self.description = description 
        self.prompt = prompt 
        self.input_type = input_type  # pydantic model for validation

    @abstractmethod
    def _execute(self, input: BaseModel) -> BaseModel: 
        pass 

    # parsing logic is shared across all tools
    def _parse(self, s: str) -> List[Tuple[BaseModel, int, int]]: 
        # find all tool_request blocks in the text
        pattern = rf'<tool_request>(.*?)</tool_request>'
        matches = list(re.finditer(pattern, s, re.DOTALL))
        
        out = []

        if not matches: 
            return None 

        # process matches in reverse order to avoid index shifting during replacement
        for match in reversed(matches): 
            try: 
                # extract json from between the xml tags
                inner = match.group(1).strip()
                json_data = json.loads(inner)

                # skip if this tool call isn't meant for us
                if json_data.get("tool_name") != self.name: continue 

                # record string positions for later replacement
                start_pos = match.start()
                end_pos = match.end()
                # validate and create input object
                input_obj = self.input_type(**json_data)
                out.append((input_obj, start_pos, end_pos))
            except: 
                print(f'FAILED PARSING JSON INTO OBJECT FOR {self.name}, json={match.group()}')
                continue 
        return out 

    def call(self, s: str) -> str: 
        # keep processing until no more tool calls for this tool
        while True:
            res = self._parse(s)
            if not res:  # no more tool calls for this tool
                break

            for parsed in res: 
                if parsed is None: 
                    print(f'WARNING: Failed parsing tool call query from LLM in SearchTool.call()...')
                    continue 
                else: 
                    tool_input, start, end = parsed
                    print(f'[Calling tool: {self.name}...]')
                    tool_output_json = self._execute(tool_input).model_dump()
                    output_s = json.dumps(tool_output_json)
                    # wrap output with metadata for model to understand what happened
                    wrapped_output_s = f"<tool_output name='{self.name}' request='{json.dumps(tool_input.model_dump())}'>" + output_s + f"</tool_output>"
                    # replace the tool_request with tool_output in the original string
                    s = s[:start] + wrapped_output_s + s[end:]

        return s